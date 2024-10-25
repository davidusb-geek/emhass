#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import copy
import pathlib
import bz2
import pickle as cPickle
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import pulp as plp
from pulp import PULP_CBC_CMD, COIN_CMD, GLPK_CMD
from math import ceil


class Optimization:
    r"""
    Optimize the deferrable load and battery energy dispatch problem using \ 
    the linear programming optimization technique. All equipement equations, \
    including the battery equations are hence transformed in a linear form.
    
    This class methods are:

    - perform_optimization

    - perform_perfect_forecast_optim

    - perform_dayahead_forecast_optim
    
    - perform_naive_mpc_optim
    
    """

    def __init__(self, retrieve_hass_conf: dict, optim_conf: dict, plant_conf: dict, 
                 var_load_cost: str, var_prod_price: str, 
                 costfun: str, emhass_conf: dict, logger: logging.Logger, 
                 opt_time_delta: Optional[int] = 24) -> None:
        r"""
        Define constructor for Optimization class.
        
        :param retrieve_hass_conf: Configuration parameters used to retrieve data \
            from hass
        :type retrieve_hass_conf: dict
        :param optim_conf: Configuration parameters used for the optimization task
        :type optim_conf: dict
        :param plant_conf: Configuration parameters used to model the electrical \
            system: PV production, battery, etc.
        :type plant_conf: dict
        :param var_load_cost: The column name for the unit load cost.
        :type var_load_cost: str
        :param var_prod_price: The column name for the unit power production price.
        :type var_prod_price: str
        :param costfun: The type of cost function to use for optimization problem
        :type costfun: str
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging object
        :param opt_time_delta: The number of hours to optimize. If days_list has \
            more than one day then the optimization will be peformed by chunks of \
            opt_time_delta periods, defaults to 24
        :type opt_time_delta: float, optional
        
        """
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.freq = self.retrieve_hass_conf['optimization_time_step']
        self.time_zone = self.retrieve_hass_conf['time_zone']
        self.timeStep = self.freq.seconds/3600 # in hours
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours") # The period of optimization
        self.var_PV = self.retrieve_hass_conf['sensor_power_photovoltaics']
        self.var_load = self.retrieve_hass_conf['sensor_power_load_no_var_loads']
        self.var_load_new = self.var_load+'_positive'
        self.costfun = costfun
        # self.emhass_conf = emhass_conf
        self.logger = logger
        self.var_load_cost = var_load_cost
        self.var_prod_price = var_prod_price
        self.optim_status = None
        if 'lp_solver' in optim_conf.keys():
            self.lp_solver = optim_conf['lp_solver']
        else:
            self.lp_solver = 'default'
        if 'lp_solver_path' in optim_conf.keys():
            self.lp_solver_path = optim_conf['lp_solver_path']
        else:
            self.lp_solver_path = 'empty'
        if self.lp_solver != 'COIN_CMD' and self.lp_solver_path != 'empty':
            self.logger.error("Use COIN_CMD solver name if you want to set a path for the LP solver")
        if self.lp_solver == 'COIN_CMD' and self.lp_solver_path == 'empty': #if COIN_CMD but lp_solver_path is empty
            self.logger.warning("lp_solver=COIN_CMD but lp_solver_path=empty, attempting to use lp_solver_path=/usr/bin/cbc")
            self.lp_solver_path = '/usr/bin/cbc'  

    def perform_optimization(self, data_opt: pd.DataFrame, P_PV: np.array, P_load: np.array, 
                             unit_load_cost: np.array, unit_prod_price: np.array,
                             soc_init: Optional[float] = None, soc_final: Optional[float] = None,
                             def_total_hours: Optional[list] = None, 
                             def_start_timestep: Optional[list] = None,
                             def_end_timestep: Optional[list] = None,
                             debug: Optional[bool] = False) -> pd.DataFrame:
        r"""
        Perform the actual optimization using linear programming (LP).
        
        :param data_opt: A DataFrame containing the input data. The results of the \
            optimization will be appended (decision variables, cost function values, etc)
        :type data_opt: pd.DataFrame
        :param P_PV: The photovoltaic power values. This can be real historical \
            values or forecasted values.
        :type P_PV: numpy.array
        :param P_load: The load power consumption values
        :type P_load: np.array
        :param unit_load_cost: The cost of power consumption for each unit of time. \
            This is the cost of the energy from the utility in a vector sampled \
            at the fixed freq value
        :type unit_load_cost: np.array
        :param unit_prod_price: The price of power injected to the grid each unit of time. \
            This is the price of the energy injected to the utility in a vector \
            sampled at the fixed freq value.
        :type unit_prod_price: np.array
        :param soc_init: The initial battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_init: float
        :param soc_final: The final battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_final: 
        :param def_total_hours: The functioning hours for this iteration for each deferrable load. \
            (For continuous deferrable loads: functioning hours at nominal power)
        :type def_total_hours: list
        :param def_start_timestep: The timestep as from which each deferrable load is allowed to operate.
        :type def_start_timestep: list
        :param def_end_timestep: The timestep before which each deferrable load should operate.
        :type def_end_timestep: list
        :return: The input DataFrame with all the different results from the \
            optimization appended
        :rtype: pd.DataFrame

        """
        # Prepare some data in the case of a battery
        if self.optim_conf['set_use_battery']:
            if soc_init is None:
                if soc_final is not None:
                    soc_init = soc_final
                else:
                    soc_init = self.plant_conf['battery_target_state_of_charge']
            if soc_final is None:
                if soc_init is not None:
                    soc_final = soc_init
                else:
                    soc_final = self.plant_conf['battery_target_state_of_charge']
        if def_total_hours is None:
            def_total_hours = self.optim_conf['operating_hours_of_each_deferrable_load']
        if def_start_timestep is None:
            def_start_timestep = self.optim_conf['start_timesteps_of_each_deferrable_load']
        if def_end_timestep is None:
            def_end_timestep = self.optim_conf['end_timesteps_of_each_deferrable_load']
        type_self_conso = 'bigm' # maxmin

        #### The LP problem using Pulp ####
        opt_model = plp.LpProblem("LP_Model", plp.LpMaximize)

        n = len(data_opt.index)
        set_I = range(n)
        M = 10e10

        ## Add decision variables
        P_grid_neg  = {(i):plp.LpVariable(cat='Continuous',
                                          lowBound=-self.plant_conf['maximum_power_to_grid'], upBound=0,
                                          name="P_grid_neg{}".format(i)) for i in set_I}
        P_grid_pos  = {(i):plp.LpVariable(cat='Continuous',
                                          lowBound=0, upBound=self.plant_conf['maximum_power_from_grid'],
                                          name="P_grid_pos{}".format(i)) for i in set_I}
        P_deferrable = []
        P_def_bin1 = []
        for k in range(self.optim_conf['number_of_deferrable_loads']):
            if type(self.optim_conf['nominal_power_of_deferrable_loads'][k]) == list:
                upBound = np.max(self.optim_conf['nominal_power_of_deferrable_loads'][k])
            else:
                upBound = self.optim_conf['nominal_power_of_deferrable_loads'][k]
            if self.optim_conf['treat_deferrable_load_as_semi_cont'][k]:
                P_deferrable.append({(i):plp.LpVariable(cat='Continuous',
                                                        name="P_deferrable{}_{}".format(k, i)) for i in set_I})
            else:
                P_deferrable.append({(i):plp.LpVariable(cat='Continuous',
                                                        lowBound=0, upBound=upBound,
                                                        name="P_deferrable{}_{}".format(k, i)) for i in set_I})
            P_def_bin1.append({(i):plp.LpVariable(cat='Binary',
                                                  name="P_def{}_bin1_{}".format(k, i)) for i in set_I})
        P_def_start = []
        P_def_bin2 = []
        for k in range(self.optim_conf['number_of_deferrable_loads']):
            P_def_start.append({(i):plp.LpVariable(cat='Binary',
                                                   name="P_def{}_start_{}".format(k, i)) for i in set_I})
            P_def_bin2.append({(i):plp.LpVariable(cat='Binary',
                                                  name="P_def{}_bin2_{}".format(k, i)) for i in set_I})
        D  = {(i):plp.LpVariable(cat='Binary',
                                 name="D_{}".format(i)) for i in set_I}
        E  = {(i):plp.LpVariable(cat='Binary',
                                 name="E_{}".format(i)) for i in set_I}
        if self.optim_conf['set_use_battery']:
            P_sto_pos  = {(i):plp.LpVariable(cat='Continuous', 
                                             lowBound=0, upBound=self.plant_conf['battery_discharge_power_max'],
                                             name="P_sto_pos_{0}".format(i)) for i in set_I}
            P_sto_neg  = {(i):plp.LpVariable(cat='Continuous', 
                                             lowBound=-self.plant_conf['battery_charge_power_max'], upBound=0,
                                             name="P_sto_neg_{0}".format(i)) for i in set_I}
        else:
            P_sto_pos  = {(i):i*0 for i in set_I}
            P_sto_neg  = {(i):i*0 for i in set_I}

        if self.costfun == 'self-consumption':
            SC  = {(i):plp.LpVariable(cat='Continuous',
                                      name="SC_{}".format(i)) for i in set_I}
        if self.plant_conf['inverter_is_hybrid']:
            P_hybrid_inverter = {(i):plp.LpVariable(cat='Continuous',
                                                    name="P_hybrid_inverter{}".format(i)) for i in set_I}
        P_PV_curtailment = {(i):plp.LpVariable(cat='Continuous', lowBound=0,
                                               name="P_PV_curtailment{}".format(i)) for i in set_I}
        
        ## Define objective
        P_def_sum= []
        for i in set_I:
            P_def_sum.append(plp.lpSum(P_deferrable[k][i] for k in range(self.optim_conf['number_of_deferrable_loads'])))
        if self.costfun == 'profit':
            if self.optim_conf['set_total_pv_sell']:
                objective = plp.lpSum(-0.001*self.timeStep*(unit_load_cost[i]*(P_load[i] + P_def_sum[i]) + \
                    unit_prod_price[i]*P_grid_neg[i]) for i in set_I)
            else:
                objective = plp.lpSum(-0.001*self.timeStep*(unit_load_cost[i]*P_grid_pos[i] + \
                    unit_prod_price[i]*P_grid_neg[i]) for i in set_I)
        elif self.costfun == 'cost':
            if self.optim_conf['set_total_pv_sell']:
                objective = plp.lpSum(-0.001*self.timeStep*unit_load_cost[i]*(P_load[i] + P_def_sum[i]) for i in set_I)
            else:
                objective = plp.lpSum(-0.001*self.timeStep*unit_load_cost[i]*P_grid_pos[i] for i in set_I)
        elif self.costfun == 'self-consumption':
            if type_self_conso == 'bigm':
                bigm = 1e3
                objective = plp.lpSum(-0.001*self.timeStep*(bigm*unit_load_cost[i]*P_grid_pos[i] + \
                    unit_prod_price[i]*P_grid_neg[i]) for i in set_I)
            elif type_self_conso == 'maxmin':
                objective = plp.lpSum(0.001*self.timeStep*unit_load_cost[i]*SC[i] for i in set_I)
            else:
                self.logger.error("Not a valid option for type_self_conso parameter")
        else:
            self.logger.error("The cost function specified type is not valid")
        # Add more terms to the objective function in the case of battery use
        if self.optim_conf['set_use_battery']:
            objective = objective + plp.lpSum(-0.001*self.timeStep*(
                self.optim_conf['weight_battery_discharge']*P_sto_pos[i] + \
                    self.optim_conf['weight_battery_charge']*P_sto_neg[i]) for i in set_I)

        # Add term penalizing each startup where configured
        if ('set_deferrable_startup_penalty' in self.optim_conf and self.optim_conf['set_deferrable_startup_penalty']):
            for k in range(self.optim_conf['number_of_deferrable_loads']):
                if (len(self.optim_conf['set_deferrable_startup_penalty']) > k and self.optim_conf['set_deferrable_startup_penalty'][k]):
                    objective = objective + plp.lpSum(
                        -0.001 * self.timeStep * self.optim_conf['set_deferrable_startup_penalty'][k] * P_def_start[k][i] *\
                            unit_load_cost[i] * self.optim_conf['nominal_power_of_deferrable_loads'][k]
                        for i in set_I)

        opt_model.setObjective(objective)

        ## Setting constraints
        # The main constraint: power balance
        if self.plant_conf['inverter_is_hybrid']:
            constraints = {"constraint_main1_{}".format(i) :
                plp.LpConstraint(
                    e = P_hybrid_inverter[i] - P_def_sum[i] - P_load[i] + P_grid_neg[i] + P_grid_pos[i] ,
                    sense = plp.LpConstraintEQ,
                    rhs = 0)
                for i in set_I}
        else:
            if self.plant_conf['compute_curtailment']:
                constraints = {"constraint_main2_{}".format(i) :
                    plp.LpConstraint(
                        e = P_PV[i] - P_PV_curtailment[i] - P_def_sum[i] - P_load[i] + P_grid_neg[i] + P_grid_pos[i] + P_sto_pos[i] + P_sto_neg[i],
                        sense = plp.LpConstraintEQ,
                        rhs = 0)
                    for i in set_I}
            else:
                constraints = {"constraint_main3_{}".format(i) :
                    plp.LpConstraint(
                        e = P_PV[i] - P_def_sum[i] - P_load[i] + P_grid_neg[i] + P_grid_pos[i] + P_sto_pos[i] + P_sto_neg[i],
                        sense = plp.LpConstraintEQ,
                        rhs = 0)
                    for i in set_I}
        
        # Constraint for hybrid inverter and curtailment cases
        if type(self.plant_conf['pv_module_model']) == list:
            P_nom_inverter = 0.0
            for i in range(len(self.plant_conf['pv_inverter_model'])):
                if type(self.plant_conf['pv_inverter_model'][i]) == str:
                    cec_inverters = bz2.BZ2File(pathlib.Path(__file__).parent / 'data/cec_inverters.pbz2', "rb")
                    cec_inverters = cPickle.load(cec_inverters)
                    inverter = cec_inverters[self.plant_conf['pv_inverter_model'][i]]
                    P_nom_inverter += inverter.Paco
                else:
                    P_nom_inverter += self.plant_conf['pv_inverter_model'][i]
        else:
            if type(self.plant_conf['pv_inverter_model'][i]) == str:
                cec_inverters = bz2.BZ2File(pathlib.Path(__file__).parent / 'data/cec_inverters.pbz2', "rb")
                cec_inverters = cPickle.load(cec_inverters)
                inverter = cec_inverters[self.plant_conf['pv_inverter_model']]
                P_nom_inverter = inverter.Paco
            else:
                P_nom_inverter = self.plant_conf['pv_inverter_model']
        if self.plant_conf['inverter_is_hybrid']:
            constraints.update({"constraint_hybrid_inverter1_{}".format(i) :
                plp.LpConstraint(
                    e = P_PV[i] - P_PV_curtailment[i] + P_sto_pos[i] + P_sto_neg[i] - P_nom_inverter,
                    sense = plp.LpConstraintLE,
                    rhs = 0)
                for i in set_I})
            constraints.update({"constraint_hybrid_inverter2_{}".format(i) :
                plp.LpConstraint(
                    e = P_PV[i] - P_PV_curtailment[i] + P_sto_pos[i] + P_sto_neg[i] - P_hybrid_inverter[i],
                    sense = plp.LpConstraintEQ,
                    rhs = 0)
                for i in set_I})
        else:
            if self.plant_conf['compute_curtailment']:
                constraints.update({"constraint_curtailment_{}".format(i) :
                    plp.LpConstraint(
                        e = P_PV_curtailment[i] - max(P_PV[i],0),
                        sense = plp.LpConstraintLE,
                        rhs = 0)
                    for i in set_I})

        # Two special constraints just for a self-consumption cost function
        if self.costfun == 'self-consumption':
            if type_self_conso == 'maxmin': # maxmin linear problem
                constraints.update({"constraint_selfcons_PV1_{}".format(i) : 
                    plp.LpConstraint(
                        e = SC[i] - P_PV[i],
                        sense = plp.LpConstraintLE,
                        rhs = 0)
                    for i in set_I})
                constraints.update({"constraint_selfcons_PV2_{}".format(i) : 
                    plp.LpConstraint(
                        e = SC[i] - P_load[i] - P_def_sum[i],
                        sense = plp.LpConstraintLE,
                        rhs = 0)
                    for i in set_I})

        # Avoid injecting and consuming from grid at the same time
        constraints.update({"constraint_pgridpos_{}".format(i) : 
            plp.LpConstraint(
                e = P_grid_pos[i] - self.plant_conf['maximum_power_from_grid']*D[i],
                sense = plp.LpConstraintLE,
                rhs = 0)
            for i in set_I})
        constraints.update({"constraint_pgridneg_{}".format(i) : 
            plp.LpConstraint(
                e = -P_grid_neg[i] - self.plant_conf['maximum_power_to_grid']*(1-D[i]),
                sense = plp.LpConstraintLE,
                rhs = 0)
            for i in set_I})

        # Treat deferrable loads constraints
        predicted_temps = {}
        for k in range(self.optim_conf['number_of_deferrable_loads']):
            
            if type(self.optim_conf['nominal_power_of_deferrable_loads'][k]) == list:
                # Constraint for sequence of deferrable
                # WARNING: This is experimental, formulation seems correct but feasibility problems.
                # Probably uncomptabile with other constraints
                power_sequence = self.optim_conf['nominal_power_of_deferrable_loads'][k]
                sequence_length = len(power_sequence)
                def create_matrix(input_list, n):
                    matrix = []
                    for i in range(n + 1):
                        row = [0] * i + input_list + [0] * (n - i)
                        matrix.append(row[:n*2])
                    return matrix
                matrix = create_matrix(power_sequence, n-sequence_length)
                y = plp.LpVariable.dicts(f"y{k}", (i for i in range(len(matrix))), cat='Binary')
                constraints.update({f"single_value_constraint_{k}" :
                    plp.LpConstraint(
                        e = plp.lpSum(y[i] for i in range(len(matrix))) - 1,
                        sense = plp.LpConstraintEQ,
                        rhs = 0)
                    })
                constraints.update({f"pdef{k}_sumconstraint_{i}" :
                    plp.LpConstraint(
                        e = plp.lpSum(P_deferrable[k][i] for i in set_I) - np.sum(power_sequence),
                        sense = plp.LpConstraintEQ,
                        rhs = 0)
                    })
                constraints.update({f"pdef{k}_positive_constraint_{i}" :
                    plp.LpConstraint(
                        e = P_deferrable[k][i],
                        sense = plp.LpConstraintGE,
                        rhs = 0)
                    for i in set_I})
                for num, mat in enumerate(matrix):
                    constraints.update({f"pdef{k}_value_constraint_{num}_{i}" :
                        plp.LpConstraint(
                            e = P_deferrable[k][i] - mat[i]*y[num],
                            sense = plp.LpConstraintEQ,
                            rhs = 0)
                        for i in set_I})
            
            elif "def_load_config" in self.optim_conf.keys():
                if "thermal_config" in self.optim_conf["def_load_config"][k]:
                    # Special case of a thermal deferrable load
                    def_load_config = self.optim_conf['def_load_config'][k]
                    if def_load_config and 'thermal_config' in def_load_config:
                        hc = def_load_config["thermal_config"]
                        start_temperature = hc["start_temperature"]
                        cooling_constant = hc["cooling_constant"]
                        heating_rate = hc["heating_rate"]
                        overshoot_temperature = hc["overshoot_temperature"]
                        outdoor_temperature_forecast = data_opt['outdoor_temperature_forecast']
                        desired_temperatures = hc["desired_temperatures"]
                        sense = hc.get('sense', 'heat')
                        predicted_temp = [start_temperature]
                        for I in set_I:
                            if I == 0:
                                continue
                            predicted_temp.append(
                                predicted_temp[I-1]
                                + (P_deferrable[k][I-1] * (heating_rate * self.timeStep / self.optim_conf['nominal_power_of_deferrable_loads'][k]))
                                - (cooling_constant * (predicted_temp[I-1] - outdoor_temperature_forecast[I-1])))
                            if len(desired_temperatures) > I and desired_temperatures[I]:
                                constraints.update({"constraint_defload{}_temperature_{}".format(k, I):
                                    plp.LpConstraint(
                                        e = predicted_temp[I],
                                        sense = plp.LpConstraintGE if sense == 'heat' else plp.LpConstraintLE,
                                        rhs = desired_temperatures[I],
                                    )
                                })
                        constraints.update({"constraint_defload{}_overshoot_temp_{}".format(k, I):
                            plp.LpConstraint(
                                e = predicted_temp[I],
                                sense = plp.LpConstraintLE if sense == 'heat' else plp.LpConstraintGE,
                                rhs = overshoot_temperature,
                            )
                        for I in set_I})
                        predicted_temps[k] = predicted_temp
            
            else:
                
                if def_total_hours[k] > 0:
                    # Total time of deferrable load
                    constraints.update({"constraint_defload{}_energy".format(k) :
                        plp.LpConstraint(
                            e = plp.lpSum(P_deferrable[k][i]*self.timeStep for i in set_I),
                            sense = plp.LpConstraintEQ,
                            rhs = def_total_hours[k]*self.optim_conf['nominal_power_of_deferrable_loads'][k])
                        })
                
            # Ensure deferrable loads consume energy between def_start_timestep & def_end_timestep
            self.logger.debug("Deferrable load {}: Proposed optimization window: {} --> {}".format(
                k, def_start_timestep[k], def_end_timestep[k]))
            def_start, def_end, warning = Optimization.validate_def_timewindow(
                def_start_timestep[k], def_end_timestep[k], ceil(def_total_hours[k]/self.timeStep), n)
            if warning is not None: 
                self.logger.warning("Deferrable load {} : {}".format(k, warning))
            self.logger.debug("Deferrable load {}: Validated optimization window: {} --> {}".format(
                k, def_start, def_end))
            if def_start > 0:                    
                constraints.update({"constraint_defload{}_start_timestep".format(k) :
                    plp.LpConstraint(
                        e = plp.lpSum(P_deferrable[k][i]*self.timeStep for i in range(0, def_start)),
                        sense = plp.LpConstraintEQ,
                        rhs = 0)
                    })
            if def_end > 0:                    
                constraints.update({"constraint_defload{}_end_timestep".format(k) :
                    plp.LpConstraint(
                        e = plp.lpSum(P_deferrable[k][i]*self.timeStep for i in range(def_end, n)),
                        sense = plp.LpConstraintEQ,
                        rhs = 0)
                    })
                
            # Treat the number of starts for a deferrable load (new method considering current state)
            current_state = 0
            if ("def_current_state" in self.optim_conf and len(self.optim_conf["def_current_state"]) > k):
                current_state = 1 if self.optim_conf["def_current_state"][k] else 0
            # P_deferrable < P_def_bin2 * 1 million
            # P_deferrable must be zero if P_def_bin2 is zero
            constraints.update({"constraint_pdef{}_start1_{}".format(k, i): 
                plp.LpConstraint(
                    e=P_deferrable[k][i] - P_def_bin2[k][i] * M,
                    sense=plp.LpConstraintLE,
                    rhs=0)
                for i in set_I})
            # P_deferrable - P_def_bin2 <= 0
            # P_def_bin2 must be zero if P_deferrable is zero
            constraints.update({"constraint_pdef{}_start1a_{}".format(k, i): 
                plp.LpConstraint(
                    e=P_def_bin2[k][i] - P_deferrable[k][i],
                    sense=plp.LpConstraintLE,
                    rhs=0)
                for i in set_I})
            # P_def_start + P_def_bin2[i-1] >= P_def_bin2[i]
            # If load is on this cycle (P_def_bin2[i] is 1) then P_def_start must be 1 OR P_def_bin2[i-1] must be 1
            # For first timestep, use current state if provided by caller.
            constraints.update({"constraint_pdef{}_start2_{}".format(k, i): 
                plp.LpConstraint(
                    e=P_def_start[k][i]
                    - P_def_bin2[k][i]
                    + (P_def_bin2[k][i - 1] if i - 1 >= 0 else current_state),
                    sense=plp.LpConstraintGE,
                    rhs=0)
                for i in set_I})
            # P_def_bin2[i-1] + P_def_start <= 1
            # If load started this cycle (P_def_start[i] is 1) then P_def_bin2[i-1] must be 0
            constraints.update({"constraint_pdef{}_start3_{}".format(k, i):
                plp.LpConstraint(
                    e=(P_def_bin2[k][i-1] if i-1 >= 0 else 0) + P_def_start[k][i],
                    sense=plp.LpConstraintLE,
                    rhs=1)
                for i in set_I})
            
            # Treat deferrable as a fixed value variable with just one startup
            if self.optim_conf['set_deferrable_load_single_constant'][k]:
                # P_def_start[i] must be 1 for exactly 1 value of i
                constraints.update({"constraint_pdef{}_start4".format(k) :
                plp.LpConstraint(
                    e = plp.lpSum(P_def_start[k][i] for i in set_I),
                    sense = plp.LpConstraintEQ,
                    rhs = 1)
                })
                # P_def_bin2 must be 1 for exactly the correct number of timesteps.
                constraints.update({"constraint_pdef{}_start5".format(k) :
                plp.LpConstraint(
                    e = plp.lpSum(P_def_bin2[k][i] for i in set_I),
                    sense = plp.LpConstraintEQ,
                    rhs = def_total_hours[k]/self.timeStep)
                })
        
            # Treat deferrable load as a semi-continuous variable
            if self.optim_conf['treat_deferrable_load_as_semi_cont'][k]:
                constraints.update({"constraint_pdef{}_semicont1_{}".format(k, i) : 
                    plp.LpConstraint(
                        e=P_deferrable[k][i] - self.optim_conf['nominal_power_of_deferrable_loads'][k]*P_def_bin1[k][i],
                        sense=plp.LpConstraintGE,
                        rhs=0)
                    for i in set_I})
                constraints.update({"constraint_pdef{}_semicont2_{}".format(k, i) :
                    plp.LpConstraint(
                        e=P_deferrable[k][i] - self.optim_conf['nominal_power_of_deferrable_loads'][k]*P_def_bin1[k][i],
                        sense=plp.LpConstraintLE,
                        rhs=0)
                    for i in set_I})
        
        
            # Treat the number of starts for a deferrable load (old method, kept here just in case)
            # if self.optim_conf['set_deferrable_load_single_constant'][k]:
            #     constraints.update({"constraint_pdef{}_start1_{}".format(k, i) : 
            #         plp.LpConstraint(
            #             e=P_deferrable[k][i] - P_def_bin2[k][i]*M,
            #             sense=plp.LpConstraintLE,
            #             rhs=0)
            #         for i in set_I})
            #     constraints.update({"constraint_pdef{}_start2_{}".format(k, i): 
            #         plp.LpConstraint(
            #             e=P_def_start[k][i] - P_def_bin2[k][i] + (P_def_bin2[k][i-1] if i-1 >= 0 else 0),
            #             sense=plp.LpConstraintGE,
            #             rhs=0)
            #         for i in set_I})
            #     constraints.update({"constraint_pdef{}_start3".format(k) :
            #     plp.LpConstraint(
            #         e = plp.lpSum(P_def_start[k][i] for i in set_I),
            #         sense = plp.LpConstraintEQ,
            #         rhs = 1)
            #     })

        # The battery constraints
        if self.optim_conf['set_use_battery']:
            # Optional constraints to avoid charging the battery from the grid
            if self.optim_conf['set_nocharge_from_grid']:
                constraints.update({"constraint_nocharge_from_grid_{}".format(i) : 
                    plp.LpConstraint(
                        e = P_sto_neg[i] + P_PV[i],
                        sense = plp.LpConstraintGE,
                        rhs = 0)
                    for i in set_I})
            # Optional constraints to avoid discharging the battery to the grid
            if self.optim_conf['set_nodischarge_to_grid']:
                constraints.update({"constraint_nodischarge_to_grid_{}".format(i) : 
                    plp.LpConstraint(
                        e = P_grid_neg[i] + P_PV[i],
                        sense = plp.LpConstraintGE,
                        rhs = 0)
                    for i in set_I})
            # Limitation of power dynamics in power per unit of time
            if self.optim_conf['set_battery_dynamic']:
                constraints.update({"constraint_pos_batt_dynamic_max_{}".format(i) : 
                    plp.LpConstraint(e = P_sto_pos[i+1] - P_sto_pos[i], 
                                     sense = plp.LpConstraintLE, 
                                     rhs = self.timeStep*self.optim_conf['battery_dynamic_max']*self.plant_conf['battery_discharge_power_max']) 
                    for i in range(n-1)})
                constraints.update({"constraint_pos_batt_dynamic_min_{}".format(i) : 
                    plp.LpConstraint(e = P_sto_pos[i+1] - P_sto_pos[i], 
                                     sense = plp.LpConstraintGE, 
                                     rhs = self.timeStep*self.optim_conf['battery_dynamic_min']*self.plant_conf['battery_discharge_power_max']) 
                    for i in range(n-1)})
                constraints.update({"constraint_neg_batt_dynamic_max_{}".format(i) : 
                    plp.LpConstraint(e = P_sto_neg[i+1] - P_sto_neg[i], 
                                     sense = plp.LpConstraintLE, 
                                     rhs = self.timeStep*self.optim_conf['battery_dynamic_max']*self.plant_conf['battery_charge_power_max']) 
                    for i in range(n-1)})
                constraints.update({"constraint_neg_batt_dynamic_min_{}".format(i) : 
                    plp.LpConstraint(e = P_sto_neg[i+1] - P_sto_neg[i], 
                                     sense = plp.LpConstraintGE, 
                                     rhs = self.timeStep*self.optim_conf['battery_dynamic_min']*self.plant_conf['battery_charge_power_max']) 
                    for i in range(n-1)})
            # Then the classic battery constraints
            constraints.update({"constraint_pstopos_{}".format(i) : 
                plp.LpConstraint(
                    e=P_sto_pos[i] - self.plant_conf['battery_discharge_efficiency']*self.plant_conf['battery_discharge_power_max']*E[i],
                    sense=plp.LpConstraintLE,
                    rhs=0)
                for i in set_I})
            constraints.update({"constraint_pstoneg_{}".format(i) : 
                plp.LpConstraint(
                    e=-P_sto_neg[i] - (1/self.plant_conf['battery_charge_efficiency'])*self.plant_conf['battery_charge_power_max']*(1-E[i]),
                    sense=plp.LpConstraintLE,
                    rhs=0)
                for i in set_I})
            constraints.update({"constraint_socmax_{}".format(i) : 
                plp.LpConstraint(
                    e=-plp.lpSum(P_sto_pos[j]*(1/self.plant_conf['battery_discharge_efficiency']) + self.plant_conf['battery_charge_efficiency']*P_sto_neg[j] for j in range(i)),
                    sense=plp.LpConstraintLE,
                    rhs=(self.plant_conf['battery_nominal_energy_capacity']/self.timeStep)*(self.plant_conf['battery_maximum_state_of_charge'] - soc_init))
                for i in set_I})
            constraints.update({"constraint_socmin_{}".format(i) : 
                plp.LpConstraint(
                    e=plp.lpSum(P_sto_pos[j]*(1/self.plant_conf['battery_discharge_efficiency']) + self.plant_conf['battery_charge_efficiency']*P_sto_neg[j] for j in range(i)),
                    sense=plp.LpConstraintLE,
                    rhs=(self.plant_conf['battery_nominal_energy_capacity']/self.timeStep)*(soc_init - self.plant_conf['battery_minimum_state_of_charge']))
                for i in set_I})
            constraints.update({"constraint_socfinal_{}".format(0) : 
                plp.LpConstraint(
                    e=plp.lpSum(P_sto_pos[i]*(1/self.plant_conf['battery_discharge_efficiency']) + self.plant_conf['battery_charge_efficiency']*P_sto_neg[i] for i in set_I),
                    sense=plp.LpConstraintEQ,
                    rhs=(soc_init - soc_final)*self.plant_conf['battery_nominal_energy_capacity']/self.timeStep)
                })
        opt_model.constraints = constraints

        ## Finally, we call the solver to solve our optimization model:
        # solving with default solver CBC
        if self.lp_solver == 'PULP_CBC_CMD':
            opt_model.solve(PULP_CBC_CMD(msg=0))
        elif self.lp_solver == 'GLPK_CMD':
            opt_model.solve(GLPK_CMD(msg=0))
        elif self.lp_solver == 'COIN_CMD':
            opt_model.solve(COIN_CMD(msg=0, path=self.lp_solver_path))
        else:
            self.logger.warning("Solver %s unknown, using default", self.lp_solver)
            opt_model.solve()

        # The status of the solution is printed to the screen
        self.optim_status = plp.LpStatus[opt_model.status]
        self.logger.info("Status: " + self.optim_status)
        if plp.value(opt_model.objective) is None:
            self.logger.warning("Cost function cannot be evaluated")
            return
        else:
            self.logger.info("Total value of the Cost function = %.02f", plp.value(opt_model.objective))

        # Build results Dataframe
        opt_tp = pd.DataFrame()
        opt_tp["P_PV"] = [P_PV[i] for i in set_I]
        opt_tp["P_Load"] = [P_load[i] for i in set_I]
        for k in range(self.optim_conf['number_of_deferrable_loads']):
            opt_tp["P_deferrable{}".format(k)] = [P_deferrable[k][i].varValue for i in set_I]
        opt_tp["P_grid_pos"] = [P_grid_pos[i].varValue for i in set_I]
        opt_tp["P_grid_neg"] = [P_grid_neg[i].varValue for i in set_I]
        opt_tp["P_grid"] = [P_grid_pos[i].varValue + P_grid_neg[i].varValue for i in set_I]
        if self.optim_conf['set_use_battery']:
            opt_tp["P_batt"] = [P_sto_pos[i].varValue + P_sto_neg[i].varValue for i in set_I]
            SOC_opt_delta = [(P_sto_pos[i].varValue*(1/self.plant_conf['battery_discharge_efficiency']) + \
                              self.plant_conf['battery_charge_efficiency']*P_sto_neg[i].varValue)*(
                                  self.timeStep/(self.plant_conf['battery_nominal_energy_capacity'])) for i in set_I]
            SOCinit = copy.copy(soc_init)
            SOC_opt = []
            for i in set_I:
                SOC_opt.append(SOCinit - SOC_opt_delta[i])
                SOCinit = SOC_opt[i]
            opt_tp["SOC_opt"] = SOC_opt
        if self.plant_conf['inverter_is_hybrid']:
            opt_tp["P_hybrid_inverter"] = [P_hybrid_inverter[i].varValue for i in set_I]
        if self.plant_conf['compute_curtailment']:
            opt_tp["P_PV_curtailment"] = [P_PV_curtailment[i].varValue for i in set_I]
        opt_tp.index = data_opt.index

        # Lets compute the optimal cost function
        P_def_sum_tp = []
        for i in set_I:
            P_def_sum_tp.append(sum(P_deferrable[k][i].varValue for k in range(self.optim_conf['number_of_deferrable_loads'])))
        opt_tp["unit_load_cost"] = [unit_load_cost[i] for i in set_I]
        opt_tp["unit_prod_price"] = [unit_prod_price[i] for i in set_I]
        if self.optim_conf['set_total_pv_sell']:
            opt_tp["cost_profit"] = [-0.001*self.timeStep*(unit_load_cost[i]*(P_load[i] + P_def_sum_tp[i]) + \
                unit_prod_price[i]*P_grid_neg[i].varValue) for i in set_I]
        else:
            opt_tp["cost_profit"] = [-0.001*self.timeStep*(unit_load_cost[i]*P_grid_pos[i].varValue + \
                unit_prod_price[i]*P_grid_neg[i].varValue) for i in set_I]

        if self.costfun == 'profit':
            if self.optim_conf['set_total_pv_sell']:
                opt_tp["cost_fun_profit"] = [-0.001*self.timeStep*(unit_load_cost[i]*(P_load[i] + P_def_sum_tp[i]) + \
                    unit_prod_price[i]*P_grid_neg[i].varValue) for i in set_I]
            else:
                opt_tp["cost_fun_profit"] = [-0.001*self.timeStep*(unit_load_cost[i]*P_grid_pos[i].varValue + \
                    unit_prod_price[i]*P_grid_neg[i].varValue) for i in set_I]
        elif self.costfun == 'cost':
            if self.optim_conf['set_total_pv_sell']:
                opt_tp["cost_fun_cost"] = [-0.001*self.timeStep*unit_load_cost[i]*(P_load[i] + P_def_sum_tp[i]) for i in set_I]
            else:
                opt_tp["cost_fun_cost"] = [-0.001*self.timeStep*unit_load_cost[i]*P_grid_pos[i].varValue for i in set_I]
        elif self.costfun == 'self-consumption':
            if type_self_conso == 'maxmin':
                opt_tp["cost_fun_selfcons"] = [-0.001*self.timeStep*unit_load_cost[i]*SC[i].varValue for i in set_I]
            elif type_self_conso == 'bigm':
                opt_tp["cost_fun_selfcons"] = [-0.001*self.timeStep*(unit_load_cost[i]*P_grid_pos[i].varValue + \
                    unit_prod_price[i]*P_grid_neg[i].varValue) for i in set_I]
        else:
            self.logger.error("The cost function specified type is not valid")

        # Add the optimization status
        opt_tp["optim_status"] = self.optim_status

        # Debug variables
        if debug:
            for k in range(self.optim_conf['number_of_deferrable_loads']):
                opt_tp[f"P_def_start_{k}"] = [P_def_start[k][i].varValue for i in set_I]
                opt_tp[f"P_def_bin2_{k}"] = [P_def_bin2[k][i].varValue for i in set_I]
        for i, predicted_temp in predicted_temps.items():
            opt_tp[f"predicted_temp_heater{i}"] = pd.Series([round(pt.value(), 2) if isinstance(pt, plp.LpAffineExpression) else pt for pt in predicted_temp], index=opt_tp.index)
            opt_tp[f"target_temp_heater{i}"] = pd.Series(self.optim_conf["def_load_config"][i]['thermal_config']["desired_temperatures"], index=opt_tp.index)

        return opt_tp

    def perform_perfect_forecast_optim(self, df_input_data: pd.DataFrame, days_list: pd.date_range) -> pd.DataFrame:
        r"""
        Perform an optimization on historical data (perfectly known PV production).
        
        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably photovoltaics and load consumption powers.
        :type df_input_data: pandas.DataFrame
        :param days_list: A list of the days of data that will be retrieved from \
            hass and used for the optimization task. We will retrieve data from \
            now and up to days_to_retrieve days
        :type days_list: list
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform optimization for perfect forecast scenario")
        self.days_list_tz = days_list.tz_convert(self.time_zone).round(self.freq)[:-1] # Converted to tz and without the current day (today)
        self.opt_res = pd.DataFrame()
        for day in self.days_list_tz:
            self.logger.info("Solving for day: "+str(day.day)+"-"+str(day.month)+"-"+str(day.year))
            # Prepare data
            day_start = day.isoformat()
            day_end = (day+self.time_delta-self.freq).isoformat()
            data_tp = df_input_data.copy().loc[pd.date_range(start=day_start, end=day_end, freq=self.freq)]
            P_PV = data_tp[self.var_PV].values
            P_load = data_tp[self.var_load_new].values
            unit_load_cost = data_tp[self.var_load_cost].values # €/kWh
            unit_prod_price = data_tp[self.var_prod_price].values # €/kWh
            # Call optimization function
            opt_tp = self.perform_optimization(data_tp, P_PV, P_load, 
                                               unit_load_cost, unit_prod_price)
            if len(self.opt_res) == 0:
                self.opt_res = opt_tp
            else:
                self.opt_res = pd.concat([self.opt_res, opt_tp], axis=0)

        return self.opt_res

    def perform_dayahead_forecast_optim(self, df_input_data: pd.DataFrame, 
                                        P_PV: pd.Series, P_load: pd.Series) -> pd.DataFrame:
        r"""
        Perform a day-ahead optimization task using real forecast data. \
        This type of optimization is intented to be launched once a day.
        
        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably the unit load cost for power consumption.
        :type df_input_data: pandas.DataFrame
        :param P_PV: The forecasted PV power production.
        :type P_PV: pandas.DataFrame
        :param P_load: The forecasted Load power consumption. This power should \
            not include the power from the deferrable load that we want to find.
        :type P_load: pandas.DataFrame
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform optimization for the day-ahead")
        unit_load_cost = df_input_data[self.var_load_cost].values # €/kWh
        unit_prod_price = df_input_data[self.var_prod_price].values # €/kWh
        # Call optimization function
        self.opt_res = self.perform_optimization(df_input_data, P_PV.values.ravel(), 
                                                 P_load.values.ravel(), 
                                                 unit_load_cost, unit_prod_price)
        return self.opt_res

    def perform_naive_mpc_optim(self, df_input_data: pd.DataFrame, P_PV: pd.Series, P_load: pd.Series,
                                prediction_horizon: int, soc_init: Optional[float] = None, soc_final: Optional[float] = None,
                                def_total_hours: Optional[list] = None,
                                def_start_timestep: Optional[list] = None,
                                def_end_timestep: Optional[list] = None) -> pd.DataFrame:
        r"""
        Perform a naive approach to a Model Predictive Control (MPC). \
        This implementaion is naive because we are not using the formal formulation \
        of a MPC. Only the sense of a receiding horizon is considered here. \
        This optimization is more suitable for higher optimization frequency, ex: 5min.
        
        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably the unit load cost for power consumption.
        :type df_input_data: pandas.DataFrame
        :param P_PV: The forecasted PV power production.
        :type P_PV: pandas.DataFrame
        :param P_load: The forecasted Load power consumption. This power should \
            not include the power from the deferrable load that we want to find.
        :type P_load: pandas.DataFrame
        :param prediction_horizon: The prediction horizon of the MPC controller in number \
            of optimization time steps.
        :type prediction_horizon: int
        :param soc_init: The initial battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_init: float
        :param soc_final: The final battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_final: 
        :param def_total_hours: The functioning hours for this iteration for each deferrable load. \
            (For continuous deferrable loads: functioning hours at nominal power)
        :type def_total_hours: list
        :param def_start_timestep: The timestep as from which each deferrable load is allowed to operate.
        :type def_start_timestep: list
        :param def_end_timestep: The timestep before which each deferrable load should operate.
        :type def_end_timestep: list
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform an iteration of a naive MPC controller")
        if prediction_horizon < 5:
            self.logger.error("Set the MPC prediction horizon to at least 5 times the optimization time step")
            return pd.DataFrame()
        else:
            df_input_data = copy.deepcopy(df_input_data)[df_input_data.index[0]:df_input_data.index[prediction_horizon-1]]
        unit_load_cost = df_input_data[self.var_load_cost].values # €/kWh
        unit_prod_price = df_input_data[self.var_prod_price].values # €/kWh
        # Call optimization function
        self.opt_res = self.perform_optimization(df_input_data, P_PV.values.ravel(), P_load.values.ravel(), 
                                                 unit_load_cost, unit_prod_price, soc_init=soc_init, 
                                                 soc_final=soc_final, def_total_hours=def_total_hours,
                                                 def_start_timestep=def_start_timestep, def_end_timestep=def_end_timestep)
        return self.opt_res

    @staticmethod
    def validate_def_timewindow(start: int, end: int, min_steps: int, window: int) -> Tuple[int,int,str]:
        r"""
        Helper function to validate (and if necessary: correct) the defined optimization window of a deferrable load.
        
        :param start: Start timestep of the optimization window of the deferrable load
        :type start: int
        :param end: End timestep of the optimization window of the deferrable load
        :type end: int
        :param min_steps: Minimal timesteps during which the load should operate (at nominal power)
        :type min_steps: int
        :param window: Total number of timesteps in the optimization window
        :type window: int
        :return: start_validated: Validated start timestep of the optimization window of the deferrable load
        :rtype: int
        :return: end_validated: Validated end timestep of the optimization window of the deferrable load
        :rtype: int
        :return: warning: Any warning information to be returned from the validation steps
        :rtype: string

        """
        start_validated = 0
        end_validated = 0
        warning = None
        # Verify that start <= end
        if start <= end or start <= 0 or end <= 0:
            # start and end should be within the optimization timewindow [0, window]
            start_validated = max(0, min(window, start))
            end_validated = max(0, min(window, end))
            if end_validated > 0:
                # If the available timeframe is shorter than the number of timesteps needed to meet the hours to operate (def_total_hours), issue a warning.
                if (end_validated-start_validated) < min_steps:
                    warning = "Available timeframe is shorter than the specified number of hours to operate. Optimization will fail."
        else:
            warning = "Invalid timeframe for deferrable load (start timestep is not <= end timestep). Continuing optimization without timewindow constraint."
        return start_validated, end_validated, warning
