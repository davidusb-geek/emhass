#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import pandas as pd
import numpy as np
import pulp as plp
from pulp import PULP_CBC_CMD, COIN_CMD, GLPK_CMD
import logging, copy


class optimization:
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
                 costfun: str, config_path: str, logger: logging.Logger, 
                 opt_time_delta: Optional[int] = 24) -> None:
        r"""
        Define constructor for optimization class.
        
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
        :param config_path: The path to the yaml configuration file
        :type config_path: str
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
        self.freq = self.retrieve_hass_conf['freq']
        self.time_zone = self.retrieve_hass_conf['time_zone']
        self.timeStep = self.freq.seconds/3600 # in hours
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours") # The period of optimization
        self.var_PV = self.retrieve_hass_conf['var_PV']
        self.var_load = self.retrieve_hass_conf['var_load']
        self.var_load_new = self.var_load+'_positive'
        self.costfun = costfun
        self.logger = logger
        self.var_load_cost = var_load_cost
        self.var_prod_price = var_prod_price
        if 'lp_solver' in optim_conf.keys():
            self.lp_solver = optim_conf['lp_solver']
        else:
            self.lp_solver = 'PULP_CBC_CMD'
        if 'lp_solver_path' in optim_conf.keys():
            self.lp_solver_path = optim_conf['lp_solver_path']
        else:
            self.lp_solver_path = 'empty'
        if self.lp_solver != 'COIN_CMD' and self.lp_solver_path != 'empty':
            self.logger.error("Use COIN_CMD solver name if you want to set a path for the LP solver")
        
    def perform_optimization(self, data_opt: pd.DataFrame, P_PV: np.array, P_load: np.array, 
                             unit_load_cost: np.array, unit_prod_price: np.array,
                             soc_init: Optional[float] = None, soc_final: Optional[float] = None,
                             def_total_hours: Optional[list] = None) -> pd.DataFrame:
        r"""
        Perform the actual optimization using linear programming (LP).
        
        :param data_tp: A DataFrame containing the input data. The results of the \
            optimization will be appended (decision variables, cost function values, etc)
        :type data_tp: pd.DataFrame
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
        :param def_total_hours: The functioning hours for this iteration for each deferrable load.
        :type def_total_hours: list
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
                    soc_init = self.plant_conf['SOCtarget']
            if soc_final is None:
                if soc_init is not None:
                    soc_final = soc_init
                else:
                    soc_final = self.plant_conf['SOCtarget']
        if def_total_hours is None:
            def_total_hours = self.optim_conf['def_total_hours']
        type_self_conso = 'bigm' # maxmin
        
        #### The LP problem using Pulp ####
        opt_model = plp.LpProblem("LP_Model", plp.LpMaximize)
        
        n = len(data_opt.index)
        set_I = range(n)
        M = 10e10
        
        ## Add decision variables
        P_grid_neg  = {(i):plp.LpVariable(cat='Continuous',
                                          lowBound=-self.plant_conf['P_grid_max'], upBound=0,
                                          name="P_grid_neg{}".format(i)) for i in set_I}
        P_grid_pos  = {(i):plp.LpVariable(cat='Continuous',
                                          lowBound=0, upBound=self.plant_conf['P_grid_max'],
                                          name="P_grid_pos{}".format(i)) for i in set_I}
        P_deferrable = []
        P_def_bin1 = []
        for k in range(self.optim_conf['num_def_loads']):
            if self.optim_conf['treat_def_as_semi_cont'][k]:
                P_deferrable.append({(i):plp.LpVariable(cat='Continuous',
                                                        name="P_deferrable{}_{}".format(k, i)) for i in set_I})
            else:
                P_deferrable.append({(i):plp.LpVariable(cat='Continuous',
                                                        lowBound=0, upBound=self.optim_conf['P_deferrable_nom'][k],
                                                        name="P_deferrable{}_{}".format(k, i)) for i in set_I})
            P_def_bin1.append({(i):plp.LpVariable(cat='Binary',
                                                  name="P_def{}_bin1_{}".format(k, i)) for i in set_I})
        P_def_start = []
        P_def_bin2 = []
        for k in range(self.optim_conf['num_def_loads']):
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
                                             lowBound=0, upBound=self.plant_conf['Pd_max'],
                                             name="P_sto_pos_{0}".format(i)) for i in set_I}
            P_sto_neg  = {(i):plp.LpVariable(cat='Continuous', 
                                             lowBound=-self.plant_conf['Pc_max'], upBound=0,
                                             name="P_sto_neg_{0}".format(i)) for i in set_I}
        else:
            P_sto_pos  = {(i):i*0 for i in set_I}
            P_sto_neg  = {(i):i*0 for i in set_I}
            
        if self.costfun == 'self-consumption':
            SC  = {(i):plp.LpVariable(cat='Continuous',
                                      name="SC_{}".format(i)) for i in set_I}
            
        ## Define objective
        P_def_sum= []
        for i in set_I:
            P_def_sum.append(plp.lpSum(P_deferrable[k][i] for k in range(self.optim_conf['num_def_loads'])))
        if self.costfun == 'profit':
            if self.optim_conf['set_total_pv_sell']:
                objective = plp.lpSum(-0.001*self.timeStep*(unit_load_cost[i]*(P_load[i] + P_def_sum[i]) + \
                                                            unit_prod_price[i]*P_grid_neg[i])
                                      for i in set_I)
            else:
                objective = plp.lpSum(-0.001*self.timeStep*(unit_load_cost[i]*P_grid_pos[i] + \
                                                            unit_prod_price[i]*P_grid_neg[i])
                                      for i in set_I)
        elif self.costfun == 'cost':
            if self.optim_conf['set_total_pv_sell']:
                objective = plp.lpSum(-0.001*self.timeStep*unit_load_cost[i]*P_grid_pos[i]
                                      for i in set_I)
            else:
                objective = plp.lpSum(-0.001*self.timeStep*unit_load_cost[i]*(P_load[i] + P_def_sum[i])
                                      for i in set_I)
        elif self.costfun == 'self-consumption':
            if type_self_conso == 'bigm':
                bigm = 1e3
                objective = plp.lpSum(-0.001*self.timeStep*(bigm*unit_load_cost[i]*P_grid_pos[i] + \
                                                            unit_prod_price[i]*P_grid_neg[i])
                                      for i in set_I)
            elif type_self_conso == 'maxmin':
                objective = plp.lpSum(0.001*self.timeStep*unit_load_cost[i]*SC[i] for i in set_I)
            else:
                self.logger.error("Not a valida option for type_self_conso parameter")
        else:
            self.logger.error("The cost function specified type is not valid")
        opt_model.setObjective(objective)
        
        ## Setting constraints
        # The main constraint: power balance
        constraints = {"constraint_main1_{}".format(i) :
                       plp.LpConstraint(
                           e = P_PV[i] - P_def_sum[i] - P_load[i] + P_grid_neg[i] + P_grid_pos[i] + P_sto_pos[i] + P_sto_neg[i],
                           sense = plp.LpConstraintEQ,
                           rhs = 0)
                       for i in set_I}
        
        # An optional constraint to avoid charging the battery from the grid
        if self.optim_conf['set_use_battery']:
            if self.optim_conf['set_nocharge_from_grid']:
                constraints.update({"constraint_nocharge_from_grid_{}".format(i) : 
                                plp.LpConstraint(
                                    e = P_sto_neg[i] + P_PV[i],
                                    sense = plp.LpConstraintGE,
                                    rhs = 0)
                                for i in set_I})
            
        # Two special constraints just for a self-consumption cost function
        if self.costfun == 'self-consumption':
            if type_self_conso == 'maxmin': # maxmin linear problem
                constraints.update({"constraint_selfcons_PV_{}".format(i) : 
                                    plp.LpConstraint(
                                        e = SC[i] - P_PV[i],
                                        sense = plp.LpConstraintLE,
                                        rhs = 0)
                                    for i in set_I})
                constraints.update({"constraint_selfcons_PV_{}".format(i) : 
                                    plp.LpConstraint(
                                        e = SC[i] - P_load[i] - P_def_sum[i],
                                        sense = plp.LpConstraintLE,
                                        rhs = 0)
                                    for i in set_I})
        
        # Avoid injecting and consuming from grid at the same time
        constraints.update({"constraint_pgridpos_{}".format(i) : 
                            plp.LpConstraint(
                                e = P_grid_pos[i] - self.plant_conf['P_grid_max']*D[i],
                                sense = plp.LpConstraintLE,
                                rhs = 0)
                            for i in set_I})
        constraints.update({"constraint_pgridneg_{}".format(i) : 
                            plp.LpConstraint(
                                e = -P_grid_neg[i] - self.plant_conf['P_grid_max']*(1-D[i]),
                                sense = plp.LpConstraintLE,
                                rhs = 0)
                            for i in set_I})
            
        # Total time of deferrable load
        for k in range(self.optim_conf['num_def_loads']):
            constraints.update({"constraint_defload{}_energy".format(k) :
                                plp.LpConstraint(
                                    e = plp.lpSum(P_deferrable[k][i]*self.timeStep for i in set_I),
                                    sense = plp.LpConstraintEQ,
                                    rhs = def_total_hours[k]*self.optim_conf['P_deferrable_nom'][k])
                                })
            
        # Treat deferrable load as a semi-continuous variable
        for k in range(self.optim_conf['num_def_loads']):
            if self.optim_conf['treat_def_as_semi_cont'][k]:
                constraints.update({"constraint_pdef{}_semicont1_{}".format(k, i) : 
                                    plp.LpConstraint(
                                        e=P_deferrable[k][i] - self.optim_conf['P_deferrable_nom'][k]*P_def_bin1[k][i],
                                        sense=plp.LpConstraintGE,
                                        rhs=0)
                                    for i in set_I})
                constraints.update({"constraint_pdef{}_semicont2_{}".format(k, i) :
                                    plp.LpConstraint(
                                        e=P_deferrable[k][i] - self.optim_conf['P_deferrable_nom'][k]*P_def_bin1[k][i],
                                        sense=plp.LpConstraintLE,
                                        rhs=0)
                                    for i in set_I})
                
        # Treat the number of starts for a deferrable load
        for k in range(self.optim_conf['num_def_loads']):
            if self.optim_conf['set_def_constant'][k]:
                constraints.update({"constraint_pdef{}_start1".format(k) : 
                                    plp.LpConstraint(
                                        e=P_def_start[k][0],
                                        sense=plp.LpConstraintEQ,
                                        rhs=0)
                                    })
                constraints.update({"constraint_pdef{}_start2_{}".format(k, i) : 
                                    plp.LpConstraint(
                                        e=P_def_start[k][i] - P_def_bin2[k][i] + P_def_bin2[k][i-1],
                                        sense=plp.LpConstraintEQ,
                                        rhs=0)
                                    for i in set_I[1:]})
                constraints.update({"constraint_pdef{}_start4_{}".format(k, i) : 
                                    plp.LpConstraint(
                                        e=P_deferrable[k][i] - P_def_bin2[k][i]*M,
                                        sense=plp.LpConstraintLE,
                                        rhs=0)
                                    for i in set_I})
                constraints.update({"constraint_pdef{}_start5_{}".format(k, i) : 
                                    plp.LpConstraint(
                                        e=-P_deferrable[k][i] + M*(P_def_bin2[k][i]-1) + 1,
                                        sense=plp.LpConstraintLE,
                                        rhs=0)
                                    for i in set_I})
        
        # The battery constraints
        if self.optim_conf['set_use_battery']:
            constraints.update({"constraint_pstopos_{}".format(i) : 
                                plp.LpConstraint(
                                    e=P_sto_pos[i] - self.plant_conf['eta_disch']*self.plant_conf['Pd_max']*E[i],
                                    sense=plp.LpConstraintLE,
                                    rhs=0)
                                for i in set_I})
            constraints.update({"constraint_pstoneg_{}".format(i) : 
                                plp.LpConstraint(
                                    e=-P_sto_neg[i] - (1/self.plant_conf['eta_ch'])*self.plant_conf['Pc_max']*(1-E[i]),
                                    sense=plp.LpConstraintLE,
                                    rhs=0)
                                for i in set_I})
            constraints.update({"constraint_socmax_{}".format(i) : 
                                plp.LpConstraint(
                                    e=-plp.lpSum(P_sto_pos[j]*(1/self.plant_conf['eta_disch']) + self.plant_conf['eta_ch']*P_sto_neg[j] for j in range(i)),
                                    sense=plp.LpConstraintLE,
                                    rhs=(self.plant_conf['Enom']/self.timeStep)*(self.plant_conf['SOCmax'] - soc_init))
                                for i in set_I})
            constraints.update({"constraint_socmin_{}".format(i) : 
                                plp.LpConstraint(
                                    e=plp.lpSum(P_sto_pos[j]*(1/self.plant_conf['eta_disch']) + self.plant_conf['eta_ch']*P_sto_neg[j] for j in range(i)),
                                    sense=plp.LpConstraintLE,
                                    rhs=(self.plant_conf['Enom']/self.timeStep)*(soc_init - self.plant_conf['SOCmin']))
                                for i in set_I})
            constraints.update({"constraint_socfinal_{}".format(0) : 
                                plp.LpConstraint(
                                    e=plp.lpSum(P_sto_pos[i]*(1/self.plant_conf['eta_disch']) + self.plant_conf['eta_ch']*P_sto_neg[i] for i in set_I),
                                    sense=plp.LpConstraintEQ,
                                    rhs=(soc_init - soc_final)*self.plant_conf['Enom']/self.timeStep)
                                })
        opt_model.constraints = constraints
    
        ## Finally, we call the solver to solve our optimization model:
        # solving with default solver CBC
        try:
            if self.lp_solver == 'PULP_CBC_CMD':
                opt_model.solve(PULP_CBC_CMD(msg=0))
            elif self.lp_solver == 'GLPK_CMD':
                opt_model.solve(GLPK_CMD(msg=0))
            elif self.lp_solver == 'COIN_CMD':
                opt_model.solve(COIN_CMD(msg=0, path=self.lp_solver_path))
            else:
                self.logger.error("Invalid solver name passed")
        except Exception:
            self.logger.error("It was not possible to find a valid solver for Pulp package")
        
        # The status of the solution is printed to the screen
        self.logger.info("Status: " + plp.LpStatus[opt_model.status])
        if plp.value(opt_model.objective) is None:
            self.logger.warning("Cost function cannot be evaluated, probably None")
        else:
            self.logger.info("Total value of the Cost function = " + str(round(plp.value(opt_model.objective),2)))
            
        # Build results Dataframe
        opt_tp = pd.DataFrame()
        opt_tp["P_PV"] = [P_PV[i] for i in set_I]
        opt_tp["P_Load"] = [P_load[i] for i in set_I]
        for k in range(self.optim_conf['num_def_loads']):
            opt_tp["P_deferrable{}".format(k)] = [P_deferrable[k][i].varValue for i in set_I]
        opt_tp["P_grid_pos"] = [P_grid_pos[i].varValue for i in set_I]
        opt_tp["P_grid_neg"] = [P_grid_neg[i].varValue for i in set_I]
        opt_tp["P_grid"] = [P_grid_pos[i].varValue + P_grid_neg[i].varValue for i in set_I]
        if self.optim_conf['set_use_battery']:
            opt_tp["P_batt"] = [P_sto_pos[i].varValue + P_sto_neg[i].varValue for i in set_I]
            SOC_opt_delta = [(P_sto_pos[i].varValue*(1/self.plant_conf['eta_disch']) + \
                              self.plant_conf['eta_ch']*P_sto_neg[i].varValue)*(
                                  self.timeStep/(self.plant_conf['Enom'])) for i in set_I]
            SOCinit = copy.copy(soc_init)
            SOC_opt = []
            for i in set_I:
                SOC_opt.append(SOCinit - SOC_opt_delta[i])
                SOCinit = SOC_opt[i]
            opt_tp["SOC_opt"] = SOC_opt
        opt_tp.index = data_opt.index
        
        # Lets compute the optimal cost function
        P_def_sum_tp = []
        for i in set_I:
            P_def_sum_tp.append(sum(P_deferrable[k][i].varValue for k in range(self.optim_conf['num_def_loads'])))
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
            opt_tp["cost_fun_cost"] = [-0.001*self.timeStep*unit_load_cost[i]*(P_load[i] + P_def_sum_tp[i]) for i in set_I]
        elif self.costfun == 'self-consumption':
            if type_self_conso == 'maxmin':
                opt_tp["cost_fun_selfcons"] = [-0.001*self.timeStep*unit_load_cost[i]*SC[i].varValue for i in set_I]
            elif type_self_conso == 'bigm':
                opt_tp["cost_fun_selfcons"] = [-0.001*self.timeStep*(unit_load_cost[i]*P_grid_pos[i].varValue + \
                    unit_prod_price[i]*P_grid_neg[i].varValue) for i in set_I]
        else:
            self.logger.error("The cost function specified type is not valid")
        
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
                                def_total_hours: Optional[list] = None) -> pd.DataFrame:
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
        :param prediction_horizon: The prediction horizon of the MPC controller in minutes.
        :type prediction_horizon: int
        :param soc_init: The initial battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_init: float
        :param soc_final: The final battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_final: 
        :param def_total_hours: The functioning hours for this iteration for each deferrable load.
        :type def_total_hours: list
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
                                                 soc_final=soc_final, def_total_hours=def_total_hours)
        return self.opt_res
    
