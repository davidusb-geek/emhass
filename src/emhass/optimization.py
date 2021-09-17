#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
import pandas as pd
import pulp as plp
from pulp import PULP_CBC_CMD

from emhass.utils import get_logger

class optimization(ABC):
    """
    Optimize the deferrable load and battery energy dispatch problem using \ 
    the linear programming optimization technique. All equipement equations, \
    including the battery equations are hence transformed in a linear form.
    
    This class methods are:

    - get_load_unit_cost

    - perform_optimization

    - perform_perfect_forecast_optim

    - perform_dayahead_forecast_optim
    
    """

    def __init__(self, retrieve_hass_conf, optim_conf, plant_conf, days_list, 
                 config_path, logger, opt_time_delta=24):
        """
        Define constructor for optimization class.
        
        :param retrieve_hass_conf: Configuration parameters used to retrieve data \
            from hass
        :type retrieve_hass_conf: dict
        :param optim_conf: Configuration parameters used for the optimization task
        :type optim_conf: dict
        :param plant_conf: Configuration parameters used to model the electrical \
            system: PV production, battery, etc.
        :type plant_conf: dict
        :param days_list: A list of the days of data that will be retrieved from \
            hass and used for the optimization task. We will retrieve data from \
            now and up to days_to_retrieve days
        :type days_list: list
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
        self.days_list = days_list
        self.var_cost = 'unit_load_cost'
        self.freq = self.retrieve_hass_conf['freq']
        self.time_zone = self.retrieve_hass_conf['time_zone']
        self.timeStep = self.freq.seconds/3600 # in hours
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours") # The period of optimization
        self.var_PV = self.retrieve_hass_conf['var_PV']
        self.var_load = self.retrieve_hass_conf['var_load']
        self.var_load_new = self.var_load+'_positive'
        # create logger
        self.logger, self.ch = get_logger(__name__, config_path, file=logger.fileSetting)
        
    def get_load_unit_cost(self, df_final):
        """
        Get the unit cost for the load consumption based on multiple tariff \
        periods. This is the cost of the energy from the utility in a vector \
        sampled at the fixed freq value.
        
        :param df_input_data: The DataFrame containing all the input data retrieved
            from hass
        :type df_input_data: pd.DataFrame
        :return: The input DataFrame with one additionnal column appended containing
            the load cost by unit of time
        :rtype: pd.DataFrame

        """
        df_final[self.var_cost] = self.optim_conf['load_cost_hc']
        list_df_hp = []
        for key, period_hp in self.optim_conf['list_hp_periods'].items():
            list_df_hp.append(df_final[self.var_cost].between_time(
                period_hp[0]['start'], period_hp[1]['end']))
        for df_hp in list_df_hp:
            df_final.loc[df_hp.index, self.var_cost] = self.optim_conf['load_cost_hp']
        return df_final
        
    def perform_optimization(self, data_opt, P_PV, P_load, unit_load_cost):
        r"""
        Perform the actual optimization using linear programming (LP).

        The LP problem for the household EMS is posed to maximize the profit. \
        In this case this is defined by the revenues from selling PV power to \
        the grid minus the consummed energy cost. This can be represented with \
        the following obtective function:

        .. math::

            \sum_{i=1}^{\Delta_{opt}/\Delta_t} 0.001*\Delta_t(-(prod_{SellPrice}*P_{gridNeg_i}+unit_{LoadCost_i}*(P_{load_i}+P_{defSum_i})))
        
        where :math:`\Delta_{opt}` is the total period of optimization in hours, \
        :math:`\Delta_t` is the optimization time step in hours, :math:`unit_{LoadCost_i}` \
        is the cost of the energy from the utility in EUR/kWh, :math:`P_{load}` is the \
        electricity load consumption, :math:`P_{defSum}` is the sum of the deferrable \
        loads defined, :math:`prod_{SellPrice}` is the price of the energy sold to \
        the utility, :math:`P_{gridNeg}` is the negative component of the grid power, \
        this is the power exported to the grid. \
        The goal is to maximize the auto-consumption from PV power.\
        The problem constraints are written as follows: \

        - The main constraint: power balance \

        .. math::

            P_{PV_i}-P_{defSum_i}-P_{load_i}+P_{gridNeg_i}+P_{gridPos_i}+P_{stoPos_i}+P_{stoNeg_i}=0

        with :math:`P_{PV}` the PV power production, :math:`P_{gridPos}` the \
        positive component of the grid power (from grid to household), \
        :math:`P_{stoPos}` and :math:`P_{stoNeg}` are the positive (discharge) \
        and negative components of the battery power (charge)
        
        :param data_tp: A DataFrame containing the input data. The results of the \
            optimization will be appended (decision variables, cost function values, etc)
        :type data_tp: pd.DataFrame
        :param P_PV: The photovoltaic power values. This can be real historical \
            values or forecasted values.
        :type P_PV: numpy.Array
        :param P_load: The load power consumption values
        :type P_load: np.Array
        :param unit_load_cost: The cost of power consumption for each unit of time. \
            This is the cost of the energy from the utility in a vector sampled \
            at the fixed freq value
        :type unit_load_cost: np.Array
        :return: The input DataFrame with all the different results from the \
            optimization appended
        :rtype: pd.DataFrame

        """
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
            
        ## Define objective
        P_def_sum= []
        for i in set_I:
            P_def_sum.append(plp.lpSum(P_deferrable[k][i] for k in range(self.optim_conf['num_def_loads'])))
        objective = plp.lpSum(-(unit_load_cost[i]*(P_load[i] + P_def_sum[i])*0.001*self.timeStep + \
                        self.optim_conf['prod_sell_price'] * P_grid_neg[i] * 0.001*self.timeStep)
            for i in set_I)
        opt_model.setObjective(objective)
        
        ## Setting constraints
        # The main constraint: power balance
        constraints = {"constraint_main1_{}".format(i) :
                       plp.LpConstraint(
                           e = P_PV[i] - P_def_sum[i] - P_load[i] + P_grid_neg[i] + P_grid_pos[i] + P_sto_pos[i] + P_sto_neg[i],
                           sense = plp.LpConstraintEQ,
                           rhs = 0)
                       for i in set_I}
        
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
                                    e = plp.lpSum(P_deferrable[k][i] for i in set_I),
                                    sense = plp.LpConstraintEQ,
                                    rhs = self.optim_conf['def_total_hours'][k]*self.optim_conf['P_deferrable_nom'][k])
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
                                    rhs=(self.plant_conf['Enom']/self.timeStep)*(self.plant_conf['SOCmax'] - self.plant_conf['SOCtarget']))
                                for i in set_I})
            constraints.update({"constraint_socmin_{}".format(i) : 
                                plp.LpConstraint(
                                    e=plp.lpSum(P_sto_pos[j]*(1/self.plant_conf['eta_disch']) + self.plant_conf['eta_ch']*P_sto_neg[j] for j in range(i)),
                                    sense=plp.LpConstraintLE,
                                    rhs=(self.plant_conf['Enom']/self.timeStep)*(self.plant_conf['SOCtarget'] - self.plant_conf['SOCmin']))
                                for i in set_I})
            constraints.update({"constraint_socfinal_{}".format(0) : 
                                plp.LpConstraint(
                                    e=plp.lpSum(P_sto_pos[i]*(1/self.plant_conf['eta_disch']) + self.plant_conf['eta_ch']*P_sto_neg[i] for i in set_I),
                                    sense=plp.LpConstraintEQ,
                                    rhs=0)
                                })
        opt_model.constraints = constraints
    
        ## Finally, we call the solver to solve our optimization model:
        # solving with default solver CBC
        opt_model.solve(PULP_CBC_CMD(msg=0))
        
        # The status of the solution is printed to the screen
        self.logger.info("Status: " + plp.LpStatus[opt_model.status])
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
                              self.plant_conf['eta_ch']*P_sto_neg[i].varValue)*(self.timeStep/(self.plant_conf['Enom'])) for i in set_I]
            SOCinit = self.plant_conf['SOCtarget']
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
        opt_tp["cost_fun"] = [-(unit_load_cost[i]*(P_load[i] + P_def_sum_tp[i]) * 0.001 * self.timeStep + \
                              self.optim_conf['prod_sell_price'] * P_grid_neg[i].varValue * 0.001 * self.timeStep)
                              for i in set_I]
        
        return opt_tp

    def perform_perfect_forecast_optim(self, df_input_data):
        """
        Perform an optimization on historical data (perfectly known PV production).
        
        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably photovoltaics and load consumption powers.
        :type df_input_data: pandas.DataFrame
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform optimization for perfect forecast scenario")
        self.days_list_tz = self.days_list.tz_convert(self.time_zone).round(self.freq)[:-1] # Converted to tz and without the current day (today)
        self.opt_res = pd.DataFrame()
        for day in self.days_list_tz:
            self.logger.info("Solving for day: "+str(day.day)+"-"+str(day.month)+"-"+str(day.year))
            # Prepare data
            day_start = day.isoformat()
            day_end = (day+self.time_delta-self.freq).isoformat()
            data_tp = df_input_data.copy().loc[pd.date_range(start=day_start, end=day_end, freq=self.freq)]
            P_PV = data_tp[self.var_PV].values
            P_load = data_tp[self.var_load_new].values
            unit_load_cost = data_tp[self.var_cost].values # €/kWh
            # Call optimization function
            opt_tp = self.perform_optimization(data_tp, P_PV, P_load, unit_load_cost)
            if len(self.opt_res) == 0:
                self.opt_res = opt_tp
            else:
                self.opt_res = pd.concat([self.opt_res, opt_tp], axis=0)
        
        return self.opt_res
        
    def perform_dayahead_forecast_optim(self, df_input_data, P_PV, P_load):
        """
        Perform a day-ahead optimization task using real forecast data.
        
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
        self.opt_res = pd.DataFrame()
        
        unit_load_cost = df_input_data[self.var_cost].values # €/kWh
        
        # Call optimization function
        self.opt_res = self.perform_optimization(P_load, P_PV.values.ravel(), 
                                                 P_load.values.ravel(), unit_load_cost)
        
        return self.opt_res
    