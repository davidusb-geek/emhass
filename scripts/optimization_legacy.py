import bz2
import copy
import logging
import os
import pickle as cPickle
from math import ceil

import numpy as np
import pandas as pd
import pulp as plp
from pulp import COIN_CMD, GLPK_CMD, PULP_CBC_CMD, HiGHS

from emhass import utils


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

    def __init__(
        self,
        retrieve_hass_conf: dict,
        optim_conf: dict,
        plant_conf: dict,
        var_load_cost: str,
        var_prod_price: str,
        costfun: str,
        emhass_conf: dict,
        logger: logging.Logger,
        opt_time_delta: int | None = 24,
    ) -> None:
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
        self.freq = self.retrieve_hass_conf["optimization_time_step"]
        self.time_zone = self.retrieve_hass_conf["time_zone"]
        self.time_step = self.freq.seconds / 3600  # in hours
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours")  # The period of optimization
        self.var_pv = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        self.var_load = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        self.var_load_new = self.var_load + "_positive"
        self.costfun = costfun
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.var_load_cost = var_load_cost
        self.var_prod_price = var_prod_price
        self.optim_status = None
        if "num_threads" in optim_conf.keys():
            if optim_conf["num_threads"] == 0:
                self.num_threads = int(os.cpu_count())
            else:
                self.num_threads = int(optim_conf["num_threads"])
        else:
            self.num_threads = int(os.cpu_count())
        if "lp_solver" in optim_conf.keys():
            self.lp_solver = optim_conf["lp_solver"]
        else:
            self.lp_solver = "default"
        if "lp_solver_path" in optim_conf.keys():
            self.lp_solver_path = optim_conf["lp_solver_path"]
        else:
            self.lp_solver_path = "empty"
        if self.lp_solver != "COIN_CMD" and self.lp_solver_path != "empty":
            self.logger.error(
                "Use COIN_CMD solver name if you want to set a path for the LP solver"
            )
        if (
            self.lp_solver == "COIN_CMD" and self.lp_solver_path == "empty"
        ):  # if COIN_CMD but lp_solver_path is empty
            self.logger.warning(
                "lp_solver=COIN_CMD but lp_solver_path=empty, attempting to use lp_solver_path=/usr/bin/cbc"
            )
            self.lp_solver_path = "/usr/bin/cbc"
        # Mask sensitive data before logging
        conf_to_log = retrieve_hass_conf.copy()
        keys_to_mask = [
            "influxdb_username",
            "influxdb_password",
            "long_lived_token",
            "Latitude",
            "Longitude",
            "solcast_api_key",
            "solcast_rooftop_id",
        ]
        for key in keys_to_mask:
            if key in conf_to_log:
                conf_to_log[key] = "***"
        self.logger.debug(f"Initialized Optimization with retrieve_hass_conf: {conf_to_log}")
        self.logger.debug(f"Optimization configuration: {optim_conf}")
        self.logger.debug(f"Plant configuration: {plant_conf}")
        self.logger.debug(
            f"Solver configuration: lp_solver={self.lp_solver}, lp_solver_path={self.lp_solver_path}"
        )
        self.logger.debug(f"Number of threads: {self.num_threads}")

    def _setup_stress_cost(self, set_i, cost_conf_key, max_power, var_name_prefix):
        """
        Generic setup for a stress cost (battery or inverter).
        """
        stress_unit_cost = self.plant_conf.get(cost_conf_key, 0)
        active = stress_unit_cost > 0 and max_power > 0

        stress_cost_vars = None
        if active:
            self.logger.debug(
                f"Stress cost enabled for {var_name_prefix}. "
                f"Unit Cost: {stress_unit_cost}/kWh at full load {max_power}W."
            )
            stress_cost_vars = {
                i: plp.LpVariable(
                    cat="Continuous",
                    lowBound=0,
                    name=f"{var_name_prefix}_stress_cost_{i}",
                )
                for i in set_i
            }

        return {
            "active": active,
            "vars": stress_cost_vars,
            "unit_cost": stress_unit_cost,
            "max_power": max_power,
            # Defaults to 10 segments if not provided in config
            "segments": self.plant_conf.get(f"{var_name_prefix}_stress_segments", 10),
        }

    def _build_stress_segments(self, max_power, stress_unit_cost, segments):
        """
        Generic builder for Piece-Wise Linear segments for a quadratic cost curve.
        """
        # Cost rate at nominal power (currency/hr)
        max_cost_rate_hr = (max_power / 1000.0) * stress_unit_cost
        max_cost_step = max_cost_rate_hr * self.time_step

        x_points = np.linspace(0, max_power, segments + 1)
        y_points = max_cost_step * (x_points / max_power) ** 2

        seg_params = []
        for k in range(segments):
            x0, x1 = x_points[k], x_points[k + 1]
            y0, y1 = y_points[k], y_points[k + 1]
            slope = (y1 - y0) / (x1 - x0)
            intercept = y0 - slope * x0
            seg_params.append((k, slope, intercept))
        return seg_params

    def _add_stress_constraints(
        self, constraints, set_i, power_var_func, stress_vars, seg_params, prefix
    ):
        """
        Generic constraint adder for stress costs.
        :param power_var_func: A function(i) that returns the LpVariable or Expression
                               representing the power to be penalized at index i.
        """
        for k, slope, intercept in seg_params:
            for sign, suffix in ((+1, "pos"), (-1, "neg")):
                for i in set_i:
                    name = f"constraint_stress_pwl_{prefix}_{suffix}_{k}_{i}"
                    constraints[name] = plp.LpConstraint(
                        e=stress_vars[i] - (sign * slope * power_var_func(i) + intercept),
                        sense=plp.LpConstraintGE,
                        rhs=0,
                    )

    def _get_clean_list(self, key, data_opt):
        """Helper to extract list from DataFrame/Series/List safely."""
        val = data_opt.get(key)
        if hasattr(val, "values"):
            return val.values.tolist()
        return val if isinstance(val, list) else []

    def _initialize_decision_variables(self, set_i, num_deferrable_loads):
        """Initialize all main decision variables for the LP problem."""
        vars_dict = {}

        # Grid power variables
        vars_dict["p_grid_neg"] = {
            (i): plp.LpVariable(
                cat="Continuous",
                lowBound=-self.plant_conf["maximum_power_to_grid"],
                upBound=0,
                name=f"P_grid_neg{i}",
            )
            for i in set_i
        }
        vars_dict["p_grid_pos"] = {
            (i): plp.LpVariable(
                cat="Continuous",
                lowBound=0,
                upBound=self.plant_conf["maximum_power_from_grid"],
                name=f"P_grid_pos{i}",
            )
            for i in set_i
        }

        # Deferrable load variables
        p_deferrable = []
        p_def_bin1 = []
        p_def_start = []
        p_def_bin2 = []

        for k in range(num_deferrable_loads):
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                up_bound = np.max(self.optim_conf["nominal_power_of_deferrable_loads"][k])
            else:
                up_bound = self.optim_conf["nominal_power_of_deferrable_loads"][k]

            if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
                p_deferrable.append(
                    {
                        (i): plp.LpVariable(cat="Continuous", name=f"P_deferrable{k}_{i}")
                        for i in set_i
                    }
                )
            else:
                p_deferrable.append(
                    {
                        (i): plp.LpVariable(
                            cat="Continuous",
                            lowBound=0,
                            upBound=up_bound,
                            name=f"P_deferrable{k}_{i}",
                        )
                        for i in set_i
                    }
                )
            p_def_bin1.append(
                {(i): plp.LpVariable(cat="Binary", name=f"P_def{k}_bin1_{i}") for i in set_i}
            )
            p_def_start.append(
                {(i): plp.LpVariable(cat="Binary", name=f"P_def{k}_start_{i}") for i in set_i}
            )
            p_def_bin2.append(
                {(i): plp.LpVariable(cat="Binary", name=f"P_def{k}_bin2_{i}") for i in set_i}
            )

        vars_dict["p_deferrable"] = p_deferrable
        vars_dict["p_def_bin1"] = p_def_bin1
        vars_dict["p_def_start"] = p_def_start
        vars_dict["p_def_bin2"] = p_def_bin2

        # Binary indicators for Grid and Battery direction
        vars_dict["D"] = {(i): plp.LpVariable(cat="Binary", name=f"D_{i}") for i in set_i}
        vars_dict["E"] = {(i): plp.LpVariable(cat="Binary", name=f"E_{i}") for i in set_i}

        # Battery power variables
        if self.optim_conf["set_use_battery"]:
            vars_dict["p_sto_pos"] = {
                (i): plp.LpVariable(
                    cat="Continuous",
                    lowBound=0,
                    upBound=self.plant_conf["battery_discharge_power_max"],
                    name=f"P_sto_pos_{i}",
                )
                for i in set_i
            }
            vars_dict["p_sto_neg"] = {
                (i): plp.LpVariable(
                    cat="Continuous",
                    lowBound=-np.abs(self.plant_conf["battery_charge_power_max"]),
                    upBound=0,
                    name=f"P_sto_neg_{i}",
                )
                for i in set_i
            }
        else:
            vars_dict["p_sto_pos"] = {(i): i * 0 for i in set_i}
            vars_dict["p_sto_neg"] = {(i): i * 0 for i in set_i}

        # Self-consumption variable
        if self.costfun == "self-consumption":
            vars_dict["SC"] = {(i): plp.LpVariable(cat="Continuous", name=f"SC_{i}") for i in set_i}

        # Hybrid Inverter variable
        if self.plant_conf["inverter_is_hybrid"]:
            vars_dict["p_hybrid_inverter"] = {
                (i): plp.LpVariable(cat="Continuous", name=f"P_hybrid_inverter{i}") for i in set_i
            }

        # Curtailment variable
        vars_dict["p_pv_curtailment"] = {
            (i): plp.LpVariable(cat="Continuous", lowBound=0, name=f"P_PV_curtailment{i}")
            for i in set_i
        }

        # Sum of deferrable loads
        p_def_sum = []
        for i in set_i:
            p_def_sum.append(plp.lpSum(p_deferrable[k][i] for k in range(num_deferrable_loads)))
        vars_dict["p_def_sum"] = p_def_sum

        return vars_dict

    def _build_objective_function(
        self,
        vars_dict,
        unit_load_cost,
        unit_prod_price,
        p_load,
        set_i,
        batt_stress_conf,
        inv_stress_conf,
        type_self_conso="bigm",
    ):
        """Construct the objective function based on configuration."""
        p_grid_pos = vars_dict["p_grid_pos"]
        p_grid_neg = vars_dict["p_grid_neg"]
        p_sto_pos = vars_dict["p_sto_pos"]
        p_sto_neg = vars_dict["p_sto_neg"]
        p_def_sum = vars_dict["p_def_sum"]
        p_def_start = vars_dict["p_def_start"]
        SC = vars_dict.get("SC", None)

        objective = plp.lpSum(0 for _ in set_i)

        # Base Cost Function
        if self.costfun == "profit":
            if self.optim_conf["set_total_pv_sell"]:
                objective = plp.lpSum(
                    -0.001
                    * self.time_step
                    * (
                        unit_load_cost[i] * (p_load[i] + p_def_sum[i])
                        + unit_prod_price[i] * p_grid_neg[i]
                    )
                    for i in set_i
                )
            else:
                objective = plp.lpSum(
                    -0.001
                    * self.time_step
                    * (unit_load_cost[i] * p_grid_pos[i] + unit_prod_price[i] * p_grid_neg[i])
                    for i in set_i
                )
        elif self.costfun == "cost":
            if self.optim_conf["set_total_pv_sell"]:
                objective = plp.lpSum(
                    -0.001 * self.time_step * unit_load_cost[i] * (p_load[i] + p_def_sum[i])
                    for i in set_i
                )
            else:
                objective = plp.lpSum(
                    -0.001 * self.time_step * unit_load_cost[i] * p_grid_pos[i] for i in set_i
                )
        elif self.costfun == "self-consumption":
            if type_self_conso == "bigm":
                bigm = 1e3
                objective = plp.lpSum(
                    -0.001
                    * self.time_step
                    * (
                        bigm * unit_load_cost[i] * p_grid_pos[i]
                        + unit_prod_price[i] * p_grid_neg[i]
                    )
                    for i in set_i
                )
            elif type_self_conso == "maxmin":
                objective = plp.lpSum(
                    0.001 * self.time_step * unit_load_cost[i] * SC[i] for i in set_i
                )

        # Battery cycle cost penalty
        if self.optim_conf["set_use_battery"]:
            objective += plp.lpSum(
                -0.001
                * self.time_step
                * (
                    self.optim_conf["weight_battery_discharge"] * p_sto_pos[i]
                    - self.optim_conf["weight_battery_charge"] * p_sto_neg[i]
                )
                for i in set_i
            )

        # Startup penalties
        if (
            "set_deferrable_startup_penalty" in self.optim_conf
            and self.optim_conf["set_deferrable_startup_penalty"]
        ):
            for k in range(self.optim_conf["number_of_deferrable_loads"]):
                if (
                    len(self.optim_conf["set_deferrable_startup_penalty"]) > k
                    and self.optim_conf["set_deferrable_startup_penalty"][k]
                ):
                    objective += plp.lpSum(
                        -0.001
                        * self.time_step
                        * self.optim_conf["set_deferrable_startup_penalty"][k]
                        * p_def_start[k][i]
                        * unit_load_cost[i]
                        * self.optim_conf["nominal_power_of_deferrable_loads"][k]
                        for i in set_i
                    )

        # Stress Costs
        if inv_stress_conf and inv_stress_conf["active"]:
            objective -= plp.lpSum(inv_stress_conf["vars"][i] for i in set_i)

        if batt_stress_conf and batt_stress_conf["active"]:
            self.logger.debug("Adding battery stress cost to objective function")
            objective -= plp.lpSum(batt_stress_conf["vars"][i] for i in set_i)

        return objective

    def _add_main_power_balance_constraints(self, constraints, vars_dict, p_pv, p_load, set_i):
        """Add the main power balance constraints."""
        p_hybrid_inverter = vars_dict.get("p_hybrid_inverter")
        p_def_sum = vars_dict["p_def_sum"]
        p_grid_neg = vars_dict["p_grid_neg"]
        p_grid_pos = vars_dict["p_grid_pos"]
        p_pv_curtailment = vars_dict["p_pv_curtailment"]
        p_sto_pos = vars_dict["p_sto_pos"]
        p_sto_neg = vars_dict["p_sto_neg"]

        if self.plant_conf["inverter_is_hybrid"]:
            for i in set_i:
                constraints[f"constraint_main1_{i}"] = plp.LpConstraint(
                    e=p_hybrid_inverter[i]
                    - p_def_sum[i]
                    - p_load[i]
                    + p_grid_neg[i]
                    + p_grid_pos[i],
                    sense=plp.LpConstraintEQ,
                    rhs=0,
                )
        else:
            if self.plant_conf["compute_curtailment"]:
                for i in set_i:
                    constraints[f"constraint_main2_{i}"] = plp.LpConstraint(
                        e=p_pv[i]
                        - p_pv_curtailment[i]
                        - p_def_sum[i]
                        - p_load[i]
                        + p_grid_neg[i]
                        + p_grid_pos[i]
                        + p_sto_pos[i]
                        + p_sto_neg[i],
                        sense=plp.LpConstraintEQ,
                        rhs=0,
                    )
            else:
                for i in set_i:
                    constraints[f"constraint_main3_{i}"] = plp.LpConstraint(
                        e=p_pv[i]
                        - p_def_sum[i]
                        - p_load[i]
                        + p_grid_neg[i]
                        + p_grid_pos[i]
                        + p_sto_pos[i]
                        + p_sto_neg[i],
                        sense=plp.LpConstraintEQ,
                        rhs=0,
                    )

        # Grid Constraints (Simultaneous import/export prevention)
        D = vars_dict["D"]
        for i in set_i:
            constraints[f"constraint_pgridpos_{i}"] = plp.LpConstraint(
                e=p_grid_pos[i] - self.plant_conf["maximum_power_from_grid"] * D[i],
                sense=plp.LpConstraintLE,
                rhs=0,
            )
            constraints[f"constraint_pgridneg_{i}"] = plp.LpConstraint(
                e=-p_grid_neg[i] - self.plant_conf["maximum_power_to_grid"] * (1 - D[i]),
                sense=plp.LpConstraintLE,
                rhs=0,
            )

    def _add_hybrid_inverter_constraints(
        self, constraints, vars_dict, p_pv, set_i, inv_stress_conf
    ):
        """Add constraints specific to hybrid inverters."""
        if not self.plant_conf["inverter_is_hybrid"]:
            return

        p_hybrid_inverter = vars_dict["p_hybrid_inverter"]
        p_pv_curtailment = vars_dict["p_pv_curtailment"]
        p_sto_pos = vars_dict["p_sto_pos"]
        p_sto_neg = vars_dict["p_sto_neg"]

        p_nom_inverter_output = self.plant_conf.get("inverter_ac_output_max", None)
        p_nom_inverter_input = self.plant_conf.get("inverter_ac_input_max", None)

        # Fallback to legacy pv_inverter_model if output power is not provided
        if p_nom_inverter_output is None:
            if "pv_inverter_model" in self.plant_conf:
                if isinstance(self.plant_conf["pv_inverter_model"], list):
                    p_nom_inverter_output = 0.0
                    for i in range(len(self.plant_conf["pv_inverter_model"])):
                        if isinstance(self.plant_conf["pv_inverter_model"][i], str):
                            cec_inverters = bz2.BZ2File(
                                self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2",
                                "rb",
                            )
                            cec_inverters = cPickle.load(cec_inverters)
                            inverter = cec_inverters[self.plant_conf["pv_inverter_model"][i]]
                            p_nom_inverter_output += inverter.Paco
                        else:
                            p_nom_inverter_output += self.plant_conf["pv_inverter_model"][i]
                else:
                    if isinstance(self.plant_conf["pv_inverter_model"], str):
                        cec_inverters = bz2.BZ2File(
                            self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2",
                            "rb",
                        )
                        cec_inverters = cPickle.load(cec_inverters)
                        inverter = cec_inverters[self.plant_conf["pv_inverter_model"]]
                        p_nom_inverter_output = inverter.Paco
                    else:
                        p_nom_inverter_output = self.plant_conf["pv_inverter_model"]

        if p_nom_inverter_input is None:
            p_nom_inverter_input = p_nom_inverter_output

        eff_dc_ac = self.plant_conf.get("inverter_efficiency_dc_ac", 1.0)
        eff_ac_dc = self.plant_conf.get("inverter_efficiency_ac_dc", 1.0)

        p_dc_ac_max = p_nom_inverter_output / eff_dc_ac
        p_ac_dc_max = p_nom_inverter_input * eff_ac_dc

        # Define internal DC power flow variables
        p_dc_ac = {
            (i): plp.LpVariable(
                cat="Continuous",
                lowBound=0,
                upBound=p_dc_ac_max,
                name=f"P_dc_ac_{i}",
            )
            for i in set_i
        }
        p_ac_dc = {
            (i): plp.LpVariable(
                cat="Continuous",
                lowBound=0,
                upBound=p_ac_dc_max,
                name=f"P_ac_dc_{i}",
            )
            for i in set_i
        }
        is_dc_sourcing = {
            (i): plp.LpVariable(cat="Binary", name=f"is_dc_sourcing_{i}") for i in set_i
        }

        for i in set_i:
            if self.plant_conf["compute_curtailment"]:
                e_dc_balance = (p_pv[i] - p_pv_curtailment[i] + p_sto_pos[i] + p_sto_neg[i]) - (
                    p_dc_ac[i] - p_ac_dc[i]
                )
            else:
                e_dc_balance = (p_pv[i] + p_sto_pos[i] + p_sto_neg[i]) - (p_dc_ac[i] - p_ac_dc[i])

            constraints[f"constraint_dc_bus_balance_{i}"] = plp.LpConstraint(
                e=e_dc_balance, sense=plp.LpConstraintEQ, rhs=0
            )

            constraints[f"constraint_ac_bus_balance_{i}"] = plp.LpConstraint(
                e=p_hybrid_inverter[i]
                - ((p_dc_ac[i] * eff_dc_ac) - (p_ac_dc[i] * (1.0 / eff_ac_dc))),
                sense=plp.LpConstraintEQ,
                rhs=0,
            )

            constraints[f"constraint_enforce_ac_dc_zero_{i}"] = plp.LpConstraint(
                e=p_ac_dc[i] - (1 - is_dc_sourcing[i]) * p_ac_dc_max,
                sense=plp.LpConstraintLE,
                rhs=0,
            )
            constraints[f"constraint_enforce_dc_ac_zero_{i}"] = plp.LpConstraint(
                e=p_dc_ac[i] - is_dc_sourcing[i] * p_dc_ac_max,
                sense=plp.LpConstraintLE,
                rhs=0,
            )

        # Inverter Stress Cost
        if inv_stress_conf and inv_stress_conf["active"]:
            seg_params = self._build_stress_segments(
                inv_stress_conf["max_power"],
                inv_stress_conf["unit_cost"],
                inv_stress_conf["segments"],
            )
            self._add_stress_constraints(
                constraints,
                set_i,
                lambda i: p_hybrid_inverter[i],
                inv_stress_conf["vars"],
                seg_params,
                "inv",
            )

    def _add_battery_constraints(
        self, constraints, vars_dict, p_pv, set_i, soc_init, soc_final, n, batt_stress_conf
    ):
        """Add all battery-related constraints."""
        if not self.optim_conf["set_use_battery"]:
            return

        p_sto_pos = vars_dict["p_sto_pos"]
        p_sto_neg = vars_dict["p_sto_neg"]
        p_grid_neg = vars_dict["p_grid_neg"]
        E = vars_dict["E"]

        # No charge from grid
        if self.optim_conf["set_nocharge_from_grid"]:
            for i in set_i:
                constraints[f"constraint_nocharge_from_grid_{i}"] = plp.LpConstraint(
                    e=p_sto_neg[i] + p_pv[i], sense=plp.LpConstraintGE, rhs=0
                )

        # No discharge to grid
        if self.optim_conf["set_nodischarge_to_grid"]:
            for i in set_i:
                constraints[f"constraint_nodischarge_to_grid_{i}"] = plp.LpConstraint(
                    e=p_grid_neg[i] + p_pv[i], sense=plp.LpConstraintGE, rhs=0
                )

        # Dynamic Power Limits
        if self.optim_conf["set_battery_dynamic"]:
            for i in range(n - 1):
                constraints[f"constraint_pos_batt_dynamic_max_{i}"] = plp.LpConstraint(
                    e=p_sto_pos[i + 1] - p_sto_pos[i],
                    sense=plp.LpConstraintLE,
                    rhs=self.time_step
                    * self.optim_conf["battery_dynamic_max"]
                    * self.plant_conf["battery_discharge_power_max"],
                )
                constraints[f"constraint_pos_batt_dynamic_min_{i}"] = plp.LpConstraint(
                    e=p_sto_pos[i + 1] - p_sto_pos[i],
                    sense=plp.LpConstraintGE,
                    rhs=self.time_step
                    * self.optim_conf["battery_dynamic_min"]
                    * self.plant_conf["battery_discharge_power_max"],
                )
                constraints[f"constraint_neg_batt_dynamic_max_{i}"] = plp.LpConstraint(
                    e=p_sto_neg[i + 1] - p_sto_neg[i],
                    sense=plp.LpConstraintLE,
                    rhs=self.time_step
                    * self.optim_conf["battery_dynamic_max"]
                    * self.plant_conf["battery_charge_power_max"],
                )
                constraints[f"constraint_neg_batt_dynamic_min_{i}"] = plp.LpConstraint(
                    e=p_sto_neg[i + 1] - p_sto_neg[i],
                    sense=plp.LpConstraintGE,
                    rhs=self.time_step
                    * self.optim_conf["battery_dynamic_min"]
                    * self.plant_conf["battery_charge_power_max"],
                )

        # Basic Battery Constraints
        for i in set_i:
            constraints[f"constraint_pstopos_{i}"] = plp.LpConstraint(
                e=p_sto_pos[i]
                - self.plant_conf["battery_discharge_efficiency"]
                * self.plant_conf["battery_discharge_power_max"]
                * E[i],
                sense=plp.LpConstraintLE,
                rhs=0,
            )
            constraints[f"constraint_pstoneg_{i}"] = plp.LpConstraint(
                e=-p_sto_neg[i]
                - (1 / self.plant_conf["battery_charge_efficiency"])
                * self.plant_conf["battery_charge_power_max"]
                * (1 - E[i]),
                sense=plp.LpConstraintLE,
                rhs=0,
            )
            constraints[f"constraint_socmax_{i}"] = plp.LpConstraint(
                e=-plp.lpSum(
                    p_sto_pos[j] * (1 / self.plant_conf["battery_discharge_efficiency"])
                    + self.plant_conf["battery_charge_efficiency"] * p_sto_neg[j]
                    for j in range(i)
                ),
                sense=plp.LpConstraintLE,
                rhs=(self.plant_conf["battery_nominal_energy_capacity"] / self.time_step)
                * (self.plant_conf["battery_maximum_state_of_charge"] - soc_init),
            )
            constraints[f"constraint_socmin_{i}"] = plp.LpConstraint(
                e=plp.lpSum(
                    p_sto_pos[j] * (1 / self.plant_conf["battery_discharge_efficiency"])
                    + self.plant_conf["battery_charge_efficiency"] * p_sto_neg[j]
                    for j in range(i)
                ),
                sense=plp.LpConstraintLE,
                rhs=(self.plant_conf["battery_nominal_energy_capacity"] / self.time_step)
                * (soc_init - self.plant_conf["battery_minimum_state_of_charge"]),
            )

        constraints[f"constraint_socfinal_{0}"] = plp.LpConstraint(
            e=plp.lpSum(
                p_sto_pos[i] * (1 / self.plant_conf["battery_discharge_efficiency"])
                + self.plant_conf["battery_charge_efficiency"] * p_sto_neg[i]
                for i in set_i
            ),
            sense=plp.LpConstraintEQ,
            rhs=(soc_init - soc_final)
            * self.plant_conf["battery_nominal_energy_capacity"]
            / self.time_step,
        )

        # Battery Stress Cost
        if batt_stress_conf and batt_stress_conf["active"]:
            self.logger.debug("Applying battery stress constraints to LP model")
            seg_params = self._build_stress_segments(
                batt_stress_conf["max_power"],
                batt_stress_conf["unit_cost"],
                batt_stress_conf["segments"],
            )
            self._add_stress_constraints(
                constraints,
                set_i,
                lambda i: p_sto_pos[i] - p_sto_neg[i],
                batt_stress_conf["vars"],
                seg_params,
                "batt",
            )

    def _add_thermal_load_constraints(
        self, constraints, vars_dict, k, data_opt, set_i, def_init_temp
    ):
        """Handle constraints for thermal deferrable loads."""
        p_deferrable = vars_dict["p_deferrable"]
        p_def_bin2 = vars_dict["p_def_bin2"]

        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_config"]
        self.logger.debug(f"Setting up Thermal Load {k}")

        start_temperature = (
            def_init_temp[k] if def_init_temp[k] is not None else hc.get("start_temperature", 20.0)
        )
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0

        outdoor_temp = self._get_clean_list("outdoor_temperature_forecast", data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = self._get_clean_list("temp_air", data_opt)

        required_len = len(data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            self.logger.warning("No outdoor temp found. Using default 15.0C.")
            outdoor_temp = [15.0] * required_len
        else:
            outdoor_temp = [15.0 if x is None else float(x) for x in outdoor_temp]
        if len(outdoor_temp) < required_len:
            outdoor_temp.extend([15.0] * (required_len - len(outdoor_temp)))

        cooling_constant = hc["cooling_constant"]
        heating_rate = hc["heating_rate"]
        overshoot_temperature = hc.get("overshoot_temperature", None)
        desired_temperatures = hc.get("desired_temperatures", [])
        min_temperatures = hc.get("min_temperatures", [])
        max_temperatures = hc.get("max_temperatures", [])
        sense = hc.get("sense", "heat")
        sense_coeff = 1 if sense == "heat" else -1

        predicted_temp = [start_temperature]
        penalty_terms = []

        for index in set_i:
            if index == 0:
                continue

            predicted_temp.append(
                predicted_temp[index - 1]
                + (
                    p_deferrable[k][index - 1]
                    * (
                        heating_rate
                        * self.time_step
                        / self.optim_conf["nominal_power_of_deferrable_loads"][k]
                    )
                )
                - (
                    cooling_constant
                    * self.time_step
                    * (predicted_temp[index - 1] - outdoor_temp[index - 1])
                )
            )

            if len(min_temperatures) > index and min_temperatures[index] is not None:
                constraints[f"constraint_defload{k}_min_temp_{index}"] = plp.LpConstraint(
                    e=predicted_temp[index], sense=plp.LpConstraintGE, rhs=min_temperatures[index]
                )
            if len(max_temperatures) > index and max_temperatures[index] is not None:
                constraints[f"constraint_defload{k}_max_temp_{index}"] = plp.LpConstraint(
                    e=predicted_temp[index], sense=plp.LpConstraintLE, rhs=max_temperatures[index]
                )

            # Overshoot logic (simplified for refactoring but keeping logic)
            if desired_temperatures and overshoot_temperature is not None:
                is_overshoot = plp.LpVariable(f"defload_{k}_overshoot_{index}")
                # Note: Constraints added directly to dict in original, mimicking that
                constraints[f"constraint_defload{k}_overshoot_{index}_1"] = plp.LpConstraint(
                    e=predicted_temp[index]
                    - overshoot_temperature
                    - (100 * sense_coeff * is_overshoot),
                    sense=plp.LpConstraintLE if sense == "heat" else plp.LpConstraintGE,
                    rhs=0,
                )
                constraints[f"constraint_defload{k}_overshoot_{index}_2"] = plp.LpConstraint(
                    e=predicted_temp[index]
                    - overshoot_temperature
                    + (100 * sense_coeff * (1 - is_overshoot)),
                    sense=plp.LpConstraintGE if sense == "heat" else plp.LpConstraintLE,
                    rhs=0,
                )
                constraints[f"constraint_defload{k}_overshoot_temp_{index}"] = plp.LpConstraint(
                    e=is_overshoot + p_def_bin2[k][index - 1],
                    sense=plp.LpConstraintLE,
                    rhs=1,
                )

                if len(desired_temperatures) > index and desired_temperatures[index]:
                    penalty_factor = hc.get("penalty_factor", 10)
                    penalty_value = (
                        (predicted_temp[index] - desired_temperatures[index])
                        * penalty_factor
                        * sense_coeff
                    )
                    penalty_var = plp.LpVariable(
                        f"defload_{k}_thermal_penalty_{index}",
                        cat="Continuous",
                        upBound=0,
                    )
                    constraints[f"constraint_defload{k}_penalty_{index}"] = plp.LpConstraint(
                        e=penalty_var - penalty_value,
                        sense=plp.LpConstraintLE,
                        rhs=0,
                    )
                    penalty_terms.append(penalty_var)

        if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
            for i in set_i:
                constraints[f"constraint_thermal_semicont_{k}_{i}"] = plp.LpConstraint(
                    e=p_deferrable[k][i]
                    - (p_def_bin2[k][i] * self.optim_conf["nominal_power_of_deferrable_loads"][k]),
                    sense=plp.LpConstraintEQ,
                    rhs=0,
                )

        return predicted_temp, None, plp.lpSum(penalty_terms)

    def _add_thermal_battery_constraints(self, constraints, vars_dict, k, data_opt, set_i):
        """Handle constraints for thermal battery loads."""
        p_deferrable = vars_dict["p_deferrable"]

        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_battery"]

        start_temperature = hc.get("start_temperature", 20.0)
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0

        outdoor_temp = self._get_clean_list("outdoor_temperature_forecast", data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = self._get_clean_list("temp_air", data_opt)

        required_len = len(data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = [15.0] * required_len
        else:
            outdoor_temp = [15.0 if x is None else float(x) for x in outdoor_temp]
        if len(outdoor_temp) < required_len:
            outdoor_temp.extend([15.0] * (required_len - len(outdoor_temp)))

        supply_temperature = hc["supply_temperature"]
        volume = hc["volume"]
        min_temperatures = hc["min_temperatures"]
        max_temperatures = hc["max_temperatures"]

        if not min_temperatures:
            raise ValueError(f"Load {k}: thermal_battery requires non-empty 'min_temperatures'")
        if not max_temperatures:
            raise ValueError(f"Load {k}: thermal_battery requires non-empty 'max_temperatures'")

        p_concr = 2400
        c_concr = 0.88
        loss = 0.045
        conversion = 3600 / (p_concr * c_concr * volume)

        heatpump_cops = utils.calculate_cop_heatpump(
            supply_temperature=supply_temperature,
            carnot_efficiency=hc.get("carnot_efficiency", 0.4),
            outdoor_temperature_forecast=outdoor_temp,
        )
        thermal_losses = utils.calculate_thermal_loss_signed(
            outdoor_temperature_forecast=outdoor_temp,
            indoor_temperature=start_temperature,
            base_loss=loss,
        )

        if all(
            key in hc
            for key in [
                "u_value",
                "envelope_area",
                "ventilation_rate",
                "heated_volume",
            ]
        ):
            indoor_target_temp = hc.get(
                "indoor_target_temperature",
                min_temperatures[0] if min_temperatures else 20.0,
            )
            window_area = hc.get("window_area", None)
            shgc = hc.get("shgc", 0.6)
            solar_irradiance = None
            if "ghi" in data_opt.columns and window_area is not None:
                solar_irradiance = data_opt["ghi"]

            heating_demand = utils.calculate_heating_demand_physics(
                u_value=hc["u_value"],
                envelope_area=hc["envelope_area"],
                ventilation_rate=hc["ventilation_rate"],
                heated_volume=hc["heated_volume"],
                indoor_target_temperature=indoor_target_temp,
                outdoor_temperature_forecast=outdoor_temp,
                optimization_time_step=int(self.freq.total_seconds() / 60),
                solar_irradiance_forecast=solar_irradiance,
                window_area=window_area,
                shgc=shgc,
            )
        else:
            base_temperature = hc.get("base_temperature", 18.0)
            annual_reference_hdd = hc.get("annual_reference_hdd", 3000.0)
            heating_demand = utils.calculate_heating_demand(
                specific_heating_demand=hc["specific_heating_demand"],
                floor_area=hc["area"],
                outdoor_temperature_forecast=outdoor_temp,
                base_temperature=base_temperature,
                annual_reference_hdd=annual_reference_hdd,
                optimization_time_step=int(self.freq.total_seconds() / 60),
            )

        predicted_temp_thermal = [start_temperature]
        for index in set_i:
            if index == 0:
                continue

            predicted_temp_thermal.append(
                predicted_temp_thermal[index - 1]
                + conversion
                * (
                    heatpump_cops[index - 1] * p_deferrable[k][index - 1] / 1000 * self.time_step
                    - heating_demand[index - 1]
                    - thermal_losses[index - 1]
                )
            )

            if len(min_temperatures) > index and min_temperatures[index] is not None:
                constraints[f"constraint_thermal_battery{k}_min_temp_{index}"] = plp.LpConstraint(
                    e=predicted_temp_thermal[index],
                    sense=plp.LpConstraintGE,
                    rhs=min_temperatures[index],
                )

            if len(max_temperatures) > index and max_temperatures[index] is not None:
                constraints[f"constraint_thermal_battery{k}_max_temp_{index}"] = plp.LpConstraint(
                    e=predicted_temp_thermal[index],
                    sense=plp.LpConstraintLE,
                    rhs=max_temperatures[index],
                )

        return predicted_temp_thermal, heating_demand

    def _add_deferrable_load_constraints(
        self,
        constraints,
        vars_dict,
        data_opt,
        set_i,
        n,
        def_total_hours,
        def_total_timestep,
        def_start_timestep,
        def_end_timestep,
        def_init_temp,
        min_power_of_deferrable_loads,
    ):
        """Master helper for all deferrable load constraints."""
        p_deferrable = vars_dict["p_deferrable"]
        p_def_bin1 = vars_dict["p_def_bin1"]
        p_def_start = vars_dict["p_def_start"]
        p_def_bin2 = vars_dict["p_def_bin2"]

        predicted_temps = {}
        heating_demands = {}
        penalty_terms_total = plp.lpSum([])
        M = 100000

        for k in range(self.optim_conf["number_of_deferrable_loads"]):
            self.logger.debug(f"Processing deferrable load {k}")

            # 1. Sequence-based Deferrable Load
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                power_sequence = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                sequence_length = len(power_sequence)

                def create_matrix(input_list, n):
                    matrix = []
                    for i in range(n + 1):
                        row = [0] * i + input_list + [0] * (n - i)
                        matrix.append(row[: n * 2])
                    return matrix

                matrix = create_matrix(power_sequence, n - sequence_length)
                y = plp.LpVariable.dicts(f"y{k}", range(len(matrix)), cat="Binary")

                constraints[f"single_value_constraint_{k}"] = plp.LpConstraint(
                    e=plp.lpSum(y[i] for i in range(len(matrix))) - 1,
                    sense=plp.LpConstraintEQ,
                    rhs=0,
                )
                constraints[f"pdef{k}_sumconstraint"] = plp.LpConstraint(
                    e=plp.lpSum(p_deferrable[k][i] for i in set_i) - np.sum(power_sequence),
                    sense=plp.LpConstraintEQ,
                    rhs=0,
                )
                for i in set_i:
                    constraints[f"pdef{k}_positive_constraint_{i}"] = plp.LpConstraint(
                        e=p_deferrable[k][i], sense=plp.LpConstraintGE, rhs=0
                    )
                for num, mat in enumerate(matrix):
                    for i in set_i:
                        constraints[f"pdef{k}_value_constraint_{num}_{i}"] = plp.LpConstraint(
                            e=p_deferrable[k][i] - mat[i] * y[num],
                            sense=plp.LpConstraintEQ,
                            rhs=0,
                        )

            # 2. Thermal Deferrable Load
            elif (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_config" in self.optim_conf["def_load_config"][k]
            ):
                pred_temp, _, penalty_term = self._add_thermal_load_constraints(
                    constraints, vars_dict, k, data_opt, set_i, def_init_temp
                )
                predicted_temps[k] = pred_temp
                if penalty_term is not None:
                    penalty_terms_total += penalty_term

            # 3. Thermal Battery Load
            elif (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_battery" in self.optim_conf["def_load_config"][k]
            ):
                pred_temp, heat_demand = self._add_thermal_battery_constraints(
                    constraints, vars_dict, k, data_opt, set_i
                )
                predicted_temps[k] = pred_temp
                heating_demands[k] = heat_demand

            # 4. Standard Deferrable Load
            elif (def_total_timestep and def_total_timestep[k] > 0) or (
                len(def_total_hours) > k and def_total_hours[k] > 0
            ):
                target_energy = 0
                if def_total_timestep and def_total_timestep[k] > 0:
                    target_energy = (self.time_step * def_total_timestep[k]) * self.optim_conf[
                        "nominal_power_of_deferrable_loads"
                    ][k]
                else:
                    target_energy = (
                        def_total_hours[k] * self.optim_conf["nominal_power_of_deferrable_loads"][k]
                    )

                constraints[f"constraint_defload{k}_energy"] = plp.LpConstraint(
                    e=plp.lpSum(p_deferrable[k][i] * self.time_step for i in set_i),
                    sense=plp.LpConstraintEQ,
                    rhs=target_energy,
                )

            # Time Window Logic
            if def_total_timestep and def_total_timestep[k] > 0:
                def_start, def_end, warning = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_timestep[k]),
                    n,
                )
            else:
                def_start, def_end, warning = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_hours[k] / self.time_step),
                    n,
                )
            if warning is not None:
                self.logger.warning(f"Deferrable load {k} : {warning}")

            if def_start > 0:
                constraints[f"constraint_defload{k}_start_timestep"] = plp.LpConstraint(
                    e=plp.lpSum(p_deferrable[k][i] * self.time_step for i in range(0, def_start)),
                    sense=plp.LpConstraintEQ,
                    rhs=0,
                )
            if def_end > 0:
                constraints[f"constraint_defload{k}_end_timestep"] = plp.LpConstraint(
                    e=plp.lpSum(p_deferrable[k][i] * self.time_step for i in range(def_end, n)),
                    sense=plp.LpConstraintEQ,
                    rhs=0,
                )

            # Minimum Power
            if min_power_of_deferrable_loads[k] > 0:
                for i in set_i:
                    constraints[f"constraint_pdef{k}_min_power_{i}"] = plp.LpConstraint(
                        e=p_deferrable[k][i]
                        - (min_power_of_deferrable_loads[k] * p_def_bin2[k][i]),
                        sense=plp.LpConstraintGE,
                        rhs=0,
                    )

            # Startup Logic
            current_state = 0
            if (
                "def_current_state" in self.optim_conf
                and len(self.optim_conf["def_current_state"]) > k
            ):
                current_state = 1 if self.optim_conf["def_current_state"][k] else 0

            for i in set_i:
                constraints[f"constraint_pdef{k}_start1_{i}"] = plp.LpConstraint(
                    e=p_deferrable[k][i] - p_def_bin2[k][i] * M,
                    sense=plp.LpConstraintLE,
                    rhs=0,
                )
                constraints[f"constraint_pdef{k}_start1a_{i}"] = plp.LpConstraint(
                    e=p_def_bin2[k][i] - p_deferrable[k][i],
                    sense=plp.LpConstraintLE,
                    rhs=0,
                )
                constraints[f"constraint_pdef{k}_start2_{i}"] = plp.LpConstraint(
                    e=p_def_start[k][i]
                    - p_def_bin2[k][i]
                    + (p_def_bin2[k][i - 1] if i - 1 >= 0 else current_state),
                    sense=plp.LpConstraintGE,
                    rhs=0,
                )
                constraints[f"constraint_pdef{k}_start3_{i}"] = plp.LpConstraint(
                    e=(p_def_bin2[k][i - 1] if i - 1 >= 0 else 0) + p_def_start[k][i],
                    sense=plp.LpConstraintLE,
                    rhs=1,
                )

            # Single Constant Start
            if self.optim_conf["set_deferrable_load_single_constant"][k]:
                constraints[f"constraint_pdef{k}_start4"] = plp.LpConstraint(
                    e=plp.lpSum(p_def_start[k][i] for i in set_i),
                    sense=plp.LpConstraintEQ,
                    rhs=1,
                )
                rhs_val = (
                    def_total_timestep[k]
                    if (def_total_timestep and def_total_timestep[k] > 0)
                    else def_total_hours[k] / self.time_step
                )
                constraints[f"constraint_pdef{k}_start5"] = plp.LpConstraint(
                    e=plp.lpSum(p_def_bin2[k][i] for i in set_i),
                    sense=plp.LpConstraintEQ,
                    rhs=rhs_val,
                )

            # Semi-continuous
            if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
                for i in set_i:
                    constraints[f"constraint_pdef{k}_semicont1_{i}"] = plp.LpConstraint(
                        e=p_deferrable[k][i]
                        - self.optim_conf["nominal_power_of_deferrable_loads"][k]
                        * p_def_bin1[k][i],
                        sense=plp.LpConstraintGE,
                        rhs=0,
                    )
                    constraints[f"constraint_pdef{k}_semicont2_{i}"] = plp.LpConstraint(
                        e=p_deferrable[k][i]
                        - self.optim_conf["nominal_power_of_deferrable_loads"][k]
                        * p_def_bin1[k][i],
                        sense=plp.LpConstraintLE,
                        rhs=0,
                    )

        return predicted_temps, heating_demands, penalty_terms_total

    def _build_results_dataframe(
        self,
        vars_dict,
        data_opt,
        set_i,
        unit_load_cost,
        unit_prod_price,
        p_load,
        soc_init,
        predicted_temps,
        heating_demands,
        debug,
    ):
        """Build the final results DataFrame."""
        opt_tp = pd.DataFrame()
        opt_tp["P_PV"] = [vars_dict.get("p_pv_curtailment", {})[i].varValue for i in set_i]
        # Note: Logic will fill correct P_PV values later in main method

        # Deferrable Loads
        for k in range(self.optim_conf["number_of_deferrable_loads"]):
            opt_tp[f"P_deferrable{k}"] = [vars_dict["p_deferrable"][k][i].varValue for i in set_i]

        opt_tp["P_grid_pos"] = [vars_dict["p_grid_pos"][i].varValue for i in set_i]
        opt_tp["P_grid_neg"] = [vars_dict["p_grid_neg"][i].varValue for i in set_i]
        opt_tp["P_grid"] = opt_tp["P_grid_pos"] + opt_tp["P_grid_neg"]

        if self.optim_conf["set_use_battery"]:
            p_sto_pos = vars_dict["p_sto_pos"]
            p_sto_neg = vars_dict["p_sto_neg"]
            opt_tp["P_batt"] = [p_sto_pos[i].varValue + p_sto_neg[i].varValue for i in set_i]

            soc_opt_delta = [
                (
                    p_sto_pos[i].varValue * (1 / self.plant_conf["battery_discharge_efficiency"])
                    + self.plant_conf["battery_charge_efficiency"] * p_sto_neg[i].varValue
                )
                * (self.time_step / (self.plant_conf["battery_nominal_energy_capacity"]))
                for i in set_i
            ]
            soc_init_curr = copy.copy(soc_init)
            soc_opt = []
            for i in set_i:
                soc_opt.append(soc_init_curr - soc_opt_delta[i])
                soc_init_curr = soc_opt[i]
            opt_tp["SOC_opt"] = soc_opt

        if self.plant_conf["inverter_is_hybrid"]:
            opt_tp["P_hybrid_inverter"] = [
                vars_dict["p_hybrid_inverter"][i].varValue for i in set_i
            ]

        if self.plant_conf["compute_curtailment"]:
            opt_tp["P_PV_curtailment"] = [vars_dict["p_pv_curtailment"][i].varValue for i in set_i]

        opt_tp.index = data_opt.index

        # Add thermal details
        for i, predicted_temp in predicted_temps.items():
            opt_tp[f"predicted_temp_heater{i}"] = pd.Series(
                [
                    round(pt.value(), 2) if isinstance(pt, plp.LpAffineExpression) else pt
                    for pt in predicted_temp
                ],
                index=opt_tp.index,
            )
            if "thermal_config" in self.optim_conf["def_load_config"][i]:
                thermal_config = self.optim_conf["def_load_config"][i]["thermal_config"]
                target_temps = (
                    thermal_config.get("desired_temperatures")
                    or thermal_config.get("min_temperatures")
                    or thermal_config.get("max_temperatures")
                )
                if target_temps:
                    opt_tp[f"target_temp_heater{i}"] = pd.Series(target_temps, index=opt_tp.index)

        for i, heating_demand in heating_demands.items():
            opt_tp[f"heating_demand_heater{i}"] = pd.Series(heating_demand, index=opt_tp.index)

        if debug:
            for k in range(self.optim_conf["number_of_deferrable_loads"]):
                opt_tp[f"P_def_start_{k}"] = [
                    vars_dict["p_def_start"][k][i].varValue for i in set_i
                ]
                opt_tp[f"P_def_bin2_{k}"] = [vars_dict["p_def_bin2"][k][i].varValue for i in set_i]

        return opt_tp

    def perform_optimization(
        self,
        data_opt: pd.DataFrame,
        p_pv: np.array,
        p_load: np.array,
        unit_load_cost: np.array,
        unit_prod_price: np.array,
        soc_init: float | None = None,
        soc_final: float | None = None,
        def_total_hours: list | None = None,
        def_total_timestep: list | None = None,
        def_start_timestep: list | None = None,
        def_end_timestep: list | None = None,
        def_init_temp: list | None = None,
        debug: bool | None = False,
    ) -> pd.DataFrame:
        r"""
        Perform the actual optimization using linear programming (LP).
        """
        # --- Initialization ---
        if self.optim_conf["set_use_battery"]:
            if soc_init is None:
                if soc_final is not None:
                    soc_init = soc_final
                else:
                    soc_init = self.plant_conf["battery_target_state_of_charge"]
            if soc_final is None:
                if soc_init is not None:
                    soc_final = soc_init
                else:
                    soc_final = self.plant_conf["battery_target_state_of_charge"]
            self.logger.debug(
                f"Battery usage enabled. Initial SOC: {soc_init}, Final SOC: {soc_final}"
            )

        if def_total_timestep is not None:
            if def_total_hours is None:
                def_total_hours = self.optim_conf["operating_hours_of_each_deferrable_load"]
            def_total_hours = [0 if x != 0 else x for x in def_total_hours]
        elif def_total_hours is None:
            def_total_hours = self.optim_conf["operating_hours_of_each_deferrable_load"]

        if def_start_timestep is None:
            def_start_timestep = self.optim_conf["start_timesteps_of_each_deferrable_load"]
        if def_end_timestep is None:
            def_end_timestep = self.optim_conf["end_timesteps_of_each_deferrable_load"]

        if def_init_temp is None:
            def_init_temp = [None] * self.optim_conf["number_of_deferrable_loads"]

        num_deferrable_loads = self.optim_conf["number_of_deferrable_loads"]
        min_power_of_deferrable_loads = self.optim_conf.get(
            "minimum_power_of_deferrable_loads", [0] * num_deferrable_loads
        )
        min_power_of_deferrable_loads = min_power_of_deferrable_loads + [0] * (
            num_deferrable_loads - len(min_power_of_deferrable_loads)
        )
        def_total_hours = def_total_hours + [0] * (num_deferrable_loads - len(def_total_hours))
        def_start_timestep = def_start_timestep + [0] * (
            num_deferrable_loads - len(def_start_timestep)
        )
        def_end_timestep = def_end_timestep + [0] * (num_deferrable_loads - len(def_end_timestep))

        # --- Variables Setup ---
        n = len(data_opt.index)
        set_i = range(n)
        opt_model = plp.LpProblem("LP_Model", plp.LpMaximize)

        vars_dict = self._initialize_decision_variables(set_i, num_deferrable_loads)

        # --- Stress Costs Setup ---
        inv_stress_conf = None
        batt_stress_conf = None
        if self.optim_conf["set_use_battery"]:
            p_batt_max = max(
                self.plant_conf.get("battery_discharge_power_max", 0),
                self.plant_conf.get("battery_charge_power_max", 0),
            )
            batt_stress_conf = self._setup_stress_cost(
                set_i, "battery_stress_cost", p_batt_max, "battery"
            )
        if self.plant_conf["inverter_is_hybrid"]:
            P_nom_inverter_max = max(
                self.plant_conf.get("inverter_ac_output_max", 0),
                self.plant_conf.get("inverter_ac_input_max", 0),
            )
            inv_stress_conf = self._setup_stress_cost(
                set_i, "inverter_stress_cost", P_nom_inverter_max, "inv"
            )

        # --- Objective Function ---
        objective = self._build_objective_function(
            vars_dict,
            unit_load_cost,
            unit_prod_price,
            p_load,
            set_i,
            batt_stress_conf,
            inv_stress_conf,
        )

        # --- Constraints ---
        constraints = {}

        self._add_main_power_balance_constraints(constraints, vars_dict, p_pv, p_load, set_i)

        self._add_hybrid_inverter_constraints(constraints, vars_dict, p_pv, set_i, inv_stress_conf)

        self._add_battery_constraints(
            constraints, vars_dict, p_pv, set_i, soc_init, soc_final, n, batt_stress_conf
        )

        if self.plant_conf["compute_curtailment"]:
            for i in set_i:
                constraints[f"constraint_curtailment_{i}"] = plp.LpConstraint(
                    e=vars_dict["p_pv_curtailment"][i] - max(p_pv[i], 0),
                    sense=plp.LpConstraintLE,
                    rhs=0,
                )

        if self.costfun == "self-consumption" and "SC" in vars_dict:
            for i in set_i:
                constraints[f"constraint_selfcons_PV1_{i}"] = plp.LpConstraint(
                    e=vars_dict["SC"][i] - p_pv[i], sense=plp.LpConstraintLE, rhs=0
                )
                constraints[f"constraint_selfcons_PV2_{i}"] = plp.LpConstraint(
                    e=vars_dict["SC"][i] - p_load[i] - vars_dict["p_def_sum"][i],
                    sense=plp.LpConstraintLE,
                    rhs=0,
                )

        predicted_temps, heating_demands, penalty_terms_total = (
            self._add_deferrable_load_constraints(
                constraints,
                vars_dict,
                data_opt,
                set_i,
                n,
                def_total_hours,
                def_total_timestep,
                def_start_timestep,
                def_end_timestep,
                def_init_temp,
                min_power_of_deferrable_loads,
            )
        )

        # Add penalty terms to the objective function
        objective += penalty_terms_total

        # Set the final objective
        opt_model.setObjective(objective)
        opt_model.constraints = constraints

        # --- Solve ---
        timeout = self.optim_conf["lp_solver_timeout"]
        if self.lp_solver == "PULP_CBC_CMD":
            opt_model.solve(PULP_CBC_CMD(msg=0, timeLimit=timeout, threads=self.num_threads))
        elif self.lp_solver == "GLPK_CMD":
            opt_model.solve(GLPK_CMD(msg=0, timeLimit=timeout))
        elif self.lp_solver == "HiGHS":
            opt_model.solve(HiGHS(msg=0, timeLimit=timeout))
        elif self.lp_solver == "COIN_CMD":
            opt_model.solve(
                COIN_CMD(
                    msg=0,
                    path=self.lp_solver_path,
                    timeLimit=timeout,
                    threads=self.num_threads,
                )
            )
        else:
            self.logger.warning("Solver %s unknown, using default", self.lp_solver)
            opt_model.solve(PULP_CBC_CMD(msg=0, timeLimit=timeout, threads=self.num_threads))

        self.optim_status = plp.LpStatus[opt_model.status]
        self.logger.info("Status: " + self.optim_status)
        if plp.value(opt_model.objective) is None:
            self.logger.warning("Cost function cannot be evaluated")
            return
        else:
            self.logger.info(
                "Total value of the Cost function = %.02f",
                plp.value(opt_model.objective),
            )

        # --- Results ---
        opt_tp = self._build_results_dataframe(
            vars_dict,
            data_opt,
            set_i,
            unit_load_cost,
            unit_prod_price,
            p_load,
            soc_init,
            predicted_temps,
            heating_demands,
            debug,
        )

        # Fill in raw input data columns
        opt_tp["P_PV"] = [p_pv[i] for i in set_i]
        opt_tp["P_Load"] = [p_load[i] for i in set_i]

        # Add stress cost results columns
        if batt_stress_conf and batt_stress_conf["active"]:
            opt_tp["batt_stress_cost"] = [batt_stress_conf["vars"][i].varValue for i in set_i]
        if inv_stress_conf and inv_stress_conf["active"]:
            opt_tp["inv_stress_cost"] = [inv_stress_conf["vars"][i].varValue for i in set_i]

        # Cost calculations
        p_def_sum_tp = []
        for i in set_i:
            p_def_sum_tp.append(
                sum(
                    vars_dict["p_deferrable"][k][i].varValue
                    for k in range(self.optim_conf["number_of_deferrable_loads"])
                )
            )
        opt_tp["unit_load_cost"] = [unit_load_cost[i] for i in set_i]
        opt_tp["unit_prod_price"] = [unit_prod_price[i] for i in set_i]

        # Cost Profit Columns
        # Recalculate cost based on actual variable values
        if self.optim_conf["set_total_pv_sell"]:
            val = [
                -0.001
                * self.time_step
                * (
                    unit_load_cost[i] * (p_load[i] + p_def_sum_tp[i])
                    + unit_prod_price[i] * vars_dict["p_grid_neg"][i].varValue
                )
                for i in set_i
            ]
        else:
            val = [
                -0.001
                * self.time_step
                * (
                    unit_load_cost[i] * vars_dict["p_grid_pos"][i].varValue
                    + unit_prod_price[i] * vars_dict["p_grid_neg"][i].varValue
                )
                for i in set_i
            ]
        opt_tp["cost_profit"] = val

        # Specific Cost Function Column
        if self.costfun == "profit":
            opt_tp["cost_fun_profit"] = val
        elif self.costfun == "cost":
            if self.optim_conf["set_total_pv_sell"]:
                opt_tp["cost_fun_cost"] = [
                    -0.001 * self.time_step * unit_load_cost[i] * (p_load[i] + p_def_sum_tp[i])
                    for i in set_i
                ]
            else:
                opt_tp["cost_fun_cost"] = [
                    -0.001
                    * self.time_step
                    * unit_load_cost[i]
                    * vars_dict["p_grid_pos"][i].varValue
                    for i in set_i
                ]
        elif self.costfun == "self-consumption":
            if vars_dict.get("SC"):  # maxmin
                opt_tp["cost_fun_selfcons"] = [
                    -0.001 * self.time_step * unit_load_cost[i] * vars_dict["SC"][i].varValue
                    for i in set_i
                ]
            else:  # bigm
                opt_tp["cost_fun_selfcons"] = [
                    -0.001
                    * self.time_step
                    * (
                        unit_load_cost[i] * vars_dict["p_grid_pos"][i].varValue
                        + unit_prod_price[i] * vars_dict["p_grid_neg"][i].varValue
                    )
                    for i in set_i
                ]

        opt_tp["optim_status"] = self.optim_status
        return opt_tp

    def perform_perfect_forecast_optim(
        self, df_input_data: pd.DataFrame, days_list: pd.date_range
    ) -> pd.DataFrame:
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
        self.days_list_tz = days_list.tz_convert(self.time_zone).round(self.freq)[
            :-1
        ]  # Converted to tz and without the current day (today)
        self.opt_res = pd.DataFrame()
        for day in self.days_list_tz:
            self.logger.info(
                "Solving for day: " + str(day.day) + "-" + str(day.month) + "-" + str(day.year)
            )
            # Prepare data
            if day.tzinfo is None:
                day = day.replace(tzinfo=self.time_zone)  # Assign timezone if naive
            else:
                day = day.astimezone(self.time_zone)
            day_start = day
            day_end = day + self.time_delta - self.freq
            if day_start.tzinfo != day_end.tzinfo:
                self.logger.warning(
                    f"Skipping day {day} as days have ddifferent timezone, probably because of DST."
                )
                continue  # Skip this day and move to the next iteration
            else:
                day_start = day_start.astimezone(self.time_zone).isoformat()
                day_end = day_end.astimezone(self.time_zone).isoformat()
                # Generate the date range for the current day
                day_range = pd.date_range(start=day_start, end=day_end, freq=self.freq)
            # Check if all timestamps in the range exist in the DataFrame index
            if not day_range.isin(df_input_data.index).all():
                self.logger.warning(
                    f"Skipping day {day} as some timestamps are missing in the data."
                )
                continue  # Skip this day and move to the next iteration
            # If all timestamps exist, proceed with the data preparation
            data_tp = df_input_data.copy().loc[day_range]
            p_pv = data_tp[self.var_pv].values
            p_load = data_tp[self.var_load_new].values
            unit_load_cost = data_tp[self.var_load_cost].values  # €/kWh
            unit_prod_price = data_tp[self.var_prod_price].values  # €/kWh
            # Call optimization function
            opt_tp = self.perform_optimization(
                data_tp, p_pv, p_load, unit_load_cost, unit_prod_price
            )
            if len(self.opt_res) == 0:
                self.opt_res = opt_tp
            else:
                self.opt_res = pd.concat([self.opt_res, opt_tp], axis=0)

        return self.opt_res

    def perform_dayahead_forecast_optim(
        self, df_input_data: pd.DataFrame, p_pv: pd.Series, p_load: pd.Series
    ) -> pd.DataFrame:
        r"""
        Perform a day-ahead optimization task using real forecast data. \
        This type of optimization is intented to be launched once a day.

        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably the unit load cost for power consumption.
        :type df_input_data: pandas.DataFrame
        :param p_pv: The forecasted PV power production.
        :type p_pv: pandas.DataFrame
        :param p_load: The forecasted Load power consumption. This power should \
            not include the power from the deferrable load that we want to find.
        :type p_load: pandas.DataFrame
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform optimization for the day-ahead")
        unit_load_cost = df_input_data[self.var_load_cost].values  # €/kWh
        unit_prod_price = df_input_data[self.var_prod_price].values  # €/kWh
        # Call optimization function
        self.opt_res = self.perform_optimization(
            df_input_data,
            p_pv.values.ravel(),
            p_load.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )
        return self.opt_res

    def perform_naive_mpc_optim(
        self,
        df_input_data: pd.DataFrame,
        p_pv: pd.Series,
        p_load: pd.Series,
        prediction_horizon: int,
        soc_init: float | None = None,
        soc_final: float | None = None,
        def_total_hours: list | None = None,
        def_total_timestep: list | None = None,
        def_start_timestep: list | None = None,
        def_end_timestep: list | None = None,
    ) -> pd.DataFrame:
        r"""
        Perform a naive approach to a Model Predictive Control (MPC). \
        This implementaion is naive because we are not using the formal formulation \
        of a MPC. Only the sense of a receiding horizon is considered here. \
        This optimization is more suitable for higher optimization frequency, ex: 5min.

        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably the unit load cost for power consumption.
        :type df_input_data: pandas.DataFrame
        :param p_pv: The forecasted PV power production.
        :type p_pv: pandas.DataFrame
        :param p_load: The forecasted Load power consumption. This power should \
            not include the power from the deferrable load that we want to find.
        :type p_load: pandas.DataFrame
        :param prediction_horizon: The prediction horizon of the MPC controller in number \
            of optimization time steps.
        :type prediction_horizon: int
        :param soc_init: The initial battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_init: float
        :param soc_final: The final battery SOC for the optimization. This parameter \
            is optional, if not given soc_init = soc_final = soc_target from the configuration file.
        :type soc_final:
        :param def_total_timestep: The functioning timesteps for this iteration for each deferrable load. \
            (For continuous deferrable loads: functioning timesteps at nominal power)
        :type def_total_timestep: list
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
            self.logger.error(
                "Set the MPC prediction horizon to at least 5 times the optimization time step"
            )
            return pd.DataFrame()
        else:
            df_input_data = copy.deepcopy(df_input_data)[
                df_input_data.index[0] : df_input_data.index[prediction_horizon - 1]
            ]
        unit_load_cost = df_input_data[self.var_load_cost].values  # €/kWh
        unit_prod_price = df_input_data[self.var_prod_price].values  # €/kWh
        # Call optimization function
        self.opt_res = self.perform_optimization(
            df_input_data,
            p_pv.values.ravel(),
            p_load.values.ravel(),
            unit_load_cost,
            unit_prod_price,
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
        )
        return self.opt_res

    @staticmethod
    def validate_def_timewindow(
        start: int, end: int, min_steps: int, window: int
    ) -> tuple[int, int, str]:
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
                if (end_validated - start_validated) < min_steps:
                    warning = "Available timeframe is shorter than the specified number of hours to operate. Optimization will fail."
        else:
            warning = "Invalid timeframe for deferrable load (start timestep is not <= end timestep). Continuing optimization without timewindow constraint."
        return start_validated, end_validated, warning
