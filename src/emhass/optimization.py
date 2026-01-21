import bz2
import copy
import logging
import os
import pickle
from math import ceil

import cvxpy as cp
import numpy as np
import pandas as pd

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

        # Configuration for Solver
        if "num_threads" in optim_conf.keys():
            if optim_conf["num_threads"] == 0:
                self.num_threads = int(os.cpu_count())
            else:
                self.num_threads = int(optim_conf["num_threads"])
        else:
            self.num_threads = int(os.cpu_count())

        # Force HiGHS solver or use configured one, defaulting to Highs if not specified
        if "lp_solver" in optim_conf.keys():
            self.lp_solver = optim_conf["lp_solver"]
        else:
            self.lp_solver = "Highs"  # Default to Highs for speed

        # Mask sensitive data before logging
        conf_to_log = retrieve_hass_conf.copy()
        keys_to_mask = utils.get_keys_to_mask()
        for key in keys_to_mask:
            if key in conf_to_log:
                conf_to_log[key] = "***"
        self.logger.debug(f"Initialized Optimization with retrieve_hass_conf: {conf_to_log}")
        self.logger.debug(f"Optimization configuration: {optim_conf}")
        self.logger.debug(f"Plant configuration: {plant_conf}")
        self.logger.debug(f"Number of threads: {self.num_threads}")

        # CVXPY Initialization
        # Calculate the fixed number of time steps (N)
        self.num_timesteps = int(self.time_delta / self.freq)
        self.logger.debug(f"CVXPY: Initialization with {self.num_timesteps} time steps.")

        # Define Parameters (Data holders)
        # These will be updated in perform_optimization without rebuilding the problem
        self.param_pv_forecast = cp.Parameter(self.num_timesteps, name="pv_forecast")
        self.param_load_forecast = cp.Parameter(self.num_timesteps, name="load_forecast")
        self.param_load_cost = cp.Parameter(self.num_timesteps, name="load_cost")
        self.param_prod_price = cp.Parameter(self.num_timesteps, name="prod_price")

        # Scalar Parameters
        self.param_soc_init = cp.Parameter(nonneg=True, name="soc_init")
        self.param_soc_final = cp.Parameter(nonneg=True, name="soc_final")

        # Initialize Variables & Bound Constraints
        self.vars, self.constraints = self._initialize_decision_variables()

        # Note: The self.prob object will be constructed in a subsequent step
        self.prob = None

    def _prepare_power_limit_array(self, limit_value, limit_name, data_length):
        """
        Convert power limit to numpy array for time-varying constraints.

        Args:
            limit_value: Scalar, list, or array of power limit values
            limit_name: Name of the limit (for logging)
            data_length: Expected length of optimization horizon

        Returns:
            numpy.ndarray: Array of power limits with length = data_length
        """
        if limit_value is None:
            self.logger.error(f"{limit_name} is None, using default value 9000 W")
            return np.full(data_length, 9000.0)

        # Convert to numpy array if it's a list
        if isinstance(limit_value, list):
            limit_array = np.array(limit_value, dtype=float)
        elif isinstance(limit_value, np.ndarray):
            limit_array = limit_value.astype(float)
        else:
            # Scalar value - broadcast to all timesteps
            return np.full(data_length, float(limit_value))

        # Validate length
        if len(limit_array) != data_length:
            self.logger.warning(
                f"{limit_name} length ({len(limit_array)}) doesn't match "
                f"optimization horizon ({data_length}). Using scalar from first value."
            )
            return np.full(data_length, float(limit_array[0]) if len(limit_array) > 0 else 9000.0)

        self.logger.info(f"{limit_name} configured as time-varying with {data_length} values")
        return limit_array

    def _setup_stress_cost(self, cost_conf_key, max_power, var_name_prefix):
        """
        Generic setup for a stress cost (battery or inverter).
        """
        stress_unit_cost = self.plant_conf.get(cost_conf_key, 0)
        active = stress_unit_cost > 0 and max_power > 0

        stress_cost_var = None
        if active:
            self.logger.debug(
                f"Stress cost enabled for {var_name_prefix}. "
                f"Unit Cost: {stress_unit_cost}/kWh at full load {max_power}W."
            )
            stress_cost_var = cp.Variable(
                self.num_timesteps, nonneg=True, name=f"{var_name_prefix}_stress_cost"
            )

        return {
            "active": active,
            "vars": stress_cost_var,
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
            seg_params.append((slope, intercept))
        return seg_params

    def _add_stress_constraints(self, constraints, power_expression, stress_var, seg_params):
        """
        Generic constraint adder for stress costs (Vectorized).

        :param constraints: List to append constraints to
        :param power_expression: CVXPY expression (vector) for the power to be penalized
        :param stress_var: CVXPY variable (vector) for the stress cost
        :param seg_params: List of (slope, intercept) tuples
        """
        for slope, intercept in seg_params:
            # Vectorized constraints for both positive and negative directions (symmetry).
            # This creates a convex envelope around |power_expression|.
            constraints.append(stress_var >= slope * power_expression + intercept)
            constraints.append(stress_var >= -slope * power_expression + intercept)

    def _get_clean_list(self, key, data_opt):
        """Helper to extract list from DataFrame/Series/List safely."""
        val = data_opt.get(key)
        if hasattr(val, "values"):
            return val.values.tolist()
        return val if isinstance(val, list) else []

    def _initialize_decision_variables(self):
        """
        Initialize all main decision variables for the CVXPY problem.

        Returns:
            vars_dict: Dictionary containing cvxpy Variables
            constraints: List of bounds constraints associated with these variables
        """
        vars_dict = {}
        constraints = []
        n = self.num_timesteps

        # Prepare Power Limits
        max_power_from_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
        )
        max_power_to_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
        )

        # Grid power variables
        # P_grid_neg <= 0
        vars_dict["p_grid_neg"] = cp.Variable(n, nonpos=True, name="p_grid_neg")
        # Apply vectorized lower bound constraint
        constraints.append(vars_dict["p_grid_neg"] >= -max_power_to_grid_arr)

        # P_grid_pos >= 0
        vars_dict["p_grid_pos"] = cp.Variable(n, nonneg=True, name="p_grid_pos")
        # Apply vectorized upper bound constraint
        constraints.append(vars_dict["p_grid_pos"] <= max_power_from_grid_arr)

        # Deferrable load variables
        num_deferrable_loads = self.optim_conf["number_of_deferrable_loads"]
        p_deferrable = []
        p_def_bin1 = []
        p_def_start = []
        p_def_bin2 = []

        for k in range(num_deferrable_loads):
            # Calculate Upper Bound
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                up_bound = np.max(self.optim_conf["nominal_power_of_deferrable_loads"][k])
            else:
                up_bound = self.optim_conf["nominal_power_of_deferrable_loads"][k]

            # Continuous/Semi-Continuous Power Variable
            var_p_def = cp.Variable(n, nonneg=True, name=f"p_deferrable_{k}")
            p_deferrable.append(var_p_def)

            # Global upper bound (specific semi-continuous logic handled in constraints)
            constraints.append(var_p_def <= up_bound)

            # Binary Variables
            p_def_bin1.append(cp.Variable(n, boolean=True, name=f"p_def_bin1_{k}"))
            p_def_start.append(cp.Variable(n, boolean=True, name=f"p_def_start_{k}"))
            p_def_bin2.append(cp.Variable(n, boolean=True, name=f"p_def_bin2_{k}"))

        vars_dict["p_deferrable"] = p_deferrable
        vars_dict["p_def_bin1"] = p_def_bin1
        vars_dict["p_def_start"] = p_def_start
        vars_dict["p_def_bin2"] = p_def_bin2

        # Binary indicators for Grid and Battery direction
        vars_dict["D"] = cp.Variable(n, boolean=True, name="D")
        vars_dict["E"] = cp.Variable(n, boolean=True, name="E")

        # Battery power variables
        if self.optim_conf["set_use_battery"]:
            vars_dict["p_sto_pos"] = cp.Variable(n, nonneg=True, name="p_sto_pos")
            constraints.append(
                vars_dict["p_sto_pos"] <= self.plant_conf["battery_discharge_power_max"]
            )

            vars_dict["p_sto_neg"] = cp.Variable(n, nonpos=True, name="p_sto_neg")
            constraints.append(
                vars_dict["p_sto_neg"] >= -np.abs(self.plant_conf["battery_charge_power_max"])
            )
        else:
            # Create dummy zero variables to preserve logic structure without conditional checks everywhere
            vars_dict["p_sto_pos"] = cp.Variable(n, name="p_sto_pos_dummy")
            vars_dict["p_sto_neg"] = cp.Variable(n, name="p_sto_neg_dummy")
            constraints.append(vars_dict["p_sto_pos"] == 0)
            constraints.append(vars_dict["p_sto_neg"] == 0)

        # Self-consumption variable
        if self.costfun == "self-consumption":
            vars_dict["SC"] = cp.Variable(n, nonneg=True, name="SC")

        # Hybrid Inverter variable
        if self.plant_conf["inverter_is_hybrid"]:
            vars_dict["p_hybrid_inverter"] = cp.Variable(n, name="p_hybrid_inverter")

        # Curtailment variable
        vars_dict["p_pv_curtailment"] = cp.Variable(n, nonneg=True, name="p_pv_curtailment")

        # Sum of deferrable loads
        if num_deferrable_loads > 0:
            # Create an expression for the sum
            vars_dict["p_def_sum"] = sum(p_deferrable)
        else:
            vars_dict["p_def_sum"] = np.zeros(n)

        return vars_dict, constraints

    def _build_objective_function(
        self,
        batt_stress_conf,
        inv_stress_conf,
        type_self_conso="bigm",
    ):
        """
        Construct the objective function based on configuration using vectorized CVXPY operations.
        Returns a CVXPY expression to be Maximized.
        """
        # Retrieve variables from self.vars (populated in _initialize_decision_variables)
        p_grid_pos = self.vars["p_grid_pos"]
        p_grid_neg = self.vars["p_grid_neg"]
        p_sto_pos = self.vars["p_sto_pos"]
        p_sto_neg = self.vars["p_sto_neg"]
        p_def_sum = self.vars["p_def_sum"]
        SC = self.vars.get("SC", None)

        # Retrieve parameters (vectors of length N)
        unit_load_cost = self.param_load_cost
        unit_prod_price = self.param_prod_price
        p_load = self.param_load_forecast

        # Common scaling factor
        # We maximize the negative cost (which is equivalent to minimizing cost)
        # or maximize Profit.
        scale = 0.001 * self.time_step

        # Initialize objective expression
        objective_terms = []

        # Base Cost Function
        if self.costfun == "profit":
            # Profit = Export Income - Import Cost
            # formulated as: -Cost - (Export_Neg_Value * Price)
            # Since p_grid_neg is negative, (Export_Neg * Price) is negative (cost-like).
            # We want to Maximize: -(ImportCost + ExportNeg*Price)
            # = -ImportCost + (-ExportNeg)*Price  <-- Positive Income

            if self.optim_conf["set_total_pv_sell"]:
                # Cost depends on Total Load (Load + Def)
                cost_term = cp.multiply(unit_load_cost, p_load + p_def_sum)
                prod_term = cp.multiply(unit_prod_price, p_grid_neg)
                objective_terms.append(-scale * cp.sum(cost_term + prod_term))
            else:
                # Cost depends on Grid Import
                cost_term = cp.multiply(unit_load_cost, p_grid_pos)
                prod_term = cp.multiply(unit_prod_price, p_grid_neg)
                objective_terms.append(-scale * cp.sum(cost_term + prod_term))

        elif self.costfun == "cost":
            if self.optim_conf["set_total_pv_sell"]:
                cost_term = cp.multiply(unit_load_cost, p_load + p_def_sum)
                objective_terms.append(-scale * cp.sum(cost_term))
            else:
                cost_term = cp.multiply(unit_load_cost, p_grid_pos)
                objective_terms.append(-scale * cp.sum(cost_term))

        elif self.costfun == "self-consumption":
            if type_self_conso == "bigm":
                bigm = 1e3
                cost_term = bigm * cp.multiply(unit_load_cost, p_grid_pos)
                prod_term = cp.multiply(unit_prod_price, p_grid_neg)
                objective_terms.append(-scale * cp.sum(cost_term + prod_term))
            elif type_self_conso == "maxmin":
                # Maximize SC
                objective_terms.append(scale * cp.sum(cp.multiply(unit_load_cost, SC)))

        # Battery Cycle Cost Penalty
        if self.optim_conf["set_use_battery"]:
            # p_sto_neg is negative. -weight*p_sto_neg is a positive penalty value.
            # We subtract this positive penalty from the maximization objective.
            weight_dis = self.optim_conf["weight_battery_discharge"]
            weight_chg = self.optim_conf["weight_battery_charge"]

            cycle_cost = (weight_dis * p_sto_pos) - (weight_chg * p_sto_neg)
            objective_terms.append(-scale * cp.sum(cycle_cost))

        # Deferrable Load Startup Penalties
        if (
            "set_deferrable_startup_penalty" in self.optim_conf
            and self.optim_conf["set_deferrable_startup_penalty"]
        ):
            p_def_start = self.vars["p_def_start"]
            for k in range(self.optim_conf["number_of_deferrable_loads"]):
                penalty = self.optim_conf["set_deferrable_startup_penalty"][k]
                if penalty > 0:
                    nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                    # Vectorized cost calculation for this load's startups
                    startup_cost_vector = cp.multiply(p_def_start[k], unit_load_cost)
                    total_startup_cost = cp.sum(startup_cost_vector)

                    term = -scale * penalty * nominal_power * total_startup_cost
                    objective_terms.append(term)

        # Stress Costs
        # These variables represent a cost to be minimized.
        # Since we are Maximizing the objective, we subtract them.
        if inv_stress_conf and inv_stress_conf["active"]:
            objective_terms.append(-cp.sum(inv_stress_conf["vars"]))

        if batt_stress_conf and batt_stress_conf["active"]:
            self.logger.debug("Adding battery stress cost to objective function")
            objective_terms.append(-cp.sum(batt_stress_conf["vars"]))

        # Sum all terms to create the final objective expression
        return cp.Maximize(cp.sum(objective_terms))

    def _add_main_power_balance_constraints(self, constraints):
        """Add the main power balance constraints (Vectorized)."""
        # Retrieve variables
        p_hybrid_inverter = self.vars.get("p_hybrid_inverter")
        p_def_sum = self.vars["p_def_sum"]
        p_grid_neg = self.vars["p_grid_neg"]
        p_grid_pos = self.vars["p_grid_pos"]
        p_pv_curtailment = self.vars["p_pv_curtailment"]
        p_sto_pos = self.vars["p_sto_pos"]
        p_sto_neg = self.vars["p_sto_neg"]
        D = self.vars["D"]

        # Retrieve parameters
        p_pv = self.param_pv_forecast
        p_load = self.param_load_forecast

        # Prepare Time-Varying Limits
        # We re-calculate them here to ensure we use the correct time-varying limits
        n = self.num_timesteps
        max_power_from_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
        )
        max_power_to_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
        )

        # Main Power Balance Constraints
        if self.plant_conf["inverter_is_hybrid"]:
            constraints.append(
                p_hybrid_inverter - p_def_sum - p_load + p_grid_neg + p_grid_pos == 0
            )
        else:
            if self.plant_conf["compute_curtailment"]:
                constraints.append(
                    p_pv
                    - p_pv_curtailment
                    - p_def_sum
                    - p_load
                    + p_grid_neg
                    + p_grid_pos
                    + p_sto_pos
                    + p_sto_neg
                    == 0
                )
            else:
                constraints.append(
                    p_pv - p_def_sum - p_load + p_grid_neg + p_grid_pos + p_sto_pos + p_sto_neg == 0
                )

        # Grid Constraints (Vectorized with Time-Varying Limits)
        # p_grid_pos <= max_from_grid[t] * D[t]
        constraints.append(p_grid_pos <= cp.multiply(max_power_from_grid_arr, D))

        # -p_grid_neg <= max_to_grid[t] * (1 - D[t])
        constraints.append(-p_grid_neg <= cp.multiply(max_power_to_grid_arr, (1 - D)))

    def _add_hybrid_inverter_constraints(self, constraints, inv_stress_conf):
        """Add constraints specific to hybrid inverters (Vectorized)."""
        if not self.plant_conf["inverter_is_hybrid"]:
            return

        # Retrieve main interface variables
        p_hybrid_inverter = self.vars["p_hybrid_inverter"]
        p_pv_curtailment = self.vars["p_pv_curtailment"]
        p_sto_pos = self.vars["p_sto_pos"]
        p_sto_neg = self.vars["p_sto_neg"]
        p_pv = self.param_pv_forecast

        # Determine Inverter Capacity (Configuration Logic)
        p_nom_inverter_output = self.plant_conf.get("inverter_ac_output_max", None)
        p_nom_inverter_input = self.plant_conf.get("inverter_ac_input_max", None)

        # (Legacy lookup logic preserved but runs once during setup)
        if p_nom_inverter_output is None:
            if "pv_inverter_model" in self.plant_conf:
                if isinstance(self.plant_conf["pv_inverter_model"], list):
                    p_nom_inverter_output = 0.0
                    for i in range(len(self.plant_conf["pv_inverter_model"])):
                        if isinstance(self.plant_conf["pv_inverter_model"][i], str):
                            with bz2.BZ2File(
                                self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2",
                                "rb",
                            ) as f:
                                cec_inverters = pickle.load(f)
                            inverter = cec_inverters[self.plant_conf["pv_inverter_model"][i]]
                            p_nom_inverter_output += inverter.Paco
                        else:
                            p_nom_inverter_output += self.plant_conf["pv_inverter_model"][i]
                else:
                    if isinstance(self.plant_conf["pv_inverter_model"], str):
                        with bz2.BZ2File(
                            self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2",
                            "rb",
                        ) as f:
                            cec_inverters = pickle.load(f)
                        inverter = cec_inverters[self.plant_conf["pv_inverter_model"]]
                        p_nom_inverter_output = inverter.Paco
                    else:
                        p_nom_inverter_output = self.plant_conf["pv_inverter_model"]

            if p_nom_inverter_output is None:
                p_nom_inverter_output = 0  # Fallback

        if p_nom_inverter_input is None:
            p_nom_inverter_input = p_nom_inverter_output

        eff_dc_ac = self.plant_conf.get("inverter_efficiency_dc_ac", 1.0)
        eff_ac_dc = self.plant_conf.get("inverter_efficiency_ac_dc", 1.0)

        p_dc_ac_max = p_nom_inverter_output / eff_dc_ac
        p_ac_dc_max = p_nom_inverter_input * eff_ac_dc

        n = self.num_timesteps

        # Define Internal Variables
        # We define them here and attach to self.vars so they persist for result extraction
        p_dc_ac = cp.Variable(n, nonneg=True, name="p_dc_ac")
        p_ac_dc = cp.Variable(n, nonneg=True, name="p_ac_dc")
        is_dc_sourcing = cp.Variable(n, boolean=True, name="is_dc_sourcing")

        self.vars["p_dc_ac"] = p_dc_ac
        self.vars["p_ac_dc"] = p_ac_dc
        self.vars["is_dc_sourcing"] = is_dc_sourcing

        # Power Balance Constraints (Vectorized)

        # DC Bus Balance
        if self.plant_conf["compute_curtailment"]:
            e_dc_balance = (p_pv - p_pv_curtailment + p_sto_pos + p_sto_neg) - (p_dc_ac - p_ac_dc)
        else:
            e_dc_balance = (p_pv + p_sto_pos + p_sto_neg) - (p_dc_ac - p_ac_dc)

        constraints.append(e_dc_balance == 0)

        # AC Bus Balance
        # p_hybrid == converted_DC_to_AC - converted_AC_to_DC
        constraints.append(
            p_hybrid_inverter == (p_dc_ac * eff_dc_ac) - (p_ac_dc * (1.0 / eff_ac_dc))
        )

        # Enforce Binary Logic (Cannot source and sink DC simultaneously)
        constraints.append(p_ac_dc <= (1 - is_dc_sourcing) * p_ac_dc_max)
        constraints.append(p_dc_ac <= is_dc_sourcing * p_dc_ac_max)

        # Stress Cost
        if inv_stress_conf and inv_stress_conf["active"]:
            seg_params = self._build_stress_segments(
                inv_stress_conf["max_power"],
                inv_stress_conf["unit_cost"],
                inv_stress_conf["segments"],
            )
            self._add_stress_constraints(
                constraints,
                p_hybrid_inverter,  # Power expression
                inv_stress_conf["vars"],  # Stress variable
                seg_params,
            )

    def _add_battery_constraints(self, constraints, batt_stress_conf):
        """Add all battery-related constraints (Vectorized)."""
        if not self.optim_conf["set_use_battery"]:
            return

        p_sto_pos = self.vars["p_sto_pos"]
        p_sto_neg = self.vars["p_sto_neg"]
        p_grid_neg = self.vars["p_grid_neg"]
        E = self.vars["E"]  # Binary: 1=Discharge, 0=Charge
        p_pv = self.param_pv_forecast

        # Parameters (Scalars)
        soc_init = self.param_soc_init
        soc_final = self.param_soc_final

        # Constants
        cap = self.plant_conf["battery_nominal_energy_capacity"]
        eff_dis = self.plant_conf["battery_discharge_efficiency"]
        eff_chg = self.plant_conf["battery_charge_efficiency"]
        max_dis = self.plant_conf["battery_discharge_power_max"]
        max_chg = self.plant_conf["battery_charge_power_max"]  # This is usually positive in config

        # Grid Interaction Constraints

        # No charge from grid: Charging power (neg) + PV must be positive (net surplus)
        if self.optim_conf["set_nocharge_from_grid"]:
            constraints.append(p_sto_neg + p_pv >= 0)

        # No discharge to grid: Grid Export (neg) + PV must be positive
        if self.optim_conf["set_nodischarge_to_grid"]:
            constraints.append(p_grid_neg + p_pv >= 0)

        # Dynamic Power Limits (Ramp Rate)
        if self.optim_conf["set_battery_dynamic"]:
            # Use slicing for vectorized ramp constraints: var[t+1] - var[t]
            # p_sto_pos ramp
            ramp_up_limit = self.time_step * self.optim_conf["battery_dynamic_max"] * max_dis
            ramp_down_limit = self.time_step * self.optim_conf["battery_dynamic_min"] * max_dis

            diff_pos = p_sto_pos[1:] - p_sto_pos[:-1]
            constraints.append(diff_pos <= ramp_up_limit)
            constraints.append(diff_pos >= ramp_down_limit)

            # p_sto_neg ramp (Note: p_sto_neg is negative, max_chg is positive magnitude)
            ramp_up_limit_neg = self.time_step * self.optim_conf["battery_dynamic_max"] * max_chg
            ramp_down_limit_neg = self.time_step * self.optim_conf["battery_dynamic_min"] * max_chg

            diff_neg = p_sto_neg[1:] - p_sto_neg[:-1]
            constraints.append(diff_neg <= ramp_up_limit_neg)
            constraints.append(diff_neg >= ramp_down_limit_neg)

        # Power & Binary Constraints
        # Discharge limit based on binary E
        constraints.append(p_sto_pos <= eff_dis * max_dis * E)

        # Charge limit based on binary E (1-E)
        # p_sto_neg >= -1/eff * max * (1-E)  --> (p_sto_neg is negative)
        constraints.append(p_sto_neg >= -(1 / eff_chg) * max_chg * (1 - E))

        # SOC Constraints (Vectorized Accumulation)

        # Calculate Energy Change per timestep (kWh)
        # Energy out = p_sto_pos / eff_dis
        # Energy in  = p_sto_neg * eff_chg  (p_sto_neg is negative, so this adds negative energy)
        power_flow = (p_sto_pos * (1 / eff_dis)) + (p_sto_neg * eff_chg)
        energy_change = power_flow * self.time_step

        # Calculate Cumulative Energy used/added
        cumulative_energy = cp.cumsum(energy_change)

        # SOC State (kWh) at every timestep t
        # SOC_t = SOC_init - Cumulative_Change
        # (Subtracting because positive flow is Discharge/Depletion)
        current_stored_energy = (soc_init * cap) - cumulative_energy

        # Min/Max SOC Bounds for all t
        constraints.append(
            current_stored_energy <= self.plant_conf["battery_maximum_state_of_charge"] * cap
        )
        constraints.append(
            current_stored_energy >= self.plant_conf["battery_minimum_state_of_charge"] * cap
        )

        # Final SOC Constraint
        # The total energy change over the whole horizon must match init -> final
        # Total Sum of power flow * dt == (Init - Final) * Capacity
        total_energy_change = cp.sum(energy_change)
        constraints.append(total_energy_change == (soc_init - soc_final) * cap)

        # Stress Cost
        if batt_stress_conf and batt_stress_conf["active"]:
            seg_params = self._build_stress_segments(
                batt_stress_conf["max_power"],
                batt_stress_conf["unit_cost"],
                batt_stress_conf["segments"],
            )
            self._add_stress_constraints(
                constraints,
                p_sto_pos - p_sto_neg,  # Total power magnitude expression
                batt_stress_conf["vars"],
                seg_params,
            )

    def _add_thermal_load_constraints(self, constraints, k, data_opt, def_init_temp):
        """
        Handle constraints for thermal deferrable loads (Vectorized).
        Replicates legacy behavior: Timestep 0 is fixed to start_temp; physics starts at t=1.
        """
        p_deferrable = self.vars["p_deferrable"][k]
        p_def_bin2 = self.vars["p_def_bin2"][k]

        # Config retrieval
        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_config"]

        start_temperature = (
            def_init_temp[k] if def_init_temp[k] is not None else hc.get("start_temperature", 20.0)
        )
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0

        # Outdoor temp handling
        outdoor_temp = self._get_clean_list("outdoor_temperature_forecast", data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = self._get_clean_list("temp_air", data_opt)

        required_len = self.num_timesteps
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = np.full(required_len, 15.0)
        else:
            outdoor_temp = np.array([15.0 if x is None else float(x) for x in outdoor_temp])

        if len(outdoor_temp) < required_len:
            pad = np.full(required_len - len(outdoor_temp), 15.0)
            outdoor_temp = np.concatenate((outdoor_temp, pad))
        outdoor_temp = outdoor_temp[:required_len]

        # Constants
        cooling_constant = hc["cooling_constant"]
        heating_rate = hc["heating_rate"]
        overshoot_temperature = hc.get("overshoot_temperature", None)
        desired_temperatures = hc.get("desired_temperatures", [])
        min_temperatures = hc.get("min_temperatures", [])
        max_temperatures = hc.get("max_temperatures", [])
        sense = hc.get("sense", "heat")
        sense_coeff = 1 if sense == "heat" else -1
        nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]

        # Define Temperature State Variable
        predicted_temp = cp.Variable(required_len, name=f"temp_load_{k}")

        constraints.append(predicted_temp[0] == start_temperature)

        heat_factor = (heating_rate * self.time_step) / nominal_power
        cool_factor = cooling_constant * self.time_step

        constraints.append(
            predicted_temp[1:]
            == predicted_temp[:-1]
            + (p_deferrable[:-1] * heat_factor)
            - (cool_factor * (predicted_temp[:-1] - outdoor_temp[:-1]))
        )

        # Min/Max Temperature Constraints
        def enforce_limit(limit_list, relation_op):
            # Filter for valid indices > 0
            valid_indices = [
                i
                for i, val in enumerate(limit_list)
                if val is not None and i < required_len and i > 0
            ]
            if valid_indices:
                limit_vals = np.array([limit_list[i] for i in valid_indices])
                constraints.append(relation_op(predicted_temp[valid_indices], limit_vals))

        if min_temperatures:
            enforce_limit(min_temperatures, lambda x, y: x >= y)
        if max_temperatures:
            enforce_limit(max_temperatures, lambda x, y: x <= y)

        # Overshoot Logic
        penalty_expr = 0
        if desired_temperatures and overshoot_temperature is not None:
            is_overshoot = cp.Variable(required_len, boolean=True, name=f"is_overshoot_{k}")
            big_m = 100
            if sense == "heat":
                constraints.append(
                    predicted_temp - overshoot_temperature - (big_m * is_overshoot) <= 0
                )
                constraints.append(
                    predicted_temp - overshoot_temperature + (big_m * (1 - is_overshoot)) >= 0
                )
            else:
                constraints.append(
                    predicted_temp - overshoot_temperature - (-big_m * is_overshoot) >= 0
                )
                constraints.append(
                    predicted_temp - overshoot_temperature + (-big_m * (1 - is_overshoot)) <= 0
                )

            constraints.append(is_overshoot[1:] + p_def_bin2[:-1] <= 1)

            # Penalty Calculation
            # Only for valid indices > 0
            valid_indices = [
                i
                for i, val in enumerate(desired_temperatures)
                if val is not None and i < required_len and i > 0
            ]
            if valid_indices:
                valid_idx = np.array(valid_indices)
                des_temps = np.array([desired_temperatures[i] for i in valid_indices])
                penalty_factor = hc.get("penalty_factor", 10)

                deviation = (predicted_temp[valid_idx] - des_temps) * sense_coeff

                penalty_expr = -cp.pos(-deviation * penalty_factor)

        # Semi-Continuous Constraint
        if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
            constraints.append(p_deferrable == p_def_bin2 * nominal_power)

        total_penalty = cp.sum(penalty_expr) if not isinstance(penalty_expr, int) else 0
        return predicted_temp, None, total_penalty

    def _add_thermal_battery_constraints(self, constraints, k, data_opt, p_load):
        """Handle constraints for thermal battery loads (Vectorized, Legacy Match)."""
        p_deferrable = self.vars["p_deferrable"][k]

        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_battery"]

        start_temperature = hc.get("start_temperature", 20.0)
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0

        # Robust Outdoor Temp Cleaning (Handle NaN)
        outdoor_temp = self._get_clean_list("outdoor_temperature_forecast", data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = self._get_clean_list("temp_air", data_opt)

        required_len = self.num_timesteps
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = np.full(required_len, 15.0)
        else:
            # Added pd.isna(x) check to handle NaNs from DataFrames
            outdoor_temp = np.array(
                [15.0 if (x is None or pd.isna(x)) else float(x) for x in outdoor_temp]
            )

        if len(outdoor_temp) < required_len:
            pad = np.full(required_len - len(outdoor_temp), 15.0)
            outdoor_temp = np.concatenate((outdoor_temp, pad))
        outdoor_temp = outdoor_temp[:required_len]

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
            outdoor_temperature_forecast=outdoor_temp.tolist(),
        )
        heatpump_cops = np.array(heatpump_cops[:required_len])

        thermal_losses = utils.calculate_thermal_loss_signed(
            outdoor_temperature_forecast=outdoor_temp.tolist(),
            indoor_temperature=start_temperature,
            base_loss=loss,
        )
        thermal_losses = np.array(thermal_losses[:required_len])

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

            # Extract optional internal gains parameter
            internal_gains_factor = hc.get("internal_gains_factor", 0.0)

            # Use p_load directly
            internal_gains_forecast = None
            if internal_gains_factor > 0:
                internal_gains_forecast = p_load

            # Solar Irradiance Logic (Numpy Array)
            solar_irradiance = None
            if "ghi" in data_opt.columns and window_area is not None:
                vals = data_opt["ghi"].values
                # Handle padding if necessary
                if len(vals) < self.num_timesteps:
                    vals = np.concatenate((vals, np.zeros(self.num_timesteps - len(vals))))
                solar_irradiance = vals[: self.num_timesteps]

            heating_demand = utils.calculate_heating_demand_physics(
                u_value=hc["u_value"],
                envelope_area=hc["envelope_area"],
                ventilation_rate=hc["ventilation_rate"],
                heated_volume=hc["heated_volume"],
                indoor_target_temperature=indoor_target_temp,
                outdoor_temperature_forecast=outdoor_temp.tolist(),
                optimization_time_step=int(self.freq.total_seconds() / 60),
                solar_irradiance_forecast=solar_irradiance,
                window_area=window_area,
                shgc=shgc,
                internal_gains_forecast=internal_gains_forecast,
                internal_gains_factor=internal_gains_factor,
            )

            # Improved Logging
            gains_info = []
            if solar_irradiance is not None:
                gains_info.append(f"solar (window_area={window_area:.1f}, shgc={shgc:.2f})")
            if internal_gains_factor > 0:
                gains_info.append(f"internal (factor={internal_gains_factor:.2f})")

            gains_str = " with " + " and ".join(gains_info) if gains_info else ""
            self.logger.debug(
                "Load %s: Using physics-based heating demand%s "
                "(u_value=%.2f, envelope_area=%.1f, ventilation_rate=%.2f, heated_volume=%.1f, "
                "indoor_target_temp=%.1f)",
                k,
                gains_str,
                hc["u_value"],
                hc["envelope_area"],
                hc["ventilation_rate"],
                hc["heated_volume"],
                indoor_target_temp,
            )
            # --- END PORTED LOGIC ---
        else:
            base_temperature = hc.get("base_temperature", 18.0)
            annual_reference_hdd = hc.get("annual_reference_hdd", 3000.0)
            heating_demand = utils.calculate_heating_demand(
                specific_heating_demand=hc["specific_heating_demand"],
                floor_area=hc["area"],
                outdoor_temperature_forecast=outdoor_temp.tolist(),
                base_temperature=base_temperature,
                annual_reference_hdd=annual_reference_hdd,
                optimization_time_step=int(self.freq.total_seconds() / 60),
            )
        heating_demand = np.array(heating_demand[:required_len])

        predicted_temp_thermal = cp.Variable(required_len, name=f"temp_thermal_batt_{k}")

        constraints.append(predicted_temp_thermal[0] == start_temperature)

        constraints.append(
            predicted_temp_thermal[1:]
            == predicted_temp_thermal[:-1]
            + conversion
            * (
                (cp.multiply(heatpump_cops[:-1], p_deferrable[:-1]) / 1000 * self.time_step)
                - heating_demand[:-1]
                - thermal_losses[:-1]
            )
        )

        def enforce_limit(limit_list, relation_op):
            valid_indices = [
                i
                for i, val in enumerate(limit_list)
                if val is not None and i < required_len and i > 0
            ]
            if valid_indices:
                limit_vals = np.array([limit_list[i] for i in valid_indices])
                constraints.append(relation_op(predicted_temp_thermal[valid_indices], limit_vals))

        if min_temperatures:
            enforce_limit(min_temperatures, lambda x, y: x >= y)
        if max_temperatures:
            enforce_limit(max_temperatures, lambda x, y: x <= y)

        return predicted_temp_thermal, heating_demand

    def _add_deferrable_load_constraints(
        self,
        constraints,
        data_opt,
        def_total_hours,
        def_total_timestep,
        def_start_timestep,
        def_end_timestep,
        def_init_temp,
        min_power_of_deferrable_loads,
        p_load,
    ):
        """Master helper for all deferrable load constraints (Vectorized)."""
        p_deferrable = self.vars["p_deferrable"]
        p_def_bin1 = self.vars["p_def_bin1"]
        p_def_start = self.vars["p_def_start"]
        p_def_bin2 = self.vars["p_def_bin2"]

        predicted_temps = {}
        heating_demands = {}
        penalty_terms_total = 0
        n = self.num_timesteps

        for k in range(self.optim_conf["number_of_deferrable_loads"]):
            self.logger.debug(f"Processing deferrable load {k}")

            # Determine Load Type & Dynamic Big-M
            # Calculate a tight Big-M value for this specific load.
            # M must be >= max possible power to allow the binary variable to work.
            # Using a dynamic tight M significantly speeds up the solver (HiGHS/CBC).
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                # Sequence load: M = max peak of the sequence
                M = np.max(self.optim_conf["nominal_power_of_deferrable_loads"][k])
                is_sequence_load = True
            else:
                # Standard load: M = nominal power
                M = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                is_sequence_load = False

            # Safety fallback if M is 0 (e.g., mock load)
            if M <= 0:
                M = 10.0

            # Load Specific Constraints

            # Sequence-based Deferrable Load
            if is_sequence_load:
                power_sequence = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                sequence_length = len(power_sequence)

                # Binary variable y: which sequence to choose?
                # We essentially slice the sequence over the horizon
                y_len = n - sequence_length + 1

                # Handle case where Horizon < Sequence Length
                if y_len < 1:
                    self.logger.warning(
                        f"Deferrable load {k}: Sequence length ({sequence_length}) is longer than "
                        f"optimization horizon ({n}). The sequence will be truncated."
                    )
                    y_len = 1

                y = cp.Variable(y_len, boolean=True, name=f"y_seq_{k}")

                # Constraint: Choose exactly one start time
                constraints.append(cp.sum(y) == 1)

                # Detailed power shape constraint (Convolution-like)
                # We build the matrix explicitly here
                mat_rows = []
                for start_t in range(y_len):
                    row = np.zeros(n)
                    end_t = min(start_t + sequence_length, n)
                    seq_slice = power_sequence[: (end_t - start_t)]
                    row[start_t:end_t] = seq_slice
                    mat_rows.append(row)

                mat_np = np.array(mat_rows)  # Shape (y_len, n)

                constraints.append(p_deferrable[k] == cp.matmul(y, mat_np))

            # Thermal Deferrable Load
            elif (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_config" in self.optim_conf["def_load_config"][k]
            ):
                pred_temp, _, penalty_term = self._add_thermal_load_constraints(
                    constraints, k, data_opt, def_init_temp
                )
                predicted_temps[k] = pred_temp
                if penalty_term is not None:
                    penalty_terms_total += penalty_term

            # Thermal Battery Load
            elif (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_battery" in self.optim_conf["def_load_config"][k]
            ):
                pred_temp, heat_demand = self._add_thermal_battery_constraints(
                    constraints, k, data_opt, p_load
                )
                predicted_temps[k] = pred_temp
                heating_demands[k] = heat_demand

            # Standard Deferrable Load
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

                # Total Energy Constraint
                constraints.append(cp.sum(p_deferrable[k]) * self.time_step == target_energy)

            # Generic Constraints (Window)

            # Time Window Logic
            # Calculate Valid Window
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

            # Apply Window Constraints (Force 0 outside window)
            if def_start > 0:
                constraints.append(p_deferrable[k][:def_start] == 0)
            if def_end > 0 and def_end < n:
                constraints.append(p_deferrable[k][def_end:] == 0)

            # Optimization: Skip Binary Logic if Possible
            # If a load is:
            # 1. Not Sequence (handled above)
            # 2. Not Semi-Continuous (variable power allowed)
            # 3. No Min Power (min=0)
            # 4. No Startup Penalty
            # 5. Not Single Constant Start
            # Then it is a pure Continuous Variable. We can skip creating/linking binary variables.
            # This dramatically speeds up solving for thermal loads which are often continuous.

            is_semi_cont = self.optim_conf["treat_deferrable_load_as_semi_cont"][k]
            is_single_const = self.optim_conf["set_deferrable_load_single_constant"][k]
            has_min_power = min_power_of_deferrable_loads[k] > 0
            has_startup_penalty = (
                "set_deferrable_startup_penalty" in self.optim_conf
                and self.optim_conf["set_deferrable_startup_penalty"][k] > 0
            )

            # Check if we MUST use binary logic
            use_binary_logic = (
                is_sequence_load
                or is_semi_cont
                or is_single_const
                or has_min_power
                or has_startup_penalty
            )

            if use_binary_logic:
                # Standard Binary/Mixed-Integer Constraints

                # Minimum Power (if active)
                if has_min_power:
                    constraints.append(
                        p_deferrable[k] >= min_power_of_deferrable_loads[k] * p_def_bin2[k]
                    )

                # Status consistency: P_def <= M * Bin2
                # Use the Dynamic M calculated above (Critical for performance)
                constraints.append(p_deferrable[k] <= M * p_def_bin2[k])

                # Startup Detection: Start[t] >= Bin[t] - Bin[t-1]
                # Retrieve State
                current_state = 0
                if (
                    "def_current_state" in self.optim_conf
                    and len(self.optim_conf["def_current_state"]) > k
                ):
                    current_state = 1 if self.optim_conf["def_current_state"][k] else 0

                constraints.append(p_def_start[k][0] >= p_def_bin2[k][0] - current_state)
                constraints.append(p_def_start[k][1:] >= p_def_bin2[k][1:] - p_def_bin2[k][:-1])

                # Startup Limit: Start[t] + Bin[t-1] <= 1
                constraints.append(p_def_start[k][0] + current_state <= 1)
                constraints.append(p_def_start[k][1:] + p_def_bin2[k][:-1] <= 1)

                if not is_sequence_load:
                    # Single Constant Start
                    if is_single_const:
                        constraints.append(cp.sum(p_def_start[k]) == 1)
                        rhs_val = (
                            def_total_timestep[k]
                            if (def_total_timestep and def_total_timestep[k] > 0)
                            else def_total_hours[k] / self.time_step
                        )
                        constraints.append(cp.sum(p_def_bin2[k]) == rhs_val)

                    # Semi-continuous
                    if is_semi_cont:
                        nominal = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                        constraints.append(p_deferrable[k] == nominal * p_def_bin1[k])
                        constraints.append(p_def_bin1[k] == p_def_bin2[k])

            else:
                # Pure Continuous Constraints (Faster!)
                # Just bound by nominal power. No binary variables involved.
                constraints.append(p_deferrable[k] >= 0)
                constraints.append(p_deferrable[k] <= M)

        return predicted_temps, heating_demands, penalty_terms_total

    def _build_results_dataframe(
        self,
        data_opt,
        unit_load_cost,
        unit_prod_price,
        p_load,
        p_pv,
        soc_init,
        predicted_temps,
        heating_demands,
        debug,
    ):
        """Build the final results DataFrame (Vectorized extraction)."""
        opt_tp = pd.DataFrame(index=data_opt.index)

        # Helper to safely get value or zeroes
        def get_val(var):
            if var is None:
                return np.zeros(self.num_timesteps)
            val = var.value
            return val if val is not None else np.zeros(self.num_timesteps)

        # Main Power Variables
        opt_tp["P_PV"] = p_pv
        opt_tp["P_Load"] = p_load

        if self.plant_conf["compute_curtailment"]:
            opt_tp["P_PV_curtailment"] = get_val(self.vars.get("p_pv_curtailment"))

        opt_tp["P_grid_pos"] = get_val(self.vars["p_grid_pos"])
        opt_tp["P_grid_neg"] = get_val(self.vars["p_grid_neg"])
        opt_tp["P_grid"] = opt_tp["P_grid_pos"] + opt_tp["P_grid_neg"]

        # Deferrable Loads
        p_def_sum = np.zeros(self.num_timesteps)
        for k in range(self.optim_conf["number_of_deferrable_loads"]):
            p_def_k = get_val(self.vars["p_deferrable"][k])
            opt_tp[f"P_deferrable{k}"] = p_def_k
            p_def_sum += p_def_k

        # Battery Results
        if self.optim_conf["set_use_battery"]:
            p_sto_pos = get_val(self.vars["p_sto_pos"])
            p_sto_neg = get_val(self.vars["p_sto_neg"])
            opt_tp["P_batt"] = p_sto_pos + p_sto_neg

            # Reconstruct SOC
            eff_dis = self.plant_conf["battery_discharge_efficiency"]
            eff_chg = self.plant_conf["battery_charge_efficiency"]
            cap = self.plant_conf["battery_nominal_energy_capacity"]

            power_flow = (p_sto_pos * (1 / eff_dis)) + (p_sto_neg * eff_chg)
            energy_change = power_flow * self.time_step
            cumulative_change = np.cumsum(energy_change)
            opt_tp["SOC_opt"] = soc_init - (cumulative_change / cap)

            # Stress Cost
            if "batt_stress_cost" in self.vars:
                opt_tp["batt_stress_cost"] = get_val(self.vars["batt_stress_cost"])

        # Hybrid Inverter Results
        if self.plant_conf["inverter_is_hybrid"]:
            opt_tp["P_hybrid_inverter"] = get_val(self.vars["p_hybrid_inverter"])
            if "inv_stress_cost" in self.vars:
                opt_tp["inv_stress_cost"] = get_val(self.vars["inv_stress_cost"])

        # Costs & Prices
        opt_tp["unit_load_cost"] = unit_load_cost
        opt_tp["unit_prod_price"] = unit_prod_price

        # Add Power Limits to Results (Required for Validation/Tests)
        n = self.num_timesteps
        opt_tp["maximum_power_from_grid"] = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
        )
        opt_tp["maximum_power_to_grid"] = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
        )

        # Cost scaling factor (kW conversion and sign flip for minimization -> profit)
        scale = -0.001 * self.time_step

        if self.optim_conf["set_total_pv_sell"]:
            cost_profit = scale * (
                unit_load_cost * (p_load + p_def_sum) + unit_prod_price * opt_tp["P_grid_neg"]
            )
        else:
            cost_profit = scale * (
                unit_load_cost * opt_tp["P_grid_pos"] + unit_prod_price * opt_tp["P_grid_neg"]
            )

        opt_tp["cost_profit"] = cost_profit

        # Specific Cost Function Breakdown
        if self.costfun == "profit":
            opt_tp["cost_fun_profit"] = cost_profit

        elif self.costfun == "cost":
            if self.optim_conf["set_total_pv_sell"]:
                opt_tp["cost_fun_cost"] = scale * unit_load_cost * (p_load + p_def_sum)
            else:
                opt_tp["cost_fun_cost"] = scale * unit_load_cost * opt_tp["P_grid_pos"]

        elif self.costfun == "self-consumption":
            if "SC" in self.vars:
                opt_tp["cost_fun_selfcons"] = scale * unit_load_cost * get_val(self.vars["SC"])
            else:
                opt_tp["cost_fun_selfcons"] = cost_profit

        # Optimization Status
        opt_tp["optim_status"] = self.optim_status

        # Thermal Details
        for k, pred_temp_var in predicted_temps.items():
            temp_values = get_val(pred_temp_var)
            opt_tp[f"predicted_temp_heater{k}"] = np.round(temp_values, 2)

            if "def_load_config" in self.optim_conf:
                # Robustly get config (support both thermal_config and thermal_battery)
                load_conf = self.optim_conf["def_load_config"][k]
                conf = load_conf.get("thermal_config") or load_conf.get("thermal_battery") or {}

                # Store Target/Desired Temperatures (Legacy behavior)
                # Only look for 'desired_temperatures'.
                targets = conf.get("desired_temperatures")

                if targets:
                    tgt_series = pd.Series(targets)
                    if len(tgt_series) > len(opt_tp):
                        tgt_series = tgt_series.iloc[: len(opt_tp)]
                    tgt_series.index = opt_tp.index[: len(tgt_series)]
                    opt_tp[f"target_temp_heater{k}"] = tgt_series

                # Store Explicit Min/Max Constraints (New request)
                for bound in ["min", "max"]:
                    key = f"{bound}_temperatures"
                    if conf.get(key):
                        bound_series = pd.Series(conf[key])
                        # Align length with optimization horizon
                        if len(bound_series) > len(opt_tp):
                            bound_series = bound_series.iloc[: len(opt_tp)]
                        bound_series.index = opt_tp.index[: len(bound_series)]
                        opt_tp[f"{bound}_temp_heater{k}"] = bound_series

        for k, heat_demand in heating_demands.items():
            opt_tp[f"heating_demand_heater{k}"] = heat_demand

        # Debug Columns
        if debug:
            for k in range(self.optim_conf["number_of_deferrable_loads"]):
                opt_tp[f"P_def_start_{k}"] = get_val(self.vars["p_def_start"][k])
                opt_tp[f"P_def_bin2_{k}"] = get_val(self.vars["p_def_bin2"][k])

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
        Perform the actual optimization using Convex Programming (CVXPY).
        """
        # 0. Dynamic Resizing (Fix for ValueError: Invalid dimensions)
        # If the input data length differs from the initialized N, we must rebuild the problem.
        current_n = len(data_opt)
        if current_n != self.num_timesteps:
            self.logger.info(
                f"Resizing optimization problem from {self.num_timesteps} to {current_n} timesteps."
            )
            self.num_timesteps = current_n

            # Re-initialize Parameters with new shape
            self.param_pv_forecast = cp.Parameter(current_n, name="pv_forecast")
            self.param_load_forecast = cp.Parameter(current_n, name="load_forecast")
            self.param_load_cost = cp.Parameter(current_n, name="load_cost")
            self.param_prod_price = cp.Parameter(current_n, name="prod_price")

            # Re-initialize Variables & Constraints
            self.vars, self.constraints = self._initialize_decision_variables()

            # Force problem rebuild
            self.prob = None

        # Data Validation & Defaults
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

        # Pad deferrable load lists
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

        def pad_list(input_list, target_len, fill=0):
            if input_list is None:
                return [fill] * target_len
            return input_list + [fill] * (target_len - len(input_list))

        min_power_of_deferrable_loads = pad_list(
            min_power_of_deferrable_loads, num_deferrable_loads
        )
        def_total_hours = pad_list(def_total_hours, num_deferrable_loads)
        def_start_timestep = pad_list(def_start_timestep, num_deferrable_loads)
        def_end_timestep = pad_list(def_end_timestep, num_deferrable_loads)

        # Parameter Updates
        self.param_pv_forecast.value = p_pv
        self.param_load_forecast.value = p_load
        self.param_load_cost.value = unit_load_cost
        self.param_prod_price.value = unit_prod_price

        if self.optim_conf["set_use_battery"]:
            self.param_soc_init.value = soc_init
            self.param_soc_final.value = soc_final

        # Build Problem (Lazy Construction)
        if self.prob is None:
            self.logger.info("Building CVXPY problem structure...")

            # Start with bound constraints
            constraints = self.constraints[:]

            # Setup Stress Costs
            inv_stress_conf = None
            batt_stress_conf = None

            if self.optim_conf["set_use_battery"]:
                p_batt_max = max(
                    self.plant_conf.get("battery_discharge_power_max", 0),
                    self.plant_conf.get("battery_charge_power_max", 0),
                )
                batt_stress_conf = self._setup_stress_cost(
                    "battery_stress_cost", p_batt_max, "battery"
                )
                if batt_stress_conf["active"]:
                    self.vars["batt_stress_cost"] = batt_stress_conf["vars"]

            if self.plant_conf["inverter_is_hybrid"]:
                P_nom_inverter_max = max(
                    self.plant_conf.get("inverter_ac_output_max", 0),
                    self.plant_conf.get("inverter_ac_input_max", 0),
                )
                inv_stress_conf = self._setup_stress_cost(
                    "inverter_stress_cost", P_nom_inverter_max, "inv"
                )
                if inv_stress_conf["active"]:
                    self.vars["inv_stress_cost"] = inv_stress_conf["vars"]

            # Add Constraints
            self._add_main_power_balance_constraints(constraints)
            self._add_hybrid_inverter_constraints(constraints, inv_stress_conf)
            self._add_battery_constraints(constraints, batt_stress_conf)

            if self.plant_conf["compute_curtailment"]:
                constraints.append(self.vars["p_pv_curtailment"] <= self.param_pv_forecast)

            if self.costfun == "self-consumption" and "SC" in self.vars:
                constraints.append(self.vars["SC"] <= self.param_pv_forecast)
                constraints.append(
                    self.vars["SC"] <= self.param_load_forecast + self.vars["p_def_sum"]
                )

            # Deferrable Loads
            self.predicted_temps, self.heating_demands, penalty_terms_total = (
                self._add_deferrable_load_constraints(
                    constraints,
                    data_opt,
                    def_total_hours,
                    def_total_timestep,
                    def_start_timestep,
                    def_end_timestep,
                    def_init_temp,
                    min_power_of_deferrable_loads,
                    p_load,
                )
            )

            # Build Objective
            objective_expr = self._build_objective_function(
                batt_stress_conf,
                inv_stress_conf,
            )

            # Add penalty term if it exists (not 0)
            # We assume penalty_terms_total is either 0 (int) or a cvxpy expression
            if not isinstance(penalty_terms_total, int) or penalty_terms_total != 0:
                objective_expr.args[0] += penalty_terms_total

            self.prob = cp.Problem(objective_expr, constraints)

        # Solve
        solver_opts = {"verbose": False}

        # Retrieve Constraints (Time & Threads)
        # We keep these config parameters as they are useful for everyone
        threads = self.optim_conf.get("num_threads", 0)
        timeout = self.optim_conf.get("lp_solver_timeout", 180)

        # Select Solver
        # We strictly default to HiGHS.
        # Advanced users can override this by setting the 'LP_SOLVER' environment variable.
        requested_solver = os.environ.get("LP_SOLVER", "HIGHS").upper()
        selected_solver = cp.HIGHS

        if requested_solver == "GUROBI":
            if "GUROBI" in cp.installed_solvers():
                selected_solver = cp.GUROBI
                # Gurobi specific options
                solver_opts["TimeLimit"] = timeout
                if threads > 0:
                    solver_opts["Threads"] = threads
            else:
                self.logger.warning(
                    "Solver 'GUROBI' requested via Env Var but not found. Falling back to HiGHS."
                )

        elif requested_solver == "CPLEX":
            if "CPLEX" in cp.installed_solvers():
                selected_solver = cp.CPLEX
                # CPLEX specific options
                cplex_params = {"timelimit": timeout}
                if threads > 0:
                    cplex_params["threads"] = threads
                solver_opts["cplex_params"] = cplex_params
            else:
                self.logger.warning(
                    "Solver 'CPLEX' requested via Env Var but not found. Falling back to HiGHS."
                )

        # Configure HiGHS (The Default)
        if selected_solver == cp.HIGHS:
            solver_opts["time_limit"] = float(timeout)
            if threads > 0:
                solver_opts["threads"] = int(threads)
            # 'run_crossover' ensures a cleaner solution (closer to simplex vertex)
            solver_opts["run_crossover"] = "on"

        # Execute Solve
        try:
            self.prob.solve(solver=selected_solver, warm_start=True, **solver_opts)
        except Exception as e:
            self.logger.warning(
                f"Solver {selected_solver} failed: {e}. Retrying with default settings."
            )
            # Fallback retry
            try:
                self.prob.solve(solver=cp.HIGHS)
            except Exception:
                pass  # Status check below will catch failure

        # Fix for Status Case: Map "optimal" -> "Optimal"
        self.optim_status = self.prob.status.title() if self.prob.status else "Failure"
        self.logger.info("Status: " + self.optim_status)

        if self.prob.value is None or self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            self.logger.warning("Cost function cannot be evaluated or Infeasible/Unbounded")
            # Return valid empty DF to match original behavior
            return pd.DataFrame()
        else:
            self.logger.info(
                "Total value of the Cost function = %.02f",
                self.prob.value,
            )

        # Results Extraction
        return self._build_results_dataframe(
            data_opt,
            unit_load_cost,
            unit_prod_price,
            p_load,
            p_pv,
            soc_init,
            self.predicted_temps,
            self.heating_demands,
            debug,
        )

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

        # List to collect results for faster one-time concatenation
        results_list = []

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
            # The new CVXPY implementation will re-use the problem structure automatically
            opt_tp = self.perform_optimization(
                data_tp, p_pv, p_load, unit_load_cost, unit_prod_price
            )

            results_list.append(opt_tp)

        # Concatenate all results at once (Much faster than appending inside the loop)
        if results_list:
            self.opt_res = pd.concat(results_list, axis=0)
        else:
            self.opt_res = pd.DataFrame()

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

        # Extract cost arrays (ensure they are flat numpy arrays)
        unit_load_cost = df_input_data[self.var_load_cost].values
        unit_prod_price = df_input_data[self.var_prod_price].values

        # Call optimization function
        # Note: .ravel() ensures 1D arrays, compatible with cvxpy Parameter shapes
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

        # Verify compatibility with Fixed Problem Size (Define Once Architecture)
        if prediction_horizon != self.num_timesteps:
            self.logger.warning(
                f"MPC Prediction Horizon ({prediction_horizon}) does not match the initialized "
                f"optimization window ({self.num_timesteps}). "
                "This may cause shape mismatch errors in the solver."
            )

        # Slice data to horizon
        subset_data = copy.deepcopy(df_input_data).iloc[:prediction_horizon]

        # Extract inputs as arrays
        # Note: We must ensure p_pv and p_load are sliced exactly like df_input_data
        p_pv_slice = p_pv.iloc[:prediction_horizon].values.ravel()
        p_load_slice = p_load.iloc[:prediction_horizon].values.ravel()
        unit_load_cost = subset_data[self.var_load_cost].values
        unit_prod_price = subset_data[self.var_prod_price].values

        # Call optimization function
        self.opt_res = self.perform_optimization(
            subset_data,
            p_pv_slice,
            p_load_slice,
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
