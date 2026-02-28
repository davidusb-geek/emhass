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
        self.var_pv = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        self.var_load = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        self.var_load_new = self.var_load + "_positive"
        self.costfun = costfun
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.var_load_cost = var_load_cost
        self.var_prod_price = var_prod_price
        self.optim_status = None

        # Prioritize config value over default arg
        if "delta_forecast_daily" in self.optim_conf:
            # If configured in days (int/float), convert to timedelta
            val = self.optim_conf["delta_forecast_daily"]
            if isinstance(val, int) or isinstance(val, float):
                self.time_delta = pd.to_timedelta(val, "days")
            else:
                # Assume it is already a timedelta or compatible
                self.time_delta = pd.to_timedelta(val)
        else:
            # Fallback to the argument (default 24h)
            self.time_delta = pd.to_timedelta(opt_time_delta, "hours")

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

        # Initialize deferrable load parameters (window masks and energy constraints)
        self._init_deferrable_load_params()

        # Initialize Variables & Bound Constraints
        self.vars, self.constraints = self._initialize_decision_variables()

        # Note: The self.prob object will be constructed in a subsequent step
        self.prob = None

    def _init_deferrable_load_params(self) -> None:
        """
        Initialize CVXPY parameters for deferrable loads (window masks and energy constraints).

        This method creates:
        - param_window_masks: Allow changing time windows without rebuilding the problem
        - param_target_energy: Target energy for Big-M energy constraints
        - param_energy_active: Flags to enable/disable energy constraints
        - param_required_timesteps: Required timesteps for binary loads
        - param_timesteps_active: Flags to enable/disable timestep constraints

        Called from __init__ and when resizing the optimization problem.
        """
        num_def_loads = self.optim_conf.get("number_of_deferrable_loads", 0)
        n = self.num_timesteps

        # Window Mask Parameters for Deferrable Loads
        # mask[t] = 0 means load must be off at timestep t
        # mask[t] = 1 means load can operate at timestep t
        self.param_window_masks = []
        for k in range(num_def_loads):
            mask = cp.Parameter(n, nonneg=True, name=f"window_mask_{k}")
            mask.value = np.ones(n)  # Default: no restriction
            self.param_window_masks.append(mask)

        # Energy Constraint Parameters for Deferrable Loads
        # Uses Big-M formulation to enable/disable the constraint
        self.param_target_energy = []  # Target energy in Wh
        self.param_energy_active = []  # 1 = constraint active, 0 = inactive (relaxed via Big-M)
        self.param_required_timesteps = []  # For binary loads: number of timesteps to run
        self.param_timesteps_active = []  # 1 = timestep constraint active, 0 = inactive
        for k in range(num_def_loads):
            # Target energy parameter
            energy_param = cp.Parameter(nonneg=True, name=f"target_energy_{k}")
            energy_param.value = 0.0
            self.param_target_energy.append(energy_param)

            # Energy constraint active flag
            energy_active = cp.Parameter(nonneg=True, name=f"energy_active_{k}")
            energy_active.value = 0.0
            self.param_energy_active.append(energy_active)

            # Required timesteps for binary loads
            timesteps_param = cp.Parameter(nonneg=True, name=f"required_timesteps_{k}")
            timesteps_param.value = 0.0
            self.param_required_timesteps.append(timesteps_param)

            # Timesteps constraint active flag
            timesteps_active = cp.Parameter(nonneg=True, name=f"timesteps_active_{k}")
            timesteps_active.value = 0.0
            self.param_timesteps_active.append(timesteps_active)

        # Thermal Parameters for warm-starting
        # Dict keyed by load index k, stores all parameters needed for thermal constraints
        # This allows updating runtime values (forecasts, temperatures) without rebuilding constraints
        self.param_thermal = {}
        def_load_config = self.optim_conf.get("def_load_config", []) or []
        for k in range(num_def_loads):
            if k < len(def_load_config) and def_load_config[k]:
                cfg = def_load_config[k]
                if "thermal_config" in cfg:
                    hc = cfg["thermal_config"]
                    init_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    min_temps = hc.get("min_temperatures", [])
                    max_temps = hc.get("max_temperatures", [])
                    desired_temps = hc.get("desired_temperatures", [])

                    self.param_thermal[k] = {
                        "type": "thermal_config",
                        "start_temp": cp.Parameter(name=f"thermal_start_temp_{k}", value=init_temp),
                        "outdoor_temp": cp.Parameter(n, name=f"thermal_outdoor_temp_{k}"),
                        "min_temps": cp.Parameter(n, name=f"thermal_min_temps_{k}"),
                        "max_temps": cp.Parameter(n, name=f"thermal_max_temps_{k}"),
                        "desired_temps": cp.Parameter(n, name=f"thermal_desired_temps_{k}"),
                    }
                    # Initialize with default values
                    self.param_thermal[k]["outdoor_temp"].value = np.full(n, 15.0)
                    self.param_thermal[k]["min_temps"].value = self._pad_temp_array(
                        min_temps, n, 18.0
                    )
                    self.param_thermal[k]["max_temps"].value = self._pad_temp_array(
                        max_temps, n, 26.0
                    )
                    self.param_thermal[k]["desired_temps"].value = self._pad_temp_array(
                        desired_temps, n, 22.0
                    )

                elif "thermal_battery" in cfg:
                    hc = cfg["thermal_battery"]
                    init_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    min_temps = hc.get("min_temperatures", [])
                    max_temps = hc.get("max_temperatures", [])

                    self.param_thermal[k] = {
                        "type": "thermal_battery",
                        "start_temp": cp.Parameter(
                            name=f"thermal_battery_start_temp_{k}", value=init_temp
                        ),
                        "outdoor_temp": cp.Parameter(n, name=f"thermal_battery_outdoor_temp_{k}"),
                        "min_temps": cp.Parameter(n, name=f"thermal_battery_min_temps_{k}"),
                        "max_temps": cp.Parameter(n, name=f"thermal_battery_max_temps_{k}"),
                        "thermal_losses": cp.Parameter(n, name=f"thermal_battery_losses_{k}"),
                        "heating_demand": cp.Parameter(
                            n, name=f"thermal_battery_heating_demand_{k}"
                        ),
                        "heatpump_cops": cp.Parameter(n, name=f"thermal_battery_cops_{k}"),
                    }
                    # Initialize with default values
                    self.param_thermal[k]["outdoor_temp"].value = np.full(n, 15.0)
                    self.param_thermal[k]["min_temps"].value = self._pad_temp_array(
                        min_temps, n, 18.0
                    )
                    self.param_thermal[k]["max_temps"].value = self._pad_temp_array(
                        max_temps, n, 26.0
                    )
                    self.param_thermal[k]["thermal_losses"].value = np.zeros(n)
                    self.param_thermal[k]["heating_demand"].value = np.zeros(n)
                    self.param_thermal[k]["heatpump_cops"].value = np.full(n, 3.0)

                    # Thermal inertia support (first-order low-pass filter on heat input)
                    # Always define q_input_start so downstream logic can rely on its presence.
                    # tau_hours controls whether inertia dynamics are applied, not whether
                    # this parameter exists.
                    q_input_init = float(hc.get("q_input_initial", 0.0) or 0.0)
                    self.param_thermal[k]["q_input_start"] = cp.Parameter(
                        name=f"thermal_battery_q_input_start_{k}", value=q_input_init
                    )

        # Legacy compatibility - keep param_thermal_start_temps as alias
        self.param_thermal_start_temps = {
            k: (params["type"], params["start_temp"]) for k, params in self.param_thermal.items()
        }

    def _pad_temp_array(self, temp_list: list, n: int, default: float) -> np.ndarray:
        """Pad/truncate temperature list to length n, replacing None with default."""
        if not temp_list:
            return np.full(n, default)
        arr = np.array([default if v is None else float(v) for v in temp_list[:n]])
        if len(arr) < n:
            arr = np.concatenate([arr, np.full(n - len(arr), default)])
        return arr

    def _persist_q_input(self, k: int, params: dict, hc: dict) -> None:
        """Auto-persist Q_input from previous solve and apply manual override.

        Called on cache hit to carry forward the thermal inertia filter state.
        Only persists when thermal inertia is currently enabled (tau > 0) AND a
        previous solve produced q_input values. If tau was changed to 0, any stale
        q_input_var is cleared to prevent surprising persistence.

        :param k: Deferrable load index
        :param params: The param_thermal[k] dict for this load
        :param hc: The thermal_battery config dict from def_load_config
        """
        tau_hours = float(hc.get("thermal_inertia_time_constant", 0.0) or 0.0)

        if tau_hours > 0 and "q_input_var" in params:
            prev_q = params["q_input_var"].value
            if prev_q is not None and len(prev_q) > 1:
                # Use index 1: in MPC the horizon shifts by one timestep,
                # so prev_q[1] becomes the new initial condition.
                new_q_start = float(prev_q[1])
                self.logger.debug(
                    "Auto-persisting q_input for load %s: %.4f -> %.4f",
                    k,
                    params["q_input_start"].value,
                    new_q_start,
                )
                params["q_input_start"].value = new_q_start
        elif tau_hours == 0 and "q_input_var" in params:
            # Inertia was disabled — clear stale variable reference
            del params["q_input_var"]
            params["q_input_start"].value = 0.0

        # Manual override via config takes priority
        if "q_input_initial" in hc:
            params["q_input_start"].value = float(hc.get("q_input_initial", 0.0) or 0.0)

    def update_thermal_start_temps(self, optim_conf: dict) -> None:
        """
        Update thermal start temperature parameters from optim_conf.

        Called on cache hit to sync runtime thermal parameters without rebuilding constraints.
        This is a convenience wrapper that only updates start_temp. For full updates including
        forecasts, use update_thermal_params().

        :param optim_conf: The optimization configuration containing def_load_config
        """
        def_load_config = optim_conf.get("def_load_config", []) or []
        for k, (thermal_type, param) in self.param_thermal_start_temps.items():
            if k < len(def_load_config) and def_load_config[k]:
                cfg = def_load_config[k]
                if thermal_type == "thermal_config" and "thermal_config" in cfg:
                    hc = cfg["thermal_config"]
                    new_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    if param.value != new_temp:
                        self.logger.debug(
                            f"Updating thermal_config start_temp for load {k}: {param.value} -> {new_temp}"
                        )
                        param.value = new_temp
                elif thermal_type == "thermal_battery" and "thermal_battery" in cfg:
                    hc = cfg["thermal_battery"]
                    new_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    if param.value != new_temp:
                        self.logger.debug(
                            f"Updating thermal_battery start_temp for load {k}: {param.value} -> {new_temp}"
                        )
                        param.value = new_temp

                    if k in self.param_thermal:
                        self._persist_q_input(k, self.param_thermal[k], hc)

    def update_thermal_params(
        self, optim_conf: dict, data_opt: pd.DataFrame, p_load: np.ndarray
    ) -> None:
        """
        Update all thermal parameters from optim_conf and data_opt.

        Called on cache hit to sync all runtime thermal parameters without rebuilding constraints.
        This includes start_temperature, outdoor_temp forecasts, min/max temps, and derived
        values like thermal_losses, heating_demand, and heatpump_cops.

        :param optim_conf: The optimization configuration containing def_load_config
        :param data_opt: DataFrame with forecast data (outdoor_temperature_forecast, ghi, etc.)
        :param p_load: Load power forecast array (for internal gains calculation)
        """
        def_load_config = optim_conf.get("def_load_config", []) or []
        n = self.num_timesteps

        for k, params in self.param_thermal.items():
            if k >= len(def_load_config) or not def_load_config[k]:
                continue

            cfg = def_load_config[k]
            thermal_type = params["type"]

            # Get outdoor temperature forecast
            outdoor_temp = self._get_clean_outdoor_temp(data_opt, n)

            if thermal_type == "thermal_config" and "thermal_config" in cfg:
                hc = cfg["thermal_config"]

                # Update start_temperature
                new_start_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                if params["start_temp"].value != new_start_temp:
                    self.logger.debug(
                        f"Updating thermal_config start_temp for load {k}: "
                        f"{params['start_temp'].value} -> {new_start_temp}"
                    )
                params["start_temp"].value = new_start_temp

                # Update outdoor_temp
                params["outdoor_temp"].value = outdoor_temp

                # Update min/max temperatures
                min_temps = hc.get("min_temperatures", [])
                max_temps = hc.get("max_temperatures", [])
                params["min_temps"].value = self._pad_temp_array(min_temps, n, 18.0)
                params["max_temps"].value = self._pad_temp_array(max_temps, n, 26.0)

                # Update desired_temperatures
                desired_temps = hc.get("desired_temperatures", [])
                params["desired_temps"].value = self._pad_temp_array(desired_temps, n, 22.0)

            elif thermal_type == "thermal_battery" and "thermal_battery" in cfg:
                hc = cfg["thermal_battery"]

                # Update start_temperature
                new_start_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                if params["start_temp"].value != new_start_temp:
                    self.logger.debug(
                        f"Updating thermal_battery start_temp for load {k}: "
                        f"{params['start_temp'].value} -> {new_start_temp}"
                    )
                params["start_temp"].value = new_start_temp

                # Update outdoor_temp
                params["outdoor_temp"].value = outdoor_temp

                # Update min/max temperatures
                min_temps = hc.get("min_temperatures", [])
                max_temps = hc.get("max_temperatures", [])
                params["min_temps"].value = self._pad_temp_array(min_temps, n, 18.0)
                params["max_temps"].value = self._pad_temp_array(max_temps, n, 26.0)

                # Compute derived arrays
                supply_temperature = hc["supply_temperature"]
                indoor_target_temp = hc.get(
                    "indoor_target_temperature",
                    min_temps[0] if min_temps else 20.0,
                )

                # Heatpump COPs
                heatpump_cops = utils.calculate_cop_heatpump(
                    supply_temperature=supply_temperature,
                    carnot_efficiency=hc.get("carnot_efficiency", 0.4),
                    outdoor_temperature_forecast=outdoor_temp.tolist(),
                )
                params["heatpump_cops"].value = np.array(heatpump_cops[:n])

                # Thermal losses
                loss = 0.045
                thermal_losses = utils.calculate_thermal_loss_signed(
                    outdoor_temperature_forecast=outdoor_temp.tolist(),
                    indoor_temperature=new_start_temp,
                    base_loss=loss,
                )
                params["thermal_losses"].value = np.array(thermal_losses[:n])

                # Heating demand (if physics-based params are available)
                if all(
                    key in hc
                    for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]
                ):
                    window_area = hc.get("window_area", None)
                    shgc = hc.get("shgc", 0.6)
                    internal_gains_factor = hc.get("internal_gains_factor", 0.0)

                    # Solar irradiance
                    solar_irradiance = None
                    if "ghi" in data_opt.columns and window_area is not None:
                        vals = data_opt["ghi"].values
                        if len(vals) < n:
                            vals = np.concatenate((vals, np.zeros(n - len(vals))))
                        solar_irradiance = vals[:n]

                    # Internal gains
                    internal_gains_forecast = None
                    if internal_gains_factor > 0:
                        internal_gains_forecast = p_load

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
                    params["heating_demand"].value = np.array(heating_demand[:n])
                else:
                    params["heating_demand"].value = np.zeros(n)

                self._persist_q_input(k, params, hc)

    def _get_clean_outdoor_temp(self, data_opt: pd.DataFrame, n: int) -> np.ndarray:
        """Extract and clean outdoor temperature from data_opt."""
        outdoor_temp = self._get_clean_list("outdoor_temperature_forecast", data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = self._get_clean_list("temp_air", data_opt)

        if not outdoor_temp or all(x is None for x in outdoor_temp):
            return np.full(n, 15.0)

        outdoor_temp = np.array(
            [15.0 if (x is None or pd.isna(x)) else float(x) for x in outdoor_temp]
        )
        if len(outdoor_temp) < n:
            pad = np.full(n - len(outdoor_temp), 15.0)
            outdoor_temp = np.concatenate((outdoor_temp, pad))
        return outdoor_temp[:n]

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

            # Handle time-varying weights with slicing for resized horizons
            if isinstance(weight_dis, list | np.ndarray) and len(weight_dis) > self.num_timesteps:
                weight_dis = weight_dis[: self.num_timesteps]
            if isinstance(weight_chg, list | np.ndarray) and len(weight_chg) > self.num_timesteps:
                weight_chg = weight_chg[: self.num_timesteps]

            cycle_cost = cp.multiply(np.array(weight_dis), p_sto_pos) - cp.multiply(
                np.array(weight_chg), p_sto_neg
            )
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
        Includes thermal inertia (lag) logic.
        Uses cp.Parameter for runtime values to enable warm-starting on cache hits.
        """
        p_deferrable = self.vars["p_deferrable"][k]
        p_def_bin2 = self.vars["p_def_bin2"][k]

        # Config retrieval
        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_config"]
        required_len = self.num_timesteps

        # Use parameterized values if available (enables warm-start on cache hit)
        if k in self.param_thermal:
            params = self.param_thermal[k]
            start_temperature = params["start_temp"]
            outdoor_temp = params["outdoor_temp"]
            min_temps_param = params["min_temps"]
            max_temps_param = params["max_temps"]
            desired_temps_param = params["desired_temps"]

            # Update param value if def_init_temp override is provided
            if def_init_temp[k] is not None:
                params["start_temp"].value = float(def_init_temp[k])

            # Initialize outdoor temp from data_opt (will be updated on subsequent calls)
            outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)
            params["outdoor_temp"].value = outdoor_temp_arr

            # Initialize min/max/desired temps from config
            min_temps_list = hc.get("min_temperatures", [])
            max_temps_list = hc.get("max_temperatures", [])
            desired_temps_list = hc.get("desired_temperatures", [])
            params["min_temps"].value = self._pad_temp_array(min_temps_list, required_len, 18.0)
            params["max_temps"].value = self._pad_temp_array(max_temps_list, required_len, 26.0)
            params["desired_temps"].value = self._pad_temp_array(
                desired_temps_list, required_len, 22.0
            )
        else:
            # Fallback for loads not in param dict (shouldn't happen normally)
            start_temperature = (
                def_init_temp[k]
                if def_init_temp[k] is not None
                else hc.get("start_temperature", 20.0)
            )
            start_temperature = float(start_temperature) if start_temperature is not None else 20.0
            outdoor_temp = self._get_clean_outdoor_temp(data_opt, required_len)
            min_temps_param = None
            max_temps_param = None
            desired_temps_param = None

        # Constants (structural - don't change between MPC iterations)
        cooling_constant = hc["cooling_constant"]
        heating_rate = hc["heating_rate"]
        overshoot_temperature = hc.get("overshoot_temperature", None)
        sense = hc.get("sense", "heat")
        nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]

        # Thermal Inertia Logic
        thermal_inertia = hc.get("thermal_inertia", 0.0)
        L = int(thermal_inertia / self.time_step)

        # Define Temperature State Variable
        predicted_temp = cp.Variable(required_len, name=f"temp_load_{k}")

        constraints.append(predicted_temp[0] == start_temperature)

        heat_factor = (heating_rate * self.time_step) / nominal_power
        cool_factor = cooling_constant * self.time_step

        # Main Dynamics (Delayed Power)
        # T[t+1] depends on T[t] and P[t-L]
        constraints.append(
            predicted_temp[1 + L :]
            == predicted_temp[L:-1]
            + (p_deferrable[: -1 - L] * heat_factor)
            - (cool_factor * (predicted_temp[L:-1] - outdoor_temp[L:-1]))
        )

        # Startup "Dead Zone" Dynamics
        if L > 0:
            constraints.append(
                predicted_temp[1 : 1 + L]
                == predicted_temp[:L] - (cool_factor * (predicted_temp[:L] - outdoor_temp[:L]))
            )

        # Min/Max Temperature Constraints
        # Only add constraints if config actually specifies min/max temps
        # Skip index 0 (already constrained by start_temperature)
        min_temps_config = hc.get("min_temperatures", [])
        max_temps_config = hc.get("max_temperatures", [])

        if min_temps_config:
            if min_temps_param is not None:
                # Use parameter (allows warm-start updates), but only for valid config indices
                valid_indices = [
                    i
                    for i, v in enumerate(min_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    constraints.append(
                        predicted_temp[valid_indices] >= min_temps_param[valid_indices]
                    )
            else:
                valid_indices = [
                    i
                    for i, v in enumerate(min_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    limit_vals = np.array([min_temps_config[i] for i in valid_indices])
                    constraints.append(predicted_temp[valid_indices] >= limit_vals)

        if max_temps_config:
            if max_temps_param is not None:
                valid_indices = [
                    i
                    for i, v in enumerate(max_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    constraints.append(
                        predicted_temp[valid_indices] <= max_temps_param[valid_indices]
                    )
            else:
                valid_indices = [
                    i
                    for i, v in enumerate(max_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    limit_vals = np.array([max_temps_config[i] for i in valid_indices])
                    constraints.append(predicted_temp[valid_indices] <= limit_vals)

        # Overshoot Logic
        penalty_expr = 0
        desired_temps_list = hc.get("desired_temperatures", [])
        sense_coeff = 1 if sense == "heat" else -1

        if desired_temps_list and overshoot_temperature is not None:
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
            # Filter for valid indices (not None, within bounds, skip index 0)
            penalty_factor = hc.get("penalty_factor", 10)
            valid_indices = [
                i
                for i, val in enumerate(desired_temps_list)
                if val is not None and i < required_len and i > 0
            ]
            if valid_indices:
                if desired_temps_param is not None:
                    # Use parameter for actual values (allows warm-start value updates)
                    deviation = (
                        predicted_temp[valid_indices] - desired_temps_param[valid_indices]
                    ) * sense_coeff
                else:
                    # Fallback to raw values
                    des_temps = np.array([desired_temps_list[i] for i in valid_indices])
                    deviation = (predicted_temp[valid_indices] - des_temps) * sense_coeff
                penalty_expr = -cp.pos(-deviation * penalty_factor)

        # Semi-Continuous Constraint
        if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
            constraints.append(p_deferrable == p_def_bin2 * nominal_power)

        total_penalty = cp.sum(penalty_expr) if not isinstance(penalty_expr, int) else 0
        return predicted_temp, None, total_penalty

    def _add_thermal_battery_constraints(self, constraints, k, data_opt, p_load):
        """
        Handle constraints for thermal battery loads (Vectorized, Legacy Match).
        Uses cp.Parameter for runtime values to enable warm-starting on cache hits.
        """
        p_deferrable = self.vars["p_deferrable"][k]

        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_battery"]
        required_len = self.num_timesteps

        # Structural parameters (don't change between MPC iterations)
        supply_temperature = hc["supply_temperature"]
        volume = hc["volume"]
        min_temperatures_list = hc["min_temperatures"]
        max_temperatures_list = hc["max_temperatures"]

        if not min_temperatures_list:
            raise ValueError(f"Load {k}: thermal_battery requires non-empty 'min_temperatures'")
        if not max_temperatures_list:
            raise ValueError(f"Load {k}: thermal_battery requires non-empty 'max_temperatures'")

        p_concr = 2400
        c_concr = 0.88
        loss = 0.045
        conversion = 3600 / (p_concr * c_concr * volume)

        # Use parameterized values if available (enables warm-start on cache hit)
        if k in self.param_thermal:
            params = self.param_thermal[k]
            start_temperature = params["start_temp"]
            heatpump_cops = params["heatpump_cops"]
            thermal_losses = params["thermal_losses"]
            heating_demand = params["heating_demand"]
            min_temps_param = params["min_temps"]
            max_temps_param = params["max_temps"]

            # Initialize parameter values from data_opt and config
            outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)
            params["outdoor_temp"].value = outdoor_temp_arr
            start_temp_float = float(params["start_temp"].value)

            # Compute and set derived parameter values
            cops = utils.calculate_cop_heatpump(
                supply_temperature=supply_temperature,
                carnot_efficiency=hc.get("carnot_efficiency", 0.4),
                outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
            )
            params["heatpump_cops"].value = np.array(cops[:required_len])

            losses = utils.calculate_thermal_loss_signed(
                outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                indoor_temperature=start_temp_float,
                base_loss=loss,
            )
            params["thermal_losses"].value = np.array(losses[:required_len])

            # Compute heating demand
            if all(
                key in hc
                for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]
            ):
                indoor_target_temp = hc.get(
                    "indoor_target_temperature",
                    min_temperatures_list[0] if min_temperatures_list else 20.0,
                )
                window_area = hc.get("window_area", None)
                shgc = hc.get("shgc", 0.6)
                internal_gains_factor = hc.get("internal_gains_factor", 0.0)

                internal_gains_forecast = p_load if internal_gains_factor > 0 else None
                solar_irradiance = None
                if "ghi" in data_opt.columns and window_area is not None:
                    vals = data_opt["ghi"].values
                    if len(vals) < required_len:
                        vals = np.concatenate((vals, np.zeros(required_len - len(vals))))
                    solar_irradiance = vals[:required_len]

                demand = utils.calculate_heating_demand_physics(
                    u_value=hc["u_value"],
                    envelope_area=hc["envelope_area"],
                    ventilation_rate=hc["ventilation_rate"],
                    heated_volume=hc["heated_volume"],
                    indoor_target_temperature=indoor_target_temp,
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    optimization_time_step=int(self.freq.total_seconds() / 60),
                    solar_irradiance_forecast=solar_irradiance,
                    window_area=window_area,
                    shgc=shgc,
                    internal_gains_forecast=internal_gains_forecast,
                    internal_gains_factor=internal_gains_factor,
                )
                params["heating_demand"].value = np.array(demand[:required_len])

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
            else:
                base_temperature = hc.get("base_temperature", 18.0)
                annual_reference_hdd = hc.get("annual_reference_hdd", 3000.0)
                demand = utils.calculate_heating_demand(
                    specific_heating_demand=hc["specific_heating_demand"],
                    floor_area=hc["area"],
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    base_temperature=base_temperature,
                    annual_reference_hdd=annual_reference_hdd,
                    optimization_time_step=int(self.freq.total_seconds() / 60),
                )
                params["heating_demand"].value = np.array(demand[:required_len])

            # Set min/max temperature parameters
            params["min_temps"].value = self._pad_temp_array(
                min_temperatures_list, required_len, 18.0
            )
            params["max_temps"].value = self._pad_temp_array(
                max_temperatures_list, required_len, 26.0
            )

        else:
            # Fallback for loads not in param dict (shouldn't happen normally)
            start_temperature = hc.get("start_temperature", 20.0)
            start_temperature = float(start_temperature) if start_temperature is not None else 20.0
            start_temp_float = start_temperature

            outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)

            heatpump_cops = np.array(
                utils.calculate_cop_heatpump(
                    supply_temperature=supply_temperature,
                    carnot_efficiency=hc.get("carnot_efficiency", 0.4),
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                )[:required_len]
            )

            thermal_losses = np.array(
                utils.calculate_thermal_loss_signed(
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    indoor_temperature=start_temp_float,
                    base_loss=loss,
                )[:required_len]
            )

            # Compute heating demand (simplified fallback)
            if all(
                key in hc
                for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]
            ):
                indoor_target_temp = hc.get("indoor_target_temperature", 20.0)
                demand = utils.calculate_heating_demand_physics(
                    u_value=hc["u_value"],
                    envelope_area=hc["envelope_area"],
                    ventilation_rate=hc["ventilation_rate"],
                    heated_volume=hc["heated_volume"],
                    indoor_target_temperature=indoor_target_temp,
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    optimization_time_step=int(self.freq.total_seconds() / 60),
                )
            else:
                demand = utils.calculate_heating_demand(
                    specific_heating_demand=hc["specific_heating_demand"],
                    floor_area=hc["area"],
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    base_temperature=hc.get("base_temperature", 18.0),
                    annual_reference_hdd=hc.get("annual_reference_hdd", 3000.0),
                    optimization_time_step=int(self.freq.total_seconds() / 60),
                )
            heating_demand = np.array(demand[:required_len])
            min_temps_param = None
            max_temps_param = None

        # Build constraints using parameters
        predicted_temp_thermal = cp.Variable(required_len, name=f"temp_thermal_batt_{k}")

        constraints.append(predicted_temp_thermal[0] == start_temperature)

        # Thermal inertia: first-order low-pass filter on heat input
        tau_hours = float(hc.get("thermal_inertia_time_constant", 0.0) or 0.0)

        if tau_hours < 0:
            raise ValueError(
                f"Load {k}: thermal_inertia_time_constant must be >= 0, got {tau_hours}"
            )
        if tau_hours > 6:
            self.logger.warning(
                "Load %s: thermal_inertia_time_constant=%.1f h is large. "
                "Ensure this value reflects your system's dynamics.",
                k,
                tau_hours,
            )

        if tau_hours > 0:
            alpha = self.time_step / tau_hours
            if alpha > 1.0:
                self.logger.warning(
                    "Load %s: thermal_inertia_time_constant (%.2f h) < time_step (%.2f h), "
                    "clamping filter coefficient to 1.0.",
                    k,
                    tau_hours,
                    self.time_step,
                )
                alpha = 1.0

            # Q_input variable: filtered heat energy per timestep (kWh)
            q_input = cp.Variable(required_len, nonneg=True, name=f"q_input_{k}")

            # Initialize Q_input[0] from CVXPY Parameter (enables warm-start updates)
            params = self.param_thermal.get(k, {})
            q_input_start = params.get("q_input_start", 0.0)
            constraints.append(q_input[0] == q_input_start)

            # Raw heat input: COP * P_hp / 1000 * dt (kWh thermal per timestep)
            raw_heat = cp.multiply(heatpump_cops[:-1], p_deferrable[:-1]) / 1000 * self.time_step

            # First-order low-pass filter
            constraints.append(q_input[1:] == q_input[:-1] + alpha * (raw_heat - q_input[:-1]))

            # Temperature uses filtered Q_input instead of raw heat
            constraints.append(
                predicted_temp_thermal[1:]
                == predicted_temp_thermal[:-1]
                + conversion * (q_input[:-1] - heating_demand[:-1] - thermal_losses[:-1])
            )

            # Store reference for auto-persistence on cache hit
            if k in self.param_thermal:
                self.param_thermal[k]["q_input_var"] = q_input
        else:
            q_input = None
            # Original Langer & Volling equation (backward compatible)
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

        # Min/Max Temperature Constraints using parameters
        if min_temps_param is not None:
            constraints.append(predicted_temp_thermal[1:] >= min_temps_param[1:])
        else:
            valid_indices = [
                i
                for i, v in enumerate(min_temperatures_list)
                if v is not None and i < required_len and i > 0
            ]
            if valid_indices:
                limit_vals = np.array([min_temperatures_list[i] for i in valid_indices])
                constraints.append(predicted_temp_thermal[valid_indices] >= limit_vals)

        if max_temps_param is not None:
            constraints.append(predicted_temp_thermal[1:] <= max_temps_param[1:])
        else:
            valid_indices = [
                i
                for i, v in enumerate(max_temperatures_list)
                if v is not None and i < required_len and i > 0
            ]
            if valid_indices:
                limit_vals = np.array([max_temperatures_list[i] for i in valid_indices])
                constraints.append(predicted_temp_thermal[valid_indices] <= limit_vals)

        # Return heating_demand array for result building
        heating_demand_arr = (
            self.param_thermal[k]["heating_demand"].value
            if k in self.param_thermal
            else heating_demand
        )
        return predicted_temp_thermal, heating_demand_arr, q_input

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
        q_inputs = {}
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
                pred_temp, heat_demand, q_input_var = self._add_thermal_battery_constraints(
                    constraints, k, data_opt, p_load
                )
                predicted_temps[k] = pred_temp
                heating_demands[k] = heat_demand
                if q_input_var is not None:
                    q_inputs[k] = q_input_var

            # Detect special load types that have their own energy/operation constraints
            is_thermal_load = (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_config" in self.optim_conf["def_load_config"][k]
            )
            is_thermal_battery = (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_battery" in self.optim_conf["def_load_config"][k]
            )

            # Standard Deferrable Load - Energy Constraint
            # Now using parameterized Big-M formulation to allow changing operating hours
            # without rebuilding the problem. The constraint is always added but relaxed
            # via Big-M when param_energy_active = 0.
            #
            # When active=1: sum(p) * dt >= target_energy AND sum(p) * dt <= target_energy
            #                (equivalent to equality constraint)
            # When active=0: sum(p) * dt >= target_energy - M AND sum(p) * dt <= target_energy + M
            #                (effectively unconstrained)
            #
            # Skip this constraint for special load types that have their own energy constraints:
            # - Sequence loads (defined by power profile)
            # - Thermal loads (controlled by temperature targets)
            # - Thermal battery loads (controlled by heat demand)
            if (
                k < len(self.param_target_energy)
                and not is_sequence_load
                and not is_thermal_load
                and not is_thermal_battery
            ):
                # Big-M value: maximum possible energy consumption
                # = max_power * num_timesteps * time_step
                nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                if isinstance(nominal_power, list):
                    nominal_power = max(nominal_power)
                M_energy = nominal_power * n * self.time_step * 2  # 2x for safety margin

                # Energy constraint: sum(p) * dt == target_energy (when active)
                # Relaxed to: target_energy - M*(1-active) <= sum(p)*dt <= target_energy + M*(1-active)
                total_energy_expr = cp.sum(p_deferrable[k]) * self.time_step
                constraints.append(
                    total_energy_expr
                    >= self.param_target_energy[k] - M_energy * (1 - self.param_energy_active[k])
                )
                constraints.append(
                    total_energy_expr
                    <= self.param_target_energy[k] + M_energy * (1 - self.param_energy_active[k])
                )

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

            # Apply Window Constraints using Parameterized Mask
            # This allows changing time windows without rebuilding the problem
            # The mask is set in perform_optimization() before solving
            # mask[t] = 0 forces p_deferrable[k][t] <= 0 (must be off)
            # mask[t] = 1 allows p_deferrable[k][t] <= nominal_power (can operate)
            if k < len(self.param_window_masks):
                nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                if isinstance(nominal_power, list):
                    # For time-series nominal power, use the max value for the constraint
                    nominal_power = max(nominal_power)
                constraints.append(p_deferrable[k] <= nominal_power * self.param_window_masks[k])

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

                        # Required timesteps constraint using Big-M parameterization
                        # When active=1: sum(bin2) == required_timesteps (tight)
                        # When active=0: sum(bin2) can be anything (relaxed)
                        if k < len(self.param_required_timesteps):
                            M_timesteps = n * 2  # Max possible timesteps * safety
                            sum_bin2 = cp.sum(p_def_bin2[k])
                            constraints.append(
                                sum_bin2
                                >= self.param_required_timesteps[k]
                                - M_timesteps * (1 - self.param_timesteps_active[k])
                            )
                            constraints.append(
                                sum_bin2
                                <= self.param_required_timesteps[k]
                                + M_timesteps * (1 - self.param_timesteps_active[k])
                            )

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

        return predicted_temps, heating_demands, penalty_terms_total, q_inputs

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
        q_inputs=None,
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

        if q_inputs:
            for k, q_input_var in q_inputs.items():
                q_values = get_val(q_input_var)
                opt_tp[f"q_input_heater{k}"] = np.round(q_values, 4)

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
        min_power_of_deferrable_loads: list | None = None,
        debug: bool | None = False,
    ) -> pd.DataFrame:
        r"""
        Perform the actual optimization using Convex Programming (CVXPY).
        Includes automatic fallback to relaxed LP if MILP fails or times out.
        """
        # Dynamic Resizing
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

            # Re-initialize deferrable load parameters (window masks and energy constraints)
            self._init_deferrable_load_params()

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

        # Ensure min_power_of_deferrable_loads is available
        if min_power_of_deferrable_loads is None:
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

        # Update Window Mask Parameters for Deferrable Loads
        # This allows warm-starting even when time windows change
        n = len(p_pv)
        for k in range(min(num_deferrable_loads, len(self.param_window_masks))):
            # Calculate validated window bounds
            if def_total_timestep and def_total_timestep[k] > 0:
                def_start, def_end, _ = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_timestep[k]),
                    n,
                )
            else:
                def_start, def_end, _ = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_hours[k] / self.time_step) if def_total_hours[k] > 0 else 0,
                    n,
                )

            # Build the window mask: 0 outside window, 1 inside window
            window_mask = np.zeros(n)
            if def_end > def_start:
                window_mask[def_start:def_end] = 1.0
            else:
                # If window is invalid or full horizon, allow operation everywhere
                window_mask[:] = 1.0

            self.param_window_masks[k].value = window_mask

        # Update Thermal Parameters for warm-starting
        # This updates all thermal parameters (outdoor_temp, heating_demand, COPs, etc.)
        # On first call, these will be set during constraint building
        # On subsequent calls (cache hit), this ensures parameters reflect new forecasts
        if self.prob is not None and self.param_thermal:
            self.update_thermal_params(self.optim_conf, data_opt, p_load)
            # Refresh heating_demands for result building (stale numpy refs from first call)
            for k, params in self.param_thermal.items():
                if params["type"] == "thermal_battery":
                    self.heating_demands[k] = params["heating_demand"].value

        # Update Energy Constraint Parameters for Deferrable Loads
        # These control the Big-M relaxation of energy/timestep constraints
        for k in range(min(num_deferrable_loads, len(self.param_target_energy))):
            # Get nominal power
            nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
            if isinstance(nominal_power, list):
                nominal_power = max(nominal_power)

            # Determine operating requirement: def_total_timestep takes priority over def_total_hours
            # def_total_timestep is specified in number of timesteps
            # def_total_hours is specified in hours
            if def_total_timestep and k < len(def_total_timestep) and def_total_timestep[k] > 0:
                # Use timestep-based specification
                required_timesteps = ceil(def_total_timestep[k])
                # Convert to energy: power * timesteps * time_step (time_step is in hours)
                target_energy = nominal_power * required_timesteps * self.time_step
                constraint_active = True
            elif def_total_hours and k < len(def_total_hours) and def_total_hours[k] > 0:
                # Use hours-based specification
                operating_hours = def_total_hours[k]
                required_timesteps = ceil(operating_hours / self.time_step)
                target_energy = nominal_power * operating_hours
                constraint_active = True
            else:
                # No constraint specified
                required_timesteps = 0
                target_energy = 0.0
                constraint_active = False

            # Set energy constraint parameters
            if constraint_active:
                self.param_target_energy[k].value = target_energy
                self.param_energy_active[k].value = 1.0  # Constraint is active
            else:
                self.param_target_energy[k].value = 0.0
                self.param_energy_active[k].value = 0.0  # Constraint is relaxed (Big-M)

            # For single-constant (binary) loads, set the required timesteps
            is_single_const = self.optim_conf["set_deferrable_load_single_constant"][k]
            if is_single_const and constraint_active:
                self.param_required_timesteps[k].value = required_timesteps
                self.param_timesteps_active[k].value = 1.0  # Constraint is active
            else:
                self.param_required_timesteps[k].value = 0.0
                self.param_timesteps_active[k].value = 0.0  # Constraint is relaxed (Big-M)

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
            self.predicted_temps, self.heating_demands, penalty_terms_total, self.q_inputs = (
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
            if not isinstance(penalty_terms_total, int) or penalty_terms_total != 0:
                objective_expr.args[0] += penalty_terms_total

            self.prob = cp.Problem(objective_expr, constraints)

        # Solver Configuration
        solver_opts = {"verbose": False}
        if debug:
            solver_opts["verbose"] = True

        # Retrieve Constraints (Time & Threads)
        threads = self.optim_conf.get("num_threads", 0)
        timeout = self.optim_conf.get("lp_solver_timeout", 180)

        # Select Solver
        # We strictly default to HiGHS.
        requested_solver = os.environ.get("LP_SOLVER", "HIGHS").upper()
        selected_solver = cp.HIGHS

        if requested_solver == "GUROBI":
            if "GUROBI" in cp.installed_solvers():
                selected_solver = cp.GUROBI
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
            # MIP gap tolerance: allows solver to stop when within X% of optimal
            # Default 0 for backward compatibility (exact optimal)
            # Recommended: Set to 0.05 (5%) for ~2x speedup with negligible quality loss
            # Benchmarks show: 5% gap gives 1.75x speedup, 10% gives 1.86x, 20% gives 2.89x
            mip_gap = self.optim_conf.get("lp_solver_mip_rel_gap", 0.0)
            # Validate MIP gap is within sensible bounds [0, 1]
            if mip_gap < 0:
                self.logger.warning(
                    f"lp_solver_mip_rel_gap={mip_gap} is negative, using 0 (exact optimal)"
                )
                mip_gap = 0.0
            elif mip_gap > 1:
                self.logger.warning(
                    f"lp_solver_mip_rel_gap={mip_gap} exceeds 1.0 (100%), clamping to 1.0"
                )
                mip_gap = 1.0
            if mip_gap > 0:
                solver_opts["mip_rel_gap"] = float(mip_gap)
                self.logger.debug(f"MIP gap tolerance set to {mip_gap:.1%}")
            else:
                self.logger.debug("MIP gap tolerance disabled (exact optimal)")

        # Solve Execution with Fallback
        try:
            self.prob.solve(solver=selected_solver, warm_start=True, **solver_opts)
        except Exception as e:
            self.logger.warning(
                f"Solver {selected_solver} failed: {e}. Checking status for fallback..."
            )

        # Check for failure or "bad" status
        # Note: "user_limit" often means timeout. "infeasible" means configuration conflict.
        fail_statuses = ["infeasible", "unbounded", "user_limit", None]
        if self.prob.status in fail_statuses or self.prob.value is None:
            self.logger.warning(
                f"Optimization failed with status: '{self.prob.status}'. "
                "Retrying with relaxed constraints (Continuous LP)..."
            )

            # Backup Configuration
            original_semi_cont = copy.deepcopy(
                self.optim_conf.get("treat_deferrable_load_as_semi_cont", [])
            )
            original_single_const = copy.deepcopy(
                self.optim_conf.get("set_deferrable_load_single_constant", [])
            )

            # Relax Configuration: Disable Binary Logic
            n_def = self.optim_conf["number_of_deferrable_loads"]
            self.optim_conf["treat_deferrable_load_as_semi_cont"] = [False] * n_def
            self.optim_conf["set_deferrable_load_single_constant"] = [False] * n_def

            # Re-build Constraints (Clean Slate)
            constraints_relaxed = self.constraints[:]  # Start with base bound constraints

            # Re-apply main constraints
            self._add_main_power_balance_constraints(constraints_relaxed)
            # (Note: We reuse previous stress configs as they don't change with relaxation)
            if inv_stress_conf:
                self._add_hybrid_inverter_constraints(constraints_relaxed, inv_stress_conf)
            if batt_stress_conf:
                self._add_battery_constraints(constraints_relaxed, batt_stress_conf)

            if self.plant_conf["compute_curtailment"]:
                constraints_relaxed.append(self.vars["p_pv_curtailment"] <= self.param_pv_forecast)
            if self.costfun == "self-consumption" and "SC" in self.vars:
                constraints_relaxed.append(self.vars["SC"] <= self.param_pv_forecast)
                constraints_relaxed.append(
                    self.vars["SC"] <= self.param_load_forecast + self.vars["p_def_sum"]
                )

            # Re-call deferrable load constraints (Skipping binary logic due to config change)
            self.predicted_temps, self.heating_demands, penalty_terms_total, self.q_inputs = (
                self._add_deferrable_load_constraints(
                    constraints_relaxed,
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

            # Re-build Objective
            objective_expr = self._build_objective_function(batt_stress_conf, inv_stress_conf)
            if not isinstance(penalty_terms_total, int) or penalty_terms_total != 0:
                objective_expr.args[0] += penalty_terms_total

            # Solve Relaxed Problem
            prob_relaxed = cp.Problem(objective_expr, constraints_relaxed)
            try:
                self.logger.info("Solving relaxed problem (LP)...")
                prob_relaxed.solve(solver=selected_solver, **solver_opts)

                if prob_relaxed.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    self.logger.info("Relaxed optimization successful!")
                    self.prob = prob_relaxed  # Use this result
                    # Mark status so user knows it was relaxed
                    self.prob._status = "Optimal (Relaxed)"
                else:
                    self.logger.error(
                        f"Relaxed optimization also failed with status: {prob_relaxed.status}"
                    )
                    self.prob = prob_relaxed
            except Exception as e:
                self.logger.error(f"Relaxed optimization crashed: {e}")

            # 5. Restore Configuration
            self.optim_conf["treat_deferrable_load_as_semi_cont"] = original_semi_cont
            self.optim_conf["set_deferrable_load_single_constant"] = original_single_const

        # Fix for Status Case: Map "optimal" -> "Optimal"
        status_raw = self.prob.status
        self.optim_status = status_raw.title() if status_raw else "Failure"

        # Helper: Ensure we return "Optimal" for tests if it was "Optimal (Relaxed)" or "Optimal_Inaccurate"
        if self.prob.value is None or self.prob.status not in [
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
            "Optimal (Relaxed)",
        ]:
            self.logger.warning("Cost function cannot be evaluated or Infeasible/Unbounded")

            # Create a DataFrame with the correct index (timestamps)
            opt_tp = pd.DataFrame(index=data_opt.index)

            # explicitely set the status column so downstream functions (like get_injection_dict)
            # don't crash when trying to access or drop it.
            opt_tp["optim_status"] = self.optim_status

            return opt_tp
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
            q_inputs=self.q_inputs,
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
                    f"Skipping day {day} as days have different timezone, probably because of DST."
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
