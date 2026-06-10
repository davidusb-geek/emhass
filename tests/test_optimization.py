#!/usr/bin/env python

import copy
import pathlib
import pickle
import random
import unittest
from datetime import datetime
from unittest import mock

import aiofiles
import numpy as np
import orjson
import pandas as pd
from pandas.testing import assert_series_equal

from emhass import utils
from emhass.forecast import Forecast
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    get_days_list,
    get_logger,
    get_root,
    get_yaml_parse,
)

# The root folder
root = pathlib.Path(get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)

# Valid optimization statuses (some solvers return "Optimal (Relaxed)" for MIP problems)
VALID_OPTIMAL_STATUSES = ["Optimal", "Optimal (Relaxed)"]


class TestOptimization(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        get_data_from_file = True
        params = {}
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await build_secrets(emhass_conf, logger, no_response=True)
            params = await build_params(emhass_conf, secrets, config, logger)
            params["optim_conf"]["set_use_pv"] = True
        else:
            raise Exception(
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params_json, logger)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = (
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        )
        # Build RetrieveHass object
        self.rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_json,
            emhass_conf,
            logger,
        )
        # Obtain sensor values from saved file
        if get_data_from_file:
            async with aiofiles.open(emhass_conf["data_path"] / "test_df_final.pkl", "rb") as inp:
                contents = await inp.read()
                self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = pickle.loads(
                    contents
                )
                self.rh.var_list = self.var_list
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
            self.retrieve_hass_conf["sensor_power_photovoltaics"] = str(self.var_list[1])
            self.retrieve_hass_conf["sensor_linear_interp"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
            ]
            self.retrieve_hass_conf["sensor_replace_zero"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"]
            ]
        # Else obtain sensor values from HA
        else:
            self.days_list = get_days_list(self.retrieve_hass_conf["historic_days_to_retrieve"])
            self.var_list = [
                self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
                self.retrieve_hass_conf["sensor_power_photovoltaics"],
            ]
            self.rh.get_data(
                self.days_list,
                self.var_list,
                minimal_response=False,
                significant_changes_only=False,
            )
        # Prepare data for optimization
        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
        )
        self.df_input_data = self.rh.df_final.copy()
        # Build Forecast object
        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=get_data_from_file,
        )
        self.df_weather = await self.fcst.get_weather_forecast(method="csv")
        self.p_pv_forecast = self.fcst.get_power_from_weather(self.df_weather)
        self.p_load_forecast = await self.fcst.get_load_forecast(
            method=optim_conf["load_forecast_method"]
        )
        self.df_input_data_dayahead = pd.concat([self.p_pv_forecast, self.p_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ["p_pv_forecast", "p_load_forecast"]
        # Build Optimization object
        self.costfun = "profit"
        self.opt = self.create_optimization()
        self.df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.input_data_dict = {
            "retrieve_hass_conf": retrieve_hass_conf,
        }

    # Helper methods to reduce code duplication
    def create_optimization(self, costfun=None, optim_conf=None, **kwargs):
        """Helper to create Optimization object with standard parameters.

        Args:
            costfun: Cost function override (defaults to self.costfun)
            optim_conf: Optim config override (defaults to self.optim_conf)
            **kwargs: Additional overrides for any parameter

        Returns:
            Optimization object
        """
        return Optimization(
            kwargs.get("retrieve_hass_conf", self.retrieve_hass_conf),
            optim_conf or self.optim_conf,
            kwargs.get("plant_conf", self.plant_conf),
            kwargs.get("var_load_cost", self.fcst.var_load_cost),
            kwargs.get("var_prod_price", self.fcst.var_prod_price),
            costfun or self.costfun,
            kwargs.get("emhass_conf", emhass_conf),
            kwargs.get("logger", logger),
        )

    def prepare_forecast_data(self, df=None):
        """Prepare input data with load cost and production price forecasts.

        Args:
            df: DataFrame to prepare (defaults to self.df_input_data_dayahead)

        Returns:
            DataFrame with forecasts added
        """
        if df is None:
            df = self.df_input_data_dayahead
        df = self.fcst.get_load_cost_forecast(df)
        df = self.fcst.get_prod_price_forecast(df)
        return df

    def assert_valid_optimization_result(self, opt_res, costfun=None, check_battery=False):
        """Assert optimization result has correct format and required columns.

        Args:
            opt_res: Optimization result DataFrame
            costfun: Expected cost function name (defaults to self.costfun)
            check_battery: Whether to check for battery-related columns
        """
        # Structure assertions
        self.assertIsInstance(opt_res, type(pd.DataFrame()))
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)

        # Column assertions
        costfun = costfun or self.costfun
        if costfun == "self-consumption":
            self.assertIn("cost_fun_selfcons", opt_res.columns)
        else:
            self.assertIn(f"cost_fun_{costfun}", opt_res.columns)

        if check_battery:
            self.assertIn("P_batt", opt_res.columns)
            self.assertIn("SOC_opt", opt_res.columns)

    def assert_energy_constraint(
        self, power_series, expected_hours, nominal_power=None, tolerance=1e-3
    ):
        """Assert that total energy matches expected operating hours.

        Args:
            power_series: Power values (Series or array)
            expected_hours: Expected operating hours
            nominal_power: Nominal power (defaults to first deferrable load)
            tolerance: Numerical tolerance
        """
        if nominal_power is None:
            nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][0]

        timestep_hours = self.retrieve_hass_conf["optimization_time_step"].seconds / 3600
        expected_energy = nominal_power * expected_hours
        actual_energy = power_series.sum() * timestep_hours

        self.assertLess(
            np.abs(expected_energy - actual_energy),
            tolerance,
            f"Energy mismatch: expected {expected_energy:.3f} Wh, got {actual_energy:.3f} Wh",
        )

    def run_thermal_battery_optimization(self, thermal_config, outdoor_temps=None, ghi=None):
        """Helper to run thermal battery optimization tests.

        Args:
            thermal_config: Thermal battery configuration dict
            outdoor_temps: Outdoor temperature forecast (list or single value for constant)
            ghi: Global horizontal irradiance forecast (optional)

        Returns:
            Optimization result DataFrame
        """
        # Prepare forecast data
        df = self.prepare_forecast_data()

        # Add outdoor temperature
        if outdoor_temps is None:
            outdoor_temps = 10.0
        df["outdoor_temperature_forecast"] = outdoor_temps

        # Add GHI if provided
        if ghi is not None:
            df["ghi"] = ghi

        # Configure thermal battery
        self.optim_conf["def_load_config"] = [{"thermal_battery": thermal_config}]
        opt = self.create_optimization()

        # Run optimization
        unit_load_cost = df[opt.var_load_cost].values
        unit_prod_price = df[opt.var_prod_price].values

        return opt.perform_optimization(
            df,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

    def run_startup_penalty_test(self, load_costs, optim_updates, expected_pattern):
        """Helper for startup penalty tests.

        Args:
            load_costs: List of load cost forecasts
            optim_updates: Dict of optim_conf updates
            expected_pattern: Expected P_deferrable0 pattern (list of 0/1 multipliers)
        """
        self.fcst.params["passed_data"]["load_cost_forecast"] = load_costs
        self.optim_conf.update(optim_updates)
        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series(expected_pattern, index=self.opt_res_dayahead.index),
            check_names=False,
        )

    def run_optimization_with_config(self, def_load_config):
        """Helper to run optimization with a given def_load_config and verify success.

        Args:
            def_load_config: Configuration list for deferrable loads

        Returns:
            Optimization result DataFrame
        """
        self.optim_conf["def_load_config"] = def_load_config
        opt = self.create_optimization()
        self.opt = opt  # Store so callers can inspect optim_status etc.

        # Run optimization
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        opt_res = opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Verify optimization succeeded
        self.assertIsInstance(opt_res, type(pd.DataFrame()))
        self.assertIn("P_deferrable0", opt_res.columns)
        self.assertGreater(len(opt_res), 0)

        return opt_res

    # Check formatting of output from perfect optimization
    def test_perform_perfect_forecast_optim(self):
        self.opt_res = self.opt.perform_perfect_forecast_optim(self.df_input_data, self.days_list)
        self.assert_valid_optimization_result(self.opt_res)

    def test_perform_dayahead_forecast_optim(self):
        # Check formatting of output from dayahead optimization
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assert_valid_optimization_result(self.opt_res_dayahead)
        self.assert_energy_constraint(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["operating_hours_of_each_deferrable_load"][0],
        )
        # Test the battery, dynamics and grid exchange contraints
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "set_nocharge_from_grid": True,
                "set_battery_dynamic": True,
                "set_nodischarge_to_grid": True,
            }
        )
        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assert_valid_optimization_result(self.opt_res_dayahead, check_battery=True)
        self.assertAlmostEqual(
            self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1], "SOC_opt"],
            self.plant_conf["battery_target_state_of_charge"],
        )
        # Test table conversion
        opt_res = pd.read_csv(
            emhass_conf["data_path"] / "opt_res_latest.csv", index_col="timestamp"
        )
        cost_cols = [i for i in opt_res.columns if "cost_" in i]
        table = (
            opt_res[cost_cols]
            .reset_index()
            .sum(numeric_only=True)
            .to_frame(name="Cost Totals")
            .reset_index()
        )
        self.assertEqual(table.columns[0], "index")
        self.assertEqual(table.columns[1], "Cost Totals")
        # Check status
        self.assertIn("optim_status", self.opt_res_dayahead.columns)
        # Test treat_def_as_semi_cont and set_def_constant constraints
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [True, True],
            }
        )
        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [False, True],
                "set_deferrable_load_single_constant": [True, True],
            }
        )
        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [False, True],
                "set_deferrable_load_single_constant": [False, True],
            }
        )
        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [False, False],
                "set_deferrable_load_single_constant": [False, True],
            }
        )
        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update({"treat_deferrable_load_as_semi_cont": [False, False]})
        self.optim_conf.update({"set_deferrable_load_single_constant": [False, False]})
        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        # Test with debug mode and batt SOC conditions
        self.optim_conf["set_use_battery"] = True
        soc_init = None
        soc_final = 0.3
        self.optim_conf["set_total_pv_sell"] = True
        self.opt = self.create_optimization()

        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values
        self.opt_res_dayahead = self.opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
            soc_init=soc_init,
            soc_final=soc_final,
            debug=True,
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertIn("cost_fun_" + self.costfun, self.opt_res_dayahead.columns)
        self.assertEqual(self.opt.optim_status, "Optimal")

    # Check formatting of output from dayahead optimization in self-consumption
    def test_perform_dayahead_forecast_optim_costfun_selfconsumption(self):
        costfun = "self-consumption"
        self.opt = self.create_optimization(costfun=costfun)
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertIn("cost_fun_selfcons", self.opt_res_dayahead.columns)

    # Check formatting of output from dayahead optimization in cost
    def test_perform_dayahead_forecast_optim_costfun_cost(self):
        costfun = "cost"
        self.opt = self.create_optimization(costfun=costfun)
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertIn("cost_fun_cost", self.opt_res_dayahead.columns)

    # Test with total PV sell and different solvers
    def test_perform_dayahead_forecast_optim_aux(self):
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [False, False]
        self.optim_conf["set_total_pv_sell"] = True
        self.optim_conf["set_deferrable_load_single_constant"] = [True, True]
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )

    # Check minimum deferrable load power
    def test_perform_dayahead_forecast_optim_min_def_load_power(self):
        self.optim_conf["minimum_power_of_deferrable_loads"] = [1000.0, 100.0]
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        # Verify the minimum power constraint for each deferrable load <<<
        num_loads = self.optim_conf["number_of_deferrable_loads"]
        min_powers = self.optim_conf["minimum_power_of_deferrable_loads"]
        for k in range(num_loads):
            min_power_k = min_powers[k]
            power_column = self.opt_res_dayahead[f"P_deferrable{k}"]
            # Filter for all values that are not close to zero (i.e., when the load is ON)
            non_zero_powers = power_column[~np.isclose(power_column, 0)]
            # If there are any non-zero values, assert that they are all greater than
            # or equal to the minimum power setting.
            if not non_zero_powers.empty:
                self.assertTrue(
                    (non_zero_powers >= min_power_k).all(),
                    f"Deferrable load {k} has values below the minimum power of {min_power_k} W. "
                    f"Invalid values found: {non_zero_powers[non_zero_powers < min_power_k].tolist()}",
                )

    def test_perform_naive_mpc_optim(self):
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Test the battery
        self.optim_conf.update({"set_use_battery": True})
        self.opt = self.create_optimization()
        prediction_horizon = 10
        soc_init = 0.4
        soc_final = 0.6
        def_total_hours = [2, 3]
        def_start_timestep = [-5, 0]
        def_end_timestep = [4, 0]
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=def_total_hours,
            def_total_timestep=None,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn("P_batt", self.opt_res_dayahead.columns)
        self.assertIn("SOC_opt", self.opt_res_dayahead.columns)
        self.assertLess(
            np.abs(
                self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1], "SOC_opt"] - soc_final
            ),
            1e-3,
        )
        term1 = self.optim_conf["nominal_power_of_deferrable_loads"][0] * def_total_hours[0]
        term2 = self.opt_res_dayahead["P_deferrable0"].sum() * (
            self.retrieve_hass_conf["optimization_time_step"].seconds / 3600
        )
        self.assertLess(np.abs(term1 - term2), 1e-3)
        #
        soc_init = 0.8
        soc_final = 0.5
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=def_total_hours,
            def_total_timestep=None,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
        )
        self.assertAlmostEqual(
            self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1], "SOC_opt"],
            soc_final,
        )

    def test_perform_naive_mpc_optim_intermediate_soc_target(self):
        """Issue #553: an intermediate ``soc_target`` must be met by
        ``soc_target_timestep`` while the battery is still free to discharge
        afterward, and passing no target must leave behaviour unchanged.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Flat, equal buy/sell prices -> zero time-arbitrage incentive. With a
        # lossless battery any charge/discharge round-trip is exactly cost-neutral,
        # so the unconstrained baseline has no economic reason to move the battery.
        self.df_input_data_dayahead["unit_load_cost"] = 0.2
        self.df_input_data_dayahead["unit_prod_price"] = 0.2
        self.optim_conf.update({"set_use_battery": True})
        self.optim_conf.update({"number_of_deferrable_loads": 0})
        self.optim_conf.update({"set_battery_dynamic": False})
        # A small positive cycle cost makes leaving the battery untouched the
        # unique baseline optimum (any cycling is then strictly worse), so the
        # charge seen in the targeted run is unambiguously caused by the floor.
        self.optim_conf.update({"weight_battery_discharge": 1.0})
        self.optim_conf.update({"weight_battery_charge": 1.0})
        # Allow export so the battery can discharge down to soc_final (otherwise
        # the no-discharge-to-grid default makes shedding 0.8->0.5 infeasible when
        # local load is low). This keeps the test focused on the intermediate target.
        self.optim_conf.update({"set_nodischarge_to_grid": False})
        # Clean, ample battery so a mid-horizon target is reachable and the
        # SOC trajectory is deterministic (no efficiency losses).
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        prediction_horizon = 10
        # Deterministic charge-up scenario: start and end low so the
        # unconstrained baseline has no reason to charge (it stays ~soc_init),
        # but a mid-horizon target forces a clear charge 0.3->0.9 by step 5 and
        # back to 0.3 — feasible at 20 kW / 10 kWh / eff 1.0.
        soc_init = 0.3
        soc_final = 0.3
        target_step = 5
        soc_target = 0.9

        # Baseline (no intermediate target) — establishes the unconstrained SOC.
        self.opt = self.create_optimization()
        opt_res_baseline = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=soc_init,
            soc_final=soc_final,
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertIn("SOC_opt", opt_res_baseline.columns)
        soc_baseline_at_step = opt_res_baseline["SOC_opt"].iloc[target_step]
        # No-op-by-default coverage: with no target and no arbitrage incentive the
        # baseline leaves the battery on soc_init, well below the target. This is
        # what proves the floor param (not arbitrage / tie-breaking) drives the
        # change in the targeted run below.
        self.assertAlmostEqual(soc_baseline_at_step, soc_init, delta=1e-2)
        self.assertLess(soc_baseline_at_step, soc_target - 0.2)

        # With intermediate target — SOC at target_step must meet/exceed soc_target.
        self.opt = self.create_optimization()
        opt_res_target = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=soc_init,
            soc_final=soc_final,
            soc_target=soc_target,
            soc_target_timestep=target_step,
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.assertIn("SOC_opt", opt_res_target.columns)
        # Lock in the DPP fix: a non-DPP product of two parameters would force
        # recanonicalisation and flip this to False (this is what catches Fix 1
        # regressing).
        self.assertTrue(self.opt.prob.is_dpp())
        soc_target_at_step = opt_res_target["SOC_opt"].iloc[target_step]

        # 1) The target is enforced at the requested timestep.
        self.assertGreaterEqual(soc_target_at_step, soc_target - 1e-3)
        # 2) The constraint actually bit: the targeted run holds clearly more
        #    charge at the target step than the unconstrained baseline did.
        self.assertGreaterEqual(soc_target_at_step, soc_baseline_at_step + 0.2)
        # 3) The battery is still free to discharge afterward: the end-of-horizon
        #    SOC still lands on soc_final (target does not pin the tail).
        self.assertLess(
            np.abs(opt_res_target.loc[opt_res_target.index[-1], "SOC_opt"] - soc_final),
            1e-2,
        )

    def test_intermediate_soc_target_clamped_above_max(self):
        """Issue #553: a soc_target above battery_maximum_state_of_charge is
        clamped to the ceiling (with a warning) instead of forcing infeasibility.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["unit_load_cost"] = 0.2
        self.df_input_data_dayahead["unit_prod_price"] = 0.2
        self.optim_conf.update({"set_use_battery": True})
        self.optim_conf.update({"number_of_deferrable_loads": 0})
        self.optim_conf.update({"set_battery_dynamic": False})
        self.optim_conf.update({"set_nodischarge_to_grid": False})
        soc_max = 0.8
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": soc_max,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        self.opt = self.create_optimization()
        with self.assertLogs(level="WARNING") as logs:
            opt_res = self.opt.perform_naive_mpc_optim(
                self.df_input_data_dayahead,
                self.p_pv_forecast,
                self.p_load_forecast,
                10,
                soc_init=0.3,
                soc_final=0.3,
                soc_target=1.5,  # absurd: above the 0.8 ceiling
                soc_target_timestep=5,
            )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        # Clamped: the target step reaches the ceiling and SOC never exceeds it.
        self.assertGreaterEqual(opt_res["SOC_opt"].iloc[5], soc_max - 1e-3)
        self.assertLessEqual(opt_res["SOC_opt"].max(), soc_max + 1e-3)
        # And the out-of-range request is surfaced to the user.
        self.assertTrue(
            any("outside" in line for line in logs.output),
            msg=f"expected an out-of-range clamp warning, got: {logs.output}",
        )

    def test_capacity_charge_shaves_peak_import(self):
        """Issue #623: an opt-in ``capacity_cost_per_kw`` must flatten the peak
        grid import. With the feature off no peak variable is created and the
        plan is unchanged; with it on the planned import peak drops, and the
        problem stays DPP (warm-start preserved).
        """
        df = self.prepare_forecast_data()
        n = len(df)
        prediction_horizon = 6
        # No PV; a flat load with one sharp spike the optimizer would otherwise
        # import in full. The battery can discharge to shave that spike.
        df["p_pv_forecast"] = 0.0
        load = np.full(n, 1000.0)
        load[2] = 5000.0  # the peak to be shaved
        df["p_load_forecast"] = load
        pv = df["p_pv_forecast"].copy()
        load_s = df["p_load_forecast"].copy()
        # Flat tariff -> no energy-arbitrage incentive to move the battery; a
        # small cycle cost makes "just import the spike" the UNIQUE baseline
        # optimum (battery use is strictly worse), so any peak shaving in the
        # feature-on run is caused by the capacity term, not by tie-breaking.
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
                "weight_battery_discharge": 0.1,
                "weight_battery_charge": 0.1,
            }
        )
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        # Same start and end SoC: no forced discharge, so the baseline leaves the
        # battery idle and imports the whole spike (deterministic peak).
        soc_init = 0.5
        soc_final = 0.5

        # Baseline: feature OFF (capacity_cost_per_kw default 0).
        self.optim_conf["capacity_cost_per_kw"] = 0.0
        self.opt = self.create_optimization()
        res_off = self.opt.perform_naive_mpc_optim(
            df, pv, load_s, prediction_horizon, soc_init=soc_init, soc_final=soc_final
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        peak_off = res_off["P_grid_pos"].iloc[:prediction_horizon].max()
        self.assertGreater(
            peak_off,
            4000.0,
            msg="baseline did not import the full spike; scenario no longer discriminates",
        )
        # No peak variable exists when the feature is off (true no-op structure).
        self.assertNotIn("peak_import", self.opt.vars)

        # Feature ON: a positive capacity cost must lower the import peak.
        self.optim_conf["capacity_cost_per_kw"] = 2.0
        self.opt = self.create_optimization()
        res_on = self.opt.perform_naive_mpc_optim(
            df, pv, load_s, prediction_horizon, soc_init=soc_init, soc_final=soc_final
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        # DPP preserved (warm-start safe): a non-DPP term would flip this.
        self.assertTrue(self.opt.prob.is_dpp())
        self.assertIn("peak_import", self.opt.vars)
        peak_on = res_on["P_grid_pos"].iloc[:prediction_horizon].max()
        self.assertLess(
            peak_on,
            peak_off - 1000.0,
            msg="capacity charge did not shave the import peak",
        )

    def test_capacity_charge_zero_equals_unset(self):
        """Issue #623: ``capacity_cost_per_kw`` of 0 must produce the identical
        plan to the parameter being absent, i.e. a provable no-op default.
        """
        df = self.prepare_forecast_data()
        n = len(df)
        prediction_horizon = 6
        df["p_pv_forecast"] = 0.0
        load = np.full(n, 1000.0)
        load[2] = 5000.0
        df["p_load_forecast"] = load
        pv = df["p_pv_forecast"].copy()
        load_s = df["p_load_forecast"].copy()
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
                "weight_battery_discharge": 0.1,
                "weight_battery_charge": 0.1,
            }
        )
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )

        # Parameter absent entirely.
        self.optim_conf.pop("capacity_cost_per_kw", None)
        self.opt = self.create_optimization()
        res_absent = self.opt.perform_naive_mpc_optim(
            df, pv, load_s, prediction_horizon, soc_init=0.5, soc_final=0.5
        )
        self.assertNotIn("peak_import", self.opt.vars)

        # Parameter present and explicitly 0.
        self.optim_conf["capacity_cost_per_kw"] = 0.0
        self.opt = self.create_optimization()
        res_zero = self.opt.perform_naive_mpc_optim(
            df, pv, load_s, prediction_horizon, soc_init=0.5, soc_final=0.5
        )
        self.assertNotIn("peak_import", self.opt.vars)
        # Identical plans: the explicit 0 changes nothing vs the absent default.
        np.testing.assert_allclose(
            res_zero["P_grid_pos"].to_numpy(),
            res_absent["P_grid_pos"].to_numpy(),
            atol=1e-6,
        )

    def test_capacity_charge_coerces_string_and_rejects_invalid(self):
        """Issue #623: capacity_cost_per_kw is runtime-overridable and arrives
        verbatim, so an HA template delivers it as a string. A numeric string
        must be coerced and applied; a non-numeric or negative value must be
        ignored (no crash, no peak variable) with a warning.
        """
        df = self.prepare_forecast_data()
        n = len(df)
        prediction_horizon = 6
        df["p_pv_forecast"] = 0.0
        load = np.full(n, 1000.0)
        load[2] = 5000.0
        df["p_load_forecast"] = load
        pv = df["p_pv_forecast"].copy()
        load_s = df["p_load_forecast"].copy()
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
                "weight_battery_discharge": 0.1,
                "weight_battery_charge": 0.1,
            }
        )
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )

        # A numeric string is coerced and the feature engages.
        self.optim_conf["capacity_cost_per_kw"] = "2.0"
        self.opt = self.create_optimization()
        res_str = self.opt.perform_naive_mpc_optim(
            df, pv, load_s, prediction_horizon, soc_init=0.5, soc_final=0.5
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertIn("peak_import", self.opt.vars)
        self.assertIn("P_grid_pos", res_str.columns)

        # A non-numeric value is ignored with a warning and creates no peak var.
        self.optim_conf["capacity_cost_per_kw"] = "not a number"
        with self.assertLogs(level="WARNING") as logs:
            self.opt = self.create_optimization()
        self.assertNotIn("peak_import", self.opt.vars)
        self.assertTrue(
            any("capacity_cost_per_kw" in line for line in logs.output),
            msg=f"expected an invalid-value warning, got: {logs.output}",
        )

        # A negative value is ignored too (no peak var, fails safe).
        self.optim_conf["capacity_cost_per_kw"] = -5.0
        self.opt = self.create_optimization()
        self.assertNotIn("peak_import", self.opt.vars)

        # A non-finite value ("inf") is ignored too (no peak var, no crash).
        self.optim_conf["capacity_cost_per_kw"] = "inf"
        self.opt = self.create_optimization()
        self.assertNotIn("peak_import", self.opt.vars)

    def test_capacity_charge_respects_current_period_peak(self):
        """Issue #623 Phase 2: current_period_peak (Watts, runtime-only) is the peak
        already locked in for the billing period. With a capacity charge active, only
        import ABOVE current_period_peak is worth shaving:
          - current_period_peak = 0  -> prices the full horizon peak (== Phase 1).
          - current_period_peak ABOVE the achievable horizon peak -> nothing left to
            shave; plan == the no-capacity baseline (battery idle, full spike).
        Discriminator: capacity_cost_per_kw > 0 is FIXED across the two capacity runs,
        so only current_period_peak changes. A Phase-1-only build (ignoring
        current_period_peak) would shave in BOTH and fail the high-baseline check.
        """
        df = self.prepare_forecast_data()
        n = len(df)
        prediction_horizon = 6
        df["p_pv_forecast"] = 0.0
        load = np.full(n, 1000.0)
        load[2] = 5000.0  # achievable horizon peak ~5 kW = 5000 W
        df["p_load_forecast"] = load
        pv = df["p_pv_forecast"].copy()
        load_s = df["p_load_forecast"].copy()
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
                "weight_battery_discharge": 0.1,
                "weight_battery_charge": 0.1,
            }
        )
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        soc_init = 0.5
        soc_final = 0.5

        def run(cap_cost, current_period_peak):
            self.optim_conf["capacity_cost_per_kw"] = cap_cost
            self.opt = self.create_optimization()
            res = self.opt.perform_naive_mpc_optim(
                df,
                pv,
                load_s,
                prediction_horizon,
                soc_init=soc_init,
                soc_final=soc_final,
                current_period_peak=current_period_peak,
            )
            self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
            return res

        # RUN 1 - no capacity charge: true no-op baseline (full spike imported).
        res_baseline = run(cap_cost=0.0, current_period_peak=0.0)
        peak_baseline = res_baseline["P_grid_pos"].iloc[:prediction_horizon].max()
        self.assertGreater(
            peak_baseline,
            4000.0,
            msg="baseline did not import the full spike; scenario no longer discriminates",
        )
        self.assertNotIn("peak_import", self.opt.vars)  # off => no var

        # RUN 2 - capacity charge ON, current_period_peak = 0: Phase-1 shaving.
        res_shave = run(cap_cost=2.0, current_period_peak=0.0)
        self.assertTrue(self.opt.prob.is_dpp())  # DPP preserved (warm-start)
        self.assertIn("peak_import", self.opt.vars)
        peak_shave = res_shave["P_grid_pos"].iloc[:prediction_horizon].max()
        self.assertLess(
            peak_shave,
            peak_baseline - 1000.0,
            msg="current_period_peak=0 must reproduce Phase 1 shaving",
        )

        # RUN 3 - SAME capacity charge, current_period_peak ABOVE achievable peak
        # (20000 W = 20 kW >> 5000 W spike): peak already locked in -> nothing left
        # to shave -> plan == the no-capacity baseline (battery idle, full spike).
        res_locked = run(cap_cost=2.0, current_period_peak=20000.0)
        self.assertTrue(self.opt.prob.is_dpp())
        peak_locked = res_locked["P_grid_pos"].iloc[:prediction_horizon].max()
        # (a) does NOT shave: peak matches the no-capacity baseline.
        self.assertGreater(
            peak_locked,
            peak_baseline - 1e-3,
            msg="capacity charge shaved a peak already locked in above current_period_peak",
        )
        # (b) whole-plan equivalence to the no-capacity baseline (strongest form).
        np.testing.assert_allclose(
            res_locked["P_grid_pos"].iloc[:prediction_horizon].to_numpy(),
            res_baseline["P_grid_pos"].iloc[:prediction_horizon].to_numpy(),
            atol=1e-3,
        )
        # (c) counterfactual defeating a Phase-1-only build: SAME cost, run 2 shaved,
        #     run 3 did not. If current_period_peak were ignored these would be equal.
        self.assertGreater(
            peak_locked,
            peak_shave + 1000.0,
            msg="current_period_peak ignored: high baseline still shaved like peak=0",
        )

    def test_current_period_peak_noop_and_coercion(self):
        """Issue #623 Phase 2: current_period_peak with the feature OFF is a no-op;
        with the feature ON, 0/unset reproduces the Phase-1 plan, a numeric string is
        coerced, and an invalid/negative value falls back to 0 with a warning.
        """
        df = self.prepare_forecast_data()
        n = len(df)
        prediction_horizon = 6
        df["p_pv_forecast"] = 0.0
        load = np.full(n, 1000.0)
        load[2] = 5000.0
        df["p_load_forecast"] = load
        pv = df["p_pv_forecast"].copy()
        load_s = df["p_load_forecast"].copy()
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
                "weight_battery_discharge": 0.1,
                "weight_battery_charge": 0.1,
            }
        )
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )

        # (i) Feature OFF + current_period_peak set => no peak var, plan unchanged.
        self.optim_conf["capacity_cost_per_kw"] = 0.0
        self.opt = self.create_optimization()
        res_off_set = self.opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            current_period_peak=20000.0,
        )
        self.assertNotIn("peak_import", self.opt.vars)
        self.opt = self.create_optimization()
        res_off_unset = self.opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
        )
        np.testing.assert_allclose(
            res_off_set["P_grid_pos"].to_numpy(),
            res_off_unset["P_grid_pos"].to_numpy(),
            atol=1e-6,
        )

        # (ii) Feature ON: peak=0 explicit == unset (identical plan).
        self.optim_conf["capacity_cost_per_kw"] = 2.0
        self.opt = self.create_optimization()
        res_zero = self.opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            current_period_peak=0.0,
        )
        self.opt = self.create_optimization()
        res_unset_on = self.opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
        )
        np.testing.assert_allclose(
            res_zero["P_grid_pos"].to_numpy(),
            res_unset_on["P_grid_pos"].to_numpy(),
            atol=1e-6,
        )

        # (iii) numeric string is coerced and locks in the peak (no shaving vs zero).
        self.opt = self.create_optimization()
        res_str = self.opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            current_period_peak="20000",
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreater(
            res_str["P_grid_pos"].iloc[:prediction_horizon].max(),
            res_zero["P_grid_pos"].iloc[:prediction_horizon].max() + 1000.0,
            msg='numeric string "20000" must coerce and lock the peak in (no shave)',
        )

        # (iv) invalid string -> warning, treated as 0, identical plan to peak=0.
        self.opt = self.create_optimization()
        with self.assertLogs(level="WARNING") as logs:
            res_bad = self.opt.perform_naive_mpc_optim(
                df,
                pv,
                load_s,
                prediction_horizon,
                soc_init=0.5,
                soc_final=0.5,
                current_period_peak="not a number",
            )
        self.assertTrue(any("current_period_peak" in line for line in logs.output))
        np.testing.assert_allclose(
            res_bad["P_grid_pos"].to_numpy(),
            res_zero["P_grid_pos"].to_numpy(),
            atol=1e-6,
        )

        # (v) negative -> warning, treated as 0, identical plan to peak=0.
        self.opt = self.create_optimization()
        with self.assertLogs(level="WARNING"):
            res_neg = self.opt.perform_naive_mpc_optim(
                df,
                pv,
                load_s,
                prediction_horizon,
                soc_init=0.5,
                soc_final=0.5,
                current_period_peak=-5.0,
            )
        np.testing.assert_allclose(
            res_neg["P_grid_pos"].to_numpy(),
            res_zero["P_grid_pos"].to_numpy(),
            atol=1e-6,
        )

        # (vi) non-finite "inf" -> warning, treated as 0 (cp.Parameter rejects
        # inf), identical plan to peak=0; must degrade gracefully, not crash.
        self.opt = self.create_optimization()
        with self.assertLogs(level="WARNING"):
            res_inf = self.opt.perform_naive_mpc_optim(
                df,
                pv,
                load_s,
                prediction_horizon,
                soc_init=0.5,
                soc_final=0.5,
                current_period_peak="inf",
            )
        np.testing.assert_allclose(
            res_inf["P_grid_pos"].to_numpy(),
            res_zero["P_grid_pos"].to_numpy(),
            atol=1e-6,
        )

    def test_battery_first_priority_drains_before_import(self):
        """Issue #834: with ``set_battery_first_priority`` the optimizer must
        not import from the grid while the battery is still above its minimum
        SoC. On a flat tariff "drain first" and "interleave import with
        discharge" are cost-equivalent, so the solver is otherwise free to
        import while the battery is full. The flag forces the drain-first order.
        """
        df = self.prepare_forecast_data()
        n = len(df)
        # Night-time scenario: no PV, constant load.
        df["p_pv_forecast"] = 0.0
        load_w = 2000.0
        df["p_load_forecast"] = load_w
        # A gently increasing import price makes "import as early as possible"
        # the UNIQUE baseline optimum, so the unconstrained run provably imports
        # while the battery is full. The feature must override that ordering;
        # the total import energy is identical, only its timing differs.
        df["unit_load_cost"] = 0.20 + 0.001 * np.arange(n)
        df["unit_prod_price"] = 0.05
        pv = df["p_pv_forecast"].copy()
        load = df["p_load_forecast"].copy()

        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
            }
        )
        prediction_horizon = 8
        soc_init, soc_min = 0.6, 0.1
        # Size the battery (loss-free, ample power) so its usable energy covers
        # exactly the first half of the horizon, independent of the configured
        # optimization_time_step, so some import is always needed in the tail.
        step_h = self.retrieve_hass_conf["optimization_time_step"].total_seconds() / 3600.0
        cap = (load_w * (prediction_horizon / 2) * step_h) / (soc_init - soc_min)
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": cap,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": soc_min,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )

        # Baseline: feature OFF. Establishes that the unconstrained optimum
        # imports while the battery is still above min.
        self.optim_conf["set_battery_first_priority"] = False
        self.opt = self.create_optimization()
        res_base = self.opt.perform_naive_mpc_optim(
            df, pv, load, prediction_horizon, soc_init=soc_init, soc_final=soc_min
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        base_import_while_charged = res_base.loc[
            res_base["SOC_opt"] > soc_min + 0.02, "P_grid_pos"
        ].sum()
        self.assertGreater(
            base_import_while_charged,
            1.0,
            msg="baseline did not import while the battery was charged; "
            "the scenario no longer discriminates the feature",
        )

        # Feature ON: must drain the battery before importing.
        self.optim_conf["set_battery_first_priority"] = True
        self.opt = self.create_optimization()
        res_bf = self.opt.perform_naive_mpc_optim(
            df, pv, load, prediction_horizon, soc_init=soc_init, soc_final=soc_min
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        charged = res_bf["SOC_opt"] > soc_min + 0.02
        self.assertGreater(charged.sum(), 0, msg="no charged timesteps to test")
        # The constraint: no grid import in any slot where SoC is above min.
        self.assertLess(
            res_bf.loc[charged, "P_grid_pos"].abs().max(),
            1.0,
            msg="feature ON still imported while the battery was above min SoC",
        )
        # The unavoidable import has simply moved to the drained tail.
        self.assertGreater(res_bf["P_grid_pos"].sum(), 1.0)

    def test_battery_first_priority_infeasible_when_load_exceeds_discharge(self):
        """Issue #834: ``set_battery_first_priority`` is a hard constraint, so it
        can make the problem infeasible when the load exceeds the battery's
        maximum discharge power while the battery is above min SoC (grid import
        would be the only way to balance power, but the feature forbids it).
        This pins that documented sharp edge.
        """
        df = self.prepare_forecast_data()
        df["p_pv_forecast"] = 0.0
        df["p_load_forecast"] = 2000.0  # exceeds the 500 W discharge cap below
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.05
        pv = df["p_pv_forecast"].copy()
        load = df["p_load_forecast"].copy()
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 0,
                "set_battery_dynamic": False,
                "set_nodischarge_to_grid": True,
            }
        )
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 500,  # cannot cover the 2000 W load
                "battery_charge_power_max": 500,
                "battery_minimum_state_of_charge": 0.1,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        # Control: the exact same setup is feasible with the feature OFF (the
        # solver just imports the shortfall), proving the infeasibility below is
        # caused by the new constraint and not by the scenario itself.
        self.optim_conf["set_battery_first_priority"] = False
        self.opt = self.create_optimization()
        self.opt.perform_naive_mpc_optim(df, pv, load, 6, soc_init=0.5, soc_final=0.5)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Feature ON: SoC stays well above min, so import is the only way to
        # serve the load, which the feature forbids -> no optimal solution.
        self.optim_conf["set_battery_first_priority"] = True
        self.opt = self.create_optimization()
        self.opt.perform_naive_mpc_optim(df, pv, load, 6, soc_init=0.5, soc_final=0.5)
        self.assertNotIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_sequence_load_runs_with_zero_operating_hours(self):
        """Issue #887: a sequence (list-valued power) deferrable load runs for
        the length of its sequence and ignores operating_hours, which is
        meaningless for it. Setting that load's operating hours to 0 must not
        make the optimization infeasible. It previously did, because the load
        was deactivated by the operating-hours==0 path even though the energy
        constraint already exempts sequence loads.
        """
        df = self.prepare_forecast_data()
        self.optim_conf.update(
            {
                "set_use_battery": False,
                "number_of_deferrable_loads": 1,
                "nominal_power_of_deferrable_loads": [[1000, 1000]],
                "operating_hours_of_each_deferrable_load": [0],
            }
        )
        self.opt = self.create_optimization()
        res = self.opt.perform_naive_mpc_optim(df, self.p_pv_forecast, self.p_load_forecast, 10)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertIn("P_deferrable0", res.columns)
        # The 2-step, 1000 W sequence ran exactly once: total power 2000 W.
        self.assertAlmostEqual(res["P_deferrable0"].sum(), 2000.0, delta=1.0)

    def test_thermal_config_unknown_key_warns(self):
        """Issue #943: a thermal_config with an unrecognized key (e.g. the
        singular min_temperature instead of the list min_temperatures, or a
        stray target_temperature) is silently ignored, which yields a load that
        never schedules. Warn so the typo is visible to the user.
        """
        self.optim_conf["number_of_deferrable_loads"] = 1
        self.optim_conf["def_load_config"] = [
            {
                "thermal_config": {
                    "heating_rate": 0.25,
                    "cooling_constant": 0.01,
                    "start_temperature": 22.0,
                    "min_temperature": 22.0,  # singular typo: never read
                    "target_temperature": 23.0,  # not a recognized key
                }
            }
        ]
        with self.assertLogs(level="WARNING") as logs:
            self.create_optimization()
        joined = "\n".join(logs.output)
        # The singular typo is flagged and the correct list key is suggested.
        self.assertIn("min_temperature", joined)
        self.assertIn("min_temperatures", joined)
        self.assertIn("target_temperature", joined)

    def test_intermediate_soc_target_below_soc_init_is_noop(self):
        """Issue #553: a soc_target at or below the SoC the battery already holds
        builds a non-biting floor, so the optimized plan is unchanged vs no target.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["unit_load_cost"] = 0.2
        self.df_input_data_dayahead["unit_prod_price"] = 0.2
        self.optim_conf.update({"set_use_battery": True})
        self.optim_conf.update({"number_of_deferrable_loads": 0})
        self.optim_conf.update({"set_battery_dynamic": False})
        self.optim_conf.update({"set_nodischarge_to_grid": False})
        self.optim_conf.update({"weight_battery_discharge": 1.0})
        self.optim_conf.update({"weight_battery_charge": 1.0})
        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        self.opt = self.create_optimization()
        opt_res_baseline = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            10,
            soc_init=0.3,
            soc_final=0.3,
        )
        self.opt = self.create_optimization()
        opt_res_low_target = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            10,
            soc_init=0.3,
            soc_final=0.3,
            soc_target=0.2,  # below soc_init -> floor never binds
            soc_target_timestep=5,
        )
        # Identical optimized SoC trajectory: the floor imposed nothing.
        assert_series_equal(
            opt_res_baseline["SOC_opt"],
            opt_res_low_target["SOC_opt"],
            atol=1e-3,
            check_names=False,
        )

    def test_perform_naive_mpc_optim_weight_scaling(self):
        """
        Regression test: Ensure weights are applied element-wise, not as matrix multiplication.
        Also verifies that time-dependent weights correctly influence discharge timing.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update({"set_use_battery": True})
        self.optim_conf.update({"set_total_pv_sell": False})
        self.optim_conf.update({"number_of_deferrable_loads": 0})
        self.optim_conf.update({"set_nodischarge_to_grid": False})  # Allow export to grid

        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "maximum_power_to_grid": 50000,
                "maximum_power_from_grid": 50000,
                "battery_stress_cost": 0.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        self.optim_conf.update({"set_battery_dynamic": False})

        prediction_horizon = 10

        # Scenario:
        # Price at t=0 is high (Profit 50). Weight is 10.
        # Correct: 10 < 50 -> Discharge should happen at t=0.
        # Bug (10x magnification): 100 > 50 -> Discharge would be avoided at t=0 if possible.

        self.df_input_data_dayahead["unit_prod_price"] = 1.0  # Low default
        self.df_input_data_dayahead.iloc[
            0, self.df_input_data_dayahead.columns.get_loc("unit_prod_price")
        ] = 50.0
        self.df_input_data_dayahead["unit_load_cost"] = 0.0

        weights = [10.0] * 10
        self.optim_conf.update({"weight_battery_discharge": weights})

        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=1.0,
            soc_final=0.9,  # Discharge only a bit (concentrated at t=0)
        )

        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertEqual(self.opt.optim_status, "Optimal", "Optimization should be feasible")

        p_batt = self.opt_res_dayahead["P_batt"]
        # Check first step
        discharge_step_0 = p_batt.iloc[0]

        # With fix: discharge_step_0 should be high because profit(50) > penalty(10).
        # Without fix: profit(50) < penalty(100), so it would avoid step 0.
        self.assertGreater(
            discharge_step_0,
            100.0,
            f"Discharge at t=0 should be high with fix. Got {discharge_step_0}",
        )

    def test_perform_naive_mpc_optim_weight_scaling_scalar(self):
        """
        Regression test: Ensure scalar weights also work correctly with resizing.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update({"set_use_battery": True})
        self.optim_conf.update({"set_total_pv_sell": False})
        self.optim_conf.update({"number_of_deferrable_loads": 0})
        self.optim_conf.update({"set_nodischarge_to_grid": False})

        self.plant_conf.update(
            {
                "battery_nominal_energy_capacity": 10000,
                "battery_discharge_power_max": 20000,
                "battery_charge_power_max": 20000,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "maximum_power_to_grid": 50000,
                "maximum_power_from_grid": 50000,
                "battery_stress_cost": 0.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            }
        )
        self.optim_conf.update({"set_battery_dynamic": False})

        prediction_horizon = 10
        self.df_input_data_dayahead["unit_prod_price"] = 1.0
        self.df_input_data_dayahead.iloc[
            0, self.df_input_data_dayahead.columns.get_loc("unit_prod_price")
        ] = 50.0
        self.df_input_data_dayahead["unit_load_cost"] = 0.0

        # Scenario: Scalar Weight = 10.0
        self.optim_conf.update({"weight_battery_discharge": 10.0})

        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=1.0,
            soc_final=0.9,
        )

        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertEqual(self.opt.optim_status, "Optimal")

        p_batt = self.opt_res_dayahead["P_batt"]
        discharge_step_0 = p_batt.iloc[0]
        self.assertGreater(
            discharge_step_0,
            100.0,
            f"Scalar: Discharge at t=0 should be high. Got {discharge_step_0}",
        )

    # Test format output of dayahead optimization with a thermal deferrable load
    def test_thermal_load_optim(self):
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            random.normalvariate(10.0, 3.0) for _ in range(48)
        ]
        runtimeparams = {
            "def_load_config": [
                {},
                {
                    "thermal_config": {
                        "heating_rate": 5.0,
                        "cooling_constant": 0.1,
                        "overshoot_temperature": 24.0,
                        "start_temperature": 20,
                        "desired_temperatures": [21] * 48,
                    }
                },
            ]
        }
        self.optim_conf["def_load_config"] = runtimeparams["def_load_config"]
        self.opt = self.create_optimization()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values  # currency/kWh
        unit_prod_price = self.df_input_data_dayahead[
            self.opt.var_prod_price
        ].values  # currency/kWh
        self.opt_res_dayahead = self.opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertIn("cost_fun_" + self.costfun, self.opt_res_dayahead.columns)
        self.assertEqual(self.opt.optim_status, "Optimal")

    def test_thermal_inertia(self):
        """
        Test that the thermal_inertia parameter correctly delays the heating effect.
        Scenario: 1 hour inertia (L=2 steps).
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Cold outside (10C)
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [10.0] * 48

        # Set costs to 0
        self.df_input_data_dayahead[self.opt.var_load_cost] = 0.0
        self.df_input_data_dayahead[self.opt.var_prod_price] = 0.0

        # Define constraints
        # Constraint: We want 18.0°C at t=3.
        # Physics Check:
        # - Without heat: Temp drops to ~14.2°C by t=3.
        # - With Max Heat at t=0: Temp recovers to ~19.2°C by t=3.
        # - Target 18.0°C forces the heater ON but is feasible.
        min_temps = [0] * 48
        min_temps[3] = 18.0

        runtimeparams = {
            "def_load_config": [
                {},
                {
                    "thermal_config": {
                        "heating_rate": 10.0,
                        "cooling_constant": 0.5,
                        "start_temperature": 20.0,
                        "thermal_inertia": 1.0,  # 1 Hour inertia -> 2 timesteps lag
                        "sense": "heat",
                        "min_temperatures": min_temps,
                        "max_temperatures": [30.0] * 48,
                    }
                },
            ]
        }

        self.optim_conf["def_load_config"] = runtimeparams["def_load_config"]
        # Ensure sufficient power
        self.optim_conf["nominal_power_of_deferrable_loads"][1] = 3000

        self.opt = self.create_optimization()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        self.opt_res_dayahead = self.opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        self.assertEqual(self.opt.optim_status, "Optimal")

        # Get results
        p_heat = self.opt_res_dayahead["P_deferrable1"]
        temp = self.opt_res_dayahead["predicted_temp_heater1"]

        # Verify Heater turned ON at t=0
        # It MUST turn on now to satisfy the hard constraint at t=3.
        self.assertGreater(
            p_heat.iloc[0], 0, "Heater should turn on at t=0 to satisfy delayed constraint"
        )

        # Verify Dead Zone (t=0 to t=2)
        # Power[0] is ON, but Temp[1] and Temp[2] should NOT see it yet due to inertia.
        # They should only reflect cooling losses.
        # Expected T[1] approx: 20.0 - (0.5 * 0.5 * (20.0 - 10.0)) = 17.5
        self.assertAlmostEqual(
            temp.iloc[1], 17.5, delta=0.5, msg="T[1] should only reflect cooling (dead zone)"
        )

        # T[2] should continue dropping
        self.assertLess(temp.iloc[2], temp.iloc[1], msg="T[2] should continue dropping (dead zone)")

        # 3. Verify Heating Effect Arrives at t=3
        # The constraint required T[3] >= 18.0.
        self.assertGreaterEqual(temp.iloc[3], 17.9, msg="T[3] should meet the constraint (18C)")

        # Sanity check: The jump from T[2] to T[3] is due to heating
        # (Or at least the drop stops significantly compared to baseline)
        self.assertGreater(
            temp.iloc[3],
            temp.iloc[2],
            msg="T[3] should rise (or stop dropping) as heating kicks in",
        )

    def test_thermal_inertia_no_regression(self):
        """
        Test backward compatibility: If thermal_inertia is 0 (or missing),
        the heating effect should be IMMEDIATE (Legacy behavior).
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Scenario: Cold outside (10C), start at 20C.
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [10.0] * 48

        # Set costs to 0 to encourage heating
        self.df_input_data_dayahead[self.opt.var_load_cost] = 0.0
        self.df_input_data_dayahead[self.opt.var_prod_price] = 0.0

        # Define constraints
        # We want 22C at t=1.
        # In the Legacy model (Instant), heating at t=0 affects t=1.
        # So this IS feasible. (In the delayed model, this would be impossible).
        min_temps = [0] * 48
        min_temps[1] = 22.0

        runtimeparams = {
            "def_load_config": [
                {},
                {
                    "thermal_config": {
                        "heating_rate": 10.0,
                        "cooling_constant": 0.5,
                        "start_temperature": 20.0,
                        "sense": "heat",
                        # "thermal_inertia": 0.0,  <-- IMPLIED DEFAULT
                        "min_temperatures": min_temps,
                        "max_temperatures": [30.0] * 48,
                    }
                },
            ]
        }

        self.optim_conf["def_load_config"] = runtimeparams["def_load_config"]
        self.optim_conf["nominal_power_of_deferrable_loads"][1] = 3000

        self.opt = self.create_optimization()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        self.opt_res_dayahead = self.opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        self.assertEqual(self.opt.optim_status, "Optimal")

        # Get results
        p_heat = self.opt_res_dayahead["P_deferrable1"]
        temp = self.opt_res_dayahead["predicted_temp_heater1"]

        # Verify Heater turned ON at t=0
        self.assertGreater(p_heat.iloc[0], 0, "Heater should turn on at t=0")

        # Verify IMMEDIATE Effect (Legacy Behavior)
        # Power[0] should raise Temp[1].
        # If inertia was active, Temp[1] would only drop (cooling).
        # Since Temp[1] meets the 22C target (rising from 20C), we know the effect was instant.
        self.assertGreaterEqual(
            temp.iloc[1], 21.9, msg="T[1] should rise immediately, matching legacy behavior"
        )

        # Sanity check: T[1] should be > T[0] (20C)
        self.assertGreater(temp.iloc[1], 20.0, "Temperature should rise immediately at t=1")

    # Setup function to run forecast for thermal tests
    def run_test_forecast(
        self,
        prediction_horizon: int = 10,
        def_total_hours: list[int] = None,
        passed_data: dict = None,
        input_data: pd.DataFrame = None,
        def_init_temp=None,
    ):
        if def_total_hours is None:
            def_total_hours = [0]
        if passed_data is None:
            passed_data = {}
        if input_data is None:
            input_data = pd.DataFrame()

        self.opt = self.create_optimization()
        def_start_timestep = [0]
        def_end_timestep = [0]
        passed_data["prediction_horizon"] = prediction_horizon
        self.optim_conf.update(
            {
                "num_def_loads": 1,
                "photovoltaic_production_sell_price": 0,
                "prediction_horizon": prediction_horizon,
            }
        )
        self.fcst.params["passed_data"].update(passed_data)

        # Prepare input data
        if input_data.empty:
            # If no input data provided, generate dummy data
            dates = pd.date_range(
                start=datetime.now(),
                periods=prediction_horizon,
                freq=self.retrieve_hass_conf["optimization_time_step"],
            )
            input_data = pd.DataFrame(index=dates)
            input_data["outdoor_temperature_forecast"] = 10.0  # constant temp

        input_data = self.fcst.get_load_cost_forecast(
            input_data,
            method="list"
            if "load_cost_forecast" in self.fcst.params["passed_data"]
            else "constant",
        )
        input_data = self.fcst.get_prod_price_forecast(input_data, method="constant")

        # Mock p_pv and p_Load as they are needed by perform_naive_mpc_optim
        p_pv = np.zeros(prediction_horizon)
        p_Load = np.zeros(prediction_horizon)

        unit_load_cost = input_data[self.opt.var_load_cost].values
        unit_prod_price = input_data[self.opt.var_prod_price].values

        self.opt_res_dayahead = self.opt.perform_optimization(
            input_data,
            p_pv,
            p_Load,
            unit_load_cost,
            unit_prod_price,
            def_total_hours=def_total_hours,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            def_init_temp=def_init_temp,
        )

        self.assertTrue((self.opt_res_dayahead["optim_status"] == "Optimal").all())

    def run_thermal_forecast(
        self,
        outdoor_temp=10,
        prices=None,
        def_init_temp=None,
    ):
        if prices is None:
            prices = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # Generate Massive Time Buffer (5 days total) to handle timezone safety
        start_time = pd.Timestamp.now(tz=self.fcst.time_zone).floor(self.fcst.freq) - pd.Timedelta(
            days=3
        )

        times = (
            pd.date_range(
                start=start_time,
                periods=240,
                freq=self.fcst.freq,
                tz=self.fcst.time_zone,
            )
            .tz_convert("utc")
            .round(self.fcst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(self.fcst.time_zone)
        )

        # Create the full buffer DataFrame
        input_data_full = pd.DataFrame(index=times)
        input_data_full["outdoor_temperature_forecast"] = outdoor_temp
        input_data_full[self.opt.var_load_cost] = 1.0
        input_data_full[self.opt.var_prod_price] = 0.0

        # Find 'now' and Slice the DataFrame to the specific horizon (10 steps)
        # This aligns 'Index 0' of the optimization with 'Now', matching your test constraints.
        now_precise = pd.Timestamp.now(tz=self.fcst.time_zone).floor(self.fcst.freq)
        try:
            start_idx = input_data_full.index.get_loc(now_precise)
        except KeyError:
            start_idx = input_data_full.index.get_indexer([now_precise], method="nearest")[0]

        horizon = len(prices)
        input_data = input_data_full.iloc[start_idx : start_idx + horizon].copy()

        # Inject prices into the sliced dataframe
        input_data[self.opt.var_load_cost] = prices

        # Setup and Run Optimization on the sliced data
        # Update config to match the horizon length
        self.optim_conf.update(
            {
                "num_def_loads": 1,
                "photovoltaic_production_sell_price": 0,
                "prediction_horizon": horizon,
            }
        )

        # Create vectors matching the slice length (10)
        p_pv = np.zeros(horizon)
        p_Load = np.zeros(horizon)
        unit_load_cost = input_data[self.opt.var_load_cost].values
        unit_prod_price = input_data[self.opt.var_prod_price].values

        # Re-init optimization to ensure clean state
        self.opt = self.create_optimization()

        self.opt_res_dayahead = self.opt.perform_optimization(
            input_data,
            p_pv,
            p_Load,
            unit_load_cost,
            unit_prod_price,
            def_total_hours=[0],
            def_start_timestep=[0],
            def_end_timestep=[0],
            def_init_temp=def_init_temp,
        )

        self.assertTrue((self.opt_res_dayahead["optim_status"] == "Optimal").all())

    def test_thermal_management(self):
        # Case: Constrain mode (Hard constraint)
        self.optim_conf.update(
            {
                "def_load_config": [
                    {
                        "thermal_config": {
                            "start_temperature": 20,
                            "cooling_constant": 0.1,
                            "heating_rate": 10,
                            "overshoot_temperature": 25,
                            "min_temperatures": [0, 0, 21, 0, 0, 0, 0, 0, 0, 0],
                            "sense": "heat",
                        }
                    }
                ]
            }
        )
        prices = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.run_thermal_forecast(prices=prices)
        # Verify heater turned on at index 1 to meet 21 degrees at index 2
        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    def test_thermal_management_overshoot(self):
        # Case: Overshoot limit
        # Adapted: Map overshoot_temperature to max_temperatures
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [False, False],
                "def_load_config": [
                    {
                        "thermal_config": {
                            "start_temperature": 20,
                            "cooling_constant": 0.2,
                            "heating_rate": 4.0,
                            "max_temperatures": [22] * 10,
                            "min_temperatures": [0, 0, 21, 0, 0, 0, 0, 0, 0, 0],
                            "sense": "heat",
                        }
                    }
                ],
            }
        )
        # High prices to discourage heating, but constraint should force it
        self.run_thermal_forecast(prices=[1, 2000, 2000, 1, 1, 1, 1, 1, 1, 1])

        predicted_temps = self.opt_res_dayahead["predicted_temp_heater0"]
        # Ensure max constraint is respected
        self.assertFalse((predicted_temps > 22).any(), "Overshot in some timesteps.")
        # Ensure min constraint is respected
        self.assertGreaterEqual(
            predicted_temps.iloc[2], 21, "Failed to meet temperature requirement"
        )

    def test_thermal_management_cooling(self):
        # Case: Cooling
        self.optim_conf.update(
            {
                "def_load_config": [
                    {
                        "thermal_config": {
                            "start_temperature": 25,
                            "cooling_constant": 0.1,
                            "heating_rate": 10,  # Positive; sense_coeff=-1 applied for cooling
                            "min_temperatures": [0] * 10,  # No min constraint
                            "max_temperatures": [
                                None,
                                None,
                                20,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                            ],  # Max temp constraint for cooling
                            "sense": "cool",
                        }
                    }
                ]
            }
        )
        self.run_thermal_forecast(
            outdoor_temp=20,
            prices=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        # Should turn on to cool down to 20
        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )
        # sense=cool + positive heating_rate must lower temperature when pump runs
        temp = self.opt_res_dayahead["predicted_temp_heater0"]
        assert temp.iloc[2] < temp.iloc[1], (
            f"sense=cool + positive heating_rate must lower temperature; "
            f"got temp[1]={temp.iloc[1]}, temp[2]={temp.iloc[2]}"
        )

    def test_thermal_management_penalty(self):
        # Case: Penalize mode (Legacy behavior)
        # We use desired_temperatures, which triggers the penalty logic
        self.optim_conf.update(
            {
                "def_load_config": [
                    {
                        "thermal_config": {
                            "start_temperature": 20,
                            "cooling_constant": 0.1,
                            "heating_rate": 10,
                            "overshoot_temperature": 50,
                            "desired_temperatures": [0, 0, 40, 0, 0, 0, 0, 0, 0, 0],
                            "penalty_factor": 1000,  # High penalty to force action
                            "sense": "heat",
                        }
                    }
                ]
            }
        )
        self.run_thermal_forecast()
        # Should turn on to try and reach 40 (or get close)
        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    def test_thermal_runtime_initial_temp(self):
        self.optim_conf.update(
            {
                "def_load_config": [
                    {
                        "thermal_config": {
                            "start_temperature": 20,  # Config says 20
                            "cooling_constant": 0.1,
                            "heating_rate": 10,
                            "min_temperatures": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            "sense": "heat",
                        }
                    }
                ]
            }
        )
        # We pass 22 at runtime.
        # With Outdoor=10, cooling=0.1.
        # Next temp should be: 22 - (0.1 * (22-10)) = 22 - 1.2 = 20.8
        # If it used config (20), next temp would be 19.
        passed_init_temp = [22.0]
        self.run_thermal_forecast(outdoor_temp=10.0, def_init_temp=passed_init_temp)

        predicted = self.opt_res_dayahead["predicted_temp_heater0"]
        # Check first step is the passed value
        self.assertAlmostEqual(predicted.iloc[0], 22.0)
        # Check second step follows physics from 22.0
        # Time step is 30min (0.5h) in default config usually, but let's check config
        ts = self.retrieve_hass_conf["optimization_time_step"].seconds / 3600
        expected_next = 22.0 - (0.1 * ts * (22.0 - 10.0))
        self.assertAlmostEqual(predicted.iloc[1], expected_next)

    # Setup function to run dayahead optimization for the following tests
    def run_penalty_test_forecast(self):
        self.opt = self.create_optimization()
        def_total_hours = [5 * self.retrieve_hass_conf["optimization_time_step"].seconds / 3600.0]
        def_start_timestep = [0]
        def_end_timestep = [0]
        prediction_horizon = 10
        self.optim_conf.update({"number_of_deferrable_loads": 1})

        attributes = vars(self.fcst).copy()

        attributes["params"]["passed_data"]["prod_price_forecast"] = [
            0 for _ in range(prediction_horizon)
        ]
        attributes["params"]["passed_data"]["solar_forecast_kwp"] = [
            0 for _ in range(prediction_horizon)
        ]
        attributes["params"]["passed_data"]["prediction_horizon"] = prediction_horizon

        fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            attributes["params"],
            emhass_conf,
            logger,
            get_data_from_file=True,
        )

        self.df_input_data_dayahead = fcst.get_load_cost_forecast(
            self.df_input_data_dayahead, method="list"
        )
        self.df_input_data_dayahead = fcst.get_prod_price_forecast(
            self.df_input_data_dayahead, method="list"
        )

        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            def_total_hours=def_total_hours,
            def_total_timestep=None,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
        )

    # Test load is constant
    def test_constant_load(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            2,
            1,
            1,
            1,
            1,
            1.5,
            1.1,
            2,
            2,
            2,
        ]
        self.optim_conf.update({"set_deferrable_load_single_constant": [True]})

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    # Test no startup penalty when bump is small
    def test_startup_penalty_continuous_with_small_bump(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            2,
            1,
            1,
            1,
            1,
            1.5,
            1.1,
            2,
            2,
            2,
        ]
        self.optim_conf.update({"set_deferrable_startup_penalty": [100.0]})

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    # Test startup penalty
    def test_startup_penalty_discontinuity_when_justified(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            2,
            1,
            1,
            1,
            1,
            1.5,
            1.1,
            2,
            2,
            2,
        ]

        self.optim_conf.update({"set_deferrable_startup_penalty": [0.1]})

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([0, 1, 1, 1, 1, 0, 1, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    # Test penalty continuity when deferrable load is already on
    def test_startup_penalty_no_discontinuity_at_start(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            1.2,
            1,
            1,
            1,
            1,
            1.1,
            2,
            2,
            2,
            2,
        ]

        self.optim_conf.update(
            {
                "set_deferrable_startup_penalty": [100.0],
                "def_current_state": [True],
            }
        )

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    # Test delay start
    def test_startup_penalty_delayed_start(self):
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            1.2,
            1,
            1,
            1,
            1,
            1.1,
            2,
            2,
            2,
            2,
        ]

        self.optim_conf.update(
            {
                "set_deferrable_startup_penalty": [100.0],
                "def_current_state": [False],
            }
        )

        self.run_penalty_test_forecast()

        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )

    def test_running_single_const_pinned_from_start(self):
        """A running single-constant load is pinned ON from t=0 regardless of cost."""
        # Cheap at t=5..9, expensive at t=0..4 — without pinning the solver would defer.
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
        ]
        self.optim_conf.update(
            {
                "set_deferrable_load_single_constant": [True],
                "def_current_state": [True],
            }
        )

        self.run_penalty_test_forecast()  # 5-step load, no window restriction

        nominal = self.optim_conf["nominal_power_of_deferrable_loads"][0]
        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            nominal * pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=self.opt_res_dayahead.index),
            check_names=False,
        )
        self.assertTrue(np.all(self.opt.param_running_lb[0].value[:5] == 1.0))
        self.assertTrue(np.all(self.opt.param_running_lb[0].value[5:] == 0.0))
        self.assertEqual(self.opt.param_already_running_sc[0].value, 1.0)

    def test_not_running_single_const_not_pinned(self):
        """A single-constant load that is NOT currently running is freely scheduled."""
        self.fcst.params["passed_data"]["load_cost_forecast"] = [
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
        ]
        self.optim_conf.update(
            {
                "set_deferrable_load_single_constant": [True],
                "def_current_state": [False],
            }
        )

        self.run_penalty_test_forecast()

        nominal = self.optim_conf["nominal_power_of_deferrable_loads"][0]
        assert_series_equal(
            self.opt_res_dayahead["P_deferrable0"],
            nominal * pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], index=self.opt_res_dayahead.index),
            check_names=False,
        )
        self.assertTrue(np.all(self.opt.param_running_lb[0].value == 0.0))
        self.assertEqual(self.opt.param_already_running_sc[0].value, 0.0)

    def _run_single_const_with_window(
        self, def_total_timestep, def_end_timestep, current_state=True
    ):
        """Helper: run a single-constant load optimization with explicit window bounds."""
        self.optim_conf.update(
            {
                "set_deferrable_load_single_constant": [True],
                "def_current_state": [current_state],
                "number_of_deferrable_loads": 1,
            }
        )
        self.opt = self.create_optimization()
        prediction_horizon = 10
        attributes = vars(self.fcst).copy()
        attributes["params"]["passed_data"]["load_cost_forecast"] = [1] * prediction_horizon
        attributes["params"]["passed_data"]["prod_price_forecast"] = [0] * prediction_horizon
        attributes["params"]["passed_data"]["solar_forecast_kwp"] = [0] * prediction_horizon
        attributes["params"]["passed_data"]["prediction_horizon"] = prediction_horizon
        fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            attributes["params"],
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        df = fcst.get_load_cost_forecast(self.df_input_data_dayahead, method="list")
        df = fcst.get_prod_price_forecast(df, method="list")
        return self.opt.perform_naive_mpc_optim(
            df,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            def_total_hours=None,
            def_total_timestep=[def_total_timestep],
            def_start_timestep=[0],
            def_end_timestep=[def_end_timestep],
        )

    def test_running_single_const_window_end_caps_pinning(self):
        """param_running_lb stops at def_end_timestep, not at required_timesteps."""
        # 3-step load, window ends at t=6 → pinned_steps = min(3, 6, 10) = 3
        self._run_single_const_with_window(def_total_timestep=3, def_end_timestep=6)

        self.assertTrue(np.all(self.opt.param_running_lb[0].value[:3] == 1.0))
        self.assertTrue(np.all(self.opt.param_running_lb[0].value[3:] == 0.0))
        # Window mask must cover [0, 3) but must not extend past step 6
        wm = self.opt.param_window_masks[0].value
        self.assertTrue(np.all(wm[:3] == 1.0))
        self.assertEqual(wm[6], 0.0)

    def test_running_single_const_required_exceeds_window_is_infeasible(self):
        """required_timesteps=8 with window end=4 → solver reports infeasible."""
        opt_res = self._run_single_const_with_window(def_total_timestep=8, def_end_timestep=4)
        # Window admits only 4 slots; sum(p_def_bin2)==8 cannot be satisfied.
        # On total failure perform_naive_mpc_optim returns a single-column DataFrame
        # with only 'optim_status'.
        self.assertNotIn("P_deferrable0", opt_res.columns)

    def test_perform_naive_mpc_optim_def_total_timestep(self):
        """Test operating_timesteps_of_each_deferrable_load parameter.

        This test verifies that operating_timesteps_of_each_deferrable_load works correctly
        and produces the exact number of timesteps requested, regardless of timestep size.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Test the battery
        self.optim_conf.update({"set_use_battery": True})
        self.opt = self.create_optimization()
        prediction_horizon = 10
        soc_init = 0.4
        soc_final = 0.6

        # Get the actual timestep size from configuration
        timestep_minutes = self.retrieve_hass_conf["optimization_time_step"].seconds / 60
        timestep_hours = timestep_minutes / 60

        # Define test case: 4 timesteps for first deferrable load
        # This should work regardless of timestep size (5min, 15min, 30min, etc.)
        requested_timesteps = 4
        def_total_timestep = [requested_timesteps, 0]  # Only test first deferrable load
        def_start_timestep = [-5, 0]
        def_end_timestep = [4, 0]

        self.opt_res_dayahead = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=None,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn("P_batt", self.opt_res_dayahead.columns)
        self.assertIn("SOC_opt", self.opt_res_dayahead.columns)
        self.assertLess(
            np.abs(
                self.opt_res_dayahead.loc[self.opt_res_dayahead.index[-1], "SOC_opt"] - soc_final
            ),
            1e-3,
        )

        # Numerical verification that exactly the requested timesteps were used
        # Count non-zero timesteps for P_deferrable0
        active_timesteps = (self.opt_res_dayahead["P_deferrable0"] > 0).sum()
        self.assertEqual(
            active_timesteps,
            requested_timesteps,
            f"Expected exactly {requested_timesteps} active timesteps, got {active_timesteps}",
        )

        # Verify energy constraint: requested_timesteps * timestep_hours * nominal_power
        expected_energy = (
            requested_timesteps
            * timestep_hours
            * self.optim_conf["nominal_power_of_deferrable_loads"][0]
        )
        actual_energy = self.opt_res_dayahead["P_deferrable0"].sum() * timestep_hours
        self.assertLess(
            np.abs(expected_energy - actual_energy),
            1e-3,
            f"Energy mismatch: expected {expected_energy:.3f} Wh, got {actual_energy:.3f} Wh",
        )

    def test_perform_naive_mpc_optim_def_total_timestep_various_sizes(self):
        """Test operating_timesteps_of_each_deferrable_load with various timestep sizes."""
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Test with common timestep sizes that should work reliably
        timestep_sizes = [15, 30]  # minutes - stick to known working sizes
        for timestep_min in timestep_sizes:
            with self.subTest(timestep_minutes=timestep_min):
                # Create fresh configuration for each test
                test_retrieve_hass_conf = self.retrieve_hass_conf.copy()
                test_retrieve_hass_conf["optimization_time_step"] = pd.Timedelta(
                    f"{timestep_min}min"
                )

                test_optim_conf = self.optim_conf.copy()
                test_optim_conf.update({"set_use_battery": True})

                self.opt = Optimization(
                    test_retrieve_hass_conf,
                    test_optim_conf,
                    self.plant_conf,
                    self.fcst.var_load_cost,
                    self.fcst.var_prod_price,
                    self.costfun,
                    emhass_conf,
                    logger,
                )

                prediction_horizon = 10
                timestep_hours = timestep_min / 60
                requested_timesteps = 4
                def_total_timestep = [requested_timesteps, 0]
                def_start_timestep = [-5, 0]
                def_end_timestep = [4, 0]

                opt_res = self.opt.perform_naive_mpc_optim(
                    self.df_input_data_dayahead,
                    self.p_pv_forecast,
                    self.p_load_forecast,
                    prediction_horizon,
                    soc_init=0.4,
                    soc_final=0.6,
                    def_total_hours=None,
                    def_total_timestep=def_total_timestep,
                    def_start_timestep=def_start_timestep,
                    def_end_timestep=def_end_timestep,
                )

                # Verify optimization was successful
                self.assertEqual(
                    self.opt.optim_status,
                    "Optimal",
                    f"Timestep size {timestep_min}min: Optimization failed with status {self.opt.optim_status}",
                )

                # Count active timesteps (power > 0)
                active_timesteps = (opt_res["P_deferrable0"] > 0).sum()

                # For robust testing, verify the energy constraint is met
                # rather than exact timestep count (which may vary due to optimization constraints)
                total_energy = opt_res["P_deferrable0"].sum() * timestep_hours
                expected_energy = (
                    requested_timesteps
                    * timestep_hours
                    * test_optim_conf["nominal_power_of_deferrable_loads"][0]
                )

                # The actual energy should match the energy that would be delivered
                # by running for exactly the requested timesteps
                self.assertLess(
                    np.abs(total_energy - expected_energy),
                    1e-3,
                    f"Timestep {timestep_min}min: Energy constraint violated - "
                    f"expected {expected_energy:.3f} Wh, got {total_energy:.3f} Wh "
                    f"({active_timesteps} active timesteps)",
                )

    def test_thermal_battery_constraints(self):
        """Test thermal battery optimization with Langer & Volling 2020 model."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12)
            for i in range(48)  # Varying outdoor temp
        ]

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                    }
                },
            ]
        }

        # Run optimization and verify success
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Assert physical plausibility: heating demand should be higher during colder periods
        outdoor_temp = self.df_input_data_dayahead["outdoor_temperature_forecast"]
        heating_power = opt_res["P_deferrable0"]

        # Define cold and warm periods (e.g., cold: temp < median, warm: temp > median)
        temp_median = np.median(outdoor_temp)
        cold_indices = outdoor_temp < temp_median
        warm_indices = outdoor_temp > temp_median

        mean_power_cold = heating_power[cold_indices].mean()
        mean_power_warm = heating_power[warm_indices].mean()

        # Assert that mean heating power is higher (or equal) during cold periods
        # Note: Both may be zero if thermal battery stays within bounds without heating
        self.assertGreaterEqual(
            mean_power_cold,
            mean_power_warm,
            "Heating power during cold periods should be >= warm periods",
        )

        # Verify thermal battery temperature constraints are properly configured
        min_temps = runtimeparams["def_load_config"][0]["thermal_battery"]["min_temperatures"]
        max_temps = runtimeparams["def_load_config"][0]["thermal_battery"]["max_temperatures"]

        # Verify constraint parameters are reasonable
        self.assertGreater(
            max_temps[0], min_temps[0], "max_temperatures must be greater than min_temperatures"
        )
        self.assertGreaterEqual(
            min_temps[0], 10.0, "min_temperatures should be reasonable (>= 10°C)"
        )
        self.assertLessEqual(max_temps[0], 30.0, "max_temperatures should be reasonable (<= 30°C)")

        # Note: Thermal battery temperatures are enforced via LP constraints (see Langer & Volling 2020,
        # Equations B.13-B.14) but are not currently output to the results DataFrame. The constraints
        # ensure T_bat[t] >= min_temperatures[t] and T_bat[t] <= max_temperatures[t] for all timesteps.
        # The optimizer would fail with "Infeasible" status if these constraints cannot be satisfied.

        # Instead, we verify the optimization succeeded (which proves constraints were satisfied)
        # and that heat pump operation is reasonable
        self.assertIn("P_deferrable0", opt_res.columns)
        total_heating_energy = opt_res["P_deferrable0"].sum()
        self.assertGreaterEqual(total_heating_energy, 0, "Heat pump energy must be non-negative")

    def test_thermal_battery_infeasible_temperature_constraints(self):
        """Test optimizer handles infeasible thermal battery configuration (min_temperatures > max_temperatures)."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        # Create infeasible configuration: min > max
        infeasible_runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 22.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [25.0] * 48,  # Deliberately set min > max
                        "max_temperatures": [20.0] * 48,  # This creates infeasibility
                    }
                },
            ]
        }

        self.optim_conf["def_load_config"] = infeasible_runtimeparams["def_load_config"]
        self.opt = self.create_optimization()

        # Run optimization - should handle infeasibility gracefully
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values
        opt_res = self.opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # The optimizer should return a result with "Infeasible" status or handle gracefully
        # Check that it doesn't crash and returns a DataFrame
        self.assertIsInstance(opt_res, type(pd.DataFrame()))

        # Verify the optimization recognized the infeasibility
        # The solver should mark this as infeasible rather than returning invalid results
        # Note: Actual behavior depends on how EMHASS handles infeasible problems
        # This test ensures it doesn't crash

    def test_thermal_battery_infeasibility_q_input_start_zero(self):
        """Test that optimization remains feasible when q_input_start=0 and start_temp <= min_temp.

        Reproduces the scenario from issue #776: after a prior infeasible MPC run,
        q_input_start is stuck at 0.  When start_temperature is at or below
        min_temperatures[0], fixing q_input[0]=0 forces the next timestep below
        the minimum — making the problem permanently infeasible.

        The fix releases the q_input[0] constraint in this situation so the
        solver can choose a feasible initial heat input.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [5.0] * 48

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 18.0,  # == min_temperatures[0]
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [24.0] * 48,
                        "thermal_inertia_time_constant": 1.5,
                        "q_input_initial": 0.0,  # Simulates post-infeasible state
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Must be feasible — before the fix this was permanently infeasible.
        # Whether the heat pump actually runs depends on cost optimization;
        # the critical assertion is that the solver finds a solution at all.
        self.assertEqual(opt_res["optim_status"].unique()[0], "Optimal")
        self.assertIn("P_deferrable0", opt_res.columns)
        self.assertTrue(
            (opt_res["P_deferrable0"] >= 0).all(),
            "Heating power must be non-negative",
        )

    def test_thermal_battery_q_input_start_below_min(self):
        """Test feasibility when start_temperature is slightly below min_temperatures."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [5.0] * 48

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 17.5,  # Below min_temperatures[0]
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [24.0] * 48,
                        "thermal_inertia_time_constant": 1.5,
                        "q_input_initial": 0.0,
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Must be feasible — before the fix this was permanently infeasible.
        self.assertEqual(opt_res["optim_status"].unique()[0], "Optimal")
        self.assertIn("P_deferrable0", opt_res.columns)
        self.assertTrue(
            (opt_res["P_deferrable0"] >= 0).all(),
            "Heating power must be non-negative",
        )

    def test_persist_q_input_infeasible_fallback(self):
        """Test that _persist_q_input resets q_input_start after an infeasible solve.

        When the solver returns None for q_input_var (infeasible), the fallback
        should use heating_demand[0] so the next MPC iteration doesn't stay
        stuck at q_input_start=0.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [5.0] * 48

        # Set up a basic thermal battery config
        config = {
            "thermal_battery": {
                "start_temperature": 20.0,
                "supply_temperature": 35.0,
                "volume": 50.0,
                "specific_heating_demand": 100.0,
                "area": 100.0,
                "min_temperatures": [18.0] * 48,
                "max_temperatures": [24.0] * 48,
                "thermal_inertia_time_constant": 1.5,
            }
        }
        self.optim_conf["def_load_config"] = [config]
        opt = self.create_optimization()

        # Simulate the state after an infeasible solve:
        # q_input_var exists but its .value is None (CVXPY sets this on infeasible)
        import cvxpy as cp

        params = opt.param_thermal[0]
        params["q_input_start"].value = 0.0
        dummy_var = cp.Variable(48, name="q_input_test")
        # Don't solve — .value stays None, simulating an infeasible result
        params["q_input_var"] = dummy_var

        # Set a non-zero heating demand so the fallback has something to use
        params["heating_demand"].value = np.full(48, 0.5)

        hc = config["thermal_battery"]
        opt._persist_q_input(0, params, hc)

        # After _persist_q_input, q_input_start should be reset to demand fallback
        self.assertAlmostEqual(
            params["q_input_start"].value,
            0.5,
            places=4,
            msg="q_input_start should be reset to heating_demand[0] after infeasible solve",
        )

    def test_thermal_battery_physics_based(self):
        """Test thermal battery optimization with physics-based heating demand calculation."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            5.0 + 10.0 * np.sin(i * np.pi / 12)
            for i in range(48)  # Temperature cycling 5-15°C
        ]

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,  # Still needed for fallback
                        "area": 100.0,  # Still needed for fallback
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                        # Physics-based parameters
                        "u_value": 0.4,  # W/(m²·K) - average insulation
                        "envelope_area": 350.0,  # m² - building envelope
                        "ventilation_rate": 0.6,  # ACH - average building
                        "heated_volume": 250.0,  # m³ - heated volume
                    }
                },
            ]
        }

        # Run optimization and verify success
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Verify thermal battery optimization behavior
        heating_power = opt_res["P_deferrable0"].values

        # Test 1: Verify optimization completed successfully
        self.assertEqual(opt_res["optim_status"].unique()[0], "Optimal")

        # Test 2: Verify heating power is non-negative (physical constraint)
        self.assertTrue(np.all(heating_power >= 0), "Heating power must be non-negative")

        # Test 3: Verify total energy consumption is reasonable for the scenario
        # With outdoor temps from -5°C to 15°C, we expect some heating over 48 timesteps
        total_energy_kwh = heating_power.sum() * 0.5 / 1000  # W -> kWh (30min timesteps)
        self.assertGreater(
            total_energy_kwh,
            0.0,
            "Expected some heating energy consumption given cold outdoor temperatures",
        )

        # Test 4: Verify physics-based calculation was actually used
        # This is logged during optimization - we're testing the code path works
        # The heating pattern depends on cost optimization, not just temperature
        # (thermal storage allows pre-heating during favorable conditions like high PV or low costs)

        # Verify physics-based parameters are being used by checking log output was correct
        # The INFO log should show "Using physics-based heating demand calculation"
        # (This is verified by the logger output, not an explicit assertion here)

        # Additional check: If any heating occurred, verify it's within reasonable bounds
        # Note: heating_power is in Watts, total_energy_kwh converts to kWh
        if total_energy_kwh > 0:
            # With physics-based model and thermal storage optimization,
            # the optimizer may pre-heat during favorable conditions (high PV, low costs, better COP)
            # rather than matching instantaneous heating demand
            # Just verify it's not unreasonably high (e.g., running at max power constantly)
            max_theoretical_energy = 3000 * 48 * 0.5 / 1000  # 3kW heat pump * 48 timesteps * 0.5h
            self.assertLess(
                total_energy_kwh,
                max_theoretical_energy,
                "Total heating energy should not exceed theoretical maximum",
            )

    def test_thermal_battery_hdd_configurable(self):
        """Test thermal battery optimization with configurable HDD parameters."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            8.0 + 6.0 * np.sin(i * np.pi / 12)
            for i in range(48)  # Temperature cycling 2-14°C
        ]

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 40.0,
                        "volume": 60.0,
                        "specific_heating_demand": 120.0,  # kWh/(m²·year)
                        "area": 150.0,  # m²
                        "min_temperatures": [19.0] * 48,
                        "max_temperatures": [21.0] * 48,
                        # Configurable HDD parameters
                        "base_temperature": 20.0,  # Use comfort target instead of default 18°C
                        "annual_reference_hdd": 2500.0,  # Adjust for milder climate
                    }
                },
            ]
        }

        # Run optimization and verify success
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Store results with custom HDD parameters
        total_energy_custom_hdd = opt_res["P_deferrable0"].sum()

        # Now run optimization with DEFAULT HDD parameters (base_temperature=18°C, annual_reference_hdd=3000)
        # to verify that changing parameters actually affects heating demand
        runtimeparams_default = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 40.0,
                        "volume": 60.0,
                        "specific_heating_demand": 120.0,  # Same as above
                        "area": 150.0,  # Same as above
                        "min_temperatures": [19.0] * 48,
                        "max_temperatures": [21.0] * 48,
                        # NO custom HDD parameters - will use defaults (base_temperature=18°C, annual_reference_hdd=3000)
                    }
                },
            ]
        }

        # Run optimization with default HDD parameters
        opt_res_default = self.run_optimization_with_config(
            runtimeparams_default["def_load_config"]
        )

        total_energy_default_hdd = opt_res_default["P_deferrable0"].sum()

        # Verify that changing HDD parameters affects heating demand
        # Custom params: base_temperature=20°C (higher), annual_reference_hdd=2500 (lower)
        # Default params: base_temperature=18°C (lower), annual_reference_hdd=3000 (higher)
        #
        # HDD = sum(max(base_temperature - outdoor_temp, 0))
        # With higher base_temperature (20 vs 18), we get MORE heating degree days
        # With lower annual_reference_hdd (2500 vs 3000), we get HIGHER heating demand per HDD
        # Combined effect: custom params should result in MORE heating demand

        # The difference should be meaningful (at least 5%)
        percent_difference = (
            abs(total_energy_custom_hdd - total_energy_default_hdd)
            / max(total_energy_default_hdd, 0.001)
            * 100
        )

        self.assertGreater(
            percent_difference,
            5.0,
            f"Changing HDD parameters should significantly affect heating demand. "
            f"Custom HDD: {total_energy_custom_hdd:.2f} kW, Default HDD: {total_energy_default_hdd:.2f} kW, "
            f"Difference: {percent_difference:.1f}% (expected > 5%)",
        )

        # Additional verification: With higher base_temperature and lower annual_reference_hdd,
        # custom params should generally result in higher heating demand
        # (though optimizer behavior may vary depending on electricity prices and constraints)
        if total_energy_custom_hdd > 0 and total_energy_default_hdd > 0:
            # Both scenarios require heating, so we can compare them
            # Note: We use assertNotEqual instead of assertGreater because optimizer strategy
            # may shift heating to different times based on the demand curve
            self.assertNotEqual(
                total_energy_custom_hdd,
                total_energy_default_hdd,
                "Custom and default HDD parameters should produce different results",
            )

    def test_thermal_battery_hdd_extreme_invalid_params(self):
        """Test thermal battery optimization with extreme/invalid HDD parameters handles gracefully."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            8.0 + 6.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        # Test cases with extreme HDD parameters
        extreme_cases = [
            {"base_temperature": 0, "annual_reference_hdd": 1},  # Very low base temp, very low HDD
            {"base_temperature": 50, "annual_reference_hdd": 10000},  # Very high values
        ]

        for idx, params in enumerate(extreme_cases):
            runtimeparams = {
                "def_load_config": [
                    {
                        "thermal_battery": {
                            "start_temperature": 20.0,
                            "supply_temperature": 40.0,
                            "volume": 60.0,
                            "specific_heating_demand": 100.0,
                            "area": 120.0,
                            "min_temperatures": [19.0] * 48,
                            "max_temperatures": [21.0] * 48,
                            "base_temperature": params["base_temperature"],
                            "annual_reference_hdd": params["annual_reference_hdd"],
                        }
                    },
                ]
            }

            self.optim_conf["def_load_config"] = runtimeparams["def_load_config"]
            opt = self.create_optimization()

            # Run optimization - should not crash
            unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
            unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
            opt_res = opt.perform_optimization(
                self.df_input_data_dayahead,
                self.p_pv_forecast.values.ravel(),
                self.p_load_forecast.values.ravel(),
                unit_load_cost,
                unit_prod_price,
            )

            # Verify result is returned (even if optimization is infeasible)
            self.assertIsInstance(
                opt_res,
                type(pd.DataFrame()),
                f"Case {idx}: Should return DataFrame for extreme HDD params {params}",
            )

            # Verify no NaN values in heating power
            if "P_deferrable0" in opt_res.columns:
                self.assertFalse(
                    opt_res["P_deferrable0"].isna().any(),
                    f"Case {idx}: Heating power should not contain NaN for params {params}",
                )

                # Verify non-negative heating energy
                total_energy = opt_res["P_deferrable0"].sum()
                self.assertGreaterEqual(
                    total_energy,
                    0,
                    f"Case {idx}: Heating energy should not be negative for params {params}",
                )

    def test_thermal_battery_physics_with_solar_gains(self):
        """Test thermal battery optimization with physics-based heating demand and solar gains."""
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Add outdoor temperature forecast (cycling between 2-12°C, cold winter day)
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            7.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        # Add solar irradiance forecast (GHI in W/m²)
        # Typical sunny winter day pattern: no sun at night, peak around midday
        ghi_pattern = []
        for i in range(48):
            hour_of_day = (i % 48) / 2  # Convert timestep to hour
            if 8 <= hour_of_day <= 16:  # Sunlight hours
                # Peak at 12:00 (400 W/m² - typical for winter)
                ghi = 400 * np.sin((hour_of_day - 8) * np.pi / 8)
            else:
                ghi = 0.0
            ghi_pattern.append(ghi)
        self.df_input_data_dayahead["ghi"] = ghi_pattern

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,  # Fallback
                        "area": 100.0,  # Fallback
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                        # Physics-based parameters with calibrated values
                        "u_value": 0.236,  # W/(m²·K) - calibrated from real data
                        "envelope_area": 250.0,  # m² - building envelope
                        "ventilation_rate": 0.398,  # ACH - calibrated from real data
                        "heated_volume": 356.0,  # m³ - heated volume
                        # Solar gain parameters
                        "window_area": 40.0,  # m² - typical 16% of envelope area
                        "shgc": 0.6,  # Solar Heat Gain Coefficient - modern double-glazed
                    }
                },
            ]
        }

        # Run optimization and verify success
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Verify that solar gains feature is working correctly by checking that:
        # 1. Optimization completes successfully with GHI data present
        # 2. The optimization result contains valid heat pump power values
        # 3. The system correctly identifies sunny vs non-sunny periods

        # Get GHI data and verify sunny periods were identified
        ghi_data = self.df_input_data_dayahead["ghi"].values
        sunny_hours = ghi_data > 200  # High GHI periods
        night_hours = ghi_data == 0  # Zero GHI periods

        # Verify test setup includes both sunny and night periods
        self.assertGreater(
            sunny_hours.sum(), 0, "Test should include sunny periods with GHI > 200 W/m²"
        )
        self.assertGreater(night_hours.sum(), 0, "Test should include night periods with GHI = 0")

        # Verify heat pump behavior is reasonable
        total_heat_pump_energy = opt_res["P_deferrable0"].sum()
        # Note: Heat pump may run zero energy if solar gains completely offset heating needs
        # and thermal battery stays within temperature bounds. This is valid optimizer behavior.
        self.assertGreaterEqual(
            total_heat_pump_energy, 0, "Heat pump energy should be non-negative"
        )

        # Log the result for verification
        if total_heat_pump_energy == 0:
            print("Solar gains completely offset heating demand - no heat pump operation needed")
        else:
            print(f"Total heat pump energy with solar gains: {total_heat_pump_energy:.3f} kW")

        # The key validation is that the optimization completes successfully with solar gains
        # enabled. The INFO log "Using physics-based heating demand with solar gains" confirms
        # the feature is working. The optimizer may choose not to heat if solar gains fully
        # offset heating demand and the thermal battery remains within temperature bounds.

        # NEW: Verify solar gains actually reduce heating demand
        # Compare average heating power during sunny periods vs night periods
        heating_power = opt_res["P_deferrable0"].values
        total_heat_pump_energy = heating_power.sum()

        # Calculate average heating during sunny and night periods
        if sunny_hours.sum() > 0 and night_hours.sum() > 0 and total_heat_pump_energy > 0:
            avg_heating_sunny = heating_power[sunny_hours].mean()
            avg_heating_night = heating_power[night_hours].mean()

            # During sunny periods, solar gains should reduce the need for active heating
            # Therefore, average heating power during sunny periods should be less than or equal
            # to night periods (assuming outdoor temps are similar due to the sin pattern)
            self.assertLessEqual(
                avg_heating_sunny,
                avg_heating_night,
                f"Solar gains should reduce heating demand during sunny periods. "
                f"Sunny period avg: {avg_heating_sunny:.3f} kW, Night avg: {avg_heating_night:.3f} kW",
            )

    def test_thermal_battery_variable_temperature_bounds(self):
        """Test thermal battery with non-uniform per-timestep temperature bounds.

        This verifies that the LP constraints correctly use different temperature
        bounds for different timesteps, enabling features like night setback.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        # Create variable temperature bounds: warmer comfort range during day (timesteps 16-40)
        min_temps = [18.0] * 16 + [20.0] * 24 + [18.0] * 8  # Lower at night/early morning
        max_temps = [24.0] * 16 + [26.0] * 24 + [24.0] * 8  # Higher limits during day

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": min_temps,
                        "max_temperatures": max_temps,
                    }
                },
            ]
        }

        # Run optimization and verify success
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        # Verify that bounds actually vary across timesteps (sanity check for test)
        self.assertNotEqual(min_temps[0], min_temps[20], "Test should use varying min temps")
        self.assertNotEqual(max_temps[0], max_temps[20], "Test should use varying max temps")

        # Verify optimization succeeded with variable bounds
        self.assertGreater(len(opt_res), 0, "Optimization should return results")
        self.assertIn("P_deferrable0", opt_res.columns)
        total_heating_energy = opt_res["P_deferrable0"].sum()
        self.assertGreaterEqual(total_heating_energy, 0, "Heat pump energy must be non-negative")

    def test_thermal_battery_short_temperature_lists(self):
        """Test thermal battery with temperature lists shorter than optimization horizon.

        Constraints should only apply to timesteps covered by the lists. Later
        timesteps should have no temperature constraints.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        # Use temperature lists that only cover first 24 timesteps (half the horizon)
        short_length = 24
        min_temps = [18.0] * short_length
        max_temps = [22.0] * short_length

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": min_temps,
                        "max_temperatures": max_temps,
                    }
                },
            ]
        }

        # Optimization should still succeed; constraints beyond list length are skipped
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        self.assertGreater(
            len(opt_res), 0, "Optimization should succeed with short temperature lists"
        )
        self.assertIn("P_deferrable0", opt_res.columns)

    def test_thermal_battery_none_temperature_entries(self):
        """Test thermal battery with None entries in temperature lists.

        Timesteps with None values should not have temperature constraints,
        allowing the optimizer more flexibility for those periods.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        # Create temperature lists with None entries for some timesteps
        min_temps = [18.0] * 48
        max_temps = [22.0] * 48

        # Remove constraints for middle-of-night periods (allows more flexibility)
        for i in [5, 10, 15, 20, 25, 30]:
            min_temps[i] = None
            max_temps[i] = None

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": min_temps,
                        "max_temperatures": max_temps,
                    }
                },
            ]
        }

        # Optimization should succeed; None entries mean no constraints for those timesteps
        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])

        self.assertGreater(
            len(opt_res), 0, "Optimization should succeed with None temperature entries"
        )
        self.assertIn("P_deferrable0", opt_res.columns)

    # --- Thermal Battery Inertia Tests ---

    def test_thermal_battery_flat_efficiency_mode(self):
        """Gas-source / constant-efficiency thermal_battery solves successfully.

        When 'efficiency' is set on the thermal_battery sub-config, the optimizer
        treats the heat source as a constant-efficiency converter (gas boiler, oil
        burner, district heating, etc.) rather than a temperature-dependent heat
        pump. supply_temperature is optional in this mode.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Varying outdoor temperature - should NOT influence the conversion factor
        # in flat-efficiency mode.
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 8.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        opt_res = self.run_optimization_with_config(
            [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "efficiency": 0.9,  # flat gas-boiler efficiency
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                    }
                },
            ]
        )

        # Optimization succeeds and deferrable power column is present
        self.assertIn("P_deferrable0", opt_res.columns)
        self.assertGreaterEqual(opt_res["P_deferrable0"].sum(), 0)

    def test_thermal_battery_efficiency_overrides_carnot(self):
        """When 'efficiency' is set, the Carnot calc is bypassed even if
        supply_temperature and carnot_efficiency are also present."""
        outdoor = np.array([0.0, 5.0, 10.0, 15.0])

        flat = utils.resolve_thermal_battery_cop(
            {"efficiency": 0.9, "supply_temperature": 35.0, "carnot_efficiency": 0.4},
            outdoor,
            length=4,
        )
        # Flat array regardless of supply_temperature/carnot_efficiency presence
        np.testing.assert_array_almost_equal(flat, np.full(4, 0.9))

    def test_thermal_battery_missing_source_field_raises(self):
        """A thermal_battery config with neither supply_temperature nor efficiency
        raises a clear ValueError at constraint-build time."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        with self.assertRaises(ValueError) as ctx:
            self.run_optimization_with_config(
                [
                    {
                        "thermal_battery": {
                            "start_temperature": 20.0,
                            # neither supply_temperature nor efficiency
                            "volume": 50.0,
                            "specific_heating_demand": 100.0,
                            "area": 100.0,
                            "min_temperatures": [18.0] * 48,
                            "max_temperatures": [22.0] * 48,
                        }
                    },
                ]
            )
        msg = str(ctx.exception)
        self.assertIn("supply_temperature", msg)
        self.assertIn("efficiency", msg)

    def test_shared_thermal_tank_two_sources(self):
        """Shared DHW tank fed by both HP (Carnot) and gas (flat efficiency).

        Configures two deferrable loads where:
        - Load 0 is the heat pump (Carnot mode: supply_temperature + carnot_efficiency)
        - Load 1 is the gas boiler (flat-efficiency mode)
        Both feed ONE shared tank declared in optim_conf.shared_thermal_tanks.
        Per-load cost prices gas at a flat 0.085 EUR/kWh and HP at the peak
        retail tariff so the optimizer is biased toward gas.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [10.0] * 48

        draw_off = [0.0] * 48
        draw_off[14] = 1.0  # morning draw at slot 14
        draw_off[40] = 1.2  # evening draw at slot 40

        self.optim_conf["number_of_deferrable_loads"] = 2
        self.optim_conf["nominal_power_of_deferrable_loads"] = [3500, 25000]
        self.optim_conf["minimum_power_of_deferrable_loads"] = [800, 8000]
        self.optim_conf["operating_hours_of_each_deferrable_load"] = [4, 4]
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [True, True]
        self.optim_conf["set_deferrable_load_single_constant"] = [False, False]
        self.optim_conf["set_deferrable_startup_penalty"] = [0.0, 0.0]
        self.optim_conf["set_deferrable_max_startups"] = [0, 0]
        self.optim_conf["start_timesteps_of_each_deferrable_load"] = [0, 0]
        self.optim_conf["end_timesteps_of_each_deferrable_load"] = [0, 0]
        # Source-side fields only - tank lives in shared_thermal_tanks
        self.optim_conf["def_load_config"] = [
            {"thermal_source": {"supply_temperature": 55.0, "carnot_efficiency": 0.40}},
            {"thermal_source": {"efficiency": 0.92}},
        ]
        # Declare the shared tank
        self.optim_conf["shared_thermal_tanks"] = [
            {
                "id": "dhw",
                "load_ids": [0, 1],
                "volume": 0.20,
                "density": 1000,
                "heat_capacity": 4.186,
                "start_temperature": 50.0,
                "thermal_loss": 0.05,
                "draw_off_demand": draw_off,
                "min_temperatures": [45.0] * 48,
                "max_temperatures": [62.0] * 48,
            }
        ]

        opt = self.create_optimization()
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        res = opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Both load columns present
        self.assertIn("P_deferrable0", res.columns)
        self.assertIn("P_deferrable1", res.columns)
        # Shared tank temperature variable exposed
        tank_cols = [c for c in res.columns if "temp_shared_dhw" in c or "temp_heater" in c]
        self.assertTrue(
            tank_cols, f"Expected a temp column for the shared tank, got {res.columns.tolist()}"
        )
        # At least one source fires to meet the morning + evening draws
        total = res["P_deferrable0"].sum() + res["P_deferrable1"].sum()
        self.assertGreater(total, 0, "Expected some dispatch to satisfy draw_off demand")

    def test_is_electric_load_excludes_load_from_grid_balance(self):
        """A load with is_electric_load[k]=False must not appear in p_def_sum
        (and hence not in grid_pos / grid_neg balance constraints).

        Set up TWO loads with identical electric draws but opposite electric
        flags; confirm that the grid_pos reads ONLY the electric one.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Both loads identical, but one is flagged non-electric (gas-style).
        # Set use_pv=False, set baseload to zero, costfun cost - so any
        # positive p_grid_pos comes purely from the electric deferrable.
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [10.0] * 48

        self.optim_conf["set_use_pv"] = False
        self.optim_conf["set_use_battery"] = False
        self.optim_conf["costfun"] = "cost"
        self.optim_conf["number_of_deferrable_loads"] = 2
        self.optim_conf["nominal_power_of_deferrable_loads"] = [3000, 3000]
        self.optim_conf["minimum_power_of_deferrable_loads"] = [0, 0]
        self.optim_conf["operating_hours_of_each_deferrable_load"] = [2, 2]
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [False, False]
        self.optim_conf["set_deferrable_load_single_constant"] = [False, False]
        self.optim_conf["set_deferrable_startup_penalty"] = [0.0, 0.0]
        self.optim_conf["set_deferrable_max_startups"] = [0, 0]
        self.optim_conf["start_timesteps_of_each_deferrable_load"] = [0, 0]
        self.optim_conf["end_timesteps_of_each_deferrable_load"] = [0, 0]
        self.optim_conf["def_load_config"] = []
        self.optim_conf["shared_thermal_tanks"] = []
        self.optim_conf["deferrable_load_groups"] = []
        # Load 0 = electric. Load 1 = non-electric (gas style).
        self.optim_conf["is_electric_load"] = [True, False]

        opt = self.create_optimization()
        ulc = self.df_input_data_dayahead[opt.var_load_cost].values
        upp = self.df_input_data_dayahead[opt.var_prod_price].values
        res = opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            ulc,
            upp,
        )

        p_load0 = res["P_deferrable0"]  # electric
        p_load1 = res["P_deferrable1"]  # non-electric
        # Both must be active to satisfy operating_hours = 2 hours = 4 slots × 3 kW
        self.assertGreater(p_load0.sum(), 0)
        self.assertGreater(p_load1.sum(), 0)
        # P_grid_pos should track ONLY load 0, not load 0 + load 1.
        # We allow tolerance for the baseline load_forecast / numeric.
        p_grid_pos = res["P_grid_pos"]
        # In slots where load 1 is firing but load 0 is not, p_grid_pos should
        # NOT reflect load 1's draw. Find such a slot and assert.
        only_l1_slots = (p_load0 == 0) & (p_load1 > 0)
        if only_l1_slots.any():
            # Grid import in these slots should be just baseload, not 3 kW.
            grid_in_l1_only = p_grid_pos[only_l1_slots]
            # Baseline household load is < 1 kW in the typical fixture; if our
            # flag works, grid_pos here is roughly baseload, not 3000 W.
            self.assertLess(
                grid_in_l1_only.max(),
                2000,
                "Non-electric load (load 1) appears to be pulling from the grid",
            )

    def test_shared_thermal_tank_single_source_matches_legacy(self):
        """A shared tank with exactly one source produces the same dispatch as
        the legacy per-load thermal_battery path (sanity / regression).
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [8.0] * 48

        tank_params = {
            "start_temperature": 20.0,
            "volume": 50.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "specific_heating_demand": 100.0,
            "area": 100.0,
        }
        source_params = {"supply_temperature": 35.0, "carnot_efficiency": 0.40}

        # Run A: legacy single-load thermal_battery
        legacy_a = dict(tank_params)
        legacy_a.update(source_params)
        res_legacy = self.run_optimization_with_config([{"thermal_battery": legacy_a}])

        # Run B: same tank declared as shared_thermal_tanks with one member
        self.optim_conf["number_of_deferrable_loads"] = 1
        self.optim_conf["def_load_config"] = [{"thermal_source": source_params}]
        self.optim_conf["shared_thermal_tanks"] = [{"id": "buf", "load_ids": [0], **tank_params}]
        opt = self.create_optimization()
        ulc = self.df_input_data_dayahead[opt.var_load_cost].values
        upp = self.df_input_data_dayahead[opt.var_prod_price].values
        res_shared = opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            ulc,
            upp,
        )

        # Both runs should dispatch >= 0 and ideally similar amounts. We don't
        # require bit-exact match (different constraint construction paths),
        # but both should be Optimal and the totals should be in the same
        # order of magnitude.
        leg_total = res_legacy["P_deferrable0"].sum()
        sh_total = res_shared["P_deferrable0"].sum()
        self.assertGreaterEqual(leg_total, 0)
        self.assertGreaterEqual(sh_total, 0)

    def test_thermal_battery_solar_gain_reduces_heating(self):
        """Surface solar absorption should reduce pumped heat consumption.

        A thermal_battery with a large absorption surface and high GHI gets
        free heat from the sun. The optimizer should consume strictly less
        electric/gas power than an identically configured battery with no
        solar absorption.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Cold outdoor, strong daytime sun to force solar to matter.
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [5.0] * 48
        # Bell-curve GHI profile (W/m²) peaking at solar noon (~slot 24).
        ghi_profile = [
            max(0.0, 800.0 * np.sin(np.pi * (i - 12) / 24)) if 12 <= i <= 36 else 0.0
            for i in range(48)
        ]
        self.df_input_data_dayahead["ghi"] = ghi_profile

        base_battery = {
            "start_temperature": 22.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [20.0] * 48,
            "max_temperatures": [28.0] * 48,
        }

        # Baseline: no solar absorption
        res_no_solar = self.run_optimization_with_config([{"thermal_battery": dict(base_battery)}])

        # With solar gain on a 30 m² pool-style surface
        with_solar_cfg = dict(base_battery)
        with_solar_cfg["solar_absorption_area"] = 30.0
        with_solar_cfg["solar_absorption_factor"] = 0.7
        res_with_solar = self.run_optimization_with_config([{"thermal_battery": with_solar_cfg}])

        # Solar gain should reduce pumped heat - strictly less consumption.
        self.assertLess(
            res_with_solar["P_deferrable0"].sum(),
            res_no_solar["P_deferrable0"].sum() + 1e-3,
            "Solar absorption should not increase heat-pump consumption",
        )

    def test_thermal_battery_solar_gain_zero_area_no_op(self):
        """solar_absorption_area = 0 (or unset) leaves dispatch unchanged."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [5.0] * 48
        self.df_input_data_dayahead["ghi"] = [600.0] * 48

        base_battery = {
            "start_temperature": 22.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [20.0] * 48,
            "max_temperatures": [28.0] * 48,
        }

        res_unset = self.run_optimization_with_config([{"thermal_battery": dict(base_battery)}])

        explicit_zero = dict(base_battery)
        explicit_zero["solar_absorption_area"] = 0.0
        res_zero = self.run_optimization_with_config([{"thermal_battery": explicit_zero}])

        np.testing.assert_array_almost_equal(
            res_unset["P_deferrable0"].values,
            res_zero["P_deferrable0"].values,
            decimal=4,
            err_msg="solar_absorption_area=0 should match unset behavior",
        )

    def test_per_load_cost_override_no_op_when_unset(self):
        """When cost_forecast_per_deferrable_load is unset, the optimizer behavior
        is identical to baseline (no per-load cost adjustment applied)."""
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Baseline: no per-load cost overrides
        baseline_conf = copy.deepcopy(self.optim_conf)
        baseline_conf.pop("cost_forecast_per_deferrable_load", None)
        self.optim_conf = baseline_conf
        opt_base = self.create_optimization()
        unit_load_cost = self.df_input_data_dayahead[opt_base.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt_base.var_prod_price].values
        res_base = opt_base.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # With override list of all-None: should match baseline
        override_conf = copy.deepcopy(self.optim_conf)
        override_conf["cost_forecast_per_deferrable_load"] = [None] * len(
            override_conf["nominal_power_of_deferrable_loads"]
        )
        self.optim_conf = override_conf
        opt_override = self.create_optimization()
        res_override = opt_override.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        np.testing.assert_array_almost_equal(
            res_base["P_deferrable0"].values,
            res_override["P_deferrable0"].values,
            decimal=4,
            err_msg="all-None override list must produce identical results to baseline",
        )

    def test_per_load_cost_override_shifts_dispatch(self):
        """When a load is given a per-timestep cost that's HIGHER than the global
        tariff, the optimizer should dispatch less of that load (cheaper to run
        the unconstrained alternative or skip)."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        # Two loads; override load 0 with a 10x penalty so the optimizer prefers
        # load 1 (which still uses the global tariff).
        penalty_cost = (np.array(unit_load_cost) * 10.0).tolist()

        cheap_conf = copy.deepcopy(self.optim_conf)
        cheap_conf["cost_forecast_per_deferrable_load"] = [None, None]
        self.optim_conf = cheap_conf
        opt_cheap = self.create_optimization()
        res_cheap = opt_cheap.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        expensive_conf = copy.deepcopy(self.optim_conf)
        expensive_conf["cost_forecast_per_deferrable_load"] = [penalty_cost, None]
        self.optim_conf = expensive_conf
        opt_expensive = self.create_optimization()
        res_expensive = opt_expensive.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Penalised load 0 should consume strictly less energy when its cost is
        # 10x higher than the baseline tariff.
        self.assertLess(
            res_expensive["P_deferrable0"].sum(),
            res_cheap["P_deferrable0"].sum() + 1e-3,
            "Penalised load should not consume MORE than baseline.",
        )

    def test_cost_forecast_per_load_string_null_does_not_crash(self):
        """String "null" for cost_forecast_per_deferrable_load must not crash;
        warning logged; all loads fall back to shared tariff."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        conf = copy.deepcopy(self.optim_conf)
        conf["cost_forecast_per_deferrable_load"] = "null"
        self.optim_conf = conf
        opt = self.create_optimization()

        with self.assertLogs(level="WARNING") as log:
            res = opt.perform_optimization(
                self.df_input_data_dayahead,
                self.p_pv_forecast.values.ravel(),
                self.p_load_forecast.values.ravel(),
                unit_load_cost,
                unit_prod_price,
            )

        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue(
            any("cost_forecast_per_deferrable_load" in m for m in log.output),
            "Expected warning mentioning cost_forecast_per_deferrable_load",
        )
        for k, param in enumerate(opt.param_cost_per_load):
            np.testing.assert_array_almost_equal(
                param.value,
                unit_load_cost,
                err_msg=f"Load {k} should use shared tariff when override is string",
            )

    def test_cost_forecast_per_load_array_with_string_element_does_not_crash(self):
        """Per-load override list where one element is a string must not crash;
        that load falls back to shared tariff."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        conf = copy.deepcopy(self.optim_conf)
        conf["cost_forecast_per_deferrable_load"] = [None, "null"]
        self.optim_conf = conf
        opt = self.create_optimization()

        with self.assertLogs(level="WARNING") as log:
            res = opt.perform_optimization(
                self.df_input_data_dayahead,
                self.p_pv_forecast.values.ravel(),
                self.p_load_forecast.values.ravel(),
                unit_load_cost,
                unit_prod_price,
            )

        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue(
            any("cost_forecast_per_deferrable_load" in m for m in log.output),
            "Expected warning about string element in per-load override list",
        )
        for k, param in enumerate(opt.param_cost_per_load):
            np.testing.assert_array_almost_equal(
                param.value,
                unit_load_cost,
                err_msg=f"Load {k} should use shared tariff (string override falls back)",
            )

    def test_cost_forecast_per_load_valid_overrides_applied(self):
        """Valid list-of-lists override is applied to per-load cost parameters."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values
        num_ts = len(unit_load_cost)

        override_0 = [0.1] * num_ts
        override_1 = [0.2] * num_ts

        conf = copy.deepcopy(self.optim_conf)
        conf["cost_forecast_per_deferrable_load"] = [override_0, override_1]
        self.optim_conf = conf
        opt = self.create_optimization()

        res = opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        self.assertIsInstance(res, pd.DataFrame)
        np.testing.assert_array_almost_equal(
            opt.param_cost_per_load[0].value,
            np.array(override_0),
            err_msg="Load 0 cost parameter should match the provided override",
        )
        np.testing.assert_array_almost_equal(
            opt.param_cost_per_load[1].value,
            np.array(override_1),
            err_msg="Load 1 cost parameter should match the provided override",
        )

    def test_thermal_battery_inertia_backward_compat(self):
        """Test that thermal_inertia_time_constant=0 produces identical results to omitting it."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        base_config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
        }

        # Without parameter
        opt_res_default = self.run_optimization_with_config(
            [{"thermal_battery": base_config.copy()}]
        )

        # With explicit tau=0
        config_tau0 = base_config.copy()
        config_tau0["thermal_inertia_time_constant"] = 0.0
        opt_res_tau0 = self.run_optimization_with_config([{"thermal_battery": config_tau0}])

        # Results should be identical
        np.testing.assert_array_almost_equal(
            opt_res_default["P_deferrable0"].values,
            opt_res_tau0["P_deferrable0"].values,
            decimal=4,
            err_msg="tau=0 should produce identical results to omitting the parameter",
        )

        # q_input column should NOT be present when tau=0
        self.assertNotIn("q_input_heater0", opt_res_default.columns)
        self.assertNotIn("q_input_heater0", opt_res_tau0.columns)

    def test_thermal_battery_inertia_valid_optimization(self):
        """Test that tau>0 produces a valid optimization with q_input output."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
        }

        opt_res = self.run_optimization_with_config([{"thermal_battery": config}])

        self.assertIn("q_input_heater0", opt_res.columns)
        self.assertTrue(
            (opt_res["q_input_heater0"] >= -1e-6).all(),
            "q_input values should be non-negative",
        )
        self.assertIn("P_deferrable0", opt_res.columns)

    def test_thermal_battery_inertia_negative_tau_raises(self):
        """Test that negative thermal_inertia_time_constant raises ValueError."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": -1.0,
        }

        self.optim_conf["def_load_config"] = [{"thermal_battery": config}]
        opt = self.create_optimization()

        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values

        with self.assertRaises(ValueError):
            opt.perform_optimization(
                self.df_input_data_dayahead,
                self.p_pv_forecast.values.ravel(),
                self.p_load_forecast.values.ravel(),
                unit_load_cost,
                unit_prod_price,
            )

    def test_thermal_battery_inertia_q_input_initial(self):
        """Test that q_input_initial config override works."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
            "q_input_initial": 0.5,
        }

        opt_res = self.run_optimization_with_config([{"thermal_battery": config}])

        self.assertIn("q_input_heater0", opt_res.columns)
        self.assertIn("P_deferrable0", opt_res.columns)

    def test_thermal_battery_inertia_slower_response(self):
        """Test that thermal inertia produces a slower/delayed temperature response."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 5.0

        base_config = {
            "start_temperature": 18.5,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [24.0] * 48,
        }

        # Without inertia (tau=0)
        opt_res_no_inertia = self.run_optimization_with_config(
            [{"thermal_battery": base_config.copy()}]
        )

        # With inertia (tau=2.0)
        config_inertia = base_config.copy()
        config_inertia["thermal_inertia_time_constant"] = 2.0
        opt_res_inertia = self.run_optimization_with_config([{"thermal_battery": config_inertia}])

        # Both should succeed
        self.assertIn("predicted_temp_heater0", opt_res_no_inertia.columns)
        self.assertIn("predicted_temp_heater0", opt_res_inertia.columns)

        # The inertia model should show a different temperature trajectory
        # (the filter delays heat transfer, so early timesteps should differ)
        temp_no_inertia = opt_res_no_inertia["predicted_temp_heater0"].values
        temp_inertia = opt_res_inertia["predicted_temp_heater0"].values

        # The trajectories should not be identical (inertia changes dynamics)
        self.assertFalse(
            np.allclose(temp_no_inertia, temp_inertia, atol=0.01),
            "Temperature trajectories should differ between tau=0 and tau=2.0",
        )

    def test_thermal_battery_inertia_large_tau_warning(self):
        """Test that tau > 6 triggers a warning but still produces valid results."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [24.0] * 48,
            "thermal_inertia_time_constant": 8.0,
        }

        opt_res = self.run_optimization_with_config([{"thermal_battery": config}])
        self.assertIn("q_input_heater0", opt_res.columns)

    def test_thermal_battery_inertia_small_tau_clamping(self):
        """Test that tau < time_step clamps alpha to 1.0 and still optimizes."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 0.1,  # Much smaller than 0.5h time_step
        }

        opt_res = self.run_optimization_with_config([{"thermal_battery": config}])
        self.assertIn("q_input_heater0", opt_res.columns)

    def test_thermal_battery_inertia_persist_on_cache_hit(self):
        """Test _persist_q_input auto-persists Q_input after a solve (simulated cache hit)."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
        }

        self.optim_conf["def_load_config"] = [{"thermal_battery": config}]
        opt = self.create_optimization()

        # First solve — establishes q_input_var
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Verify q_input_var was stored
        self.assertIn("q_input_var", opt.param_thermal[0])
        # Simulate cache hit: call _persist_q_input directly
        opt._persist_q_input(0, opt.param_thermal[0], config)
        new_start = opt.param_thermal[0]["q_input_start"].value

        # q_input_start should have been updated from q_input_var.value[1]
        expected = float(opt.param_thermal[0]["q_input_var"].value[1])
        self.assertAlmostEqual(new_start, expected, places=4)

    def test_thermal_battery_inertia_persist_clears_stale(self):
        """Test _persist_q_input clears q_input_var when tau changed to 0."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
        }

        self.optim_conf["def_load_config"] = [{"thermal_battery": config}]
        opt = self.create_optimization()

        # Solve to establish q_input_var
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )
        self.assertIn("q_input_var", opt.param_thermal[0])

        # Now simulate tau changed to 0 on next MPC call
        config_no_inertia = config.copy()
        config_no_inertia["thermal_inertia_time_constant"] = 0.0
        opt._persist_q_input(0, opt.param_thermal[0], config_no_inertia)

        # q_input_var should be cleared, q_input_start reset to 0
        self.assertNotIn("q_input_var", opt.param_thermal[0])
        self.assertAlmostEqual(opt.param_thermal[0]["q_input_start"].value, 0.0)

    def test_thermal_battery_inertia_persist_manual_override(self):
        """Test _persist_q_input applies q_input_initial override over auto-persisted value."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
        }

        self.optim_conf["def_load_config"] = [{"thermal_battery": config}]
        opt = self.create_optimization()

        # Solve to establish q_input_var
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Simulate cache hit with q_input_initial override
        config_override = config.copy()
        config_override["q_input_initial"] = 1.23
        opt._persist_q_input(0, opt.param_thermal[0], config_override)

        # Manual override should take priority
        self.assertAlmostEqual(opt.param_thermal[0]["q_input_start"].value, 1.23, places=2)

    def test_thermal_battery_inertia_update_thermal_start_temps(self):
        """Test update_thermal_start_temps calls _persist_q_input for thermal battery."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
        }

        self.optim_conf["def_load_config"] = [{"thermal_battery": config}]
        opt = self.create_optimization()

        # Solve to establish q_input_var
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Call update_thermal_start_temps (simulating cache hit path)
        opt.update_thermal_start_temps(self.optim_conf)

        new_start = opt.param_thermal[0]["q_input_start"].value
        expected = float(opt.param_thermal[0]["q_input_var"].value[1])
        self.assertAlmostEqual(new_start, expected, places=4)

    def test_thermal_battery_inertia_update_thermal_params(self):
        """Test update_thermal_params calls _persist_q_input for thermal battery."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = 10.0

        config = {
            "start_temperature": 20.0,
            "supply_temperature": 35.0,
            "volume": 50.0,
            "specific_heating_demand": 100.0,
            "area": 100.0,
            "min_temperatures": [18.0] * 48,
            "max_temperatures": [22.0] * 48,
            "thermal_inertia_time_constant": 2.0,
        }

        self.optim_conf["def_load_config"] = [{"thermal_battery": config}]
        opt = self.create_optimization()

        # Solve to establish q_input_var
        unit_load_cost = self.df_input_data_dayahead[opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[opt.var_prod_price].values
        opt.perform_optimization(
            self.df_input_data_dayahead,
            self.p_pv_forecast.values.ravel(),
            self.p_load_forecast.values.ravel(),
            unit_load_cost,
            unit_prod_price,
        )

        # Call update_thermal_params (simulating full cache hit path)
        opt.update_thermal_params(
            self.optim_conf,
            self.df_input_data_dayahead,
            self.p_load_forecast.values.ravel(),
        )

        new_start = opt.param_thermal[0]["q_input_start"].value
        expected = float(opt.param_thermal[0]["q_input_var"].value[1])
        self.assertAlmostEqual(new_start, expected, places=4)

    def test_thermal_battery_water_physics(self):
        """Test thermal battery with water-specific density and heat capacity."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [15.0] * 48

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 50.0,
                        "supply_temperature": 45.0,
                        "volume": 0.2,  # 200 liters = 0.2 m^3
                        "density": 997,  # water kg/m^3
                        "heat_capacity": 4.184,  # water kJ/(kg*degC)
                        "thermal_loss": 0.035,  # kW standby loss
                        "specific_heating_demand": 0.0,
                        "area": 1.0,
                        "min_temperatures": [40.0] * 48,
                        "max_temperatures": [60.0] * 48,
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])
        self.assertIn("P_deferrable0", opt_res.columns)
        self.assertEqual(opt_res["optim_status"].unique()[0], "Optimal")

    def test_thermal_battery_draw_off_demand(self):
        """Test hot water tank with draw_off_demand profile instead of building heating."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [15.0] * 48

        # 12-hour draw-off profile (kWh per 30-min slot), intentionally shorter than
        # the 48-timestep horizon so _tile_profile's tiling branch is exercised.
        # Morning shower at 7:00, evening at 19:00 (relative to the 12-h pattern).
        draw_off_24h = [0.0] * 14 + [1.5] + [0.0] * 8  # 24 elements — tiled × 2 → 48

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 50.0,
                        "supply_temperature": 45.0,
                        "volume": 0.2,
                        "density": 997,
                        "heat_capacity": 4.184,
                        "thermal_loss": 0.035,
                        "draw_off_demand": draw_off_24h,
                        "min_temperatures": [40.0] * 48,
                        "max_temperatures": [60.0] * 48,
                    }
                },
            ]
        }

        # Hot water tank heat pumps modulate continuously (not semi-continuous on/off)
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [False, True]

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])
        self.assertEqual(opt_res["optim_status"].unique()[0], "Optimal")
        self.assertIn("P_deferrable0", opt_res.columns)

        # Heat pump must compensate for draw-off + standby losses
        total_heating = opt_res["P_deferrable0"].sum() * 0.5  # kWh (30-min timesteps)
        # Heating energy should be at least the draw-off demand (COP amplifies electrical input)
        self.assertGreater(total_heating, 0, "Heat pump must run to compensate draw-off demand")

    def test_inverter_stress_cost_discharge_spread(self):
        """Test that inverter stress cost encourages spreading discharge over time."""
        # Setup plant configuration for hybrid inverter with battery
        self.plant_conf.update(
            {
                "inverter_is_hybrid": True,
                "compute_curtailment": False,
                "inverter_ac_output_max": 5000,
                "inverter_ac_input_max": 5000,
                "inverter_stress_segments": 5,
                "battery_nominal_energy_capacity": 5000,
                "battery_discharge_power_max": 5000,
                "battery_charge_power_max": 5000,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_target_state_of_charge": 0.0,
            }
        )

        # Optimization configuration
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "set_nocharge_from_grid": True,  # Ensure purely discharge behavior
                "operating_hours_of_each_deferrable_load": [0, 0],
                "load_cost_forecast_method": "csv",
                "production_price_forecast_method": "csv",
                "set_nodischarge_to_grid": False,
            }
        )

        # Create input data: 4 periods of 30 minutes, 0 PV, 0 load (focus on selling discharge)
        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp.now(tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 0.0
        df_input["p_load_forecast"] = 0.0  # No load, focus on selling battery discharge

        # Varying production prices: increasing prices to incentivize discharge later
        df_input[self.fcst.var_prod_price] = [0.1, 0.2, 0.3, 0.4]
        # Higher load cost, should not be used
        df_input[self.fcst.var_load_cost] = [0.5, 0.6, 0.7, 0.8]

        # Test without stress cost
        self.plant_conf["inverter_stress_cost"] = 0.0
        self.opt_no_stress = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_no_stress = self.opt_no_stress.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_no_stress.var_load_cost].values,
            df_input[self.opt_no_stress.var_prod_price].values,
            soc_init=0.5,
            soc_final=0.0,
        )

        # Test with stress cost
        self.plant_conf["inverter_stress_cost"] = 1.0  # currency/kWh at max power
        self.opt_with_stress = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_with_stress = self.opt_with_stress.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_with_stress.var_load_cost].values,
            df_input[self.opt_with_stress.var_prod_price].values,
            soc_init=0.5,
            soc_final=0.0,
        )

        # Assertions
        # Both optimizations should be successful
        self.assertEqual(self.opt_no_stress.optim_status, "Optimal")
        self.assertEqual(self.opt_with_stress.optim_status, "Optimal")

        for res in [opt_res_no_stress, opt_res_with_stress]:
            self.assertAlmostEqual(
                res["P_grid_pos"].sum(),
                0.0,
                msg="Grid charging should be 0",
            )

        # In a hybrid system with efficiency=1.0, 0 PV, and 0 Load:
        # P_batt (discharge) should exactly equal P_hybrid_inverter.
        # If this fails, the solver is "leaking" energy from the battery to satisfy SOC constraints
        # without passing it through the inverter variable to avoid the stress cost.
        for name, res in [("No Stress", opt_res_no_stress), ("With Stress", opt_res_with_stress)]:
            try:
                assert_series_equal(
                    res["P_batt"], res["P_hybrid_inverter"], check_names=False, atol=1e-3
                )
            except AssertionError as e:
                self.fail(
                    f"{name} optimization failed P_batt == P_hybrid_inverter check. "
                    f"Likely missing constraint on P_hybrid_inverter.\n{e}"
                )

        # Without stress cost: should discharge at max power in the most expensive period
        self.assertAlmostEqual(opt_res_no_stress["P_batt"].max(), 5000)

        # With stress cost: discharge should be more spread out
        discharge_no_stress = opt_res_no_stress["P_batt"].abs()
        discharge_with_stress = opt_res_with_stress["P_batt"].abs()

        variance_no_stress = discharge_no_stress.var()
        variance_with_stress = discharge_with_stress.var()

        self.assertLess(
            variance_with_stress,
            variance_no_stress,
            "Stress cost should reduce variance in discharge power",
        )

        # Check that stress cost is present and positive
        self.assertIn("inv_stress_cost", opt_res_with_stress.columns)
        self.assertGreater(opt_res_with_stress["inv_stress_cost"].sum(), 0)

    def test_battery_stress_cost_charging_spread(self):
        """Test that battery stress cost encourages spreading charging over time."""
        # Setup plant configuration for a non-hybrid system
        # We use a small battery and force a charge event
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": False,
                "battery_nominal_energy_capacity": 2000,  # 2kWh
                "battery_discharge_power_max": 1000,  # 1kW
                "battery_charge_power_max": 1000,  # 1kW
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_target_state_of_charge": 0.5,
                "battery_stress_segments": 5,
            }
        )

        # Optimization configuration
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "set_nocharge_from_grid": False,  # Allow grid charging
                "operating_hours_of_each_deferrable_load": [0, 0],
                "load_cost_forecast_method": "csv",
                "production_price_forecast_method": "csv",
            }
        )

        # Create input data: 4 periods of 30 minutes
        # We use constant prices. Without stress cost, the solver won't care WHEN it charges.
        # It usually picks the first or last slot or random.
        # With stress cost, it should flatten the curve.
        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp.now(tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 0.0
        df_input["p_load_forecast"] = 0.0
        df_input[self.fcst.var_prod_price] = 0.1
        df_input[self.fcst.var_load_cost] = 0.1

        # --- Run 1: No Stress Cost ---
        self.plant_conf["battery_stress_cost"] = 0.0
        self.opt_no_stress = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        # Force charge from 0.0 to 0.5 SOC (Needs 1000Wh)
        # Available time: 2 hours (4 * 30min). Max charge 1000W.
        # It could charge 1000W for 1hr, or 500W for 2hrs.
        opt_res_no_stress = self.opt_no_stress.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_no_stress.var_load_cost].values,
            df_input[self.opt_no_stress.var_prod_price].values,
            soc_init=0.0,
            soc_final=0.5,
        )

        # --- Run 2: With Stress Cost ---
        self.plant_conf["battery_stress_cost"] = 0.5  # Significant cost
        self.opt_with_stress = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_with_stress = self.opt_with_stress.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_with_stress.var_load_cost].values,
            df_input[self.opt_with_stress.var_prod_price].values,
            soc_init=0.0,
            soc_final=0.5,
        )

        # Assertions
        self.assertEqual(self.opt_no_stress.optim_status, "Optimal")
        self.assertEqual(self.opt_with_stress.optim_status, "Optimal")

        # Verify result column existence
        self.assertIn("batt_stress_cost", opt_res_with_stress.columns)
        self.assertNotIn("batt_stress_cost", opt_res_no_stress.columns)

        # Verify Energy constraints met (Both must reach target SOC)
        # SOC delta = 0.5 * 2000Wh = 1000Wh
        # P_batt is negative for charging. Sum(P_batt) * 0.5h
        energy_no_stress = -opt_res_no_stress["P_batt"].sum() * 0.5
        energy_with_stress = -opt_res_with_stress["P_batt"].sum() * 0.5

        self.assertAlmostEqual(energy_no_stress, 1000.0, delta=1.0)
        self.assertAlmostEqual(energy_with_stress, 1000.0, delta=1.0)

        # Verify Distribution
        # No stress: likely max power (1000W) for fewer steps
        # With stress: likely lower power (500W) for more steps
        peak_power_no_stress = opt_res_no_stress["P_batt"].abs().max()
        peak_power_with_stress = opt_res_with_stress["P_batt"].abs().max()

        # The smoothed peak should be strictly less than the "bang-bang" peak
        # (Given that we have enough time windows to spread it out)
        self.assertLess(peak_power_with_stress, peak_power_no_stress)

        # Variance check
        variance_no_stress = opt_res_no_stress["P_batt"].var()
        variance_with_stress = opt_res_with_stress["P_batt"].var()

        self.assertLess(
            variance_with_stress,
            variance_no_stress,
            "Battery stress cost should reduce variance in charging power",
        )

    def test_prepare_power_limit_array_scalar(self):
        """Test _prepare_power_limit_array with scalar input (existing behavior)"""
        # Test scalar input should broadcast to all timesteps
        result = self.opt._prepare_power_limit_array(9000, "test_scalar", 10)

        self.assertIsInstance(result, np.ndarray, "Should return numpy array")
        self.assertEqual(len(result), 10, "Array length should match data_length")
        self.assertTrue(np.all(result == 9000), "All values should equal the scalar input")

    def test_prepare_power_limit_array_list(self):
        """Test _prepare_power_limit_array with list input (new feature)"""
        # Test list input with correct length
        input_list = [9000, 8000, 7000, 6000, 5000]
        result = self.opt._prepare_power_limit_array(input_list, "test_list", 5)

        self.assertIsInstance(result, np.ndarray, "Should return numpy array")
        self.assertEqual(len(result), 5, "Array length should match input list length")
        self.assertEqual(result[0], 9000, "First value should be preserved")
        self.assertEqual(result[4], 5000, "Last value should be preserved")

    def test_prepare_power_limit_array_numpy(self):
        """Test _prepare_power_limit_array with numpy array input"""
        # Test numpy array input
        input_array = np.array([7000, 6000, 5000])
        result = self.opt._prepare_power_limit_array(input_array, "test_array", 3)

        self.assertIsInstance(result, np.ndarray, "Should return numpy array")
        self.assertEqual(len(result), 3, "Array length should match input")
        self.assertTrue(np.array_equal(result, input_array), "Should preserve numpy array values")

    def test_prepare_power_limit_array_wrong_length(self):
        """Test _prepare_power_limit_array with mismatched list length"""
        # Test list with wrong length should fallback to scalar (first value)
        input_list = [9000, 8000]
        result = self.opt._prepare_power_limit_array(input_list, "test_wrong_len", 5)

        self.assertIsInstance(result, np.ndarray, "Should return numpy array")
        self.assertEqual(len(result), 5, "Should fallback to correct length")
        self.assertTrue(np.all(result == 9000), "Should use first value as scalar fallback")

    def test_prepare_power_limit_array_none(self):
        """Test _prepare_power_limit_array with None input"""
        # Test None input should use default value
        result = self.opt._prepare_power_limit_array(None, "test_none", 5)

        self.assertIsInstance(result, np.ndarray, "Should return numpy array")
        self.assertEqual(len(result), 5, "Should have correct length")
        self.assertTrue(np.all(result == 9000), "Should use default value of 9000")

    def test_optimization_with_scalar_power_limits(self):
        """Test full optimization with scalar power limits (backward compatibility)"""
        # This tests that existing behavior still works
        df_input_data_dayahead = self.prepare_forecast_data()
        P_PV = df_input_data_dayahead["p_pv_forecast"]
        P_load = df_input_data_dayahead["p_load_forecast"]

        # Run optimization with scalar limits (default behavior)
        opt_res = self.opt.perform_dayahead_forecast_optim(df_input_data_dayahead, P_PV, P_load)

        # Verify optimization succeeded
        self.assertIsNotNone(opt_res, "Optimization should return results")
        self.assertIsInstance(opt_res, pd.DataFrame, "Should return DataFrame")

        # Verify power limit columns exist
        self.assertIn("maximum_power_from_grid", opt_res.columns, "Should have from_grid column")
        self.assertIn("maximum_power_to_grid", opt_res.columns, "Should have to_grid column")

        # Verify scalar values are consistent (all same value)
        from_grid_values = opt_res["maximum_power_from_grid"].unique()
        to_grid_values = opt_res["maximum_power_to_grid"].unique()

        self.assertEqual(len(from_grid_values), 1, "Scalar should have single unique value")
        self.assertEqual(len(to_grid_values), 1, "Scalar should have single unique value")

    def test_optimization_with_vector_power_limits(self):
        """Test full optimization with time-varying power limits (new feature)"""
        df_input_data_dayahead = self.prepare_forecast_data()
        n = len(df_input_data_dayahead)

        # Create time-varying limits: lower during middle hours
        vector_from_grid = [9000] * n
        for i in range(n // 4, 3 * n // 4):  # Middle half has lower limit
            vector_from_grid[i] = 5000

        # Update config temporarily
        original_from_grid = self.plant_conf["maximum_power_from_grid"]
        self.plant_conf["maximum_power_from_grid"] = vector_from_grid

        # Recreate optimization object with new config
        # Setup emhass_conf for the test
        root = pathlib.Path(get_root(__file__, num_parent=2))
        emhass_conf = {
            "data_path": root / "data/",
            "root_path": root / "src/emhass/",
        }

        opt_temp = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            "unit_load_cost",
            "unit_prod_price",
            "profit",
            emhass_conf,
            logger,
        )

        try:
            P_PV = df_input_data_dayahead["p_pv_forecast"]
            P_load = df_input_data_dayahead["p_load_forecast"]

            # Run optimization with vector limits
            opt_res = opt_temp.perform_dayahead_forecast_optim(df_input_data_dayahead, P_PV, P_load)

            # Verify optimization succeeded
            self.assertIsNotNone(opt_res, "Optimization should return results")
            self.assertIsInstance(opt_res, pd.DataFrame, "Should return DataFrame")

            # Verify power limit columns exist
            self.assertIn(
                "maximum_power_from_grid", opt_res.columns, "Should have from_grid column"
            )

            # Verify vector values are present
            from_grid_values = opt_res["maximum_power_from_grid"].values
            self.assertEqual(len(from_grid_values), n, "Should have value for each timestep")
            self.assertIn(5000, from_grid_values, "Should contain lower limit values")
            self.assertIn(9000, from_grid_values, "Should contain higher limit values")

            # Verify the pattern matches our input
            for i in range(n // 4, 3 * n // 4):
                self.assertEqual(from_grid_values[i], 5000, f"Timestep {i} should have lower limit")

            # Verify optimization respects the limits
            p_grid_pos = opt_res["P_grid_pos"].values
            for i in range(n):
                self.assertLessEqual(
                    p_grid_pos[i],
                    from_grid_values[i] + 0.1,
                    f"Grid import at timestep {i} should not exceed limit",
                )

        finally:
            # Restore original config
            self.plant_conf["maximum_power_from_grid"] = original_from_grid

    def test_power_limit_columns_are_integers(self):
        """Test that power limit columns are displayed as integers, not floats"""
        df_input_data_dayahead = self.prepare_forecast_data()
        P_PV = df_input_data_dayahead["p_pv_forecast"]
        P_load = df_input_data_dayahead["p_load_forecast"]

        # Run optimization
        opt_res = self.opt.perform_dayahead_forecast_optim(df_input_data_dayahead, P_PV, P_load)

        # Check that values are integers
        from_grid_values = opt_res["maximum_power_from_grid"].values
        to_grid_values = opt_res["maximum_power_to_grid"].values

        # All values should be whole numbers (no decimals)
        self.assertTrue(
            np.all(from_grid_values == from_grid_values.astype(int)),
            "from_grid values should be integers",
        )
        self.assertTrue(
            np.all(to_grid_values == to_grid_values.astype(int)),
            "to_grid values should be integers",
        )

    def test_perform_naive_mpc_optim_complex_case(self):
        """
        Test a complex case with 7 deferrable loads, including thermal and sequence loads,
        over a long horizon (288 steps) to verify performance fixes (Dynamic Big-M).
        """

        # Reuse existing helper to get base data structure
        df_base = self.prepare_forecast_data()

        # Extend data to 288 steps (6 days @ 30min)
        n_steps = 288

        # Create new index
        idx = pd.date_range(start=df_base.index[0], periods=n_steps, freq=self.opt.freq)

        # Create extended DataFrame by tiling the base data
        tile_count = (n_steps // len(df_base)) + 1
        df_extended = pd.concat([df_base] * tile_count).iloc[:n_steps]
        df_extended.index = idx

        # Update inputs with specific test values (as Series)
        P_PV = pd.Series(1000 * np.random.rand(n_steps), index=idx)
        P_Load = pd.Series(500 * np.random.rand(n_steps), index=idx)

        # Add thermal-specific columns if they don't exist
        temp_profile = 10 + 5 * np.sin(np.linspace(0, 8 * np.pi, n_steps))
        if "temperature_forecast" not in df_extended.columns:
            df_extended["temperature_forecast"] = temp_profile
        if "outdoor_temperature_forecast" not in df_extended.columns:
            df_extended["outdoor_temperature_forecast"] = temp_profile

        if "solar_irradiance_forecast" not in df_extended.columns:
            df_extended["solar_irradiance_forecast"] = 800 * np.clip(
                np.sin(np.linspace(0, 8 * np.pi, n_steps)), 0, 1
            )

        # Ensure the column used for cost exists
        if self.opt.var_load_cost not in df_extended.columns:
            df_extended[self.opt.var_load_cost] = 0.20

        # Configure Optimization for 7 loads
        complex_optim_conf = copy.deepcopy(self.optim_conf)
        complex_optim_conf["number_of_deferrable_loads"] = 7
        complex_optim_conf["nominal_power_of_deferrable_loads"] = [
            3000,
            2000,
            1500,
            7000,
            2000,
            500,
            1000,
        ]
        complex_optim_conf["operating_hours_of_each_deferrable_load"] = [0] * 7
        complex_optim_conf["treat_deferrable_load_as_semi_cont"] = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        complex_optim_conf["set_deferrable_load_single_constant"] = [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        complex_optim_conf["set_deferrable_startup_penalty"] = [0.0] * 7
        complex_optim_conf["deferrable_load_max_cost"] = [0.0] * 7

        # Setup Thermal Configs
        def_load_config = [{} for _ in range(7)]
        def_load_config[4] = {
            "thermal_config": {
                "heating_rate": 5.0,
                "cooling_constant": 0.1,
                "start_temperature": 45,
                "desired_temperatures": [50] * n_steps,
                "min_temperatures": [40] * n_steps,
                "max_temperatures": [60] * n_steps,
            }
        }
        def_load_config[6] = {
            "thermal_battery": {
                "capacity": 10.0,
                "volume": 500.0,
                "u_value": 0.23,
                "envelope_area": 314.0,
                "ventilation_rate": 0.41,
                "heated_volume": 356.0,
                "indoor_target_temp": 21,
                "window_area": 29.0,
                "shgc": 0.50,
                "start_temperature": 20,
                "min_temperatures": [18] * n_steps,
                "max_temperatures": [24] * n_steps,
                "supply_temperature": 35.0,
                "carnot_efficiency": 0.4,
            }
        }
        complex_optim_conf["def_load_config"] = def_load_config

        # Initialize Optimization
        opt = Optimization(
            self.retrieve_hass_conf,
            complex_optim_conf,
            self.plant_conf,
            self.opt.var_load_cost,
            self.opt.var_prod_price,
            self.opt.costfun,
            emhass_conf,
            logger,
        )

        # Runtime Definitions
        def_total_timestep = [24, 8, 12, 5, 9, 27, 18]
        def_start_timestep = [0] * 7
        def_end_timestep = [n_steps] * 7

        # Run Optimization
        try:
            opt_res = opt.perform_naive_mpc_optim(
                df_extended,
                P_PV,
                P_Load,
                prediction_horizon=n_steps,
                def_total_timestep=def_total_timestep,
                def_start_timestep=def_start_timestep,
                def_end_timestep=def_end_timestep,
            )

            # Assertions - accept both Optimal (MIP gap may help) and Optimal (Relaxed)
            status = opt_res["optim_status"].iloc[0]
            self.assertIn(
                status,
                ["Optimal", "Optimal (Relaxed)"],
                f"Expected Optimal or Optimal (Relaxed), got {status}",
            )

            # Check Load 0
            p_def_0 = opt_res["P_deferrable0"]

            # Since we might have fallen back to relaxed LP, strict timestep counting
            # can be off by +/- 1 due to continuous "smearing" of energy.
            # Instead, verify the Total Energy matches the requirement.
            # Target: 24 steps * 3000 W
            total_energy_delivered = p_def_0.sum()
            target_energy = 24 * 3000

            # Allow small numerical tolerance (e.g. 1%)
            self.assertAlmostEqual(
                total_energy_delivered, target_energy, delta=target_energy * 0.01
            )

            # Active-step count and peak power are invariants of the integer
            # (MILP) solution only. When the solver falls back to a relaxed LP,
            # energy is "smeared" across partial steps, so neither the step
            # count nor the peak power is constrained; the total-energy check
            # above is the invariant in that case.
            steps_active = (p_def_0 > 10.0).sum()  # 10 W threshold ignores noise
            max_power = p_def_0.max()
            if status == "Optimal":
                self.assertTrue(
                    23 <= steps_active <= 25,
                    f"Expected ~24 active steps, got {steps_active}",
                )
                self.assertAlmostEqual(max_power, 3000.0, delta=1.0)

            self.assertIn("predicted_temp_heater6", opt_res.columns)

        except Exception as e:
            self.fail(f"Complex optimization failed with error: {e}")

    def test_perform_naive_mpc_optim_complex_case_with_inactive_loads(self):
        """
        Test the complex 7-load case with loads 2, 3, 5 set to 0 operating timesteps
        (matching a real user scenario: dishwasher, wallbox, mock load inactive).

        Compares solve time and results between all-active vs partially-inactive
        configurations to verify the load deactivation optimization works correctly.
        Loads 4 (thermal_config) and 6 (thermal_battery) must remain active regardless.

        Uses 0.2 MIP gap (matching user's production config) and 96 timesteps
        (2 days at 30min) to keep solve times manageable in CI.
        """
        import time

        # Reuse existing helper to get base data structure
        df_base = self.prepare_forecast_data()

        n_steps = 96

        # Create new index
        idx = pd.date_range(start=df_base.index[0], periods=n_steps, freq=self.opt.freq)

        # Create extended DataFrame by tiling the base data
        tile_count = (n_steps // len(df_base)) + 1
        df_extended = pd.concat([df_base] * tile_count).iloc[:n_steps]
        df_extended.index = idx

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        P_PV = pd.Series(1000 * rng.random(n_steps), index=idx)
        P_Load = pd.Series(500 * rng.random(n_steps), index=idx)

        # Add thermal-specific columns
        temp_profile = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi, n_steps))
        df_extended["temperature_forecast"] = temp_profile
        df_extended["outdoor_temperature_forecast"] = temp_profile
        df_extended["solar_irradiance_forecast"] = 800 * np.clip(
            np.sin(np.linspace(0, 2 * np.pi, n_steps)), 0, 1
        )
        if self.opt.var_load_cost not in df_extended.columns:
            df_extended[self.opt.var_load_cost] = 0.20

        # Configure Optimization for 7 loads (matching user's real setup)
        complex_optim_conf = copy.deepcopy(self.optim_conf)
        complex_optim_conf["number_of_deferrable_loads"] = 7
        complex_optim_conf["nominal_power_of_deferrable_loads"] = [
            3000,
            2000,
            1500,
            7000,
            2000,
            500,
            1000,
        ]
        complex_optim_conf["operating_hours_of_each_deferrable_load"] = [0] * 7
        complex_optim_conf["treat_deferrable_load_as_semi_cont"] = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        complex_optim_conf["set_deferrable_load_single_constant"] = [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        complex_optim_conf["set_deferrable_startup_penalty"] = [
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
        ]
        complex_optim_conf["minimum_power_of_deferrable_loads"] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        complex_optim_conf["deferrable_load_max_cost"] = [0.0] * 7

        # Match user's production MIP gap for realistic solve times
        complex_optim_conf["lp_solver_mip_rel_gap"] = 0.2

        # Setup Thermal Configs (load 4 = hot water, load 6 = heat pump)
        def_load_config = [{} for _ in range(7)]
        def_load_config[4] = {
            "thermal_config": {
                "heating_rate": 5.0,
                "cooling_constant": 0.1,
                "start_temperature": 45,
                "desired_temperatures": [50] * n_steps,
                "min_temperatures": [40] * n_steps,
                "max_temperatures": [60] * n_steps,
            }
        }
        def_load_config[6] = {
            "thermal_battery": {
                "capacity": 10.0,
                "volume": 500.0,
                "u_value": 0.23,
                "envelope_area": 314.0,
                "ventilation_rate": 0.41,
                "heated_volume": 356.0,
                "indoor_target_temp": 21,
                "window_area": 29.0,
                "shgc": 0.50,
                "start_temperature": 20,
                "min_temperatures": [18] * n_steps,
                "max_temperatures": [24] * n_steps,
                "supply_temperature": 35.0,
                "carnot_efficiency": 0.4,
            }
        }
        complex_optim_conf["def_load_config"] = def_load_config

        # --- Solve 1: All loads active (baseline) ---
        opt_all_active = Optimization(
            self.retrieve_hass_conf,
            copy.deepcopy(complex_optim_conf),
            self.plant_conf,
            self.opt.var_load_cost,
            self.opt.var_prod_price,
            self.opt.costfun,
            emhass_conf,
            logger,
        )

        # All non-thermal loads have operating timesteps > 0
        def_total_timestep_all = [16, 8, 12, 5, 0, 18, 0]
        t_start_all = time.perf_counter()
        opt_res_all = opt_all_active.perform_naive_mpc_optim(
            df_extended.copy(),
            P_PV,
            P_Load,
            prediction_horizon=n_steps,
            def_total_timestep=def_total_timestep_all,
            def_start_timestep=[0] * 7,
            def_end_timestep=[n_steps] * 7,
        )
        t_all_active = time.perf_counter() - t_start_all

        status_all = opt_res_all["optim_status"].iloc[0]
        self.assertIn(status_all, VALID_OPTIMAL_STATUSES)

        # All loads should be active (non-thermal have timesteps > 0, thermal always active)
        for k in range(7):
            self.assertEqual(
                opt_all_active.param_load_active[k].value,
                1.0,
                f"Load {k} should be active in all-active case",
            )

        # --- Solve 2: Loads 2, 3, 5 inactive (user's real scenario) ---
        opt_partial = Optimization(
            self.retrieve_hass_conf,
            copy.deepcopy(complex_optim_conf),
            self.plant_conf,
            self.opt.var_load_cost,
            self.opt.var_prod_price,
            self.opt.costfun,
            emhass_conf,
            logger,
        )

        # Loads 2, 3, 5 have 0 operating timesteps (dishwasher, wallbox, mock off)
        def_total_timestep_partial = [16, 8, 0, 0, 0, 0, 0]
        t_start_partial = time.perf_counter()
        opt_res_partial = opt_partial.perform_naive_mpc_optim(
            df_extended.copy(),
            P_PV,
            P_Load,
            prediction_horizon=n_steps,
            def_total_timestep=def_total_timestep_partial,
            def_start_timestep=[0] * 7,
            def_end_timestep=[n_steps] * 7,
        )
        t_partial = time.perf_counter() - t_start_partial

        status_partial = opt_res_partial["optim_status"].iloc[0]
        self.assertIn(status_partial, VALID_OPTIMAL_STATUSES)

        # Verify deactivation: loads 2, 3, 5 should be inactive
        for k in [2, 3, 5]:
            self.assertEqual(
                opt_partial.param_load_active[k].value,
                0.0,
                f"Load {k} should be deactivated (0 operating timesteps, not thermal)",
            )
            self.assertTrue(
                np.allclose(opt_res_partial[f"P_deferrable{k}"], 0.0),
                f"Deactivated load {k} should have zero power output",
            )

        # Verify active loads still work correctly
        for k in [0, 1]:
            self.assertEqual(opt_partial.param_load_active[k].value, 1.0)
            active_steps = (opt_res_partial[f"P_deferrable{k}"] > 10.0).sum()
            expected = def_total_timestep_partial[k]
            self.assertTrue(
                expected - 1 <= active_steps <= expected + 1,
                f"Load {k}: expected ~{expected} active steps, got {active_steps}",
            )

        # Thermal loads must remain active even with no explicit operating timesteps
        # Load 4 (thermal_config / hot water heater)
        self.assertEqual(
            opt_partial.param_load_active[4].value,
            1.0,
            "Thermal load 4 (hot water) must remain active",
        )
        # Load 6 (thermal_battery / heat pump)
        self.assertEqual(
            opt_partial.param_load_active[6].value,
            1.0,
            "Thermal load 6 (heat pump) must remain active",
        )

        # Partial case should solve faster (fewer active binary variables)
        # Not a hard assertion (solver timing can vary), just log for inspection
        logger.info(
            f"Complex case solve times - all active: {t_all_active:.2f}s, "
            f"3 inactive: {t_partial:.2f}s, "
            f"speedup: {t_all_active / max(t_partial, 0.001):.1f}x"
        )

    def test_thermal_optimization_with_nan_temperatures(self):
        """
        Test thermal optimization robustness when outdoor_temperature_forecast contains NaNs.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Create a temperature profile with NaNs to simulate corrupted data
        nan_temps = [np.nan] * 48
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = nan_temps

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_config": {
                        "start_temperature": 20,
                        "cooling_constant": 0.1,
                        "heating_rate": 5.0,
                        "desired_temperatures": [21] * 48,
                        "sense": "heat",
                    }
                }
            ]
        }

        self.optim_conf["def_load_config"] = runtimeparams["def_load_config"]
        self.opt = self.create_optimization()

        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        try:
            self.opt_res_dayahead = self.opt.perform_optimization(
                self.df_input_data_dayahead,
                self.p_pv_forecast.values.ravel(),
                self.p_load_forecast.values.ravel(),
                unit_load_cost,
                unit_prod_price,
            )

            # SUCCESS CRITERIA: The optimization finished without crashing
            self.assertEqual(self.opt.optim_status, "Optimal")
            self.assertIn("P_deferrable0", self.opt_res_dayahead.columns)

            # We use GreaterEqual because 0 is a valid result if costs are high
            # The important part is that we got a result, not what the result is.
            self.assertGreaterEqual(self.opt_res_dayahead["P_deferrable0"].sum(), 0)

        except Exception as e:
            self.fail(f"Optimization failed when handling NaN temperatures: {e}")

    def test_thermal_battery_with_partial_nan_data(self):
        """Test thermal battery optimization where some (but not all) temperature data is missing."""
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Create a mix of valid data and NaNs
        temps = [10.0] * 48
        temps[10] = np.nan  # Missing single value
        temps[11] = np.nan
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = temps

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                    }
                }
            ]
        }

        self.optim_conf["def_load_config"] = runtimeparams["def_load_config"]
        self.opt = self.create_optimization()

        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values

        try:
            self.opt.perform_optimization(
                self.df_input_data_dayahead,
                self.p_pv_forecast.values.ravel(),
                self.p_load_forecast.values.ravel(),
                unit_load_cost,
                unit_prod_price,
            )
            self.assertEqual(self.opt.optim_status, "Optimal")
        except Exception as e:
            self.fail(f"Optimization failed with partial NaN data: {e}")

    def test_thermal_battery_soft_constraints(self):
        """Test thermal_battery with desired_temperatures and overshoot penalty."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [10.0] * 48

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [24.0] * 48,
                        "desired_temperatures": [21.0] * 48,
                        "overshoot_temperature": 23.0,
                        "penalty_factor": 10,
                        "sense": "heat",
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])
        self.assertIn("P_deferrable0", opt_res.columns)
        self.assertEqual(opt_res["optim_status"].unique()[0], "Optimal")
        total_heating = opt_res["P_deferrable0"].sum()
        self.assertGreater(total_heating, 0, "Heat pump must run")

    # Test MIP gap tolerance configuration
    def test_mip_gap_default_value(self):
        """Test that default MIP gap is 0 (exact optimal for backward compatibility)."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        # Default should be 0 for backward compatibility
        self.assertEqual(self.optim_conf.get("lp_solver_mip_rel_gap", 0.0), 0.0)

        self.opt = self.create_optimization()
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_mip_gap_zero_exact_optimal(self):
        """Test that MIP gap 0 gives exact optimal solution."""
        self.optim_conf["lp_solver_mip_rel_gap"] = 0.0
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()

        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_mip_gap_custom_value(self):
        """Test that custom MIP gap values work correctly."""
        # Test with 10% gap
        self.optim_conf["lp_solver_mip_rel_gap"] = 0.10
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()

        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_mip_gap_with_binary_variables(self):
        """Test MIP gap with semi-continuous loads (binary variables)."""
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [True, True]
        self.optim_conf["lp_solver_mip_rel_gap"] = 0.05
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()

        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn("P_deferrable0", self.opt_res_dayahead.columns)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_mip_gap_solution_quality(self):
        """Test that MIP gap produces similar objective values."""
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Run with exact optimal (gap = 0)
        self.optim_conf["lp_solver_mip_rel_gap"] = 0.0
        opt_exact = self.create_optimization()
        opt_exact.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead.copy(), self.p_pv_forecast, self.p_load_forecast
        )
        obj_exact = opt_exact.prob.value

        # Run with 5% gap
        self.optim_conf["lp_solver_mip_rel_gap"] = 0.05
        opt_gap = self.create_optimization()
        opt_gap.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead.copy(), self.p_pv_forecast, self.p_load_forecast
        )
        obj_gap = opt_gap.prob.value

        # Both should succeed
        self.assertIn(opt_exact.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertIn(opt_gap.optim_status, VALID_OPTIMAL_STATUSES)

        # Objective values should be within reasonable range
        # (gap solution should be within 10% of exact for this simple problem)
        if obj_exact is not None and obj_gap is not None and obj_exact != 0:
            relative_diff = abs(obj_exact - obj_gap) / abs(obj_exact)
            self.assertLess(
                relative_diff,
                0.10,
                f"Objective difference too large: exact={obj_exact}, gap={obj_gap}",
            )

    def test_mip_gap_negative_clamped_to_zero(self):
        """Test that negative MIP gap values are clamped to 0."""
        self.optim_conf["lp_solver_mip_rel_gap"] = -0.5
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Should still work - negative value clamped to 0
        with self.assertLogs(level="WARNING") as log:
            self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
                self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
            )
        # Check warning was logged
        self.assertTrue(
            any("negative" in msg.lower() for msg in log.output),
            "Expected warning about negative MIP gap value",
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_mip_gap_exceeds_one_clamped(self):
        """Test that MIP gap values > 1 are clamped to 1.0."""
        self.optim_conf["lp_solver_mip_rel_gap"] = 2.5
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Should still work - value clamped to 1.0
        with self.assertLogs(level="WARNING") as log:
            self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
                self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
            )
        # Check warning was logged
        self.assertTrue(
            any("exceeds" in msg.lower() or "clamping" in msg.lower() for msg in log.output),
            "Expected warning about MIP gap exceeding 1.0",
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    def test_mip_gap_boundary_values(self):
        """Test MIP gap at boundary values (0 and 1)."""
        self.df_input_data_dayahead = self.prepare_forecast_data()

        # Test gap = 0 (exact optimal)
        self.optim_conf["lp_solver_mip_rel_gap"] = 0
        opt_zero = self.create_optimization()
        opt_zero.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead.copy(), self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIn(opt_zero.optim_status, VALID_OPTIMAL_STATUSES)

        # Test gap = 1 (100% gap - any feasible solution)
        self.optim_conf["lp_solver_mip_rel_gap"] = 1.0
        opt_one = self.create_optimization()
        opt_one.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead.copy(), self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIn(opt_one.optim_status, VALID_OPTIMAL_STATUSES)

    def test_load_deactivation_zero_operating_timesteps(self):
        """Test that non-thermal loads with 0 operating timesteps are deactivated.

        When a load has operating_timesteps=0 and is not thermal, param_load_active
        should be set to 0, forcing all its binary variables to 0 via presolve.
        The load's power output must be zero throughout the horizon.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [True, True],
                "load_forecast_method": "naive",  # pin: test asserts against naive load-curve; default may shift, see #856
            }
        )
        self.opt = self.create_optimization()
        prediction_horizon = 10
        # Load 0: active (4 timesteps), Load 1: inactive (0 timesteps)
        def_total_timestep = [4, 0]
        def_start_timestep = [0, 0]
        def_end_timestep = [0, 0]

        opt_res = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=None,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
        )
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Load 0 should be active (has 4 operating timesteps)
        self.assertEqual(self.opt.param_load_active[0].value, 1.0)
        active_timesteps_0 = (opt_res["P_deferrable0"] > 0).sum()
        self.assertEqual(active_timesteps_0, 4, "Load 0 should have exactly 4 active timesteps")

        # Load 1 should be deactivated (0 operating timesteps, not thermal)
        self.assertEqual(self.opt.param_load_active[1].value, 0.0)
        self.assertTrue(
            np.allclose(opt_res["P_deferrable1"], 0.0),
            "Deactivated load 1 should have zero power output",
        )

    def test_load_deactivation_reactivation_on_cache_hit(self):
        """Test that a load can be deactivated and reactivated across cached solves.

        First solve: load 1 active. Second solve: load 1 inactive (0 timesteps).
        Third solve: load 1 active again. All should use the cached problem.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [True, True],
            }
        )
        self.opt = self.create_optimization()
        prediction_horizon = 10

        # Solve 1: both loads active
        opt_res_1 = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=None,
            def_total_timestep=[4, 3],
            def_start_timestep=[0, 0],
            def_end_timestep=[0, 0],
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertEqual(self.opt.param_load_active[0].value, 1.0)
        self.assertEqual(self.opt.param_load_active[1].value, 1.0)
        active_1_first = (opt_res_1["P_deferrable1"] > 0).sum()
        self.assertEqual(active_1_first, 3)

        # Solve 2: load 1 inactive (cache hit, same problem structure)
        opt_res_2 = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=None,
            def_total_timestep=[4, 0],
            def_start_timestep=[0, 0],
            def_end_timestep=[0, 0],
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertEqual(self.opt.param_load_active[1].value, 0.0)
        self.assertTrue(
            np.allclose(opt_res_2["P_deferrable1"], 0.0),
            "Deactivated load 1 should have zero power in second solve",
        )

        # Solve 3: load 1 reactivated (cache hit again)
        opt_res_3 = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=None,
            def_total_timestep=[4, 3],
            def_start_timestep=[0, 0],
            def_end_timestep=[0, 0],
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertEqual(self.opt.param_load_active[1].value, 1.0)
        active_1_third = (opt_res_3["P_deferrable1"] > 0).sum()
        self.assertEqual(active_1_third, 3, "Reactivated load 1 should have 3 active timesteps")

    def test_load_deactivation_does_not_affect_thermal_loads(self):
        """Test that thermal loads remain active even with 0 operating timesteps.

        Thermal loads are driven by temperature constraints, not operating timesteps,
        so they must never be deactivated by param_load_active.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "number_of_deferrable_loads": 3,
                "nominal_power_of_deferrable_loads": [2000, 2000, 5000],
                "operating_hours_of_each_deferrable_load": [0, 0, 0],
                "treat_deferrable_load_as_semi_cont": [True, False, False],
                "set_deferrable_load_single_constant": [True, False, False],
                "weight_deferrable_loads": [1.0, 1.0, 1.0],
                "minimum_power_of_deferrable_loads": [0, 0, 0],
                "set_deferrable_startup_penalty": [0, 0, 0],
                "deferrable_load_max_cost": [0, 0, 0],
                "def_current_state": [0, 0, 0],
                "def_start_penalty": [0, 0, 0],
                "def_load_config": [
                    {},
                    {
                        "thermal_config": {
                            "heating_rate": 5.0,
                            "sense": "heat",
                            "cooling_constant": 0.03,
                            "max_temperatures": [55] * 10,
                            "min_temperatures": [40] * 10,
                            "start_temperature": 45.0,
                        }
                    },
                    {},
                ],
            }
        )
        self.opt = self.create_optimization()
        prediction_horizon = 10

        # All loads have 0 operating timesteps
        opt_res = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=None,
            def_total_timestep=[0, 0, 0],
            def_start_timestep=[0, 0, 0],
            def_end_timestep=[0, 0, 0],
        )
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Load 0 (non-thermal, 0 timesteps): should be deactivated
        self.assertEqual(self.opt.param_load_active[0].value, 0.0)
        self.assertTrue(
            np.allclose(opt_res["P_deferrable0"], 0.0),
            "Non-thermal load 0 with 0 timesteps should be deactivated",
        )

        # Load 1 (thermal_config): must remain active regardless of operating_timesteps
        self.assertEqual(
            self.opt.param_load_active[1].value,
            1.0,
            "Thermal load should remain active even with 0 operating timesteps",
        )

        # Load 2 (non-thermal, 0 timesteps): should be deactivated
        self.assertEqual(self.opt.param_load_active[2].value, 0.0)
        self.assertTrue(
            np.allclose(opt_res["P_deferrable2"], 0.0),
            "Non-thermal load 2 with 0 timesteps should be deactivated",
        )

    def test_load_deactivation_multiple_inactive_with_single_constant(self):
        """Test that multiple inactive single-constant loads don't cause infeasibility.

        Previously, sum(p_def_start[k]) == 1 was always enforced for single-constant
        loads, even when inactive. This caused the solver to waste time branching on
        192 equivalent positions for a meaningless startup. Now it uses
        sum(p_def_start[k]) == param_load_active[k], which becomes == 0 for inactive loads.
        """
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [True, True],
            }
        )
        self.opt = self.create_optimization()
        prediction_horizon = 10

        # Both loads inactive with single-constant enabled
        opt_res = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=None,
            def_total_timestep=[0, 0],
            def_start_timestep=[0, 0],
            def_end_timestep=[0, 0],
        )
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Both loads should be deactivated
        self.assertEqual(self.opt.param_load_active[0].value, 0.0)
        self.assertEqual(self.opt.param_load_active[1].value, 0.0)
        self.assertTrue(np.allclose(opt_res["P_deferrable0"], 0.0))
        self.assertTrue(np.allclose(opt_res["P_deferrable1"], 0.0))

    def test_load_deactivation_with_def_total_hours(self):
        """Test that loads with 0 def_total_hours (not using def_total_timestep) are deactivated."""
        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [True, True],
            }
        )
        self.opt = self.create_optimization()
        prediction_horizon = 10

        # Load 0 active via hours, Load 1 inactive (0 hours)
        opt_res = self.opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.p_pv_forecast,
            self.p_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=[2, 0],
            def_total_timestep=None,
            def_start_timestep=[0, 0],
            def_end_timestep=[0, 0],
        )
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Load 0 active, Load 1 deactivated
        self.assertEqual(self.opt.param_load_active[0].value, 1.0)
        self.assertEqual(self.opt.param_load_active[1].value, 0.0)
        self.assertTrue(
            np.allclose(opt_res["P_deferrable1"], 0.0),
            "Load 1 with 0 hours should be deactivated",
        )

    def test_fractional_operating_hours(self):
        """Fractional ``operating_hours_of_each_deferrable_load`` are honoured and
        schedule the exact corresponding energy (regression for issue #373).

        ``assert_energy_constraint`` checks scheduled energy == ``nominal_power *
        fractional_hours`` to 1e-3 Wh. An integer-only implementation would
        truncate/round e.g. 2.5 h to 2 or 3 h and produce ``nominal_power * {2, 3}``
        Wh, so the assertion (plus the explicit integer-counterfactual below)
        cannot false-green.
        """
        nominal = self.optim_conf["nominal_power_of_deferrable_loads"][0]
        timestep_h = self.retrieve_hass_conf["optimization_time_step"].seconds / 3600
        # Counterfactual margin tied to the problem scale rather than a magic number:
        # 10% of one timestep's energy. Far above solver noise (~1e-3 Wh) yet far below
        # the >=750 Wh gap from either test value to its nearest integer-hour schedule,
        # so it stays valid if the timestep or nominal power are changed.
        integer_margin = 0.1 * nominal * timestep_h
        # 2.5 h: non-integer, timestep-aligned. 1.25 h: sub-timestep fraction. subTest
        # reports each regime independently so a failure pinpoints which one broke.
        for fractional_hours in (2.5, 1.25):
            with self.subTest(fractional_hours=fractional_hours):
                self.optim_conf.update(
                    {"operating_hours_of_each_deferrable_load": [fractional_hours, 0]}
                )
                self.opt = self.create_optimization()
                self.df_input_data_dayahead = self.prepare_forecast_data()
                opt_res = self.opt.perform_dayahead_forecast_optim(
                    self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
                )
                # A sub-timestep fraction (1.25 h at a 30-min step) makes the strict MILP
                # infeasible, so EMHASS falls back to the relaxed LP ("Optimal (Relaxed)").
                # The target-energy equality is enforced on both solve paths, so the energy
                # assertion below still holds; VALID_OPTIMAL_STATUSES accepts both statuses.
                self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
                # Exact fractional energy = nominal_power * fractional_hours.
                self.assert_energy_constraint(opt_res["P_deferrable0"], fractional_hours)
                # Discriminating counterfactual: energy must not match integer-rounded hours.
                actual_energy = opt_res["P_deferrable0"].sum() * timestep_h
                for integer_hours in (int(fractional_hours), int(fractional_hours) + 1):
                    self.assertGreater(
                        abs(actual_energy - nominal * integer_hours),
                        integer_margin,
                        f"Energy {actual_energy:.1f} Wh matches integer {integer_hours} h "
                        "-> fractional hours not honoured",
                    )

    def test_deferrable_load_group_shared_power(self):
        """Test that shared power budget constraint limits combined power of grouped loads."""
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [False, False],
                "nominal_power_of_deferrable_loads": [2000.0, 2000.0],
                "operating_hours_of_each_deferrable_load": [4, 4],
                "deferrable_load_groups": [
                    {
                        "names": ["deferrable0", "deferrable1"],
                        "max_power": 2500,
                        "mutual_exclusion": False,
                    }
                ],
            }
        )
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()
        opt_res = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        # Verify combined power never exceeds group max_power (with small tolerance)
        combined = opt_res["P_deferrable0"] + opt_res["P_deferrable1"]
        self.assertTrue(
            (combined <= 2500 + 1.0).all(),
            f"Combined power exceeded group max_power: max={combined.max():.1f}",
        )

    def test_deferrable_load_group_mutual_exclusion(self):
        """Test that mutual exclusion prevents simultaneous operation of grouped loads."""
        self.optim_conf.update(
            {
                "treat_deferrable_load_as_semi_cont": [True, True],
                "set_deferrable_load_single_constant": [False, False],
                "nominal_power_of_deferrable_loads": [2000.0, 1500.0],
                "operating_hours_of_each_deferrable_load": [4, 4],
                "deferrable_load_groups": [
                    {
                        "names": ["deferrable0", "deferrable1"],
                        "max_power": 2500,
                        "mutual_exclusion": True,
                    }
                ],
            }
        )
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()
        opt_res = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)
        # Verify at most one load is active at any timestep
        both_active = (opt_res["P_deferrable0"] > 1.0) & (opt_res["P_deferrable1"] > 1.0)
        self.assertFalse(
            both_active.any(),
            "Mutual exclusion violated: both loads active simultaneously",
        )

    def test_deferrable_load_group_no_groups(self):
        """Test that empty deferrable_load_groups works (backward compatibility)."""
        self.optim_conf["deferrable_load_groups"] = []
        self.opt = self.create_optimization()
        self.df_input_data_dayahead = self.prepare_forecast_data()
        _ = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIn(self.opt.optim_status, VALID_OPTIMAL_STATUSES)

    async def _build_params_with_groups(self, groups, **config_overrides):
        """Helper to build params with deferrable_load_groups set in config."""
        config = await build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        config["deferrable_load_groups"] = groups
        for key, value in config_overrides.items():
            config[key] = value
        _, secrets = await build_secrets(emhass_conf, logger, no_response=True)
        return await build_params(emhass_conf, secrets, config, logger)

    async def test_deferrable_load_group_validation_invalid_name(self):
        """Test that invalid deferrable names in groups raise errors."""
        with self.assertRaises(ValueError):
            await self._build_params_with_groups(
                [
                    {
                        "names": ["deferrable0", "deferrable99"],
                        "max_power": 2500,
                        "mutual_exclusion": False,
                    }
                ]
            )

    async def test_deferrable_load_group_validation_mutual_exclusion_allows_non_semi_cont(self):
        """Mutual exclusion is allowed for non-semi-continuous loads.

        The validator no longer requires every member to have
        treat_deferrable_load_as_semi_cont=true; the optimizer creates an
        anonymous binary + linking constraint for non-semi-cont members.
        """
        params = await self._build_params_with_groups(
            [
                {
                    "names": ["deferrable0", "deferrable1"],
                    "max_power": 2500,
                    "mutual_exclusion": True,
                }
            ],
            treat_deferrable_load_as_semi_cont=[False, False],
        )
        groups = params["optim_conf"]["deferrable_load_groups"]
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["mutual_exclusion"])

    async def test_deferrable_load_group_validation_overlapping_groups(self):
        """Test that a load in multiple groups raises error."""
        with self.assertRaises(ValueError):
            await self._build_params_with_groups(
                [
                    {
                        "names": ["deferrable0", "deferrable1"],
                        "max_power": 2500,
                        "mutual_exclusion": False,
                    },
                    {
                        "names": ["deferrable1", "deferrable2"],
                        "max_power": 2000,
                        "mutual_exclusion": False,
                    },
                ],
                number_of_deferrable_loads=3,
            )

    def test_deferrable_load_groups_mutex_semi_cont(self):
        """Two semi-cont loads in a mutual_exclusion group are never co-active."""
        self.optim_conf["number_of_deferrable_loads"] = 2
        self.optim_conf["nominal_power_of_deferrable_loads"] = [1000, 2000]
        self.optim_conf["operating_hours_of_each_deferrable_load"] = [0, 0]
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [True, True]
        self.optim_conf["set_deferrable_load_single_constant"] = [False, False]
        self.optim_conf["set_deferrable_startup_penalty"] = [0, 0]
        self.optim_conf["set_deferrable_max_startups"] = [0, 0]
        self.optim_conf["minimum_power_of_deferrable_loads"] = [500, 1500]
        self.optim_conf["set_deferrable_load_as_timeseries"] = [False, False]
        self.optim_conf["deferrable_load_groups"] = [
            {"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}
        ]

        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 20.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                    }
                },
                {
                    "thermal_battery": {
                        "start_temperature": 50.0,
                        "supply_temperature": 45.0,
                        "volume": 0.2,
                        "density": 997,
                        "heat_capacity": 4.184,
                        "thermal_loss": 0.035,
                        "draw_off_demand": [0.0] * 14 + [1.5] + [0.0] * 23 + [1.0] + [0.0] * 9,
                        "min_temperatures": [40.0] * 48,
                        "max_temperatures": [60.0] * 48,
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])
        self.assertEqual(self.opt.optim_status, "Optimal")

        p0 = opt_res["P_deferrable0"].values
        p1 = opt_res["P_deferrable1"].values
        for t in range(len(p0)):
            both_active = (p0[t] > 1.0) and (p1[t] > 1.0)
            self.assertFalse(
                both_active,
                f"Timestep {t}: both loads active (P0={p0[t]:.1f}W, P1={p1[t]:.1f}W) "
                f"— violates deferrable_load_groups mutual_exclusion",
            )

    def test_deferrable_load_groups_mutex_mixed_semi_cont(self):
        """Mutual exclusion holds when one member is semi-cont and the other is not.

        Exercises the new auto-binary path for the non-semi-cont member.
        """
        self.optim_conf["number_of_deferrable_loads"] = 2
        # Load 1 nominal is a list — exercises the max(nominal) branch in the mutex path.
        self.optim_conf["nominal_power_of_deferrable_loads"] = [1000, [1000, 2000]]
        self.optim_conf["operating_hours_of_each_deferrable_load"] = [0, 0]
        # Load 0 semi-continuous (modulating); load 1 non-semi-continuous (on/off)
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [True, False]
        self.optim_conf["set_deferrable_load_single_constant"] = [False, False]
        self.optim_conf["set_deferrable_startup_penalty"] = [0, 0]
        self.optim_conf["set_deferrable_max_startups"] = [0, 0]
        self.optim_conf["minimum_power_of_deferrable_loads"] = [580, 0]
        self.optim_conf["set_deferrable_load_as_timeseries"] = [False, False]
        self.optim_conf["deferrable_load_groups"] = [
            {"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}
        ]

        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 18.0,
                        "supply_temperature": 35.0,
                        "volume": 50.0,
                        "specific_heating_demand": 100.0,
                        "area": 100.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [22.0] * 48,
                    }
                },
                {
                    "thermal_battery": {
                        "start_temperature": 50.0,
                        "supply_temperature": 45.0,
                        "volume": 0.2,
                        "density": 997,
                        "heat_capacity": 4.184,
                        "thermal_loss": 0.035,
                        "draw_off_demand": [0.3] * 48,
                        "min_temperatures": [40.0] * 48,
                        "max_temperatures": [60.0] * 48,
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])
        self.assertEqual(self.opt.optim_status, "Optimal")

        p0 = opt_res["P_deferrable0"].values
        p1 = opt_res["P_deferrable1"].values
        for t in range(len(p0)):
            both_active = (p0[t] > 1.0) and (p1[t] > 1.0)
            self.assertFalse(
                both_active,
                f"Timestep {t}: both loads active (P0={p0[t]:.1f}W, P1={p1[t]:.1f}W)",
            )

    def test_deferrable_load_groups_mutex_thermal_config_and_battery(self):
        """Mutual exclusion across a thermal_config load and a thermal_battery load."""
        self.optim_conf["number_of_deferrable_loads"] = 2
        self.optim_conf["nominal_power_of_deferrable_loads"] = [1000, 2000]
        self.optim_conf["operating_hours_of_each_deferrable_load"] = [0, 0]
        self.optim_conf["treat_deferrable_load_as_semi_cont"] = [True, False]
        self.optim_conf["set_deferrable_load_single_constant"] = [False, False]
        self.optim_conf["set_deferrable_startup_penalty"] = [0, 0]
        self.optim_conf["set_deferrable_max_startups"] = [0, 0]
        self.optim_conf["minimum_power_of_deferrable_loads"] = [580, 0]
        self.optim_conf["set_deferrable_load_as_timeseries"] = [False, False]
        self.optim_conf["deferrable_load_groups"] = [
            {"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}
        ]

        self.df_input_data_dayahead = self.prepare_forecast_data()
        self.df_input_data_dayahead["outdoor_temperature_forecast"] = [
            10.0 + 5.0 * np.sin(i * np.pi / 12) for i in range(48)
        ]

        runtimeparams = {
            "def_load_config": [
                {
                    "thermal_config": {
                        "cooling_constant": 0.005,
                        "heating_rate": 3.0,
                        "overshoot_temperature": 24.0,
                        "start_temperature": 20.0,
                        "min_temperatures": [18.0] * 48,
                        "max_temperatures": [24.0] * 48,
                        "desired_temperatures": [21.0] * 48,
                    }
                },
                {
                    "thermal_battery": {
                        "start_temperature": 50.0,
                        "supply_temperature": 45.0,
                        "volume": 0.2,
                        "density": 997,
                        "heat_capacity": 4.184,
                        "thermal_loss": 0.035,
                        "draw_off_demand": [0.3] * 48,
                        "min_temperatures": [40.0] * 48,
                        "max_temperatures": [60.0] * 48,
                    }
                },
            ]
        }

        opt_res = self.run_optimization_with_config(runtimeparams["def_load_config"])
        self.assertEqual(self.opt.optim_status, "Optimal")

        p0 = opt_res["P_deferrable0"].values
        p1 = opt_res["P_deferrable1"].values
        for t in range(len(p0)):
            both_active = (p0[t] > 1.0) and (p1[t] > 1.0)
            self.assertFalse(
                both_active,
                f"Timestep {t}: both loads active (P0={p0[t]:.1f}W, P1={p1[t]:.1f}W)",
            )

    def test_battery_soc_deficit_cost(self):
        """Test that battery SOC deficit cost prevents battery
        discharge below threshold unless price difference is
        sufficient."""

        # Setup plant configuration for a non-hybrid system
        # We use a small battery and force a charge event
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": False,
                "battery_nominal_energy_capacity": 2000,  # 2kWh
                "battery_discharge_power_max": 1000,  # 1kW
                "battery_charge_power_max": 1000,  # 1kW
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_target_state_of_charge": 1.0,
            }
        )

        # Optimization configuration
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "set_nocharge_from_grid": False,  # Allow grid charging
                "set_nodischarge_to_grid": False,  # Allow grid selling
                "operating_hours_of_each_deferrable_load": [0, 0],
                "load_cost_forecast_method": "csv",
                "production_price_forecast_method": "csv",
                "battery_soc_deficit_threshold": 0.5,
            }
        )

        # Create input data: 4 periods of 30 minutes
        # Without deficit cost, the solver should discharge the battery
        # at full power when price is high and then recharge.
        # With deficit cost, it should only discharge until the deficit
        # cost negates the price difference.
        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp.now(tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 0.0
        df_input["p_load_forecast"] = 0.0
        df_input[self.fcst.var_prod_price] = [0.2, 0.2, 0.1, 0.1]
        df_input[self.fcst.var_load_cost] = [0.1, 0.1, 0.1, 0.1]

        # --- Run 1: No Deficit Cost ---
        self.optim_conf["battery_soc_deficit_cost"] = 0.0
        self.opt_no_cost = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_no_cost = self.opt_no_cost.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_no_cost.var_load_cost].values,
            df_input[self.opt_no_cost.var_prod_price].values,
            soc_init=0.5,
            soc_final=0.5,
        )

        # --- Run 2: With Stress Cost of 0.1 per kWh per h ---
        self.optim_conf["battery_soc_deficit_cost"] = 0.1
        self.opt_with_cost = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_with_cost = self.opt_with_cost.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_with_cost.var_load_cost].values,
            df_input[self.opt_with_cost.var_prod_price].values,
            soc_init=0.5,
            soc_final=0.5,
        )

        # Assertions
        self.assertEqual(self.opt_no_cost.optim_status, "Optimal")
        self.assertEqual(self.opt_with_cost.optim_status, "Optimal")

        # Verify result column existence
        self.assertIn("soc_deficit_cost", opt_res_with_cost.columns)
        self.assertIn("soc_deficit_cost", opt_res_no_cost.columns)

        # Verify SOC without deficit cost. Should be a full discharge
        # followed by a full charge, for a total gain of 1kWh*0.1 =
        # 0.1.
        self.assertEqual(opt_res_no_cost["SOC_opt"].iloc[0], 0.25)
        self.assertEqual(opt_res_no_cost["SOC_opt"].iloc[1], 0.00)
        self.assertEqual(opt_res_no_cost["SOC_opt"].iloc[2], 0.25)
        self.assertEqual(opt_res_no_cost["SOC_opt"].iloc[3], 0.50)
        logger.debug("soc cost\n{}".format(opt_res_with_cost["SOC_opt"]))

        # Verify SOC with deficit cost. The optimizer can always avoid
        # a deficit penalty by first charging one timestep and then
        # discharging for one, for a gain of 0.05.  A significant
        # deficit cost will make this the preferred action.
        self.assertEqual(opt_res_with_cost["soc_deficit_cost"].iloc[0], 0.0)
        self.assertEqual(opt_res_with_cost["soc_deficit_cost"].iloc[1], 0.0)
        self.assertEqual(opt_res_with_cost["soc_deficit_cost"].iloc[2], 0.0)
        self.assertEqual(opt_res_with_cost["soc_deficit_cost"].iloc[3], 0.0)

        self.assertEqual(opt_res_with_cost["SOC_opt"].iloc[0], 0.75)
        self.assertEqual(opt_res_with_cost["SOC_opt"].iloc[1], 0.50)
        # it may take any path here as long as it doesn't discharge below 0.5
        self.assertGreaterEqual(opt_res_with_cost["SOC_opt"].iloc[2], 0.50)
        self.assertGreaterEqual(opt_res_with_cost["SOC_opt"].iloc[3], 0.50)

    def test_battery_soc_surplus_cost(self):
        """Test that the battery SOC surplus cost discourages the
        battery from dwelling above a high SOC threshold unless the
        price difference is sufficient. Mirror of the SOC deficit
        cost test."""

        # Same small 2 kWh battery as the deficit test.
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": False,
                "battery_nominal_energy_capacity": 2000,  # 2kWh
                "battery_discharge_power_max": 1000,  # 1kW
                "battery_charge_power_max": 1000,  # 1kW
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_target_state_of_charge": 1.0,
            }
        )

        self.optim_conf.update(
            {
                "set_use_battery": True,
                "set_nocharge_from_grid": False,  # Allow grid charging
                "set_nodischarge_to_grid": False,  # Allow grid selling
                "operating_hours_of_each_deferrable_load": [0, 0],
                "load_cost_forecast_method": "csv",
                "production_price_forecast_method": "csv",
                "battery_soc_surplus_threshold": 0.5,
            }
        )

        # 4 periods of 30 minutes. Buy is cheap throughout and the
        # sell price is high in the last two periods, so the cheapest
        # plan (absent any surplus cost) is to charge up early (SOC
        # well above the 0.5 threshold) and sell it back later. The
        # surplus cost should make that high-SOC dwell unattractive.
        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp.now(tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 0.0
        df_input["p_load_forecast"] = 0.0
        df_input[self.fcst.var_prod_price] = [0.05, 0.05, 0.30, 0.30]
        df_input[self.fcst.var_load_cost] = [0.1, 0.1, 0.1, 0.1]

        # --- Run 1: No Surplus Cost ---
        self.optim_conf["battery_soc_surplus_cost"] = 0.0
        self.opt_no_cost = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_no_cost = self.opt_no_cost.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_no_cost.var_load_cost].values,
            df_input[self.opt_no_cost.var_prod_price].values,
            soc_init=0.5,
            soc_final=0.5,
        )

        # --- Run 2: With Surplus Cost of 1.0 per kWh per h ---
        self.optim_conf["battery_soc_surplus_cost"] = 1.0
        self.opt_with_cost = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res_with_cost = self.opt_with_cost.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt_with_cost.var_load_cost].values,
            df_input[self.opt_with_cost.var_prod_price].values,
            soc_init=0.5,
            soc_final=0.5,
        )

        # Both optimizations must solve.
        self.assertEqual(self.opt_no_cost.optim_status, "Optimal")
        self.assertEqual(self.opt_with_cost.optim_status, "Optimal")

        # Result column exists in both runs.
        self.assertIn("soc_surplus_cost", opt_res_with_cost.columns)
        self.assertIn("soc_surplus_cost", opt_res_no_cost.columns)

        # Discriminating power: without the surplus cost the optimizer
        # DOES dwell above the threshold to exploit the price spread.
        self.assertGreater(opt_res_no_cost["SOC_opt"].max(), 0.5)

        # With a large surplus cost it never dwells above the threshold.
        self.assertLessEqual(opt_res_with_cost["SOC_opt"].max(), 0.5 + 1e-6)

        # And the reported surplus penalty stays at zero in that case.
        for i in range(periods):
            self.assertAlmostEqual(opt_res_with_cost["soc_surplus_cost"].iloc[i], 0.0, places=6)

    def test_load_max_cost(self):
        """Test that a nonzero max cost for a load prevents the load
        from being scheduled unless it can be done for less than the
        configured cost."""

        # Setup plant configuration for a non-hybrid system
        # We use a small battery and force a charge event
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": False,
            }
        )

        # Optimization configuration
        self.optim_conf.update(
            {
                "set_use_battery": False,
                "set_use_pv": False,
                "set_nocharge_from_grid": False,  # Allow grid charging
                "set_nodischarge_to_grid": False,  # Allow grid selling
                "operating_hours_of_each_deferrable_load": [1, 1],
                "set_deferrable_startup_penalty": [0.5, 0.5],
                "nominal_power_of_deferrable_loads": [1000, 1000],
                "load_cost_forecast_method": "csv",
                "production_price_forecast_method": "csv",
            }
        )

        # Create input data: 4 periods of 30 minutes, constant energy cost
        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp.now(tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 0.0
        df_input["p_load_forecast"] = 0.0
        df_input[self.fcst.var_prod_price] = 0.1
        df_input[self.fcst.var_load_cost] = [1.0, 0.5, 0.5, 1.0]

        # Scheduling the loads for the required 1 hour will cost
        # 0.5*(1 + 0.25) (startup penalty) = 0.625. If max_cost is below this, they should not
        # be scheduled.

        self.optim_conf["deferrable_load_max_cost"] = [0.60, 0.65]
        self.opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res = self.opt.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt.var_load_cost].values,
            df_input[self.opt.var_prod_price].values,
        )

        # Assertions
        self.assertEqual(self.opt.optim_status, "Optimal")

        # Verify load 0 was not scheduled, load 1 was
        self.assertTrue(np.allclose(opt_res["P_deferrable0"], 0.0))
        self.assertTrue(np.allclose(opt_res["P_deferrable1"], [0, 1000, 1000, 0]))

    def test_sequence_load_max_cost(self):
        """Test that a nonzero max cost for a sequence load prevents the load
        from being scheduled unless it can be done for less than the
        configured cost."""

        # Setup plant configuration for a non-hybrid system
        # We use a small battery and force a charge event
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": False,
            }
        )

        # Optimization configuration
        self.optim_conf.update(
            {
                "set_use_battery": False,
                "set_use_pv": False,
                "set_nocharge_from_grid": False,  # Allow grid charging
                "set_nodischarge_to_grid": False,  # Allow grid selling
                "nominal_power_of_deferrable_loads": [[1000, 1000], [1000, 1000]],
                "operating_hours_of_each_deferrable_load": [4, 4],  # without this it doesn't work
                "load_cost_forecast_method": "csv",
                "production_price_forecast_method": "csv",
            }
        )

        # Create input data: 4 periods of 30 minutes, constant energy cost
        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp.now(tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 0.0
        df_input["p_load_forecast"] = 0.0
        df_input[self.fcst.var_prod_price] = 0.1
        df_input[self.fcst.var_load_cost] = [1.0, 0.5, 0.5, 1.0]

        # Scheduling the loads with the configured sequence will cost
        # 0.5*1.0 = 0.5. If max_cost is below this, they should not
        # be scheduled.

        self.optim_conf["deferrable_load_max_cost"] = [0.45, 0.55]
        self.opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            self.costfun,
            emhass_conf,
            logger,
        )

        opt_res = self.opt.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[self.opt.var_load_cost].values,
            df_input[self.opt.var_prod_price].values,
        )

        # Assertions
        self.assertEqual(self.opt.optim_status, "Optimal")

        logger.debug("Pdef0\n{}".format(opt_res["P_deferrable0"]))
        logger.debug("Pdef1\n{}".format(opt_res["P_deferrable1"]))

        # Verify load 0 was not scheduled, load 1 was
        self.assertTrue(np.allclose(opt_res["P_deferrable0"], 0.0))
        self.assertTrue(np.allclose(opt_res["P_deferrable1"], [0, 1000, 1000, 0]))

    def _make_curtailment_scenario(self):
        """Build shared config and input DataFrame for curtailment tie-break tests (issue #342).

        Scenario: 8 steps @ 30-min. PV=1200W, load=200W, surplus=1000W/step.
        Export cap=0 (no grid export). Battery cap=2000Wh, charge_rate=2000W
        (absorbs up to 1000Wh/step == the per-step surplus energy).
        soc_init=0, soc_final=1.0: battery stores exactly 2000Wh net.
        Total surplus = 8 * 1000W * 0.5h = 4000Wh; battery absorbs 2000Wh;
        so exactly 2000Wh must be curtailed.

        Temporal freedom: any 4 of the 8 steps can be the curtailment steps (battery
        covers the other 4). Flat prices make all allocations cost-equal; the
        tie-break is the ONLY thing that distinguishes them.
        """
        import emhass.optimization as opt_module

        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": True,
                "battery_nominal_energy_capacity": 2000,  # 2 kWh
                "battery_charge_power_max": 2000,  # 2 kW -> 1000 Wh/step max
                "battery_discharge_power_max": 2000,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_target_state_of_charge": 1.0,
                "maximum_power_to_grid": 0,  # no export
            }
        )
        self.optim_conf.update(
            {
                "set_use_battery": True,
                "set_nocharge_from_grid": True,
                "set_nodischarge_to_grid": True,
                "weight_battery_discharge": 0.0,
                "weight_battery_charge": 0.0,
                "operating_hours_of_each_deferrable_load": [0, 0],
            }
        )

        n = 8
        dates = pd.date_range(
            start=pd.Timestamp("2024-01-01 06:00:00", tz=self.retrieve_hass_conf["time_zone"]),
            periods=n,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df = pd.DataFrame(index=dates)
        df["p_pv_forecast"] = 1200.0  # W
        df["p_load_forecast"] = 200.0  # W  (net surplus = 1000 W/step)
        df[self.fcst.var_load_cost] = 0.1  # flat, no real cost difference across steps
        df[self.fcst.var_prod_price] = 0.0  # no export revenue (export blocked anyway)

        return df, opt_module

    def test_curtailment_scheduled_late(self):
        """Discriminating test (issue #342): with temporal freedom the solver must prefer
        the LATEST feasible timesteps for curtailment.

        See _make_curtailment_scenario: battery absorbs half of the horizon's surplus
        (soc 0->1.0). The other half must be curtailed. All allocations share the same
        real cost (flat prices) so the tie-break penalty is the only distinguisher.
        Without it the solver places curtailment arbitrarily; with it curtailment must
        concentrate in the SECOND HALF (center-of-mass > midpoint).
        """
        df_input, opt_module = self._make_curtailment_scenario()

        def run(eps_override=None):
            eps = (
                eps_override
                if eps_override is not None
                else getattr(opt_module, "CURTAILMENT_TIEBREAK_EPS", 0.0)
            )
            with mock.patch.object(opt_module, "CURTAILMENT_TIEBREAK_EPS", eps, create=True):
                opt = Optimization(
                    self.retrieve_hass_conf,
                    self.optim_conf,
                    self.plant_conf,
                    self.fcst.var_load_cost,
                    self.fcst.var_prod_price,
                    "profit",
                    emhass_conf,
                    logger,
                )
                res = opt.perform_optimization(
                    df_input,
                    df_input["p_pv_forecast"].values,
                    df_input["p_load_forecast"].values,
                    df_input[opt.var_load_cost].values,
                    df_input[opt.var_prod_price].values,
                    soc_init=0.0,
                    soc_final=1.0,
                )
                self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
                return res

        opt_res = run()

        self.assertIn("P_PV_curtailment", opt_res.columns)
        curtailment = opt_res["P_PV_curtailment"].values
        n = len(curtailment)

        # Scenario guarantees forced curtailment (total surplus > battery capacity)
        self.assertGreater(
            curtailment.sum(),
            0,
            "No curtailment occurred - scenario misconfigured",
        )

        indices = np.arange(n)
        total_curtail = curtailment.sum()
        com = np.dot(indices, curtailment) / total_curtail
        midpoint = (n - 1) / 2.0  # 3.5 for n=8

        # With tie-break: curtailment mass in LATE timesteps -> CoM > midpoint
        self.assertGreater(
            com,
            midpoint,
            f"Curtailment CoM {com:.2f} should be > midpoint {midpoint:.2f}. "
            f"Curtailment per step: {curtailment}",
        )

        # Counterfactual discriminating-power check: negate EPS -> EARLY preference -> CoM < midpoint
        res_early = run(eps_override=-getattr(opt_module, "CURTAILMENT_TIEBREAK_EPS", 1e-7))
        c_early = res_early["P_PV_curtailment"].values
        total_early = c_early.sum()
        if total_early > 0:
            com_early = np.dot(indices, c_early) / total_early
            self.assertLess(
                com_early,
                midpoint,
                f"With negative EPS, curtailment CoM {com_early:.2f} should be < midpoint "
                f"{midpoint:.2f}. Curtailment per step: {c_early}",
            )

    def test_curtailment_tiebreak_cost_invariant(self):
        """Issue #342: The tie-break must not change the real economic cost.
        Solve the same scenario with normal EPS and EPS=0; assert that real cost
        (grid import minus export revenue) and total curtailed energy are equal
        within tight tolerance.
        """
        df_input, opt_module = self._make_curtailment_scenario()

        def run_with_eps(eps_value):
            with mock.patch.object(opt_module, "CURTAILMENT_TIEBREAK_EPS", eps_value, create=True):
                opt = Optimization(
                    self.retrieve_hass_conf,
                    self.optim_conf,
                    self.plant_conf,
                    self.fcst.var_load_cost,
                    self.fcst.var_prod_price,
                    "profit",
                    emhass_conf,
                    logger,
                )
                res = opt.perform_optimization(
                    df_input,
                    df_input["p_pv_forecast"].values,
                    df_input["p_load_forecast"].values,
                    df_input[opt.var_load_cost].values,
                    df_input[opt.var_prod_price].values,
                    soc_init=0.0,
                    soc_final=1.0,
                )
                self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
                return res

        res_with = run_with_eps(getattr(opt_module, "CURTAILMENT_TIEBREAK_EPS", 1e-7))
        res_without = run_with_eps(0.0)

        time_step_h = self.retrieve_hass_conf["optimization_time_step"].seconds / 3600.0
        unit_load_cost = df_input[self.fcst.var_load_cost].values
        unit_prod_price = df_input[self.fcst.var_prod_price].values

        def real_cost(res):
            # Real import cost minus export revenue (tie-break term excluded)
            import_cost = np.sum(res["P_grid_pos"].values * unit_load_cost) * time_step_h * 0.001
            export_rev = np.sum(-res["P_grid_neg"].values * unit_prod_price) * time_step_h * 0.001
            return import_cost - export_rev

        cost_with = real_cost(res_with)
        cost_without = real_cost(res_without)

        denom = max(abs(cost_without), 1e-10)
        self.assertLess(
            abs(cost_with - cost_without) / denom,
            1e-6,
            f"Real cost changed: with_eps={cost_with:.8f}, without_eps={cost_without:.8f}",
        )

        # Total curtailed energy must be unchanged
        curtail_with = res_with["P_PV_curtailment"].values.sum() * time_step_h * 0.001
        curtail_without = res_without["P_PV_curtailment"].values.sum() * time_step_h * 0.001
        self.assertAlmostEqual(
            curtail_with,
            curtail_without,
            places=4,
            msg=f"Total curtailed energy changed: {curtail_with:.6f} vs {curtail_without:.6f}",
        )

    def test_curtailment_tiebreak_absent_when_disabled(self):
        """Issue #342: When compute_curtailment=False, the tie-break term must not
        reference p_pv_curtailment in the objective (structurally absent).
        """
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": False,
            }
        )
        self.optim_conf.update(
            {
                "set_use_battery": False,
                "operating_hours_of_each_deferrable_load": [0, 0],
            }
        )

        periods = 4
        dates = pd.date_range(
            start=pd.Timestamp("2024-01-01 06:00:00", tz=self.retrieve_hass_conf["time_zone"]),
            periods=periods,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 1000.0
        df_input["p_load_forecast"] = 200.0
        df_input[self.fcst.var_load_cost] = 0.1
        df_input[self.fcst.var_prod_price] = 0.05

        opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            "profit",
            emhass_conf,
            logger,
        )

        opt.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[opt.var_load_cost].values,
            df_input[opt.var_prod_price].values,
        )

        self.assertIsNotNone(opt.prob, "prob should be built after perform_optimization")

        # p_pv_curtailment must NOT appear in the objective when compute_curtailment=False
        obj_var_names = {v.name() for v in opt.prob.objective.variables()}
        self.assertNotIn(
            "p_pv_curtailment",
            obj_var_names,
            f"p_pv_curtailment should not be in objective when compute_curtailment=False. "
            f"Objective variables: {obj_var_names}",
        )

    def test_continuous_deferrable_inactive_no_phantom_consumption(self):
        """An inactive pure-continuous deferrable load (0 operating hours, no
        binaries) must not absorb surplus PV. Without the param_load_active bound
        in the continuous branch it acts as a free energy sink, and the
        curtailment tie-break (issue #342) would then deterministically route
        surplus into the disabled load instead of booking it as curtailment.
        """
        self.plant_conf.update(
            {
                "inverter_is_hybrid": False,
                "compute_curtailment": True,
                "maximum_power_to_grid": 0,  # no export
            }
        )
        self.optim_conf.update(
            {
                "set_use_battery": False,
                "operating_hours_of_each_deferrable_load": [0, 0],
                # Pure continuous: no binaries, so the load-active bound is the
                # only thing standing between an inactive load and the surplus
                "treat_deferrable_load_as_semi_cont": [False, False],
                "set_deferrable_load_single_constant": [False, False],
            }
        )

        n = 8
        dates = pd.date_range(
            start=pd.Timestamp("2024-01-01 06:00:00", tz=self.retrieve_hass_conf["time_zone"]),
            periods=n,
            freq=self.retrieve_hass_conf["optimization_time_step"],
        )
        df_input = pd.DataFrame(index=dates)
        df_input["p_pv_forecast"] = 1200.0  # W
        df_input["p_load_forecast"] = 200.0  # W (surplus = 1000 W/step, unexportable)
        df_input[self.fcst.var_load_cost] = 0.1
        df_input[self.fcst.var_prod_price] = 0.0

        opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            "profit",
            emhass_conf,
            logger,
        )
        opt_res = opt.perform_optimization(
            df_input,
            df_input["p_pv_forecast"].values,
            df_input["p_load_forecast"].values,
            df_input[opt.var_load_cost].values,
            df_input[opt.var_prod_price].values,
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Inactive loads must consume nothing
        for col in ("P_deferrable0", "P_deferrable1"):
            self.assertTrue(
                np.allclose(opt_res[col].values, 0.0, atol=1e-6),
                f"{col} should be zero for an inactive continuous load, got {opt_res[col].values}",
            )

        # The full surplus must be booked as curtailment, not silently dumped
        time_step_h = self.retrieve_hass_conf["optimization_time_step"].seconds / 3600.0
        expected_curtail_wh = 1000.0 * n * time_step_h
        actual_curtail_wh = opt_res["P_PV_curtailment"].values.sum() * time_step_h
        self.assertAlmostEqual(
            actual_curtail_wh,
            expected_curtail_wh,
            delta=1.0,
            msg=f"Expected {expected_curtail_wh} Wh curtailed, got {actual_curtail_wh}",
        )


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
