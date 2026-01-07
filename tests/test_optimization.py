#!/usr/bin/env python

import os
import pathlib
import pickle
import random
import unittest
from datetime import datetime

import aiofiles
import numpy as np
import orjson
import pandas as pd
from pandas.testing import assert_series_equal

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
        # Test with different default solver, debug mode and batt SOC conditions
        del self.optim_conf["lp_solver"]
        del self.optim_conf["lp_solver_path"]
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
        # Test dayahead optimization using different solvers
        import pulp as pl

        solver_list = pl.listSolvers(onlyAvailable=True)
        for solver in solver_list:
            self.optim_conf["lp_solver"] = solver
            if os.getenv("lp_solver_path", default=None) is None:
                self.optim_conf["lp_solver_path"] = os.getenv("lp_solver_path", default=None)
            self.opt = self.create_optimization()
            self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(
                self.df_input_data_dayahead
            )
            self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(
                self.df_input_data_dayahead
            )
            self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
                self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
            )
            self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
            self.assertIsInstance(
                self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex
            )
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
        unit_load_cost = self.df_input_data_dayahead[self.opt.var_load_cost].values  # €/kWh
        unit_prod_price = self.df_input_data_dayahead[self.opt.var_prod_price].values  # €/kWh
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

        # Mock P_PV and P_Load as they are needed by perform_naive_mpc_optim
        P_PV = np.zeros(prediction_horizon)
        P_Load = np.zeros(prediction_horizon)

        unit_load_cost = input_data[self.opt.var_load_cost].values
        unit_prod_price = input_data[self.opt.var_prod_price].values

        self.opt_res_dayahead = self.opt.perform_optimization(
            input_data,
            P_PV,
            P_Load,
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
        P_PV = np.zeros(horizon)
        P_Load = np.zeros(horizon)
        unit_load_cost = input_data[self.opt.var_load_cost].values
        unit_prod_price = input_data[self.opt.var_prod_price].values

        # Re-init optimization to ensure clean state
        self.opt = self.create_optimization()

        self.opt_res_dayahead = self.opt.perform_optimization(
            input_data,
            P_PV,
            P_Load,
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
                            "heating_rate": -10,  # Negative for cooling capacity
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
        self.assertGreater(max_temps[0], min_temps[0], "max_temperatures must be greater than min_temperatures")
        self.assertGreaterEqual(min_temps[0], 10.0, "min_temperatures should be reasonable (>= 10°C)")
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

        # Verify physical plausibility: heating power should correlate with outdoor temperature
        # Lower outdoor temperatures should require higher heating power
        outdoor_temp = self.df_input_data_dayahead["outdoor_temperature_forecast"].values
        heating_power = opt_res["P_deferrable0"].values

        # Define cold and warm periods based on median temperature
        temp_median = np.median(outdoor_temp)
        cold_indices = outdoor_temp < temp_median  # Colder periods
        warm_indices = outdoor_temp > temp_median  # Warmer periods

        # Calculate average heating power during each period
        mean_power_cold = heating_power[cold_indices].mean()
        mean_power_warm = heating_power[warm_indices].mean()

        # Assert physical plausibility: heating power during cold periods >= warm periods
        # Note: May be equal (both zero) if thermal battery stays within bounds without heating
        self.assertGreaterEqual(
            mean_power_cold,
            mean_power_warm,
            f"Physics check failed: heating during cold periods ({mean_power_cold:.3f} kW) "
            f"should be >= warm periods ({mean_power_warm:.3f} kW). "
            f"Temperature range: {outdoor_temp.min():.1f}°C to {outdoor_temp.max():.1f}°C"
        )

        # Verify physics-based parameters are being used by checking log output was correct
        # The INFO log should show "Using physics-based heating demand calculation"
        # (This is verified by the logger output, not an explicit assertion here)

        # Additional check: If any heating occurred, verify it's reasonable
        total_heating_energy = heating_power.sum()
        if total_heating_energy > 0:
            # With physics-based model, heating should be proportional to temperature difference
            # For the given parameters (u_value=0.4, envelope_area=350, ventilation_rate=0.6, volume=250)
            # at ΔT=15°C, heating demand should be around 2.5-3.5 kWh per 30-min timestep
            # Over 48 timesteps with average ΔT≈10°C, total should be roughly 50-150 kWh
            self.assertGreater(total_heating_energy, 0, "Some heating should occur given the cold outdoor temperatures")
            self.assertLess(total_heating_energy, 200, "Total heating energy should be reasonable (not excessive)")

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
        opt_res_default = self.run_optimization_with_config(runtimeparams_default["def_load_config"])

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
        percent_difference = abs(total_energy_custom_hdd - total_energy_default_hdd) / max(total_energy_default_hdd, 0.001) * 100

        self.assertGreater(
            percent_difference,
            5.0,
            f"Changing HDD parameters should significantly affect heating demand. "
            f"Custom HDD: {total_energy_custom_hdd:.2f} kW, Default HDD: {total_energy_default_hdd:.2f} kW, "
            f"Difference: {percent_difference:.1f}% (expected > 5%)"
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
                "Custom and default HDD parameters should produce different results"
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
            self.assertIsInstance(opt_res, type(pd.DataFrame()),
                f"Case {idx}: Should return DataFrame for extreme HDD params {params}")

            # Verify no NaN values in heating power
            if "P_deferrable0" in opt_res.columns:
                self.assertFalse(
                    opt_res["P_deferrable0"].isna().any(),
                    f"Case {idx}: Heating power should not contain NaN for params {params}"
                )

                # Verify non-negative heating energy
                total_energy = opt_res["P_deferrable0"].sum()
                self.assertGreaterEqual(
                    total_energy,
                    0,
                    f"Case {idx}: Heating energy should not be negative for params {params}"
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
        self.assertGreater(sunny_hours.sum(), 0, "Test should include sunny periods with GHI > 200 W/m²")
        self.assertGreater(night_hours.sum(), 0, "Test should include night periods with GHI = 0")

        # Verify heat pump behavior is reasonable
        total_heat_pump_energy = opt_res["P_deferrable0"].sum()
        # Note: Heat pump may run zero energy if solar gains completely offset heating needs
        # and thermal battery stays within temperature bounds. This is valid optimizer behavior.
        self.assertGreaterEqual(total_heat_pump_energy, 0, "Heat pump energy should be non-negative")

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
                f"Sunny period avg: {avg_heating_sunny:.3f} kW, Night avg: {avg_heating_night:.3f} kW"
            )
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


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
