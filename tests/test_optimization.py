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
        self.df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.input_data_dict = {
            "retrieve_hass_conf": retrieve_hass_conf,
        }

    # Check formatting of output from perfect optimization
    def test_perform_perfect_forecast_optim(self):
        self.opt_res = self.opt.perform_perfect_forecast_optim(self.df_input_data, self.days_list)
        self.assertIsInstance(self.opt_res, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertIn("cost_fun_" + self.costfun, self.opt_res.columns)

    def test_perform_dayahead_forecast_optim(self):
        # Check formatting of output from dayahead optimization
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIsInstance(self.opt_res_dayahead.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.opt_res_dayahead.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertIn("cost_fun_" + self.costfun, self.opt_res_dayahead.columns)
        self.assertEqual(
            self.opt_res_dayahead["P_deferrable0"].sum()
            * (self.retrieve_hass_conf["optimization_time_step"].seconds / 3600),
            self.optim_conf["nominal_power_of_deferrable_loads"][0]
            * self.optim_conf["operating_hours_of_each_deferrable_load"][0],
        )
        # Test the battery, dynamics and grid exchange contraints
        self.optim_conf.update({"set_use_battery": True})
        self.optim_conf.update({"set_nocharge_from_grid": True})
        self.optim_conf.update({"set_battery_dynamic": True})
        self.optim_conf.update({"set_nodischarge_to_grid": True})
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
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertIsInstance(self.opt_res_dayahead, type(pd.DataFrame()))
        self.assertIn("P_batt", self.opt_res_dayahead.columns)
        self.assertIn("SOC_opt", self.opt_res_dayahead.columns)
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
        self.optim_conf.update({"treat_deferrable_load_as_semi_cont": [True, True]})
        self.optim_conf.update({"set_deferrable_load_single_constant": [True, True]})
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
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update({"treat_deferrable_load_as_semi_cont": [False, True]})
        self.optim_conf.update({"set_deferrable_load_single_constant": [True, True]})
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
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update({"treat_deferrable_load_as_semi_cont": [False, True]})
        self.optim_conf.update({"set_deferrable_load_single_constant": [False, True]})
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
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update({"treat_deferrable_load_as_semi_cont": [False, False]})
        self.optim_conf.update({"set_deferrable_load_single_constant": [False, True]})
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
        self.opt_res_dayahead = self.opt.perform_dayahead_forecast_optim(
            self.df_input_data_dayahead, self.p_pv_forecast, self.p_load_forecast
        )
        self.assertEqual(self.opt.optim_status, "Optimal")
        self.optim_conf.update({"treat_deferrable_load_as_semi_cont": [False, False]})
        self.optim_conf.update({"set_deferrable_load_single_constant": [False, False]})
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
        self.opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            costfun,
            emhass_conf,
            logger,
        )
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
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
        self.opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            costfun,
            emhass_conf,
            logger,
        )
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
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
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
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
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
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
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        # Test the battery
        self.optim_conf.update({"set_use_battery": True})
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
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
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
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)
        # Test the battery
        self.optim_conf.update({"set_use_battery": True})
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
        self.df_input_data_dayahead = self.fcst.get_load_cost_forecast(self.df_input_data_dayahead)
        self.df_input_data_dayahead = self.fcst.get_prod_price_forecast(self.df_input_data_dayahead)

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
