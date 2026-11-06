#!/usr/bin/env python
"""Per-plant PV forecast inputs (#1150): nested runtime payloads, csv and list columns, the
Solcast rooftop count.

Two parts. At one inverter, a nested ``pv_power_forecast`` is refused and the check runs through
the real ``treat_runtimeparams``. Above one inverter nothing is reachable end to end while the
refusal of ``number_of_inverters > 1`` is in place, so the helpers and the Forecast methods are
called directly.
"""

import logging
import tracemalloc
import unittest
import unittest.mock

import numpy as np
import orjson
import pandas as pd

from emhass import utils
from tests import test_forecast_validity_contract as validity
from tests.aioresponses_compat import aioresponses
from tests.test_per_plant_pv import PerPlantBase, logger

TZ = "Europe/Paris"
STEP_MINUTES = 30
HORIZON = 4
GRID = pd.date_range("2026-06-26 09:00", periods=HORIZON, freq="30min", tz=TZ)
STAMPS = [ts.isoformat() for ts in GRID]


def assert_error_says(test, captured, *fragments):
    """The captured ERROR log has every fragment, so an unrelated earlier error cannot pass."""
    text = "\n".join(captured.output)
    for fragment in fragments:
        test.assertIn(fragment, text)


def peak_memory(call):
    """Run ``call`` and return its result and the peak traced allocation in bytes.

    A huge ``number_of_inverters`` must cost nothing: an allocation sized by it (a list, or a
    numpy array, which tracemalloc also sees) would show up here long before it finished.
    """
    tracemalloc.start()
    try:
        result = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


# Far above what these calls need, far below what anything sized by 10**9 inverters would take.
PEAK_LIMIT_BYTES = 50 * 1024 * 1024


def series(base, n=HORIZON):
    """A short ramp of watts, ``n`` long, starting at ``base``."""
    return [float(base + 10 * i) for i in range(n)]


def stamped(base, n=HORIZON):
    """The same ramp as a timestamped mapping on the test grid."""
    return {STAMPS[i]: value for i, value in enumerate(series(base, n))}


# ---------------------------------------------------------------------------
# One inverter: a nested payload is refused through the real treat_runtimeparams
# ---------------------------------------------------------------------------


class TestNestedPayloadAtOneInverter(validity._PinnedNow):
    async def treat(self, runtimeparams):
        return await validity._treat(runtimeparams)

    async def assert_refused(self, payload, key="pv_power_forecast"):
        with self.assertLogs(validity.logger, level="ERROR") as captured:
            params, optim_conf = await self.treat({key: payload})
        self.assertIsNone(params["passed_data"][key])
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        named = validity._errors_mentioning(captured, "number_of_inverters")
        self.assertEqual(len(named), 1, [r.getMessage() for r in captured.records])
        message = named[0].getMessage()
        self.assertIn("flat list", message)
        self.assertLess(len(message), 600)
        # The refusal is not the old fall-back log, and the payload is not echoed.
        self.assertFalse(validity._errors_mentioning(captured, "the length is not correct"))

    async def test_nested_list_shorter_than_the_horizon_is_refused(self):
        await self.assert_refused([series(100, 3), series(200, 3)])

    async def test_nested_list_longer_than_the_horizon_is_refused(self):
        n = self.horizon
        await self.assert_refused([series(100, n), series(200, n)] * (n // 2 + 2))

    async def test_nested_list_with_one_inner_list_is_refused(self):
        await self.assert_refused([series(100, self.horizon), 5.0])

    async def test_list_of_mappings_is_refused(self):
        await self.assert_refused([stamped(100), stamped(200)])

    async def test_list_of_mappings_longer_than_the_horizon_is_refused(self):
        n = self.horizon
        await self.assert_refused([stamped(100, 2)] * (n + 3))

    async def test_stringified_nested_list_is_refused(self):
        await self.assert_refused(repr([series(100, 3), series(200, 3)]))

    async def test_a_nested_list_for_another_key_keeps_the_old_fall_back(self):
        """Only the PV forecast has the per-inverter shape; the other keys are unchanged."""
        with self.assertLogs(validity.logger, level="ERROR") as captured:
            params, optim_conf = await self.treat({"load_power_forecast": [[1.0, 2.0], [3.0, 4.0]]})
        self.assertIsNone(params["passed_data"]["load_power_forecast"])
        self.assertTrue(validity._errors_mentioning(captured, "the length is not correct"))
        self.assertFalse(validity._errors_mentioning(captured, "number_of_inverters"))

    async def test_a_flat_list_holding_a_mapping_is_refused(self):
        """A list of the right length with one mapping among the numbers is not a flat list.

        Before this change it fell back or failed on a message about a non-numeric value; it
        is now treated like the other nested shapes.
        """
        flat = series(0, self.horizon)
        await self.assert_refused([*flat[:-1], stamped(100)])
        await self.assert_refused([stamped(100), *flat[1:]])
        await self.assert_refused([*series(0, self.horizon + 5), {"x": 1.0}])

    async def test_a_nested_payload_is_refused_under_naive_mpc_on_the_cut_grid(self):
        """naive-mpc-optim cuts the grid to prediction_horizon; the refusal does not depend on
        the payload matching either grid length."""
        cut = 6
        self.assertLess(cut, self.horizon)
        payloads = (
            [series(100, cut), series(200, cut)],
            [series(100, self.horizon), series(200, self.horizon)],
            [stamped(100), stamped(200)],
            repr([series(100, cut), series(200, cut)]),
        )
        for payload in payloads:
            with self.subTest(payload=type(payload).__name__, outer=len(payload)):
                with self.assertLogs(validity.logger, level="ERROR") as captured:
                    params, optim_conf = await validity._treat(
                        {"pv_power_forecast": payload, "prediction_horizon": cut},
                        "naive-mpc-optim",
                    )
                self.assertIsNone(params["passed_data"]["pv_power_forecast"])
                self.assertEqual(optim_conf["weather_forecast_method"], "list")
                named = validity._errors_mentioning(captured, "number_of_inverters")
                self.assertEqual(len(named), 1, [r.getMessage() for r in captured.records])
                self.assertIn("flat list", named[0].getMessage())
        # A flat list of the cut length is still accepted on that grid.
        flat = series(0, cut)
        params, optim_conf = await validity._treat(
            {"pv_power_forecast": flat, "prediction_horizon": cut}, "naive-mpc-optim"
        )
        self.assertEqual(params["passed_data"]["pv_power_forecast"], flat)
        self.assertEqual(optim_conf["weather_forecast_method"], "list")

    async def test_a_valid_flat_list_is_unchanged(self):
        values = series(0, self.horizon + 5)
        params, optim_conf = await self.treat({"pv_power_forecast": values})
        self.assertEqual(params["passed_data"]["pv_power_forecast"], values)
        self.assertEqual(optim_conf["weather_forecast_method"], "list")

    async def test_a_flat_timestamped_mapping_is_unchanged(self):
        mapping = {str(ts): 100.0 for ts in self.forecast_dates}
        params, optim_conf = await self.treat({"pv_power_forecast": mapping})
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertEqual(len(params["passed_data"]["pv_power_forecast"]), self.horizon)

    async def test_a_flat_list_of_the_wrong_length_still_falls_back(self):
        """Pin: the generic wrong-length fallback is unchanged (no value, method kept)."""
        _, baseline = await self.treat({})
        with self.assertLogs(validity.logger, level="ERROR") as captured:
            params, optim_conf = await self.treat(
                {"pv_power_forecast": series(0, self.horizon - 1)}
            )
        self.assertIsNone(params["passed_data"]["pv_power_forecast"])
        self.assertEqual(optim_conf["weather_forecast_method"], baseline["weather_forecast_method"])
        self.assertNotEqual(optim_conf["weather_forecast_method"], "list")
        self.assertTrue(validity._errors_mentioning(captured, "the length is not correct"))
        self.assertFalse(validity._errors_mentioning(captured, "number_of_inverters"))

    async def test_the_paired_path_stays_fail_closed(self):
        """Pin: a nested P50/P10 pair keeps the pair sentinel (p50 None, p10 empty)."""
        n = self.horizon
        nested = [series(100, n), series(200, n)]
        with self.assertLogs(validity.logger, level="ERROR"):
            params, optim_conf = await self.treat(
                {"pv_power_forecast": nested, "pv_power_forecast_p10": nested}
            )
        self.assertIsNone(params["passed_data"]["pv_power_forecast"])
        self.assertEqual(params["passed_data"]["pv_power_forecast_p10"], [])
        self.assertEqual(optim_conf["weather_forecast_method"], "list")


# ---------------------------------------------------------------------------
# Above one inverter: the standalone helpers
# ---------------------------------------------------------------------------


class TestPvCarryingInverters(unittest.TestCase):
    def pv(self, conf):
        return utils.pv_carrying_inverters(conf)

    def test_distinct_sorted_values(self):
        conf = {"number_of_inverters": 3, "pv_plant_inverter_index": [2, 0]}
        self.assertEqual(self.pv(conf), [0, 2])

    def test_pvlib_style_index_with_two_plants_on_one_inverter(self):
        conf = {"number_of_inverters": 2, "pv_plant_inverter_index": [0, 0, 1]}
        self.assertEqual(self.pv(conf), [0, 1])

    def test_one_inverter_does_not_read_the_index(self):
        self.assertEqual(self.pv({"number_of_inverters": 1, "pv_plant_inverter_index": "x"}), [])
        self.assertEqual(self.pv({}), [])

    def test_a_bad_index_is_a_bounded_value_error(self):
        huge = (10**5000, 10**400, 1e300, -(10**400), "a" * 10**5)
        bad_indexes = (None, "0,1", [True], [1.5], [-1], [2], [[0]], *([value] for value in huge))
        for case, bad in enumerate(bad_indexes):
            with self.subTest(case=case):
                conf = {"number_of_inverters": 2, "pv_plant_inverter_index": bad}
                with self.assertRaises(utils.InverterConfigError) as ctx:
                    self.pv(conf)
                self.assertLess(len(str(ctx.exception)), 300)
                self.assertIn("pv_plant_inverter_index", str(ctx.exception))

    def test_a_huge_inverter_count_costs_nothing(self):
        conf = {"number_of_inverters": 10**9, "pv_plant_inverter_index": [0, 999_999_999, 0]}
        result, peak = peak_memory(lambda: self.pv(conf))
        self.assertEqual(result, [0, 999_999_999])
        self.assertLess(peak, PEAK_LIMIT_BYTES)


class PerInverterHelpers(unittest.TestCase):
    """Common call shapes for the two helpers."""

    def align(self, payload, n_inverters, pv_inverters, **kwargs):
        func = utils.validate_and_align_per_inverter_pv_forecast
        return func(payload, n_inverters, pv_inverters, GRID, STEP_MINUTES, TZ, **kwargs)

    def pair(self, p50, p10, n_inverters, pv_inverters):
        func = utils.validate_and_align_per_inverter_pv_pair
        return func(p50, p10, n_inverters, pv_inverters, GRID, STEP_MINUTES, TZ)

    def assert_fault(self, result, *fragments):
        values, fault = result
        self.assertIsNone(values)
        self.assertIsInstance(fault, str)
        for fragment in fragments:
            self.assertIn(fragment, fault)
        self.assertLess(len(fault), 500)
        return fault


class TestPerInverterForecast(PerInverterHelpers):
    def test_nested_lists_are_accepted_and_cut_to_the_horizon(self):
        payload = [series(100, 6), [], series(300, 5)]
        aligned, fault = self.align(payload, 3, [0, 2])
        self.assertIsNone(fault)
        self.assertEqual(aligned, [series(100), [], series(300)])

    def test_list_of_mappings_is_aligned_onto_the_grid(self):
        payload = [stamped(100), {}, stamped(300)]
        aligned, fault = self.align(payload, 3, [0, 2])
        self.assertIsNone(fault)
        self.assertEqual(aligned[1], [])
        np.testing.assert_allclose(aligned[0], series(100))
        np.testing.assert_allclose(aligned[2], series(300))

    def test_an_empty_list_is_fine_for_a_non_pv_inverter_in_a_mapping_payload(self):
        aligned, fault = self.align([[], stamped(100)], 2, [1])
        self.assertIsNone(fault)
        self.assertEqual(aligned[0], [])
        self.assertEqual(len(aligned[1]), HORIZON)

    def test_a_pvlib_style_index_gives_the_same_inverters(self):
        pv = utils.pv_carrying_inverters(
            {"number_of_inverters": 2, "pv_plant_inverter_index": [0, 0, 1]}
        )
        aligned, fault = self.align([series(1), series(2)], 2, pv)
        self.assertIsNone(fault)
        self.assertEqual(aligned, [series(1), series(2)])

    def test_the_entry_for_a_non_pv_inverter_must_be_empty(self):
        self.assert_fault(
            self.align([series(1), series(2)], 2, [0]), "pv_power_forecast[1]", "no PV plant"
        )

    def test_the_entry_for_a_pv_inverter_must_not_be_empty(self):
        self.assert_fault(
            self.align([series(1), []], 2, [0, 1]), "pv_power_forecast[1]", "is empty"
        )
        self.assert_fault(self.align([[], {}], 2, [0, 1]), "pv_power_forecast[0]", "is empty")

    def test_the_outer_length_must_match(self):
        self.assert_fault(
            self.align([series(1), series(2)], 3, [0, 1]), "2 entries", "number_of_inverters"
        )
        self.assert_fault(
            self.align([series(1)] * 4, 3, [0, 1]), "4 entries", "number_of_inverters"
        )

    def test_a_flat_list_is_refused_whatever_its_length(self):
        self.assert_fault(self.align(series(1, 3), 2, [0, 1]), "3 entries")
        self.assert_fault(self.align(series(1, 2), 2, [0, 1]), "[0]", "flat list")

    def test_a_single_mapping_is_refused(self):
        self.assert_fault(self.align(stamped(1), 2, [0, 1]), "dict", "single mapping")

    def test_none_and_strings_are_refused(self):
        self.assert_fault(self.align(None, 2, [0, 1]), "NoneType")
        self.assert_fault(self.align("[[1],[2]]", 2, [0, 1]), "str")

    def test_lists_and_mappings_cannot_be_mixed(self):
        self.assert_fault(
            self.align([series(1), stamped(2)], 2, [0, 1]), "mixes lists and timestamped mappings"
        )

    def test_an_inner_list_shorter_than_the_horizon_is_refused(self):
        self.assert_fault(
            self.align([series(1), series(2, HORIZON - 1)], 2, [0, 1]),
            "pv_power_forecast[1]",
            f"{HORIZON - 1} values for {HORIZON} required steps",
        )

    def test_a_non_finite_value_names_the_inverter_and_the_position(self):
        for label, bad in (
            ("NaN", float("nan")),
            ("Inf", float("inf")),
            ("bool", True),
            ("str", "5"),
        ):
            with self.subTest(bad=label):
                inner = series(2)
                inner[2] = bad
                fault = self.assert_fault(
                    self.align([series(1), inner], 2, [0, 1]), "pv_power_forecast[1]", "position 2"
                )
                self.assertNotIn("pv_power_forecast[0]", fault)

    def test_a_non_finite_value_in_a_mapping_names_the_inverter_and_the_timestamp(self):
        mapping = stamped(2)
        mapping[STAMPS[1]] = float("nan")
        self.assert_fault(
            self.align([stamped(1), mapping], 2, [0, 1]), "pv_power_forecast[1]", STAMPS[1]
        )

    def test_a_bad_timestamp_is_a_fault_not_an_exception(self):
        self.assert_fault(
            self.align([stamped(1), {"not a time": 1.0}], 2, [0, 1]), "pv_power_forecast[1]"
        )

    def test_the_input_is_not_modified(self):
        payload = [series(1, 6), [], series(3, 6)]
        before = [list(entry) for entry in payload]
        self.align(payload, 3, [0, 2])
        self.assertEqual(payload, before)

    def test_a_huge_inverter_count_with_a_short_payload_is_a_fast_error(self):
        result, peak = peak_memory(lambda: self.align([series(1), series(2)], 10**9, [0, 1]))
        self.assert_fault(result, "2 entries", "1000000000")
        self.assertLess(peak, PEAK_LIMIT_BYTES)

    def test_the_pair_helper_also_ignores_a_huge_inverter_count(self):
        result, peak = peak_memory(
            lambda: self.pair([series(1), series(2)], [series(1), series(2)], 10**9, [0, 1])
        )
        self.assertLess(peak, PEAK_LIMIT_BYTES)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIn("2 entries", result[2])

    def test_a_huge_inner_list_gives_a_bounded_message(self):
        big = [1.0] * (10**6)
        big[-1] = float("nan")
        fault = self.assert_fault(
            self.align([big, series(2)], 2, [0, 1]), "pv_power_forecast[0]", "position 999999"
        )
        self.assertLess(len(fault), 500)
        # A value that is itself huge is not echoed in full either.
        huge_value = series(1)
        huge_value[3] = "x" * 10**6
        fault = self.assert_fault(
            self.align([huge_value, series(2)], 2, [0, 1]), "pv_power_forecast[0]"
        )
        self.assertLessEqual(len(fault), 400)
        # A huge entry where an empty one belongs is described by its length only.
        fault = self.assert_fault(
            self.align([series(1), big], 2, [0]), "pv_power_forecast[1]", "no PV plant"
        )
        self.assertLess(len(fault), 500)


class TestPerInverterPair(PerInverterHelpers):
    def test_valid_pair_of_lists(self):
        p50 = [series(100, 6), [], series(300)]
        p10 = [series(50, 6), [], series(150)]
        aligned50, aligned10, fault = self.pair(p50, p10, 3, [0, 2])
        self.assertIsNone(fault)
        self.assertEqual(aligned50, [series(100), [], series(300)])
        self.assertEqual(aligned10, [series(50), [], series(150)])

    def test_valid_pair_of_mappings(self):
        aligned50, aligned10, fault = self.pair(
            [stamped(100), stamped(300)], [stamped(50), stamped(150)], 2, [0, 1]
        )
        self.assertIsNone(fault)
        np.testing.assert_allclose(aligned50[1], series(300))
        np.testing.assert_allclose(aligned10[0], series(50))

    def assert_pair_fault(self, result, *fragments):
        p50, p10, fault = result
        self.assertIsNone(p50)
        self.assertIsNone(p10)
        self.assertIsInstance(fault, str)
        for fragment in fragments:
            self.assertIn(fragment, fault)
        self.assertLess(len(fault), 500)

    def test_a_missing_p50_is_refused(self):
        self.assert_pair_fault(self.pair(None, [series(1)], 1, [0]), "requires pv_power_forecast")

    def test_p10_must_have_the_p50_shape_per_inverter(self):
        p50 = [series(100), series(200)]
        cases = {
            "p10 flat": (series(1, 2), "pv_power_forecast_p10"),
            "p10 single mapping": (stamped(1), "pv_power_forecast_p10"),
            "p10 wrong outer length": ([series(1)] * 3, "3 entries"),
            "p10 none": (None, "pv_power_forecast_p10"),
            "p10 empty where the p50 has values": ([series(1), []], "pv_power_forecast_p10[1]"),
        }
        for label, (p10, fragment) in cases.items():
            with self.subTest(case=label):
                self.assert_pair_fault(self.pair(p50, p10, 2, [0, 1]), fragment)
        self.assert_pair_fault(
            self.pair([series(100), []], [series(1), series(2)], 2, [0]),
            "pv_power_forecast_p10[1]",
            "no PV plant",
        )
        self.assert_pair_fault(
            self.pair([series(100), series(200)], [series(1), []], 2, [0, 1]),
            "pv_power_forecast_p10[1]",
            "is empty",
        )

    def test_each_inverter_is_a_pair_of_the_same_kind(self):
        self.assert_pair_fault(
            self.pair([series(100), series(200)], [series(1), stamped(2)], 2, [0, 1]),
            "mixes lists and timestamped mappings",
        )
        self.assert_pair_fault(
            self.pair([series(100), stamped(200)], [series(1), stamped(2)], 2, [0, 1]),
            "mixes lists and timestamped mappings",
        )

    def test_the_inner_lengths_of_a_pair_must_match(self):
        self.assert_pair_fault(
            self.pair([series(100), series(200, 6)], [series(1), series(2, 5)], 2, [0, 1]),
            "inverter 1",
            "length mismatch",
        )

    def test_an_inner_pair_that_does_not_cover_the_horizon_is_refused(self):
        self.assert_pair_fault(
            self.pair([series(100), series(200, 3)], [series(1), series(2, 3)], 2, [0, 1]),
            "inverter 1",
            "do not cover the full forecast horizon",
        )

    def test_a_non_finite_p10_value_names_the_inverter(self):
        bad = series(2)
        bad[1] = float("inf")
        self.assert_pair_fault(
            self.pair([series(100), series(200)], [series(1), bad], 2, [0, 1]),
            "inverter 1",
            "pv_power_forecast_p10",
            "position 1",
        )

    def test_mismatched_mapping_timelines_are_refused(self):
        other = {STAMPS[0]: 1.0, STAMPS[1]: 2.0, STAMPS[2]: 3.0, "2026-06-26T12:00:00+02:00": 4.0}
        self.assert_pair_fault(
            self.pair([stamped(100), stamped(200)], [stamped(1), other], 2, [0, 1]),
            "inverter 1",
            "timestamp mismatch",
        )

    def test_the_wrong_outer_length_is_refused_before_anything_else(self):
        self.assert_pair_fault(self.pair([series(1)], [series(1)], 2, [0, 1]), "1 entries")

    def test_a_pvlib_style_index(self):
        pv = utils.pv_carrying_inverters(
            {"number_of_inverters": 2, "pv_plant_inverter_index": [0, 0, 1]}
        )
        p50, p10, fault = self.pair([series(100), series(200)], [series(1), series(2)], 2, pv)
        self.assertIsNone(fault)
        self.assertEqual((p50, p10), ([series(100), series(200)], [series(1), series(2)]))


class TestWiringIntoTreatRuntimeParams(validity._PinnedNow):
    """The helpers are called from the forecast list block above one inverter. That branch is
    behind the refusal of more than one inverter, so it is exercised by patching the inverter
    count the block reads, which is all the refusal protects."""

    async def treat_with(self, runtimeparams, n_inverters=2, index=(0, 1)):
        params = await validity._build_params()
        params["plant_conf"]["number_of_inverters"] = n_inverters
        params["plant_conf"]["pv_plant_inverter_index"] = list(index)
        params_json = orjson.dumps(params).decode("utf-8")
        rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, validity.logger)
        with unittest.mock.patch.object(utils, "require_usable_inverter_config"):
            treated, _, optim_conf, _ = await utils.treat_runtimeparams(
                runtimeparams,
                params_json,
                rh_conf,
                optim_conf,
                plant_conf,
                "dayahead-optim",
                validity.logger,
                validity.EMHASS_CONF,
            )
        return orjson.loads(treated), optim_conf

    async def test_a_valid_nested_payload_reaches_passed_data(self):
        n = self.horizon
        params, optim_conf = await self.treat_with(
            {"pv_power_forecast": [series(1, n), series(2, n)]}
        )
        self.assertEqual(params["passed_data"]["pv_power_forecast"], [series(1, n), series(2, n)])
        self.assertEqual(optim_conf["weather_forecast_method"], "list")

    async def test_a_flat_list_fails_closed_with_the_unpaired_sentinel(self):
        with self.assertLogs(validity.logger, level="ERROR") as captured:
            params, optim_conf = await self.treat_with(
                {"pv_power_forecast": series(1, self.horizon)}
            )
        self.assertIsNone(params["passed_data"]["pv_power_forecast"])
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertTrue(validity._errors_mentioning(captured, "number_of_inverters"))

    async def test_a_nested_pair_reaches_passed_data(self):
        n = self.horizon
        params, _ = await self.treat_with(
            {
                "pv_power_forecast": [series(1, n), series(2, n)],
                "pv_power_forecast_p10": [series(0, n), series(1, n)],
            }
        )
        self.assertEqual(params["passed_data"]["pv_power_forecast"], [series(1, n), series(2, n)])
        self.assertEqual(
            params["passed_data"]["pv_power_forecast_p10"], [series(0, n), series(1, n)]
        )

    async def test_a_bad_pair_fails_closed_with_the_pair_sentinel(self):
        n = self.horizon
        with self.assertLogs(validity.logger, level="ERROR"):
            params, optim_conf = await self.treat_with(
                {
                    "pv_power_forecast": [series(1, n), series(2, n)],
                    "pv_power_forecast_p10": series(0, n),
                }
            )
        self.assertIsNone(params["passed_data"]["pv_power_forecast"])
        self.assertEqual(params["passed_data"]["pv_power_forecast_p10"], [])
        self.assertEqual(optim_conf["weather_forecast_method"], "list")

    async def test_an_unusable_plant_index_fails_closed(self):
        n = self.horizon
        with self.assertLogs(validity.logger, level="ERROR"):
            params, _ = await self.treat_with(
                {"pv_power_forecast": [series(1, n), series(2, n)]}, index=(0, 7)
            )
        self.assertIsNone(params["passed_data"]["pv_power_forecast"])


# ---------------------------------------------------------------------------
# Above one inverter: the Forecast side
# ---------------------------------------------------------------------------


class MultiInverterBase(PerPlantBase):
    def set_inverters(self, n_inverters, index):
        self.plant_conf["number_of_inverters"] = n_inverters
        self.plant_conf["pv_plant_inverter_index"] = list(index)

    def split_plants(self, p_pv):
        return self.fcst.get_pv_plant_split(p_pv)

    def expected_count(self):
        return self.fcst._expected_pv_plant_count()

    def n_steps(self):
        return len(self.fcst.forecast_dates_tz)

    def passed(self, p50, p10=None):
        self.fcst.params = {
            "passed_data": {
                "pv_power_forecast": p50,
                "pv_power_forecast_p10": p10,
                "prediction_horizon": None,
            }
        }


class TestListAboveOneInverter(MultiInverterBase):
    async def test_two_pv_inverters_give_the_sum_and_two_plants(self):
        self.set_inverters(2, [0, 1])
        n = self.n_steps()
        first, second = series(100, n), series(300, n)
        self.passed([first, second])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame, "the nested payload must give a frame")
        self.assertEqual(list(frame.columns), ["yhat", "yhat_plant0", "yhat_plant1"])
        np.testing.assert_allclose(frame["yhat"], np.add(first, second))
        np.testing.assert_allclose(frame["yhat_plant0"], first)
        np.testing.assert_allclose(frame["yhat_plant1"], second)
        self.assertEqual(len(frame), n)

        p_pv = self.fcst.get_power_from_weather(frame)
        plants = self.split_plants(p_pv)
        self.assertIsNotNone(plants)
        self.assertEqual(list(plants.columns), ["P_PV_plant0", "P_PV_plant1"])
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12)
        np.testing.assert_allclose(plants["P_PV_plant0"], first, rtol=1e-12)
        np.testing.assert_allclose(plants["P_PV_plant1"], second, rtol=1e-12)

    async def test_only_the_pv_inverters_are_read_and_the_order_follows_the_inverters(self):
        """Inverter 1 has no PV plant and an empty entry; plant 0 is inverter 0, plant 1 is
        inverter 2. A pvlib-style index with two plants on inverter 0 gives the same two."""
        self.set_inverters(3, [0, 0, 2])
        n = self.n_steps()
        self.passed([series(100, n), [], series(300, n)])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame)
        self.assertEqual(list(frame.columns), ["yhat", "yhat_plant0", "yhat_plant1"])
        np.testing.assert_allclose(frame["yhat_plant1"], series(300, n))
        self.assertEqual(self.expected_count(), 2)

    async def test_one_pv_inverter_of_two_gives_no_plant_data(self):
        self.set_inverters(2, [1])
        n = self.n_steps()
        self.passed([[], series(300, n)])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame)
        self.assertEqual(list(frame.columns), ["yhat"])
        np.testing.assert_allclose(frame["yhat"], series(300, n))
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertIsNone(self.fcst.pv_plant_raw)
        self.assertIsNone(self.split_plants(p_pv))
        self.assertEqual(self.expected_count(), 1)

    async def test_the_plant_data_does_not_alias_the_aggregate(self):
        """The negative clip writes into ``yhat``; the plant columns must not follow it."""
        self.set_inverters(2, [0, 1])
        n = self.n_steps()
        first, second = series(100, n), series(300, n)
        first[0], second[0] = -30.0, 10.0
        self.passed([first, second])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame)
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertEqual(p_pv.iloc[0], 0.0)
        raw = self.fcst.pv_plant_raw
        self.assertIsNotNone(raw)
        self.assertEqual(raw["yhat_plant0"].iloc[0], -30.0)
        plants = self.split_plants(p_pv)
        self.assertTrue((plants >= 0).all().all())
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, atol=1e-9)

    async def test_the_p10_companion_is_blended_into_each_series(self):
        self.set_inverters(2, [0, 1])
        self.optim_conf["weather_forecast_pv_quantile_bias"] = 0.5
        n = self.n_steps()
        self.passed([[100.0] * n, [200.0] * n], [[50.0] * n, [100.0] * n])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame["yhat_plant0"], 75.0)
        np.testing.assert_allclose(frame["yhat_plant1"], 150.0)
        np.testing.assert_allclose(frame["yhat"], 225.0)

    async def test_bias_zero_keeps_the_p50_series(self):
        self.set_inverters(2, [0, 1])
        self.optim_conf["weather_forecast_pv_quantile_bias"] = 0.0
        n = self.n_steps()
        self.passed([[100.0] * n, [200.0] * n], [[50.0] * n, [100.0] * n])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame["yhat"], 300.0)

    async def test_a_payload_that_does_not_fit_gives_no_frame(self):
        self.set_inverters(2, [0, 1])
        n = self.n_steps()
        cases = {
            "flat list": (series(100, n), "needs one entry per inverter"),
            "one entry": ([series(100, n)], "needs one entry per inverter"),
            "three entries": ([series(100, n)] * 3, "needs one entry per inverter"),
            "short inner list": ([series(100, n), series(100, n - 1)], "is not long enough"),
            "non-finite value": (
                [series(100, n), [float("nan")] * n],
                "not a series of finite numbers",
            ),
            "not a list": ({"a": 1}, "needs one entry per inverter"),
            "none": (None, "needs one entry per inverter"),
        }
        for label, (payload, fragment) in cases.items():
            with self.subTest(case=label):
                self.passed(payload)
                with self.assertLogs(logger, level="ERROR") as captured:
                    frame = await self.fcst.get_weather_forecast(method="list")
                self.assertIsNone(frame)
                assert_error_says(self, captured, fragment)

    async def test_a_p10_that_does_not_fit_gives_no_frame(self):
        self.set_inverters(2, [0, 1])
        n = self.n_steps()
        p50 = [series(100, n), series(200, n)]
        for label, (p10, fragment) in {
            "wrong outer length": ([series(1, n)], "pv_power_forecast_p10 needs one entry"),
            "short inner": ([series(1, n), series(1, n - 1)], "length mismatch for inverter 1"),
            "flat": (series(1, n), "pv_power_forecast_p10 needs one entry"),
        }.items():
            with self.subTest(case=label):
                self.passed(p50, p10)
                with self.assertLogs(logger, level="ERROR") as captured:
                    frame = await self.fcst.get_weather_forecast(method="list")
                self.assertIsNone(frame)
                assert_error_says(self, captured, fragment)

    async def test_an_unusable_plant_index_gives_no_frame(self):
        self.set_inverters(2, [0, 9])
        n = self.n_steps()
        self.passed([series(100, n), series(200, n)])
        with self.assertLogs(logger, level="ERROR") as captured:
            self.assertIsNone(await self.fcst.get_weather_forecast(method="list"))
        assert_error_says(self, captured, "pv_plant_inverter_index")

    async def test_an_index_with_no_inverter_gives_no_frame(self):
        self.set_inverters(2, [])
        self.passed([[], []])
        with self.assertLogs(logger, level="ERROR") as captured:
            self.assertIsNone(await self.fcst.get_weather_forecast(method="list"))
        assert_error_says(self, captured, "No inverter has a PV plant")

    async def test_one_inverter_is_unchanged(self):
        """Pin: with one inverter the list method reads one flat series and has no plant data,
        even if the plant index is left in the configuration."""
        self.set_inverters(1, [0, 1])
        n = self.n_steps()
        self.passed(series(100, n))
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertEqual(list(frame.columns), ["yhat"])
        np.testing.assert_allclose(frame["yhat"], series(100, n))
        self.fcst.get_power_from_weather(frame)
        self.assertIsNone(self.fcst.pv_plant_raw)
        self.assertEqual(self.expected_count(), 1)

    async def test_the_zero_kwp_branch_keeps_the_plants_consistent(self):
        self.set_inverters(2, [0, 1])
        self.retrieve_hass_conf["solar_forecast_kwp"] = 0
        n = self.n_steps()
        self.passed([series(100, n), series(300, n)])
        frame = await self.fcst.get_weather_forecast(method="list")
        self.assertIsNotNone(frame)
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertTrue((p_pv == 0).all())
        plants = self.split_plants(p_pv)
        self.assertIsNotNone(plants)
        self.assertTrue((plants == 0).all().all())


class TestCsvAboveOneInverter(MultiInverterBase):
    def write_csv(self, columns):
        n = len(self.fcst.forecast_dates)
        stamps = [ts.isoformat() for ts in self.fcst.forecast_dates]
        rows = [",".join([stamps[i], *(str(column[i]) for column in columns)]) for i in range(n)]
        (self.data_dir / "pv.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    async def test_two_pv_inverters_read_two_columns(self):
        self.set_inverters(2, [0, 1])
        n = len(self.fcst.forecast_dates)
        first, second = series(100, n), series(300, n)
        self.write_csv([first, second])
        frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNotNone(frame)
        self.assertEqual(set(frame.columns), {"yhat", "yhat_plant0", "yhat_plant1"})
        np.testing.assert_allclose(frame["yhat"], np.add(first, second))
        np.testing.assert_allclose(frame["yhat_plant0"], first)
        np.testing.assert_allclose(frame["yhat_plant1"], second)
        self.assertTrue(frame.index.equals(self.fcst.forecast_dates))
        p_pv = self.fcst.get_power_from_weather(frame)
        plants = self.split_plants(p_pv)
        self.assertIsNotNone(plants)
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12)
        self.assertEqual(self.expected_count(), 2)

    async def test_three_pv_inverters_of_four(self):
        self.set_inverters(4, [0, 0, 1, 3])
        n = len(self.fcst.forecast_dates)
        self.write_csv([series(1, n), series(2, n), series(3, n)])
        frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(
            frame["yhat"], np.add(np.add(series(1, n), series(2, n)), series(3, n))
        )
        self.assertIn("yhat_plant2", frame.columns)

    async def test_one_pv_inverter_of_two_reads_one_column(self):
        self.set_inverters(2, [1])
        n = len(self.fcst.forecast_dates)
        self.write_csv([series(100, n)])
        frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNotNone(frame)
        self.assertEqual(list(frame.columns), ["yhat"])
        np.testing.assert_allclose(frame["yhat"], series(100, n))

    async def test_one_inverter_reads_one_column_as_before(self):
        self.set_inverters(1, [0, 1])
        n = len(self.fcst.forecast_dates)
        self.write_csv([series(100, n)])
        frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertEqual(list(frame.columns), ["yhat"])
        self.fcst.get_power_from_weather(frame)
        self.assertIsNone(self.fcst.pv_plant_raw)
        self.assertEqual(self.expected_count(), 1)

    async def test_a_file_shorter_than_the_grid_is_still_refused(self):
        self.set_inverters(2, [0, 1])
        n = len(self.fcst.forecast_dates)
        self.write_csv([series(100, n), series(300, n)])
        lines = (self.data_dir / "pv.csv").read_text(encoding="utf-8").splitlines()
        (self.data_dir / "pv.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertLogs(logger, level="ERROR") as captured:
            frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNone(frame)
        assert_error_says(self, captured, "not long enough")

    async def test_a_file_with_the_wrong_number_of_columns_is_refused(self):
        """Two PV inverters need a timestamp and two value columns, nothing else."""
        n = len(self.fcst.forecast_dates)
        self.set_inverters(2, [0, 1])
        for columns in (
            [series(100, n)],
            [series(100, n), series(200, n), series(300, n)],
        ):
            with self.subTest(value_columns=len(columns)):
                self.write_csv(columns)
                with self.assertLogs(logger, level="ERROR") as captured:
                    frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
                self.assertIsNone(frame)
                text = "\n".join(captured.output)
                self.assertIn(f"has {len(columns) + 1} columns", text)
                self.assertIn("3 are expected", text)
                self.assertIn("pv_plant_inverter_index", text)

    async def test_the_column_count_follows_the_pv_inverters_not_the_inverters(self):
        """Four inverters but two with PV: three columns in all; a column per inverter fails."""
        n = len(self.fcst.forecast_dates)
        self.set_inverters(4, [0, 3])
        self.write_csv([series(1, n), series(2, n)])
        frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNotNone(frame)
        self.assertIn("yhat_plant1", frame.columns)
        self.write_csv([series(1, n)] * 4)
        with self.assertLogs(logger, level="ERROR") as captured:
            self.assertIsNone(await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv"))
        assert_error_says(self, captured, "has 5 columns but 3 are expected")

    async def test_an_unusable_plant_index_gives_no_frame(self):
        self.set_inverters(2, [0, 9])
        n = len(self.fcst.forecast_dates)
        self.write_csv([series(100, n), series(300, n)])
        with self.assertLogs(logger, level="ERROR") as captured:
            frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNone(frame)
        assert_error_says(self, captured, "pv_plant_inverter_index")

    async def test_an_index_with_no_inverter_gives_no_frame(self):
        self.set_inverters(2, [])
        n = len(self.fcst.forecast_dates)
        self.write_csv([series(100, n)])
        with self.assertLogs(logger, level="ERROR") as captured:
            frame = await self.fcst.get_weather_forecast(method="csv", csv_path="pv.csv")
        self.assertIsNone(frame)
        assert_error_says(self, captured, "No inverter has a PV plant")


class TestSolcastRooftopCount(MultiInverterBase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.two_plant_pvlib_conf()
        self.set_grid("15min")
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "key"

    async def fetch(self, roof_ids):
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = roof_ids
        with aioresponses() as mocked:
            result = await self.fcst.get_weather_forecast(method="solcast")
            return result, mocked

    async def test_a_count_that_differs_from_the_plants_is_an_error_before_any_request(self):
        self.set_inverters(2, [0, 1])
        self.fcst._solcast_rate_limit_ok = unittest.mock.Mock(return_value=True)
        for roof_ids in ("a,b,c", "a", "a b c d"):
            with self.subTest(roof_ids=roof_ids):
                with self.assertLogs(logger, level="ERROR") as captured:
                    result, mocked = await self.fetch(roof_ids)
                self.assertIs(result, False)
                self.assertEqual(mocked.requests, {})
                self.fcst._solcast_rate_limit_ok.assert_not_called()
                text = "\n".join(captured.output)
                self.assertIn("solcast_rooftop_id", text)
                self.assertIn("pv_module_model", text)

    async def test_an_index_of_the_wrong_length_is_an_error_even_with_a_cache(self):
        """The index needs one inverter number per plant, checked before the cache is read."""
        self.set_inverters(2, [0])
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "a,b"
        self.write_cache(self.cache_frame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0}))
        before = self.cache_path.read_bytes()
        with aioresponses() as mocked:
            with self.assertLogs(logger, level="ERROR") as captured:
                result = await self.fcst.get_weather_forecast(method="solcast")
            self.assertEqual(mocked.requests, {})
        self.assertIs(result, False)
        self.assertEqual(self.cache_path.read_bytes(), before)
        assert_error_says(
            self, captured, "pv_plant_inverter_index has 1 entries", "pv_module_model"
        )

    async def test_a_rooftop_id_that_is_not_text_is_left_to_the_fetch_path(self):
        self.set_inverters(2, [0, 1])
        for value in (None, 5, ["a", "b"]):
            with self.subTest(value=value):
                self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = value
                self.assertTrue(self.fcst._solcast_plant_count_agrees())

    async def test_a_cached_forecast_does_not_skip_the_count_check(self):
        """A fresh cache with three plant columns must not be served for two PV plants, and a
        refused check neither fetches nor drops the cache."""
        self.set_inverters(2, [0, 1])
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "a,b,c"
        self.write_cache(
            self.cache_frame(
                {"yhat": 60.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0, "yhat_plant2": 30.0}
            )
        )
        before = self.cache_path.read_bytes()
        self.fcst._solcast_rate_limit_ok = unittest.mock.Mock(return_value=True)
        with aioresponses() as mocked:
            with self.assertLogs(logger, level="ERROR") as captured:
                result = await self.fcst.get_weather_forecast(method="solcast")
            self.assertEqual(mocked.requests, {})
        self.assertIs(result, False)
        self.fcst._solcast_rate_limit_ok.assert_not_called()
        self.assertTrue(self.cache_path.exists(), "the check must not drop the cache")
        self.assertEqual(self.cache_path.read_bytes(), before)
        text = "\n".join(captured.output)
        self.assertIn("solcast_rooftop_id", text)
        self.assertIn("pv_module_model", text)

    async def test_a_cached_forecast_with_an_equal_count_is_still_served(self):
        self.set_inverters(2, [0, 1])
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "a,b"
        self.write_cache(self.cache_frame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0}))
        with aioresponses() as mocked:
            frame = await self.fcst.get_weather_forecast(method="solcast")
            self.assertEqual(mocked.requests, {})
        self.assertIsNot(frame, False)
        self.assertIn("yhat_plant1", frame.columns)

    async def test_one_inverter_serves_a_cache_whatever_the_counts(self):
        """Pin: at one inverter the cache is read first and nothing is compared."""
        self.set_inverters(1, [0, 1])
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "a,b,c"
        self.write_cache(self.cache_frame({"yhat": 30.0}))
        with aioresponses() as mocked:
            frame = await self.fcst.get_weather_forecast(method="solcast")
            self.assertEqual(mocked.requests, {})
        self.assertIsNot(frame, False)
        self.assertIsNotNone(frame)

    async def test_an_equal_count_fetches_and_gives_plants(self):
        self.set_inverters(2, [0, 1])
        frame = await self.fetch_solcast(
            {"a": self.solcast_payload(1.0, 2.0, 3.0), "b": self.solcast_payload(0.5, 1.0, 1.5)}
        )
        self.assertIsNot(frame, False)
        self.assertIsNotNone(frame)
        self.assertIn("yhat_plant1", frame.columns)

    async def test_one_inverter_does_not_compare_the_counts(self):
        """Pin: with one inverter three rooftops and two plants fetch as before."""
        self.set_inverters(1, [0, 1])
        frame = await self.fetch_solcast(
            {
                "a": self.solcast_payload(1.0, 2.0, 3.0),
                "b": self.solcast_payload(0.5, 1.0, 1.5),
                "c": self.solcast_payload(0.5, 1.0, 1.5),
            }
        )
        self.assertIsNot(frame, False)
        self.assertIsNotNone(frame)

    async def test_a_single_plant_model_counts_as_one_plant(self):
        self.plant_conf["pv_module_model"] = self.plant_conf["pv_module_model"][0]
        self.set_inverters(2, [0])
        with self.assertLogs(logger, level="ERROR") as captured:
            result, mocked = await self.fetch("a,b")
        self.assertIs(result, False)
        self.assertEqual(mocked.requests, {})
        assert_error_says(self, captured, "solcast_rooftop_id")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
