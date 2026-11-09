#!/usr/bin/env python
"""Two inverters from a real optimization to the published sensors (#1150).

dayahead-optim saves the plan, publish-data reads it back and posts with entity save,
then publish_json replays every saved entity the way continual publish does. Inverter 0
is a hybrid with the battery and curtailment off (so it has a spill column), inverter 1
is PV-only with curtailment on.
"""

import copy
import json

import orjson

from emhass import command_line
from tests.fixture_clock import fixture_clock
from tests.test_multi_inverter_actions import (
    COSTFUN,
    MultiInverterActionBase,
    logger,
)

N = 48


class TestTwoInvertersOptimizeThenPublish(MultiInverterActionBase):
    async def input_dict(self, action, params, runtime):
        """Build the input data dict for an action from the given params and runtime data."""
        params = copy.deepcopy(params)
        params["passed_data"] = runtime
        return await command_line.set_input_data_dict(
            self.emhass_conf,
            COSTFUN,
            orjson.dumps(params).decode("utf-8"),
            orjson.dumps(runtime).decode("utf-8"),
            action,
            logger,
            get_data_from_file=True,
        )

    @staticmethod
    def capture(input_data_dict):
        """Wrap rh.post_data to record each POST body by entity_id; return that dict."""
        posts: dict[str, dict] = {}
        real = input_data_dict["rh"].post_data

        async def post_data(*args, **kwargs):
            """Call the real post_data and record the body it built."""
            result = await real(*args, **kwargs)
            posts[kwargs.get("entity_id", args[2] if len(args) > 2 else None)] = result[1]
            return result

        input_data_dict["rh"].post_data = post_data
        return posts

    @fixture_clock()
    async def test_optimize_publish_and_replay(self):
        """Optimize two inverters for real, publish the plan, then replay every saved entity.

        Checks the per-inverter sensors, the battery's inverter attribute in the posts and the
        metadata, and that a continual publish posts the same attribute keys again.
        """
        params = await self.params(
            plant={
                "compute_curtailment": [False, True],
                "maximum_power_to_grid": 0,
            }
        )
        runtime = self.runtime(N, [9000.0, 2000.0])
        res = await command_line.dayahead_forecast_optim(
            await self.input_dict("dayahead-optim", params, runtime), logger
        )
        self.assertTrue((res["optim_status"] == "Optimal").all())
        for column in ("P_PV_spill_0", "P_PV_curtailment_1"):
            self.assertIn(column, res.columns)

        publish = await self.input_dict("publish-data", params, runtime)
        posts = self.capture(publish)
        frame = await command_line.publish_data(publish, logger, entity_save=True)

        per_inverter = {e for e in posts if e[-1].isdigit() and "_inverter" in e}
        self.assertEqual(
            per_inverter,
            {
                "sensor.p_pv_forecast_inverter0",
                "sensor.p_pv_forecast_inverter1",
                "sensor.p_pv_curtailment_inverter1",
                "sensor.p_pv_spill_inverter0",
                "sensor.p_hybrid_inverter_inverter0",
                "sensor.p_hybrid_inverter_inverter1",
            },
        )
        for column in ("P_PV_0", "P_PV_1", "P_PV_curtailment_1", "P_PV_spill_0"):
            self.assertIn(column, frame.columns)
        self.assertGreater(float(posts["sensor.p_pv_spill_inverter0"]["state"]), 0.0)
        for entity in ("sensor.p_batt_forecast", "sensor.soc_batt_forecast"):
            self.assertEqual(posts[entity]["attributes"]["inverter"], 0)
        self.assertNotIn("inverter", posts["sensor.p_pv_forecast_inverter0"]["attributes"])

        entities = self.emhass_conf["data_path"] / "entities"
        metadata = json.loads((entities / "metadata.json").read_text())
        self.assertEqual(metadata["sensor.p_batt_forecast"]["attributes"], {"inverter": 0})
        self.assertNotIn("attributes", metadata["sensor.p_pv_spill_inverter0"])

        # continual publish: every saved entity posts the same attribute keys again
        reader = await self.input_dict("publish-data", params, runtime)
        replays = self.capture(reader)
        for path in sorted(entities.glob("sensor.*.json")):
            await command_line.publish_json(path.name, reader, entities, logger)
        self.assertEqual(replays.keys(), posts.keys())
        for entity, body in posts.items():
            self.assertEqual(
                replays[entity]["attributes"].keys(), body["attributes"].keys(), entity
            )
        self.assertEqual(replays["sensor.soc_batt_forecast"]["attributes"]["inverter"], 0)
