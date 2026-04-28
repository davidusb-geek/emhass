# Rolling-horizon control with naive-mpc-optim

> **Type:** How-To Guide — task-oriented, follow when you need a live, continuously-updated optimization plan instead of a static day-ahead schedule.

The `dayahead-optim` action computes a single 24 h schedule once per day. For systems where forecasts and state change throughout the day (especially battery SOC, EV state, dynamic prices), you want **Model Predictive Control**: re-run the optimization on a rolling window every N minutes, using the latest measurements as the new initial state.

EMHASS implements this with the `naive-mpc-optim` action. This page walks through wiring it up.

## Scenario

| Component | Value |
|-----------|-------|
| PV | 5 kWp |
| Battery | 5 kWh nominal, defaults: `battery_minimum_state_of_charge: 0.3`, `battery_maximum_state_of_charge: 0.9` |
| Deferrable loads | as previous tutorials |
| MPC re-run cadence | every 30 minutes |
| Prediction horizon | 24 h (number of timesteps = `24 × 60 / optimization_time_step`; with the default 30-minute step, that is 48) |

## Configuration

In addition to the parameters from [Basic — PV + Battery](basic_pv_battery.md), MPC needs runtime parameters per call. The following keys go into the request body, not the static config:

| Runtime key | Value |
|-------------|-------|
| `prediction_horizon` | number of timesteps to plan ahead |
| `soc_init` | current battery SOC, fraction of nominal capacity (read from your battery sensor) |
| `soc_final` | end-of-horizon SOC target (see note below) |
| `def_total_hours` | remaining required operating hours per deferrable load |

For the full list of runtime keys, see [Passing data](../passing_data.md).

```{note}
`soc_init` and `soc_final` are read from `runtimeparams` independently. If one is omitted, EMHASS substitutes `battery_target_state_of_charge` (default `0.6`) for that single value; it does **not** mirror the passed value onto the missing one. Always pass both explicitly. Two production-tested rolling-MPC patterns: `soc_final = soc_init` (current SOC for both, neutral trailing edge) or `soc_final = 0` (with a 24 h horizon re-run every 30 min, the deadline is always 24 h away and never reached, so this behaves the same in practice). For a hard end-of-horizon target, pass that value and extend `prediction_horizon` so the deadline sits at the real point in time.
```

## Run

REST (recommended for HA `rest_command` automation):

```yaml
rest_command:
  emhass_mpc:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers:
      content-type: application/json
    payload: >
      {
        "prediction_horizon": 48,
        "soc_init": {{ states('sensor.battery_soc') | float / 100 }},
        "def_total_hours": [
          {{ states('sensor.water_heater_remaining_hours') }},
          {{ states('sensor.pool_pump_remaining_hours') }}
        ]
      }
```

(`soc_final` is intentionally omitted; see the note above. Adjust the literal `48` if your `optimization_time_step` is not 30 minutes.)

Trigger it every 30 minutes:

```yaml
automation:
  - alias: EMHASS MPC every 30 min
    trigger:
      - platform: time_pattern
        minutes: "/30"
    action:
      - service: rest_command.emhass_mpc
      - service: shell_command.emhass_publish_data
```

For the matching `shell_command.emhass_publish_data`, see [Automations](../automations.md).

## Output

After each MPC run, EMHASS publishes the same sensors as `dayahead-optim` (`sensor.p_deferrable0`, `sensor.soc_optim`, etc.), but the schedule covering the prediction horizon replaces the previous one. Your HA automations follow the current state of `sensor.p_deferrable*` to switch real loads on/off; they don't care whether the underlying schedule came from `dayahead-optim` or `naive-mpc-optim`.

The plan-cycle behavior with a 30-minute re-run cadence and a 24 h horizon: at minute 0 the optimizer plans 24 h forward; at minute 30 it plans 24 h forward from the new "now"; at minute 60 again; and so on. Each plan **partially overlaps** the previous one for the next 23.5 h, but the new plan reflects the latest SOC, latest `def_total_hours`, latest forecast.

## Interpretation

- MPC's main advantage over day-ahead is **resilience to forecast error**: when actual PV generation diverges from the morning forecast, the next MPC iteration corrects course immediately.
- MPC's main cost is solver time × frequency. With CVXPY/HiGHS the solve is typically 0.1 – 0.5 s; running every 30 min is well within budget. Running every 1 min is overkill for most homes. See [Good Practices](good_practices.md).
- For deferrable loads with strict windows (EV must charge by 07:00), pass `start_timesteps_of_each_deferrable_load` / `end_timesteps_of_each_deferrable_load` per load in the runtime payload. See [EV walkthrough](ev.md).

## See also

- Tutorial: [Basic — PV + Battery](basic_pv_battery.md) (the static day-ahead version)
- Reference: [Passing data](../passing_data.md) for the full runtime params list
- Reference: [Automations](../automations.md) for `shell_command` / `rest_command` patterns
- How-to: [Heat-pump walkthrough](heat_pump_walkthrough.md)
- How-to: [EV walkthrough](ev.md)
- Explanation: [Good Practices](good_practices.md)
- Explanation: [Advanced math model](../advanced_math_model.md)
