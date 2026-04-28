# Good Practices

> **Type:** Explanation — understanding-oriented. Hard-learned wisdom about EMHASS that is not obvious from the parameter reference.

This page collects insights that come from running EMHASS in production over months: what matters, what doesn't, and what surprises new users. Treat it as the *why* behind several recommendations scattered across the tutorials and how-to guides.

## 1. Forecast quality dominates parameter tuning

A common newcomer mistake is to spend hours tuning battery parameters (`battery_minimum_state_of_charge`, `battery_charge_power_max`, `battery_charge_efficiency`, …) while leaving the PV forecast and load forecast at defaults.

The dominant factor in real-world cost-minimization is the **gap between forecasted and actual** PV power, load power, and price. A 10% improvement in PV forecast accuracy typically yields more cost savings than retuning battery parameters from default to "perfect".

In practice:
- The default PV forecast method is `open-meteo` (free, no API key, configured as `weather_forecast_method: open-meteo`). Community feedback in EMHASS discussions repeatedly favors **Solcast** (free tier: 10 calls/day, `weather_forecast_method: solcast`) when accuracy matters most. **Forecast.Solar** is another free option (`weather_forecast_method: solar.forecast`). EMHASS also supports a clearoutside.com scraper (`weather_forecast_method: scrapper`) as a fallback some users still prefer.
- Load forecast 1-day-persistence is fine for routine households. It breaks on holiday weekends. The [ML Forecaster](../mlforecaster.md) helps if you have ≥ 1 month of history.
- Dynamic price forecasts must come from the tariff provider (Tibber, aWATTar, Octopus, Stromee, Nordpool, etc.) via runtime payload; there is no useful default for these.

Before tuning anything else, **measure your forecast errors** for at least a week and address the largest one. The [HA forum thread](https://community.home-assistant.io/t/emhass-an-energy-management-for-home-assistant/338126) has many user reports comparing forecast methods.

## 2. Timestep alignment

EMHASS internally works with a fixed `optimization_time_step` (default: 30 minutes). Three numbers must agree:

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `optimization_time_step` | 30 min | EMHASS internal |
| HA sensor publish interval | 5 min | should be ≤ `optimization_time_step` |
| Forecast list length | (matches horizon) | `prediction_horizon × optimization_time_step / forecast_step` items |

If your `prediction_horizon` is `N` and `optimization_time_step` is 30 minutes, lists like `pv_power_forecast` must be **`N` items long** (one per timestep, not hourly). EMHASS does not auto-resample. A length mismatch causes silent truncation or padding with zeros, which then poisons the optimization.

Concrete example: a 24-hour horizon at 30-minute step = 48 items per list; at 15-minute step = 96 items.

## 3. SOC convention: fraction of nominal capacity

EMHASS expresses SOC as a fraction `[0.0, 1.0]` of the **nominal** battery capacity (`battery_nominal_energy_capacity`, `Enom`). The bounds `battery_minimum_state_of_charge` (default 0.3) and `battery_maximum_state_of_charge` (default 0.9) are operational *limits* the optimizer respects; they do **not** rescale the reported SOC value.

So a published `sensor.soc_optim = 0.45` means 45% of nominal capacity, regardless of what the bounds are. This matches what your battery management system reports (assuming it also uses 0..100% of nominal). No conversion is needed when comparing the two.

Source: `optimization.py` constraints `min_energy = battery_minimum_state_of_charge × cap`, `max_energy = battery_maximum_state_of_charge × cap`.

If your downstream automation or display does need a different convention (e.g. percentage of *usable* range), apply the transform yourself in a HA template. Don't assume EMHASS already did it.

## 4. `soc_init` and `soc_final` defaults are forgiving

For day-ahead optimization, setting `soc_final` to a desired end-of-day SOC (e.g. 0.6) ensures you don't end the day empty. EMHASS uses `battery_target_state_of_charge` (default 0.6) as the fallback when neither `soc_init` nor `soc_final` is passed.

For rolling MPC, the often-cited concern is that a fixed `soc_final` reserves capacity at the trailing edge of every horizon and biases the optimizer toward conservative mid-day behavior. The EMHASS code helps here for the simple case: if you pass only `soc_init` at runtime (and omit `soc_final`), EMHASS sets `soc_final = soc_init` for you, so passing just `soc_init` is enough for a basic rolling-MPC setup.

If you pass both `soc_init` and `soc_final` explicitly with different values, EMHASS uses them as-is. That is appropriate when you have a hard end-of-horizon target (e.g. "must be at 60% by tomorrow 06:00 to absorb morning PV"), or when your runtime layer computes a target SOC dynamically. In that case you may also want to extend the horizon so the constraint sits at the actual deadline rather than at the trailing edge of a fixed 24 h window.

A common new-user trap is the opposite: starting with very low actual SOC where `soc_init` is below `battery_minimum_state_of_charge`. The optimization becomes infeasible because the initial state already violates a constraint. See Section 5 below (item 4) for the full triage and [discussion #359](https://github.com/davidusb-geek/emhass/discussions/359) for the canonical thread.

## 5. `optim_status: Infeasible` triage order

When EMHASS returns `optim_status: "Infeasible"` and publishes nothing, work through this list in order. The first match is almost always the cause.

1. Forecast NaN. A sensor publishing `unavailable` becomes a NaN in the forecast list and the solver chokes. Check `pv_power_forecast`, `load_power_forecast`, `load_cost_forecast`, `prod_price_forecast` for NaN/None entries. Fix at the HA-template level (use `default(0)` or filter out unavailable states).
2. Timestep mismatch. List lengths don't match `prediction_horizon`. See section 2 above.
3. Windowed-deferrable infeasibility. A windowed deferrable load (EV, washing machine with hard deadline) requires more energy than the window×nominal-power product allows. Either widen the window, increase the power, or reduce the required energy.
4. Battery state contradicts limits. `soc_init = 0.05` but `battery_minimum_state_of_charge = 0.30` (the default): the optimizer cannot find any valid trajectory because the *initial* state already violates a constraint. Either lower `battery_minimum_state_of_charge`, clamp `soc_init` before passing it, or accept that the battery genuinely cannot be used until it recovers above the minimum. See [discussion #359](https://github.com/davidusb-geek/emhass/discussions/359) for the canonical thread.
5. Thermal-battery infeasibility. `start_temperature` outside the `min_temperatures[0]` / `max_temperatures[0]` range, or a heating/cooling rate that physically cannot reach `min_temperatures` from `start_temperature` within the available timesteps. Widen the comfort range or increase the heat pump power.
6. Solver time-limit. `lp_solver_timeout` (default 45 s) exceeded for very large problems (long horizon × many deferrable loads × many timesteps). Reduce horizon or increase timeout.

The new stage-timing banner introduced in upstream PR [#806](https://github.com/davidusb-geek/emhass/pull/806) makes the per-stage timing visible in the logs, useful for distinguishing forecast errors (early stages) from solver issues (late stage).

## 6. Update intervals

| Action | Recommended interval |
|--------|----------------------|
| `naive-mpc-optim` | every `optimization_time_step` minutes (default 30); set your HA automation trigger to match |
| `dayahead-optim` | once per day, around 05:30 local time (after spot prices publish) |
| `publish-data` | every 5 min (or use `continual_publish: true`) |
| Forecast refresh | every 30–60 min |

Running `naive-mpc-optim` every 1 minute is overkill for residential systems and burns CPU for no measurable cost gain. Conversely, running it only every 4 hours leaves the system slow to react to forecast errors.

## 7. Logs and stage timing

The CLI and Add-on log to `data/logger_emhass.log`. The Add-on web UI exposes them under the *Logs* tab.

At the default log level (INFO), each optimization run emits a runtime banner and a one-line summary:

```
2026-04-26 17:00:00 INFO     EMHASS 0.17.2 | Python 3.11.9 | CVXPY 1.5.3 (Highs) | Linux-x86_64
2026-04-26 17:00:01 INFO     Optimization completed in 0.96s (top: optim_solve=0.52s, 54%)
```

The banner is emitted by `log_runtime_banner` (added in PR [#806](https://github.com/davidusb-geek/emhass/pull/806)) and is useful when filing bug reports: the EMHASS / Python / CVXPY / solver / platform combination is printed at the start of every run.

The summary identifies which stage dominated total runtime. Common patterns:
- `top: pv_forecast=...` consistently dominating usually means a remote API timeout (Solcast / Forecast.Solar). Check network access and API keys.
- `top: optim_solve=...` taking more than 30 s usually means the problem size has grown: either reduce the prediction horizon or increase `lp_solver_timeout`.
- `top: input_data=...` or `top: load_forecast=...` taking long usually means a slow Home Assistant database query: check sensor history depth.

Per-stage timings are recorded internally (in `input_data_dict["stage_times"]`) for every run but are only logged at DEBUG level. To see them, raise the log level to DEBUG temporarily (in `config_emhass.yaml` set `logging_level: DEBUG`, or use the Add-on advanced options). Each stage is logged as `Stage [<name>] completed in <X.XXX>s`.

## See also

- Tutorial: [Basic — PV + Battery](basic_pv_battery.md) (start here if you're new)
- How-to: [MPC walkthrough](mpc.md)
- Reference: [Forecasts](../forecasts.md)
- Reference: [ML Forecaster](../mlforecaster.md)
- Reference: [Configuration](../config.md)
- Reference: [Reference Configurations](reference_configs.md)
