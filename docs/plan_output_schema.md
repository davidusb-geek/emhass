# Plan / Optimization Output Schema

This page documents the columns of the optimization result: the DataFrame returned by
`publish_data` (see [API Reference](emhass.md)) and, where marked below, the diagnostic
columns that only the CSV file and the plan carry.
Downstream consumers — Home Assistant cards, Node-RED flows, EVCC adapters, third-party
automations — can rely on this schema as a versioned contract.

## Schema version

Every result DataFrame carries the schema version as a pandas attribute:

```python
result = await publish_data(...)
result.attrs["emhass_schema_version"]
# "1.1"
```

The version string is also exposed as a module-level constant:

```python
from emhass.command_line import EMHASS_SCHEMA_VERSION
```

### Semver rules

| Bump kind | Trigger |
|-----------|---------|
| **Patch** (`1.0` → `1.0.1`) | Documentation clarification, sign convention confirmed, description sharpened. No consumer code change required. |
| **Minor** (`1.0` → `1.1`) | New column added without removing or renaming existing columns. Consumers using `.get(col)` keep working. |
| **Major** (`1.0` → `2.0`) | Column removed, column renamed, sign convention flipped, unit changed, or HA-scaling factor changed for a column. |

Consumers pin to a major version. Code that requires schema ≥ 1.0 should check:

```python
major = int(result.attrs.get("emhass_schema_version", "0").split(".")[0])
if major != 1:
    raise RuntimeError(f"Unexpected EMHASS schema major {major}")
```

## Column table

11 fixed columns plus the variable groups: `P_deferrable{k}`,
`predicted_temp_heater{k}`, `heating_demand_heater{k}` (for each configured
deferrable / thermal load), and `cost_fun_<name>` (one column per cost-function
component the chosen `costfun` decomposes into). A `heat_topology` with
tank-to-tank transfers also adds one `P_transfer_{from}_{to}` column per transfer
(delivered heat, W, non-negative). A storage that is only fed by a transfer has
no load of its own: its temperature is in `predicted_temp_heater{n + i}`, where
`n` is the number of deferrable loads and `i` the storage's position in the
tank list (its position in `storage`, after any manual `shared_thermal_tanks` in
extend mode). Adding a source flow changes `n`, and so this index. Neither is published to Home Assistant.

With two or more PV plants, the optimization result also has one `P_PV_plant{j}` column per plant
(W, never negative, `j` counting from 0 in plant order), written next to `P_PV`. The plant
columns add up to `P_PV` at every timestep, up to float rounding (above one inverter, not after a
PV-only clip: see below); where `P_PV` is not a
finite number, neither are the plant columns. They split `P_PV` as forecast:
`P_PV_curtailment` is not split per plant. They are diagnostic output. They are in
`opt_res_latest.csv` and in the plan returned by `/api/v1/plan`, not in the frame that
`publish_data` returns (it keeps only the published columns), and they are not published
to Home Assistant. They are absent with a single plant, and absent when the cached weather
forecast in use was saved before this version and has no per-plant data (see
[Forecasts](forecasts.md) for how to refresh it); in both cases the result has the same
columns as before. See [Forecasts](forecasts.md) for what a plant is for each forecast method.

Above one inverter (`number_of_inverters` above 1), the result also has per-inverter columns,
`i` counting from 0 in inverter order. `P_PV_{i}` is the PV forecast the model was given for
each inverter that carries PV (W; `P_PV` is their sum). The PV that inverter actually uses is
`P_PV_{i}` minus its `P_PV_curtailment_{i}` or `P_PV_spill_{i}`, whichever it has, just as
`P_PV_curtailment` comes off `P_PV`. For a PV-only inverter with
`compute_curtailment` off, the forecast is clipped to what the inverter can convert, so
`P_PV` and `P_PV_{i}` are clipped too, while the `P_PV_plant{j}` columns keep the forecast as it
came in and no longer always add up to `P_PV`. `P_PV_curtailment_{i}` is the curtailment of
each inverter that carries PV and has `compute_curtailment` on (W, never negative), and
`P_PV_curtailment` is then the sum over those inverters. `P_PV_spill_{i}` is the PV a hybrid
inverter with `compute_curtailment` off cannot use in a step (W, never negative); it appears
only for those inverters. `P_hybrid_inverter_{i}` is the AC-side
power of every inverter, with the same sign convention as `P_hybrid_inverter` (positive = DC to
AC, negative = AC to DC; an inverter with no PV and no battery is 0, and a PV-only inverter is
never negative), and `P_hybrid_inverter` is the sum over inverters. `inv_stress_cost_{i}` appears
for each inverter with `inverter_stress_cost` above 0 and a cap above 0 that has a PV plant or a
battery on it. The site totals `P_PV`, `P_PV_curtailment` and `P_hybrid_inverter` are in the plan
as at one inverter, but `P_hybrid_inverter` is always present here. The per-inverter columns are
diagnostic output like `P_PV_plant{j}`: they are in `opt_res_latest.csv` and in the plan, not in
the frame that `publish_data` returns, and they are not published to Home Assistant.

With more than one battery, the result has `P_batt_{k}` (W, same sign convention as
`P_batt`) and `SOC_opt_{k}` (fraction 0 to 1, the same scaling as `SOC_opt`) for each
battery `k` counting from 0, and no bare `SOC_opt`. `P_batt` is then the total of all
batteries. These columns are published to Home Assistant as the per-battery sensors
`sensor.p_batt_forecast_battery{k}` and `sensor.soc_batt_forecast_battery{k}`. The other
per-battery diagnostic columns that only the CSV file and the plan carry (such as
`batt_stress_cost_{k}`) are not listed on this page.

| Column | Source helper | Unit | Sign convention | Conditional | HA scaling | `type_var` | Notes |
|--------|---------------|------|-----------------|-------------|------------|------------|-------|
| `P_Load` | `_publish_standard_forecasts` | W | positive = consumption | always | 1:1 | `power` | DataFrame column read directly |
| `P_PV` | `_publish_standard_forecasts` | W | positive = production (PV → DC bus) | when `"P_PV"` is in DataFrame | 1:1 | `power` | `custom_pv_forecast_id`. Assigned from `p_pv` forecast input at `optimization.py:2291`. |
| `P_PV_curtailment` | `_publish_standard_forecasts` | W | positive = curtailed delta (subtracts from gross PV in DC balance) | `plant_conf.compute_curtailment = true` (one inverter); above one inverter, when any inverter that carries PV has it on | 1:1 | `power` | `custom_pv_curtailment_id`. CVXPY `nonneg=True` at `optimization.py:876`; appears as `(p_pv - p_pv_curtailment)` in DC-balance constraint at line 1131; bounded by `<= param_pv_forecast` at line 2761. |
| `P_hybrid_inverter` | `_publish_standard_forecasts` | W | AC-side power; positive = DC → AC delivery, negative = AC → DC (charging-from-grid via DC bus) | `plant_conf.inverter_is_hybrid = true` (one inverter); always above one inverter | 1:1 | `power` | `custom_hybrid_inverter_id`. Defined in `_add_hybrid_inverter_constraints` as `p_hybrid_inverter == ac_out - ac_in`. By default (scalar efficiencies) `ac_out = p_dc_ac * eff_dc_ac` and `ac_in = p_ac_dc / eff_ac_dc`; when `inverter_power_curve_dc_ac` / `inverter_power_curve_ac_dc` is configured, that direction's DC-bus flow is instead the exact piecewise-linear function of the AC power, converted from the configured `[ac_power_w, efficiency]` points. `P_hybrid_inverter` is the AC side and `P_batt` the DC-bus (battery) side. Unit and sign are unchanged. |
| `P_deferrable{k}` | `_publish_deferrable_loads` | W | positive = consumption | `k ∈ [0, number_of_deferrable_loads)`; row skipped (error log) if column missing | 1:1 | `deferrable` | `custom_deferrable_forecast_id[k]` |
| `predicted_temp_heater{k}` | `_publish_thermal_loads` | °C | n/a (state) | per `k` where `def_load_config[k]` has `thermal_config`, `thermal_battery` or `thermal_source` (a heat topology or shared-tank member, which reports its storage's temperature) | 1:1 | `temperature` | `custom_predicted_temperature_id[k]`. Semantics depend on heater type: **room (air) temperature** for `thermal_config` loads (kept within `min_temps`/`max_temps` band toward `desired_temps`); **thermal-storage/tank temperature** for `thermal_battery` loads. |
| `heating_demand_heater{k}` | `_publish_thermal_loads` | **kWh** | **thermal energy** the building exchanges with its surroundings, not electrical input. For a heating load (`sense: "heat"`) this is positive (heat the building loses and the source must replace). For a cooling load (`sense: "cool"`) it is zero or **negative**, representing the heat the building gains and the source must remove. | set for `thermal_battery` loads and for `thermal_source` loads (a heat topology or shared-tank member), not for bare `thermal_config`. Every load that feeds a shared storage reports that storage's whole demand, so do not add them up; a storage without a demand model reports 0 | 1:1 | `energy` | `custom_heating_demand_id[k]`. The thermal-battery model carries a separate `heatpump_cops` parameter (`optimization.py:310`); electrical input = thermal / COP, so the two are decoupled. Unit `"kWh"` confirmed at `utils.py` (`heating_demand_friendly_name`). |
| `P_batt` | `_publish_battery_data` | W | positive = discharge (battery → house), negative = charge (house/grid → battery) | `optim_conf.set_use_battery = true`; row skipped if column missing | 1:1 | `batt` | `custom_batt_forecast_id`. Sum of `p_sto_pos + p_sto_neg` at `optimization.py:2312`. SOC reconstruction at lines 2319-2322 confirms direction: `power_flow = p_sto_pos / eff_dis + p_sto_neg * eff_chg`, then `SOC_opt = soc_init - cumulative_change / cap` (so positive `P_batt` drives SOC down → discharge). |
| `SOC_opt` | `_publish_battery_data` | **fraction (0..1) in CSV; ×100 in HA** | n/a (state) | `optim_conf.set_use_battery = true` | **×100** | `SOC` | See [SOC_opt scaling callout](#soc_opt-scaling-callout) |
| `P_grid` | `_publish_grid_and_costs` | W | positive = import (grid → house), negative = export (house → grid) | always | 1:1 | `power` | `custom_grid_forecast_id`. `P_grid = P_grid_pos + P_grid_neg` at `optimization.py:2299`. `p_grid_pos` is CVXPY `nonneg=True` bounded by `maximum_power_from_grid` (line 803); `p_grid_neg` is `nonpos=True` bounded by `-maximum_power_to_grid` (line 798). |
| `cost_fun_<name>` | `_publish_grid_and_costs` | € | n/a (cost) | always; multi-column (filter on substring `"cost_fun_"`) | 1:1 | `cost_fun` | Components depend on `costfun` (`profit`, `cost`, `self-consumption`). HA single entity `custom_cost_fun_id` aggregates them. |
| `optim_status` | `_publish_grid_and_costs` | text | n/a | always (defaulted to `"Optimal"` with WARN log if missing) | n/a | `optim_status` | `Optimal`, `Optimal_Inaccurate`, `Optimal (Incumbent)` and `Optimal (Relaxed)` come with a plan (see [publish_data](publish_data.md)); other CVXPY status strings (`Infeasible`, `Unbounded`, `User_Limit`, etc.) come without one. `device_class=""`, `unit_of_measurement=""`. |
| `unit_load_cost` | `_publish_grid_and_costs` | €/kWh | n/a (price) | always | 1:1 | `unit_load_cost` | per-timestep tariff series for load |
| `unit_prod_price` | `_publish_grid_and_costs` | €/kWh | n/a (price) | always | 1:1 | `unit_prod_price` | per-timestep sell price series |

### SOC_opt scaling callout

> ⚠️ **`SOC_opt` is the most common consumer bug.**
>
> The DataFrame value and the CSV export carry the state-of-charge as a **fraction in
> [0, 1]**. When the same column is published to Home Assistant it is multiplied by
> **100** to display as a percentage. Consumers reading the result DataFrame or the
> exported CSV must therefore multiply by 100 themselves if they expect percent;
> consumers reading the HA entity get percent already and must not double-scale.

## Sign conventions

All sign conventions are derived from `src/emhass/optimization.py` (the CVXPY model
definition). Per-column citations are inline in the column table above. Summary:

| Quantity | Positive means | Negative means |
|----------|----------------|----------------|
| `P_PV` | PV production (panels → DC bus) | (always ≥ 0; PV does not consume) |
| `P_PV_curtailment` | curtailed delta (subtracted from gross PV) | (always ≥ 0; CVXPY `nonneg=True`) |
| `P_hybrid_inverter` | DC → AC flow (delivered to house/grid) | AC → DC flow (charging via DC bus) |
| `P_deferrable{k}` | load consumption | (always ≥ 0; loads do not export) |
| `P_batt` | battery discharge (battery → house) | battery charge (house/grid → battery) |
| `P_grid` | import (grid → house) | export (house → grid) |

## Consumer recipes

### Python (pandas)

```python
from emhass.command_line import publish_data, EMHASS_SCHEMA_VERSION

result = await publish_data(input_data_dict, logger)
assert result.attrs["emhass_schema_version"].startswith("1.")

if "SOC_opt" in result.columns:
    soc_percent = result["SOC_opt"] * 100  # CSV holds fraction
```

### Node-RED

The published HA entities carry the result columns by their friendly names (set per
`custom_*_id` config). A flow that depends on the schema should hard-code the major
version it was authored against and refuse to run if the EMHASS instance advertises a
different major (read from the EMHASS log line at startup, or via the `/api/last-run`
endpoint once that lands — see board item AC-3).

## Source helpers

Per-column source helpers in `src/emhass/command_line.py`:

| Helper | Approximate line range | Columns published |
|--------|------------------------|-------------------|
| `_publish_standard_forecasts` | 2152-2214 | `P_Load`, `P_PV`, `P_PV_curtailment`, `P_hybrid_inverter` |
| `_publish_deferrable_loads` | 2215-2259 | `P_deferrable{k}` for `k ∈ [0, N)` |
| `_publish_thermal_loads` | 2260-2304 | `predicted_temp_heater{k}`, `heating_demand_heater{k}` |
| `_publish_battery_data` | 2305-2341 | `P_batt`, `SOC_opt`, and with more than one battery `P_batt_{k}`, `SOC_opt_{k}` |
| `_publish_grid_and_costs` | 2342-2405 | `P_grid`, `cost_fun_<name>`, `optim_status`, `unit_load_cost`, `unit_prod_price` |

The aggregator `publish_data` (~line 2408) calls these in order and subsets the result
DataFrame to the published columns before returning.

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-05-10 | Initial published schema. Sign conventions derived from `src/emhass/optimization.py` with inline citations. |
| 1.1 | 2026-11-06 | Added the optional `P_PV_plant{j}` columns (only with two or more PV plants). Documented the `P_batt_{k}` and `SOC_opt_{k}` columns that appear with more than one battery. Added the per-inverter columns `P_PV_{i}`, `P_PV_curtailment_{i}`, `P_hybrid_inverter_{i}` and `inv_stress_cost_{i}` (only with more than one inverter). Added `P_PV_spill_{i}` (only for a hybrid inverter with curtailment off, above one inverter). Above one inverter, a PV-only inverter with curtailment off has its `P_PV_{i}` (and so `P_PV`) clipped, and the `P_PV_plant{j}` columns then no longer always add up to `P_PV`. No existing column changed. |
