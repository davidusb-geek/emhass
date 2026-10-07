# Power-dependent hybrid inverter efficiency

## Goal

Make EMHASS plan with the fact that a hybrid inverter wastes a larger share of low power than of high power, in both directions (DC bus to AC when discharging or exporting PV, AC to DC bus when charging from the grid). After this recipe, `inverter_power_curve_dc_ac` and `inverter_power_curve_ac_dc` hold validated efficiency curves built from your own datasheet or measurements, and you can check that the plan obeys them.

## Prerequisites

- EMHASS version: a release that includes the `inverter_power_curve_*` parameters (issue #746).
- `inverter_is_hybrid: true` and a battery (`set_use_battery: true`).
- Your own measurements or datasheet points. No curve is shipped: there are no vendor or site defaults.
- Configuration only: there is no transport or orchestration step, so nothing here depends on Node-RED, Home Assistant or another stack.

## Step 1: Decide whether you need it

The default (scalar `inverter_efficiency_dc_ac` and `inverter_efficiency_ac_dc`) is a good model when your inverter's efficiency is roughly flat over the power range you actually use. A curve is worth the added solver work only if the loss at low power is material for your hardware and tariff, for example when EMHASS would otherwise choose between slow self-consumption discharge and fast export discharge on the assumption that both cost the same percentage. A more faithful physical model does not by itself mean better economics: if a well-calibrated scalar gives the same schedule, keep it.

Existing configurations need no migration: with both curve parameters empty (the default) the optimization is exactly the one of earlier releases.

## Step 2: Pick the electrical boundary and orient the points

<!-- source: src/emhass/optimization.py:2864 (exact transfer between the DC-side variable and the AC side) -->

A curve describes the inverter only, between the DC bus and the AC terminals. It is not the battery's efficiency: `battery_charge_efficiency` and `battery_discharge_efficiency` stay separate state-of-charge stage efficiencies and are never replaced by a curve.

Each point is `[dc_power_w, efficiency]`: a DC-side power in Watts and the inverter efficiency at that power as a **fraction** (`0.97`, not `97`):

- `inverter_power_curve_dc_ac`: `[dc_input_power_w, efficiency]`, where the DC power is PV plus battery discharge reaching the inverter and the efficiency is AC output / DC input. `0 <= efficiency <= 1`: a measured 0 % at a positive power such as `[50, 0.0]` is valid (50 W in, 0 W out).
- `inverter_power_curve_ac_dc`: `[dc_output_power_w, efficiency]`, where the DC power is what reaches the DC bus for charging and the efficiency is DC output / AC input. `0 < efficiency <= 1`: zero efficiency at a positive DC output would need infinite AC input and cannot be represented, so a device with a zero-output or minimum-power region is outside this curve.

The efficiency must already contain the losses at that operating point. It must not include standby consumption, and you do not enter a point at 0 W (efficiency there is undefined): EMHASS adds the origin (0 W DC, 0 W AC) itself, so idle has no standby term. Keep standby where you account for it today (for example in the load forecast).

## Step 3: Build the points from your data

<!-- source: src/emhass/utils.py:4332 (validate_inverter_power_curve) -->

Datasheets and measurements usually give an efficiency at a power level: copy those pairs as `[power, efficiency]` with the efficiency as a fraction. No arithmetic is needed when the datasheet's power axis is the DC side.

If your data is against **AC** power instead, convert only the power (x) coordinate to the DC side and keep the efficiency as it is:

- `inverter_power_curve_dc_ac` (AC is the output): `dc_power_w = ac_power_w / efficiency`;
- `inverter_power_curve_ac_dc` (AC is the input): `dc_power_w = ac_power_w * efficiency`.

The values below are illustrative only.

```python
# (dc_side_power_w, efficiency) - illustrative numbers, replace with your own
discharge = [(100, 0.60), (250, 0.75), (800, 0.85), (2500, 0.90), (5000, 0.93)]
charge = [(100, 0.55), (250, 0.72), (800, 0.84), (2500, 0.90), (4000, 0.92)]

dc_ac = [[w, eta] for w, eta in discharge]
ac_dc = [[w, eta] for w, eta in charge]
```

EMHASS turns each point into a power-transfer point (`ac_output = dc * eta` for DC to AC, `ac_input = dc / eta` for AC to DC) and enforces the exact piecewise-linear transfer between them. Efficiency itself is **not** linearly interpolated between your points, because that would make the AC power a quadratic function of the DC power.

For the discharge example the internal transfer points are `(0, 0), (100, 60), (250, 187.5), (800, 680), (2500, 2250), (5000, 4650)` W (DC, AC). For the charge example they are `(0, 0), (100, 181.8), (250, 347.2), (800, 952.4), (2500, 2777.8), (4000, 4347.8)` W (DC, AC).

A flat segment is fine. If a measurement shows 0 % at 50 W, enter `[50, 0.0]` in `inverter_power_curve_dc_ac`: the transfer then has a dead zone (50 W DC gives 0 W AC) and the optimizer will not discharge inside it unless it has to.

Prefer measured data at the powers you actually use, and few points. Every segment of a curved direction after the first adds one binary variable per time step to the optimization. With the internal origin, N configured points make N segments and N-1 binaries per step (a 10-point curve adds 9 per step: 432 over 48 steps, 2592 over 288), so three to six points per direction are usually plenty.

## Step 4: Configure and validate

<!-- source: src/emhass/data/config_defaults.json:129-130 -->
<!-- source: src/emhass/utils.py:4422 (check_inverter_power_curves) -->

```yaml
plant_conf:
  inverter_is_hybrid: true
  inverter_ac_output_max: 4600  # at or below the last AC point of each curve
  inverter_ac_input_max: 4300
  inverter_power_curve_dc_ac: [[100, 0.60], [250, 0.75], [800, 0.85], [2500, 0.90], [5000, 0.93]]
  inverter_power_curve_ac_dc: [[100, 0.55], [250, 0.72], [800, 0.84], [2500, 0.90], [4000, 0.92]]
```

A curve is validated whenever it is saved or passed at runtime, and an invalid one is rejected with an error naming the parameter: saving in the configuration page fails with the reason shown and keeps the previous configuration, and a runtime request fails. Only a curve that is already stored in a bad state (for example in a hand-edited `config.json`) is cleared at startup, with an error in the log, so the page stays reachable and the scalar efficiency applies until you fix it. Common faults and what they mean:

| Message contains | Cause |
|---|---|
| `must be a list of at least 2` | one point, an empty string, or not a list |
| `expected a ... pair` | a point does not have exactly two numbers |
| `strictly positive` | a `dc_power_w` is `0` or negative (do not enter a 0 W point) |
| `ascend strictly` | `dc_power_w` repeats or goes backwards |
| `0 <= efficiency <= 1` | efficiency is negative or above 1, for example `97` instead of `0.97` |
| `infinite AC input` | an `inverter_power_curve_ac_dc` point has efficiency `0` at a positive DC output |
| `non-decreasing` | the converted AC-side power falls as DC power rises (a flat segment is allowed) |
| `non-finite` | an efficiency so small that the converted AC power overflows |
| `requires inverter_is_hybrid` | a runtime request passed a curve while `inverter_is_hybrid` is false; a stored curve is kept but ignored in that case |

Expected: EMHASS starts with no error. A warning that the curve ends below `inverter_ac_output_max` or `inverter_ac_input_max` means the last point is a lower power limit than your inverter rating (see the caveat on range below).

## Step 5: Check the plan

After a dayahead or MPC run, take an interval where the battery charges or discharges and PV is zero. `P_hybrid_inverter` must equal the converted transfer evaluated at `P_batt` (linear between the converted points): `P_hybrid_inverter = f(P_batt)` while discharging, with `f(dc) = dc * efficiency` at each of your points, and `P_hybrid_inverter = -g(-P_batt)` while charging, with `g(dc) = dc / efficiency` at each of your points.

Expected: the published values lie on the transfer your points define. With the curves empty they lie on the scalar efficiency instead.

## Caveats

- **Range.** The last point is the supported domain: the optimizer never plans more DC power through that direction than the last point's DC power, and `inverter_ac_output_max` / `inverter_ac_input_max` still limit the AC side, so the lower of the two wins. A curve that ends well below the inverter rating therefore lowers the usable power. If PV can exceed the curve's last DC point, enable curtailment (`compute_curtailment`) or extend the curve with measured points.
- **Zero power.** The origin (0 W DC, 0 W AC) is added internally to every curve and you do not enter it. Idle has no standby term in the model.
- **Negative prices.** The curve is an equality, not an inequality, so negative import prices cannot be exploited to draw AC power that never reaches the DC bus.
- **Charge-power derating.** `battery_charge_power_derating` keeps limiting charge power by state of charge on top of the curve; a curve does not replace it.
- **Several batteries.** There is one hybrid inverter and one shared DC bus: one curve per direction acts on the combined battery and PV power, not per battery.
- **Direction independence.** You may set one curve and keep the other direction scalar.
- **Solver time.** Curved directions add binaries; with several segments over a long horizon, solve time rises. Watch `lp_solver_timeout` and keep curves compact.

## Credits

- Request and design discussion: davidusb-geek/emhass#746.
- Equations: [Hybrid inverter conversion](../advanced_math_model.md#hybrid-inverter-conversion); parameter reference: [Hybrid inverter power curves](../config.md#hybrid-inverter-power-curves).
- Field names verified against `src/emhass/data/config_defaults.json` and `src/emhass/utils.py` on 2026-10-05.
