# Power-dependent hybrid inverter efficiency

## Goal

Make EMHASS plan with the fact that a hybrid inverter wastes a larger share of low power than of high power, in both directions (DC bus to AC when discharging or exporting PV, AC to DC bus when charging from the grid), indexed on the AC side where the inverter is commanded and measured. After this recipe, `inverter_power_curve_dc_ac` and `inverter_power_curve_ac_dc` hold validated efficiency curves built from your own datasheet or measurements, and you can check that the plan obeys them.

## Prerequisites

- EMHASS version: a release that includes the `inverter_power_curve_*` parameters (issue #746).
- `inverter_is_hybrid: true` and a battery (`set_use_battery: true`).
- Your own measurements or datasheet points. No curve is shipped: there are no vendor or site defaults.
- Configuration only: there is no transport or orchestration step, so nothing here depends on Node-RED, Home Assistant or another stack.

## Step 1: Decide whether you need it

The default (scalar `inverter_efficiency_dc_ac` and `inverter_efficiency_ac_dc`) is a good model when your inverter's efficiency is roughly flat over the power range you actually use. A curve is worth the added solver work only if the loss at low power is material for your hardware and tariff, for example when EMHASS would otherwise choose between slow self-consumption discharge and fast export discharge on the assumption that both cost the same percentage. A more faithful physical model does not by itself mean better economics: if a well-calibrated scalar gives the same schedule, keep it.

Existing configurations need no migration: with both curve parameters empty (the default) the optimization is exactly the one of earlier releases.

## Step 2: Pick the electrical boundary and orient the points

<!-- source: src/emhass/optimization.py:2869 (exact transfer between the AC-side power and the DC-bus variable) -->

A curve describes the inverter only, between the DC bus and the AC terminals. It is not the battery's efficiency: `battery_charge_efficiency` and `battery_discharge_efficiency` stay separate state-of-charge stage efficiencies and are never replaced by a curve.

Each point is `[ac_power_w, efficiency]`: an **AC-side** power in Watts, the power you command and measure at the inverter's AC terminals, and the inverter efficiency at that power as a **fraction** (`0.97`, not `97`). The DC bus is internal; `P_batt` in the plan is a DC-bus power, while `P_hybrid_inverter` is the AC side.

- `inverter_power_curve_dc_ac` (discharge): `[ac_output_power_w, efficiency]`, where the efficiency is AC output / DC input, so the DC input is `ac_output / efficiency`. `0 < efficiency <= 1`: delivering positive AC output always takes DC power, so 0 % at a positive output would need infinite DC input and is rejected.
- `inverter_power_curve_ac_dc` (charge): `[ac_input_power_w, efficiency]`, where the efficiency is DC output / AC input, so the DC output is `ac_input * efficiency`. `0 <= efficiency <= 1`: at low power, fixed losses can consume the whole AC input, so a measured `[50, 0.0]` (50 W of AC input reaches the DC bus as 0 W) is valid.

The efficiency must already contain the losses at that operating point. It must not include standby consumption, and you do not enter a point at 0 W (efficiency there is undefined): EMHASS adds the origin (0 W AC, 0 W DC) itself, so idle has no standby term. Keep standby where you account for it today (for example in the load forecast).

## Step 3: Build the points from your data

<!-- source: src/emhass/utils.py:4527 (validate_inverter_power_curve) -->

Datasheets and measurements usually give an efficiency at a power level: copy those pairs as `[power, efficiency]` with the efficiency as a fraction. No arithmetic is needed when the power axis is the AC side, which is how most inverter and AC-coupled battery data is given.

If your data is against **DC** power instead, convert only the power (x) coordinate to the AC side and keep the efficiency as it is:

- `inverter_power_curve_dc_ac` (AC is the output): `ac_power_w = dc_power_w * efficiency`;
- `inverter_power_curve_ac_dc` (AC is the input): `ac_power_w = dc_power_w / efficiency`.

The values below are illustrative only.

```python
# (ac_side_power_w, efficiency) - illustrative numbers, replace with your own
discharge = [(100, 0.60), (250, 0.75), (800, 0.85), (2500, 0.90), (4500, 0.93)]
charge = [(50, 0.0), (100, 0.55), (250, 0.72), (800, 0.84), (2500, 0.90), (4000, 0.92)]

dc_ac = [[w, eta] for w, eta in discharge]
ac_dc = [[w, eta] for w, eta in charge]
```

EMHASS turns each point into a power-transfer point (`dc_input = ac / eta` for DC to AC, `dc_output = ac * eta` for AC to DC) and enforces the exact piecewise-linear transfer between them. Efficiency itself is **not** linearly interpolated between your points, because that would make the DC power a quadratic function of the AC power.

For the discharge example the internal transfer points are `(0, 0), (100, 166.7), (250, 333.3), (800, 941.2), (2500, 2777.8), (4500, 4838.7)` W (AC, DC). For the charge example they are `(0, 0), (50, 0), (100, 55), (250, 180), (800, 672), (2500, 2250), (4000, 3680)` W (AC, DC).

A flat segment is fine. In the charge example, `[50, 0.0]` means 50 W of AC input gives 0 W of DC output: the transfer has a dead zone, and the optimizer will not draw AC inside it unless the import price pays it to (see the negative-price caveat).

Prefer measured data at the powers you actually use, and few points. Every segment of a curved direction after the first adds one binary variable per time step to the optimization. With the internal origin, N configured points make N segments and N-1 binaries per step (a 10-point curve adds 9 per step: 432 over 48 steps, 2592 over 288), so three to six points per direction are usually plenty. At most 20 points per curve are accepted; in a 288-step benchmark, two 20-point curves took about 21 s to solve at a 1 % MIP gap against about 2 s for two 2-point curves, so keep a curve a compact calibration, not a high-resolution trace.

## Step 4: Configure and validate

<!-- source: src/emhass/data/config_defaults.json:129-130 -->
<!-- source: src/emhass/utils.py:4666 (check_inverter_power_curves) -->

```yaml
plant_conf:
  inverter_is_hybrid: true
  inverter_ac_output_max: 4500  # at or below the last AC point of each curve
  inverter_ac_input_max: 4000
  inverter_power_curve_dc_ac: [[100, 0.60], [250, 0.75], [800, 0.85], [2500, 0.90], [4500, 0.93]]
  inverter_power_curve_ac_dc: [[50, 0.0], [100, 0.55], [250, 0.72], [800, 0.84], [2500, 0.90], [4000, 0.92]]
```

A curve is validated whenever it is saved or passed at runtime, and an invalid one is rejected with an error naming the parameter: saving in the configuration page fails with the reason shown and keeps the previous configuration, and a runtime request fails. Only a curve that is already stored in a bad state (for example in a hand-edited `config.json`) is cleared at startup, with an error in the log, so the page stays reachable and the scalar efficiency applies until you fix it. Common faults and what they mean:

| Message contains | Cause |
|---|---|
| `must be a list of at least 2` | one point, an empty string, `null`/`false`/`0`, or not a list (only an absent parameter or `[]` means "no curve") |
| `at most 20 are supported` | more than 20 points |
| `expected a ... pair` | a point does not have exactly two numbers |
| `expected a finite number` | `NaN`, infinity, or an integer too large for a float |
| `strictly positive` | an `ac_power_w` is `0` or negative (do not enter a 0 W point) |
| `ascend strictly` | `ac_power_w` repeats or goes backwards |
| `0 <= efficiency <= 1` | efficiency is negative or above 1, for example `97` instead of `0.97` |
| `infinite DC input` | an `inverter_power_curve_dc_ac` point has efficiency `0` at a positive AC output |
| `non-decreasing` | the converted DC-side power falls as AC power rises (a flat segment is allowed) |
| `non-finite` / `too close together` | an efficiency or point spacing so small that a converted power or slope overflows |
| `requires inverter_is_hybrid` | a runtime request passed a curve while `inverter_is_hybrid` is false; a stored curve is kept but ignored in that case |

Expected: EMHASS starts with no error. A warning that the curve ends below `inverter_ac_output_max` or `inverter_ac_input_max` means the last point is a lower power limit than your inverter rating (see the caveat on range below).

## Step 5: Check the plan

After a dayahead or MPC run, take an interval where the battery charges or discharges and PV is zero. The DC-bus power `P_batt` must equal the transfer evaluated at the AC power `P_hybrid_inverter` (linear between the converted points): `P_batt = f(P_hybrid_inverter)` while discharging, with `f(ac) = ac / efficiency` at each of your points, and `-P_batt = g(-P_hybrid_inverter)` while charging, with `g(ac) = ac * efficiency` at each of your points.

Expected: the published values lie on the transfer your points define. With the curves empty they lie on the scalar efficiency instead.

## Caveats

- **Range.** The last point is the supported domain: the optimizer never plans more AC power through that direction than the last point's AC power, and `inverter_ac_output_max` / `inverter_ac_input_max` still limit the AC side, so the lower of the two wins. A curve that ends well below the inverter rating therefore lowers the usable power. If PV can exceed what the discharge curve covers, enable curtailment (`compute_curtailment`) or extend the curve with measured points.
- **Zero power.** The origin (0 W AC, 0 W DC) is added internally to every curve and you do not enter it. Idle has no standby term in the model.
- **Negative prices.** The curve is an equality, not an inequality, so negative import prices cannot be exploited to draw AC power that is not accounted for. A dead zone you declare yourself is different: with `[50, 0.0]` in the charge curve and a sufficiently negative import price, EMHASS may draw those 50 W and lose them as inverter heat, because under your curve that is physically what happens and it is paid for. EMHASS does not hide this with a minimum-power rule.
- **Minimum operating power.** EMHASS has no setting for a minimum battery charge or discharge power, so "never operate below N W" cannot be expressed here; a charge dead zone only describes the losses, it does not forbid operating in it.
- **Charge-power derating.** `battery_charge_power_derating` keeps limiting charge power by state of charge on top of the curve; a curve does not replace it.
- **Several batteries.** There is one hybrid inverter and one shared DC bus: one curve per direction acts on the combined battery and PV power, not per battery.
- **Direction independence.** You may set one curve and keep the other direction scalar.
- **Solver time.** Curved directions add binaries; with several segments over a long horizon, solve time rises. Watch `lp_solver_timeout` and keep curves compact.

## Credits

- Request and design discussion: davidusb-geek/emhass#746.
- Equations: [Hybrid inverter conversion](../advanced_math_model.md#hybrid-inverter-conversion); parameter reference: [Hybrid inverter power curves](../config.md#hybrid-inverter-power-curves).
- Field names verified against `src/emhass/data/config_defaults.json` and `src/emhass/utils.py` on 2026-10-07.
