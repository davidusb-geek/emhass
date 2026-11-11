# Two hybrid inverters, each with its own battery

## Goal

Plan a site with two hybrid inverters, each with its own battery and PV strings, as two separate inverters rather than one big DC bus, then send each battery's setpoint to the inverter it sits on, using the `inverter` attribute EMHASS publishes so the executor keeps no battery-to-inverter mapping of its own.

## Prerequisites

- EMHASS version: a release that includes `number_of_inverters` above 1 and the per-inverter sensors (issue [#1150](https://github.com/davidusb-geek/emhass/issues/1150)).
- Optimization mode: `naive-mpc-optim`, followed by `publish-data`.
- Two batteries (`number_of_batteries: 2`), one behind each inverter.
- Transport stack tested against: Home Assistant 2026.10. The templates were rendered on its template engine; the scripts have not been run against real inverters (untested on hardware, contribution welcome).

This recipe is an example of the routing, not a complete safety layer: monitoring the inverters themselves (a failed write, a limit left in place, a stale input) is up to your own watchdogs.

The numbers in this recipe are illustrative, not from a real site. The plan excerpts come from a real run of this config and the Step 2 payload, with a load forecast and the prices named in Step 3 added.

## Step 1: Describe the two inverters

<!-- source: src/emhass/data/config_defaults.json:129-141 -->

Inverter 0 is a 5 kW hybrid with a 10 kWh battery and more PV than it can convert, and it cannot curtail. Inverter 1 is a 3.6 kW hybrid with a 5 kWh battery, and it can curtail. Above one inverter the two AC caps and the two index lists are required lists. The efficiencies, stress cost and curtailment flag take one value for every inverter or one entry per inverter, and `inverter_is_hybrid` is not read: an inverter with a battery and a PV plant on it is a hybrid.

```yaml
optimization_time_step: 30
method_ts_round: first                      # the first plan step and the published row are the step that contains now
set_use_pv: true
set_use_battery: true
number_of_batteries: 2
battery_nominal_energy_capacity: [10000, 5000]
battery_charge_power_max: [5000, 2500]
battery_discharge_power_max: [5000, 2500]
battery_charge_efficiency: 0.95             # one value is used for both batteries
battery_discharge_efficiency: 0.95
battery_minimum_state_of_charge: 0.1
battery_maximum_state_of_charge: 0.95
battery_target_state_of_charge: 0.5
weight_battery_charge: 0.01                 # small, non-zero: see the caveat on churn
weight_battery_discharge: 0.01
number_of_inverters: 2
battery_inverter_index: [0, 1]              # battery 0 on inverter 0, battery 1 on inverter 1
pv_plant_inverter_index: [0, 1]             # one entry per PV plant, in pv_module_model order
inverter_ac_output_max: [5000, 3600]        # a list above one inverter, never a site total
inverter_ac_input_max: [5000, 3600]
inverter_efficiency_dc_ac: [0.97, 0.96]
inverter_efficiency_ac_dc: [0.97, 0.96]
compute_curtailment: [false, true]          # inverter 0 cannot curtail, inverter 1 can
pv_module_model:                            # the PV plant lists, one entry per plant
  - CSUN_Eurasia_Energy_Systems_Industry_and_Trade_CSUN295_60M
  - CSUN_Eurasia_Energy_Systems_Industry_and_Trade_CSUN295_60M
pv_inverter_model:                          # the model for the PV forecast; the AC caps are above
  - Fronius_International_GmbH__Fronius_Primo_5_0_1_208_240__240V_
  - Fronius_International_GmbH__Fronius_Primo_5_0_1_208_240__240V_
surface_tilt: [25, 25]
surface_azimuth: [205, 115]
modules_per_string: [11, 7]
strings_per_inverter: [2, 2]
```

`method_ts_round: first` matters for Steps 2 and 4. With the default `nearest`, a run in the second half of a step starts one step later, so a PV list that begins with the current step is shifted by one, and a publish late in a step posts the next step's setpoint before that step starts.

Keep the efficiencies below 1.0. With a lossless DC-to-AC side on one inverter and a lossless AC-to-DC side on the other, energy can go round between the two for free, and EMHASS logs a warning saying so. The full rules (broadcast, lengths, what is refused) are in [Per-inverter settings](../config.md#per-inverter-settings-and-the-one-inverter-default), and what each inverter's shape means for the plan is in [How the optimizer models more than one inverter](../config.md#how-the-optimizer-models-more-than-one-inverter).

Expected: the configuration saves with no error, and the first optimization run logs no warning about lossless conversion.

## Step 2: Send the PV forecast per inverter, then publish

<!-- source: src/emhass/utils.py:2400 (_per_inverter_shape_fault) -->
<!-- transport: Home Assistant 2026.10 (template rendered, not run on hardware) -->

Above one inverter a runtime `pv_power_forecast` is a list with one list per inverter, in inverter order, each the sum of the PV plants on that inverter: `[[0, 450, 1200, 2100], [0, 300, 900, 1500]]` for a horizon of 4. A flat list is refused. `soc_init` and `soc_final` take one value per battery, as they do for any multi-battery plant. Load and prices keep coming from your configured methods. In the Home Assistant examples, inverter A is inverter 0 and inverter B is inverter 1.

The model treats each inverter's PV forecast as DC power and applies that inverter's `inverter_efficiency_dc_ac` to it. The forecast EMHASS builds itself with PVLib is already AC power (it includes the `pv_inverter_model` loss), and a forecast service's figure is often AC too, so here the plan works with 3 % and 4 % less PV than the forecast says. On a hybrid that efficiency also covers battery discharge, so do not set it to 1.0 to close the gap. If the gap matters to you, send a DC forecast, before the inverter, if your forecast service has one. Dividing an AC forecast by the efficiency is only an approximation: an AC forecast is usually also clipped at its inverter model's rating, and dividing cannot bring back the clipped PV, which on inverter 0 is the surplus that could charge battery 0. The built-in PVLib forecast cannot be adjusted that way.

```yaml
script:
  emhass_mpc_two_inverters:
    sequence:
      - variables:
          run_started: "{{ now().isoformat() }}"   # Step 4 only trusts sensors posted after this
          horizon: 48                       # sent as prediction_horizon; each PV list needs at least this many values
          # One plain list of W per step (not a timestamp mapping) for each inverter,
          # from your PV forecast (placeholder entities and attribute; adapt them).
          # The first value must be the 30-minute step that contains now.
          pv_a: "{{ state_attr('sensor.pv_forecast_inverter_a', 'watts') }}"
          pv_b: "{{ state_attr('sensor.pv_forecast_inverter_b', 'watts') }}"
      - if: >
          {{ not (pv_a is list and pv_b is list
                  and (pv_a[:horizon] + pv_b[:horizon]) | select('number') | reject('boolean')
                      | select('is_number') | list | count == 2 * horizon) }}
        then:
          - stop: PV forecast missing, too short or not numeric; skipping this MPC run
            error: true
      - action: rest_command.emhass_mpc_two_inverters
        data:
          horizon: "{{ horizon }}"
          pv: "{{ [pv_a[:horizon] | map('float') | list, pv_b[:horizon] | map('float') | list] }}"
        response_variable: mpc
      - if: "{{ mpc.status >= 400 }}"
        then:
          - stop: EMHASS refused the MPC run (see its log); not publishing or routing
            error: true
      - action: rest_command.emhass_publish
        response_variable: published
      - if: "{{ published.status >= 400 }}"
        then:
          - stop: EMHASS did not publish the plan (see its log); not routing
            error: true
      - action: script.emhass_battery_setpoints   # Step 4
        data:
          run_started: "{{ run_started }}"

rest_command:
  emhass_mpc_two_inverters:
    url: http://localhost:5000/action/naive-mpc-optim
    method: POST
    timeout: 120
    headers: {content-type: application/json}
    payload: >
      {
        "prediction_horizon": {{ horizon }},
        "pv_power_forecast": {{ pv | tojson }},
        "soc_init": [ {{ states('sensor.battery_a_soc') | float / 100 }}, {{ states('sensor.battery_b_soc') | float / 100 }} ],
        "soc_final": [0.5, 0.5]
      }
  emhass_publish:
    url: http://localhost:5000/action/publish-data
    method: POST
    timeout: 60
    headers: {content-type: application/json}
    payload: "{}"

automation:
  - alias: EMHASS MPC two inverters
    triggers:
      - trigger: time_pattern
        minutes: "/30"                      # = optimization_time_step
        seconds: 20                         # a little after the step boundary
    actions:
      - action: script.emhass_mpc_two_inverters
```

The script stops rather than send a forecast that is short or holds anything but finite numbers (see the [Forecast input contract](../passing_data.md#forecast-input-contract)). It stops again if EMHASS refuses the run with HTTP 400 (a wrong shape, a bad index list, a plan that did not solve) or another error status, because a Home Assistant `rest_command` only logs a failed status, and the publish would otherwise post the previous plan again. It checks the publish's status the same way before it routes. A publish with no usable plan logs an error and answers 400, so the script stops there. A 200 does not prove that every sensor was posted, though, since a sensor post that Home Assistant answers with an error status only logs a warning; Step 4's freshness test is what catches that, per inverter. `sensor.battery_a_soc` and `sensor.battery_b_soc` are placeholders reporting percent; drop the `/ 100` if yours report a fraction. They have no default, so an unavailable SOC sensor fails the call instead of planning from an invented SOC. A sensor that is available but stale, and a PV forecast attribute that stopped updating, still pass these checks: if your integrations can freeze, add an age check on them (`last_reported`, which moves on every update even when the value stays the same) before the MPC call. `naive-mpc-optim` does not post sensors by itself, so the script calls `publish-data` next, and only then routes the setpoints. Keep `continual_publish` off with this recipe, so that sensors are only posted by this sequence, and run no other optimization (a `dayahead-optim`, another MPC) at other times: `publish-data` posts the latest result, whichever run wrote it. A Home Assistant script runs one at a time by default, so the sequence itself does not overlap. The automation starts it once per step, just after the step begins, so the run and its publish fall in the same step; run the script by hand once to test it. EMHASS does not clip a hybrid's PV forecast to its AC cap before the solve, whatever the source, so inverter 0's list can sit above its 5 kW cap at midday; that is the case Step 5 handles. The full rules for this list, and its timestamped form, are in [Passing data](../passing_data.md#forecast-data-at-runtime).

Expected: the plan has `P_PV_0` and `P_PV_1`, `P_PV` is their sum, and the sensors in Step 3 exist after the publish.

## Step 3: Read the plan per inverter

<!-- source: src/emhass/command_line.py:4037 (_publish_per_inverter_sensors), src/emhass/command_line.py:4372 (_inverter_attribute) -->

The unsuffixed columns stay site totals. Next to them the plan has one set per inverter `i` and per battery `k`:

| Column | Sensor | What it is |
|---|---|---|
| `P_PV_{i}` | `sensor.p_pv_forecast_inverter{i}` | the PV forecast given for inverter `i` |
| `P_PV_curtailment_{i}` | `sensor.p_pv_curtailment_inverter{i}` | PV the plan curtails, only on an inverter with `compute_curtailment` on |
| `P_PV_spill_{i}` | `sensor.p_pv_spill_inverter{i}` | PV the plan does not use on a hybrid with curtailment off, only on such an inverter |
| `P_hybrid_inverter_{i}` | `sensor.p_hybrid_inverter_inverter{i}` | AC side of inverter `i`, positive = DC to AC |
| `P_batt_{k}` | `sensor.p_batt_forecast_battery{k}` | battery `k`, positive = discharge |
| `SOC_opt_{k}` | `sensor.soc_batt_forecast_battery{k}` | battery `k` state of charge; a fraction in the plan, percent on the sensor |

Here are the steps from 11:00 to 14:00 of the run. The run's export price is -0.02 for the steps 11:00 to 13:00 and 0.05 from 13:30, so you will only see this pattern with a negative export price (W, SOC as a fraction, `P_grid` negative = export, prices per kWh). This plan is one of several that cost exactly the same, so another solver version can split the charge between the batteries and the steps differently; expect the same pattern, not these exact numbers:

```text
       P_PV_0  P_PV_1  P_PV_spill_0  P_PV_curtailment_1  P_hybrid_inverter_0  P_hybrid_inverter_1  P_batt_0  P_batt_1  SOC_opt_0  SOC_opt_1   P_grid
11:00  5542.6  2989.8           0.0              2989.8               3114.2              -2604.2   -2332.1   -2500.0      0.544      0.338      0.0
11:30  5912.8  2989.8           0.0              2989.8               2717.6              -2177.6   -3111.2   -2090.5      0.692      0.536      0.0
12:00  6181.9  2908.2           0.0              2908.2               5000.0              -1934.3   -1027.3   -1856.9      0.740      0.712  -2615.7
12:30  6345.2  2747.3           0.0              2747.3               3084.2              -2604.2   -3165.6   -2500.0      0.891      0.950      0.0
13:00  6400.0  2511.5           0.0              2511.5               5000.0                  0.0   -1245.4       0.0      0.950      0.950  -4490.0
13:30  6345.2  2207.2        1190.6                 0.0               5000.0               2118.9       0.0       0.0      0.950      0.950  -6578.9
14:00  6181.9  1842.6        1027.3                 0.0               5000.0               1768.9       0.0       0.0      0.950      0.950  -6318.9
```

While export costs money, inverter 1 curtails all of its own PV and, from 11:00 to 12:30 until battery 1 is full, takes AC in (`P_hybrid_inverter_1` negative) to charge battery 1 with PV from inverter 0, which cannot curtail. That PV crosses through both inverters and pays both efficiencies. Both batteries together have less room than inverter 0's surplus in that window, so some of it is still exported at a loss, and the plan is indifferent between the steps in which that loss falls. From 13:30 battery 0 is full and inverter 0 sits at its 5000 W cap, so the PV above that is spill.

With more than one inverter, every per-battery power and SOC sensor carries an `inverter` attribute. From the run:

```json
"sensor.p_batt_forecast_battery1": {
  "state": "0.00",
  "attributes": {"unit_of_measurement": "W", "friendly_name": "Battery Power Forecast Battery 1", "inverter": 1, "...": "..."}
}
```

The fleet total `sensor.p_batt_forecast` has no `inverter` attribute, since it spans both. The column and sensor reference is in [Plan output schema](../plan_output_schema.md) and [Publishing data](../publish_data.md).

Expected: `sensor.p_batt_forecast_battery0` has `inverter: 0` and `sensor.p_batt_forecast_battery1` has `inverter: 1`, and they keep it on every later publish.

## Step 4: Route each battery setpoint to its inverter

<!-- transport: Home Assistant 2026.10 (template rendered, not run on hardware) -->

The executor commands inverters, not batteries. This script finds the batteries on each inverter through the `inverter` attribute and sends their sum to that inverter, so moving a battery to the other inverter, or putting a second battery behind one, only changes the EMHASS config. The one map it keeps is from an inverter number to the entity that commands that inverter, because that is the device you write to. The Step 2 script calls it once, after the publish has succeeded, and passes the time the run started. Only battery sensors posted after that time are summed, so if a sensor's post failed, its inverter does not get the previous plan's value again. Put it under the same `script:` key as the Step 2 script.

```yaml
script:
  emhass_battery_setpoints:
    variables:
      # inverter number -> the entity that takes its battery power setpoint in W (placeholders)
      inverters:
        - {inverter: 0, setpoint: number.inverter_a_battery_power_setpoint}
        - {inverter: 1, setpoint: number.inverter_b_battery_power_setpoint}
    sequence:
      - repeat:
          for_each: "{{ inverters }}"
          sequence:
            - variables:
                # every battery sensor on this inverter, fresh or not; the bare sensor
                # counts only with one battery, when it carries the inverter attribute
                group: >
                  {{ states.sensor
                     | selectattr('entity_id', 'match', 'sensor[.]p_batt_forecast(_battery[0-9]+)?$')
                     | selectattr('attributes.inverter', 'defined')
                     | selectattr('attributes.inverter', 'eq', repeat.item.inverter)
                     | map(attribute='entity_id') | list }}
                # the ones this run posted
                batt_states: >
                  {{ expand(group)
                     | selectattr('last_reported', 'gt', run_started | as_datetime)
                     | map(attribute='state') | list }}
            - if: >
                {{ batt_states | count > 0 and batt_states | count == group | count
                   and batt_states | reject('is_number') | list | count == 0 }}
              then:
                - action: number.set_value
                  continue_on_error: true   # a refused value must not stop the other inverter
                  target: {entity_id: "{{ repeat.item.setpoint }}"}
                  data:
                    value: "{{ batt_states | map('float') | sum | round(0) }}"
```

`P_batt_{k}` is positive for discharge. Many inverters take the opposite sign, or separate charge and discharge entities, so adapt the last step to your device. An inverter gets no new setpoint, rather than an invented 0 W or a partial sum, if any battery sensor on it is unavailable or older than this run. With two batteries behind one inverter, both have to be fresh. An inverter that gets no new setpoint keeps its last one, which may be a full-rate charge or discharge: if your inverter has a safe self-use mode, send that from an `else:` branch of the `if:`. That branch only runs when Step 2 got this far. A run that stops in Step 2 (no forecast, a refused or unsolved run, EMHASS unreachable) never reaches it, so also add a watchdog: an automation that sends the safe mode when the battery sensors have not been posted for more than two steps.

If your inverter takes an AC-side power target rather than a battery power, send `sensor.p_hybrid_inverter_inverter{i}` instead, which is the AC side of that inverter with its PV included.

If you lower `number_of_batteries`, give `soc_init` and `soc_final` in Step 2 one value per remaining battery, or EMHASS refuses the run. Then restart Home Assistant. A battery sensor EMHASS no longer posts stays in Home Assistant, with its `inverter` attribute, until a restart (these sensors have no registry entry, so the only other way to remove one is `DELETE /api/states/<entity_id>`), and until then it keeps its inverter from getting a setpoint. Going down to a single battery, that is all the numbered ones: EMHASS then publishes the one battery on the bare `sensor.p_batt_forecast`, which carries the `inverter` attribute, so the script finds it there. If you set `custom_batt_forecast_id`, that single battery is published under your entity id instead, so change the pattern to match it.

Expected: on the run's first step (06:00, before the Step 3 excerpt), the two setpoints add up to about 464 W of discharge. In this run inverter A (inverter 0) gets all of it (`sensor.p_batt_forecast_battery0` at 463.92) and inverter B (inverter 1) gets 0 W; another solver version may give it to battery 1 instead at the same cost.

## Step 5: Curtailment and spill per inverter

<!-- source: src/emhass/optimization.py:685 (_log_spill) -->

The two flags give two different things to act on.

- Inverter 1 has `compute_curtailment` on, so the plan can ask it to produce less. If it takes a PV power limit, the PV the plan uses on it is `sensor.p_pv_forecast_inverter1` minus `sensor.p_pv_curtailment_inverter1` (a DC-side figure, before the inverter's efficiency). Set that as the limit while the curtailment sensor is above 0, and lift the limit when it is back at 0. Apply the same freshness test as Step 4 (`last_reported` after `run_started`) before you set a limit, and have the Step 4 watchdog lift the limit too, because a run that stops in Step 2 never gets here.
- Inverter 0 has curtailment off, so there is no PV limit to send. It still gets its battery setpoint from Step 4, and `sensor.p_pv_spill_inverter0` shows where the plan expects PV to go unused. The log names each step and the power in it:

```text
Inverter 0 spilled PV in 4 steps that its battery and the inverter could not or would not take: 2025-06-15 13:30:00+02:00: 1191 W, 2025-06-15 14:00:00+02:00: 1027 W, 2025-06-15 14:30:00+02:00: 758 W, 2025-06-15 15:00:00+02:00: 388 W
```

Most spill is the cap: the battery is full and the AC side is at its limit, so the inverter clips by itself. The plan can also spill PV the battery could take, when taking it would leave the battery above its `soc_final` at the end of the horizon; then the battery setpoint from Step 4 is what holds the battery back. A battery on an inverter that spills does not discharge in that step. If spill from the cap shows up most sunny days, the inverter is the bottleneck: a larger AC cap, or curtailment on that inverter if the hardware supports it, are the fixes.

Expected: the curtailment limit only applies in steps where `sensor.p_pv_curtailment_inverter1` is above 0, and inverter 0's battery setpoint is 0 W or charging in its spill steps.

## Caveats

- **Churn between the units.** Two hybrids can move energy through the AC bus, one discharging while the other charges. Both inverter efficiencies make that cost something, and small non-zero battery weights remove the rest. At a negative import price the plan may still do it on purpose, because there the loss pays (see the paragraph on battery weights with more than one battery in [Configuration](../config.md)).
- **`set_nodischarge_to_grid` changes meaning.** If it is true, one hybrid blocks battery discharge while exporting. With two inverters it becomes a single export ceiling (PV the inverters can deliver, minus the load), so a discharging battery may feed the other battery or a deferrable load while the site exports.
- **`set_nocharge_from_grid` stays plant-wide.** It compares total PV with total charge and does not look at which inverter either is on.
- **The sensor state is the current step, with `method_ts_round: first`.** Each per-inverter and per-battery sensor's state is the value of the plan row `publish-data` picks, and with `first` that is the step that contains the publish time; the whole horizon is in its list attribute. With the default `nearest`, a publish in the second half of a step picks the next step's row.
- **Other shapes.** A PV-only string inverter or an AC-coupled battery with its own inverter use the same two index lists: an inverter with only PV, or only a battery, takes that shape. See [How the optimizer models more than one inverter](../config.md#how-the-optimizer-models-more-than-one-inverter).
- **Fail-safe and security.** The Step 4 script already skips anything this run did not post. Add the watchdog from Step 4 for runs that stop early or stop running at all, and keep the EMHASS `/action/*` endpoints and the inverter controls off the internet.

## Credits

- Design and discussion: issue [#1150](https://github.com/davidusb-geek/emhass/issues/1150), from [#870](https://github.com/davidusb-geek/emhass/issues/870) and [#610](https://github.com/davidusb-geek/emhass/issues/610), with the two-hybrid site from [#1020](https://github.com/davidusb-geek/emhass/issues/1020).
- Reference: [Configuration](../config.md#per-inverter-settings-and-the-one-inverter-default), [Passing data](../passing_data.md#forecast-data-at-runtime), [Plan output schema](../plan_output_schema.md), [Publishing data](../publish_data.md).
- Field names verified against `src/emhass/data/config_defaults.json`, `src/emhass/utils.py`, `src/emhass/optimization.py` and `src/emhass/command_line.py` on 2026-11-10.
