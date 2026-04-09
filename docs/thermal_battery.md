# Thermal battery for heat pump optimization

EMHASS supports modeling thermal batteries for heat pump systems with thermal mass storage. This is useful when you have a heat pump with underfloor heating or a radiator system with a thermal storage medium like a concrete slab or water-based thermal mass.

The thermal battery model allows you to optimize heat pump operation by taking advantage of:
- Variable electricity pricing (run the heat pump during cheap periods)
- Solar PV production (use excess solar energy for heating)
- Building thermal inertia (pre-heat during low-cost periods)
- Heat pump efficiency variations with outdoor temperature

This implementation is based on the methodology from Langer & Volling (2020) "An optimal home energy management system for modular battery electric vehicles (MBEV) and the power grid".

## The thermal battery model

A thermal battery stores heat energy in a thermal mass (concrete slab, screed, water, etc.) which can be heated by a heat pump during periods of low electricity cost or high solar production. The stored heat is then gradually released to maintain comfortable indoor temperatures.

The model calculates the evolution of the thermal storage temperature based on:
- Heat pump operation (with COP depending on outdoor temperature)
- Heating demand from the building
- Thermal losses from the storage

The optimizer then determines when to run the heat pump to minimize costs while maintaining the storage temperature within acceptable bounds.

## Configuration parameters

To use a thermal battery, you need to configure it within the `def_load_config` list. The thermal battery is defined using the `thermal_battery` key.

### Required core parameters

These parameters define the basic thermal battery system:

* **supply_temperature**: The heat pump supply/flow temperature in °C. This is the temperature of water/fluid leaving the heat pump.
    * Underfloor heating: typically 30-40°C
    * Radiator systems: typically 50-70°C
    * Lower temperatures = better heat pump efficiency
    * Example: `35.0` for underfloor heating

* **volume**: Volume of the thermal storage medium in m³.
    * For underfloor heating: concrete slab volume (floor area × screed thickness)
    * For buffer tanks: tank volume in m³
    * Example: `20.0` for a 100m² floor with 20cm screed (100 × 0.2 = 20 m³)

* **start_temperature**: Initial temperature of the thermal storage in °C at the start of optimization.
    * Should match the current actual temperature
    * For underfloor: measure floor surface temperature
    * Example: `22.0`

* **min_temperatures**: Minimum allowed storage temperatures in °C per timestep (list).
    * Lower bound of the comfort range for each optimization timestep
    * Temperature should not drop below these values
    * Can be constant or vary by time of day (e.g., lower at night)
    * Example: `[20.0] * 48` for constant 20°C comfort over 48 timesteps
    * Example: `[18.0]*16 + [20.0]*16 + [18.0]*16` for night setback

* **max_temperatures**: Maximum allowed storage temperatures in °C per timestep (list).
    * Upper bound of the comfort range for each optimization timestep
    * Safety limit for the storage medium
    * Can be constant or vary by time of day
    * Example: `[28.0] * 48` for underfloor heating (avoid too hot floors)
    * Example: `[26.0]*24 + [28.0]*24` for different day/night limits

### Optional efficiency parameter

* **carnot_efficiency**: Real-world heat pump efficiency as a fraction of the ideal Carnot cycle (default: 0.4).
    * Typical range: 0.35-0.50 (35-50% of ideal Carnot efficiency)
    * Modern inverter heat pumps: 0.40-0.50
    * Standard on/off heat pumps: 0.35-0.42
    * Example: `0.42`
    * See the calibration section below for how to determine this

### Heating demand calculation

EMHASS needs to know how much heat your building requires. There are two methods to calculate this, and EMHASS automatically selects the appropriate method based on which parameters you provide.

#### Method 1: Physics-based (recommended)

This method models the actual building physics and is more accurate. It requires these parameters:

* **u_value**: Overall heat transfer coefficient of your building in W/(m²·K).
    * Lower values = better insulation
    * Typical values:
        * Modern passive house: 0.2-0.3
        * Well-insulated new build: 0.4-0.6
        * Average existing building: 0.8-1.2
        * Poorly insulated: 1.5-2.5
    * Example: `0.45`

* **envelope_area**: Total building envelope surface area in m².
    * Sum of all exterior walls, roof, and floor areas
    * Example: `380.0` for a typical single-family home

* **ventilation_rate**: Air change rate per hour (ACH).
    * Natural ventilation: 0.3-0.5
    * Mechanical ventilation with heat recovery: 0.3-0.4
    * Mechanical ventilation without heat recovery: 0.5-1.0
    * Example: `0.4`

* **heated_volume**: Interior volume being heated in m³.
    * Floor area × ceiling height
    * Example: `240.0` (e.g., 100m² × 2.4m height)

Optional solar gain parameters (if you want to account for passive solar heating):

* **window_area**: Total window area in m² (default: none).
    * Only include south/east/west-facing windows that receive significant sun
    * Example: `25.0`

* **shgc**: Solar Heat Gain Coefficient, fraction of solar radiation passing through windows (default: 0.6).
    * Modern double-glazed: 0.5-0.6
    * Low-e triple-glazed: 0.25-0.4
    * Example: `0.6`

Optional internal gains parameter (accounts for heat generated by electrical appliances):

* **internal_gains_factor**: Fraction of electrical load that becomes useful internal heat gains (default: 0.0).
    * Electrical appliances, lighting, and equipment generate heat when operating
    * This parameter uses the load power forecast to reduce heating demand
    * Typical values:
        * 0.0: No internal gains considered (default, backwards compatible)
        * 0.5-0.7: Conservative estimate (some heat lost to ventilation/drains)
        * 0.8-0.9: Most electrical energy becomes heat (well-insulated building)
        * 1.0: All electrical energy becomes internal heat (theoretical maximum)
    * Example: `0.7`

#### Method 2: Heating Degree Days (simpler, less accurate)

This method uses historical heating consumption data. It requires:

* **specific_heating_demand**: Annual heating energy demand in kWh/(m²·year).
    * Get this from your energy bills or energy certificate
    * Typical values:
        * Passive house: <15
        * Modern low-energy: 15-50
        * Standard modern: 50-100
        * Older buildings: 100-200
    * Example: `65.0`

* **area**: Heated floor area in m².
    * Example: `100.0`

Optional HDD parameters:

* **base_temperature**: Base temperature for heating degree day calculation in °C (default: 18.0).
* **annual_reference_hdd**: Annual heating degree days for your location (default: 3000.0).
    * Northern Europe: 3500-5000
    * Central Europe: 2500-3500
    * Southern Europe: 1500-2500

### Physical constants

By default, the thermal battery assumes concrete as the storage medium. You can override the physical constants to model other media such as water (for hot water tanks).

* **density**: Density of the thermal storage medium in kg/m³ (default: 2400).
    * Concrete: 2400
    * Water: 997
    * Example: `997` for a hot water tank

* **heat_capacity**: Specific heat capacity of the thermal storage medium in kJ/(kg·°C) (default: 0.88).
    * Concrete: 0.88
    * Water: 4.184
    * Example: `4.184` for a hot water tank

* **thermal_loss**: Constant standby heat loss rate from storage to environment in kW (default: 0.045).
    * For underfloor heating in concrete slab: 0.045
    * For a well-insulated hot water tank: 0.02-0.04
    * Example: `0.035`

All three values must be positive. Invalid values raise an error.

### Hot water tank mode (draw-off demand)

When `draw_off_demand` is present, the thermal battery switches to hot water tank mode. In this mode, the building heating demand calculation is skipped entirely. Instead, the tank has:

- A **constant standby loss** (`thermal_loss`) — heat escaping the tank to the surrounding room.
- A **draw-off demand profile** — energy withdrawn by hot water usage (showers, taps, etc.).

This is appropriate because a hot water tank sits in a room at roughly constant temperature — there are no outdoor-temperature-dependent losses, no solar gains, and no internal gains.

* **draw_off_demand**: Daily profile of hot water draw-off energy in kWh per timestep (default: none).
    * A list of energy values representing one day of hot water consumption
    * The profile is automatically tiled (repeated) to fill the optimization horizon
    * Each value represents the energy withdrawn during that timestep
    * Example for 30-minute timesteps: `[0,0,0,0,0,0, 0,0,0,0,0,0, 0.5,0.3,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0.8,0.5,0.3,0,0,0, 0,0,0,0,0,0]`
      (morning shower around 06:00, evening shower around 18:00)

When `draw_off_demand` is absent (or empty), the existing behavior applies — building heating demand with outdoor-temperature-dependent losses. This preserves backward compatibility for floor heating use cases.

The temperature dynamics in hot water tank mode:

```
conversion = 3600 / (density * heat_capacity * volume)

predicted_temp[t+1] = predicted_temp[t]
    + conversion * (cop[t] * p_deferrable[t] / 1000 * dt - draw_off_demand[t] - thermal_loss)
```

### Soft constraints (desired temperature / overshoot)

By default, `min_temperatures` and `max_temperatures` are enforced as hard constraints — the optimizer must keep the storage temperature strictly within bounds. You can optionally add soft constraints using desired temperatures and overshoot detection, following the same pattern as `thermal_config`.

When configured, the optimizer penalizes deviations from the desired temperature and suppresses heating when the storage temperature overshoots:

* **desired_temperatures**: Target temperatures in °C per timestep (default: none).
    * When present, enables soft comfort constraints
    * The penalty for deviating from these targets is added to the objective
    * Example: `[50.0] * 48` for constant 50°C target in a hot water tank

* **overshoot_temperature**: Temperature threshold above which heating is suppressed in °C (default: none).
    * When the predicted temperature exceeds this value, the heat pump is forced off
    * Prevents wasteful overheating
    * Example: `55.0`

* **penalty_factor**: Weight of the comfort deviation penalty in the objective (default: 10).
    * Higher values make the optimizer try harder to hit desired temperatures
    * Lower values give the optimizer more flexibility to shift heating for cost savings
    * Example: `10`

* **sense**: Direction of thermal control (default: `"heat"`).
    * `"heat"`: Penalizes temperatures below desired (standard heating)
    * `"cool"`: Penalizes temperatures above desired (cooling mode)
    * Example: `"heat"`

### Heat pump group coupling

If a single heat pump serves multiple thermal loads (e.g., underfloor heating AND a hot water tank), you can declare them as a group. The optimizer then ensures at most one load is active per timestep — matching the real-world constraint where a valve switches the heat pump between circuits.

* **heatpump_group**: Group identifier string (default: none).
    * All loads sharing the same group ID get mutual exclusivity constraints
    * Can be any string — loads are grouped by matching values
    * Works across `thermal_config` and `thermal_battery` load types
    * Works with both semi-continuous and non-semi-continuous loads
    * Example: `"hp1"`

For semi-continuous loads (`treat_deferrable_load_as_semi_cont: true`), the existing on/off binary (`p_def_bin2`) is reused. For non-semi-continuous loads (e.g., a hot water tank that always runs at fixed power), a new binary variable `hp_active` is created and linked to the power variable.

### Advanced parameters

* **thermal_loss**: Base thermal loss coefficient from storage to environment in kW (default: 0.045).
    * Only adjust this if you have measured data showing different loss rates
    * Example: `0.045`

### Thermal inertia (optional)

Real heating systems have a delay between the heat pump operating and the heat actually reaching the thermal mass (e.g., water circulating through underfloor pipes, slab warming up). The default model treats heat input as instantaneous, which can cause the optimizer to schedule heating at suboptimal times — especially for short pre-heating windows in MPC mode.

The `thermal_inertia_time_constant` parameter adds a first-order low-pass filter on the heat input, modeling this physical delay. When set, a new state variable `Q_input` tracks the *effective* heat energy reaching the thermal mass, smoothing out the raw heat pump output.

* **thermal_inertia_time_constant**: Time constant (τ) of the thermal inertia filter in hours (default: 0.0).
    * `0.0` (default): No filter, original instantaneous model — fully backward compatible
    * `0.5-1.0`: Light filtering, suitable for well-coupled radiator systems
    * `1.0-3.0`: Moderate filtering, typical for underfloor heating with concrete screed
    * `3.0-4.0`: Heavy filtering, thick slabs or poorly coupled systems
    * Values above 6.0 trigger a warning (unusually large)
    * Negative values raise an error
    * Example: `2.0` for a typical underfloor heating system

The filter equation at each timestep is:

```
Q_input[t+1] = Q_input[t] + α × (raw_heat[t] - Q_input[t])
```

where `α = time_step / τ` (clamped to 1.0 if τ < time_step). The temperature equation then uses `Q_input` instead of the raw heat pump output.

#### Warm-starting Q_input in MPC mode

When using MPC (repeated optimizations), `Q_input` automatically persists between solves via the optimization cache. The value from timestep 1 of the previous solve becomes the initial value for the next solve. This means the optimizer "remembers" the thermal state of the system without any manual intervention.

For manual control, you can override the initial Q_input value:

* **q_input_initial**: Manual override for the initial Q_input value in kWh (default: 0.0).
    * Only needed if you want to explicitly set the starting thermal energy in the filter
    * When set, it takes priority over the auto-persisted value from the previous solve
    * Example: `0.5`

#### Published sensors

When `thermal_inertia_time_constant > 0`, an additional sensor is published:

* **sensor.q_input_heater{k}** — Filtered heat input reaching the thermal mass (kWh per timestep)

This sensor shows the effective heat delivery after accounting for the system's thermal lag. Compare it with `P_deferrable{k}` to see the smoothing effect.

## Example configurations

### Example 1: Modern home with underfloor heating, solar and internal gains

This example shows a well-insulated modern home with underfloor heating, using the physics-based heating demand calculation including solar gains through windows and internal gains from electrical appliances.

```python
{
  "def_load_config": [
    {
      "thermal_battery": {
        "supply_temperature": 35.0,
        "volume": 18.0,
        "start_temperature": 22.0,
        "min_temperatures": [20.0] * 48,  # Constant 20°C minimum
        "max_temperatures": [28.0] * 48,  # Constant 28°C maximum
        "carnot_efficiency": 0.45,
        "u_value": 0.35,
        "envelope_area": 380.0,
        "ventilation_rate": 0.4,
        "heated_volume": 240.0,
        "window_area": 28.0,
        "shgc": 0.6,
        "internal_gains_factor": 0.7
      }
    }
  ]
}
```

This configuration:
- Models a 90m² underfloor heating slab with 20cm screed (90 × 0.2 = 18 m³)
- Uses 35°C supply temperature (typical for underfloor heating)
- Efficient modern heat pump (45% Carnot efficiency)
- Well-insulated building (U-value 0.35)
- Includes solar gains through 28m² of south-facing windows
- Accounts for 70% of electrical load becoming internal heat gains
- Maintains floor temperature between 20-28°C

### Example 2: Older home with radiators, simple configuration

This example uses the simpler HDD-based approach for an older home with radiator heating.

```python
{
  "def_load_config": [
    {
      "thermal_battery": {
        "supply_temperature": 50.0,
        "volume": 12.0,
        "start_temperature": 45.0,
        "min_temperatures": [40.0] * 48,  # Constant 40°C minimum
        "max_temperatures": [65.0] * 48,  # Constant 65°C maximum
        "carnot_efficiency": 0.38,
        "specific_heating_demand": 95.0,
        "area": 120.0,
        "base_temperature": 18.0,
        "annual_reference_hdd": 2800.0
      }
    }
  ]
}
```

This configuration:
- Models a 60m² concrete floor as thermal mass (60 × 0.2 = 12 m³)
- Uses 50°C supply temperature (needed for radiators)
- Standard efficiency heat pump (38% Carnot efficiency, lower due to higher supply temperature)
- Older building consuming 95 kWh/m²/year
- 120m² heated floor area
- Maintains thermal mass between 40-65°C

### Example 3: Underfloor heating with thermal inertia

This example shows how to use the thermal inertia filter for a system where there is a measurable delay between heat pump operation and temperature change in the slab.

```python
{
  "def_load_config": [
    {
      "thermal_battery": {
        "supply_temperature": 35.0,
        "volume": 18.0,
        "start_temperature": 22.0,
        "min_temperatures": [20.0] * 48,
        "max_temperatures": [28.0] * 48,
        "carnot_efficiency": 0.45,
        "u_value": 0.35,
        "envelope_area": 380.0,
        "ventilation_rate": 0.4,
        "heated_volume": 240.0,
        "thermal_inertia_time_constant": 2.0
      }
    }
  ]
}
```

This configuration:
- Uses a 2-hour thermal inertia time constant, modeling the delay in heat transfer through the concrete slab
- The optimizer will schedule heating earlier to account for the lag, resulting in better pre-heating behavior
- The `q_input_heater0` sensor shows the filtered heat delivery to the slab
- In MPC mode, Q_input automatically persists between solves for continuity

### Example 4: Multiple deferrable loads

If you have other deferrable loads (EV charger, dishwasher, etc.) along with your thermal battery:

```python
{
  "def_load_config": [
    {},
    {
      "thermal_battery": {
        "supply_temperature": 35.0,
        "volume": 15.0,
        "start_temperature": 22.0,
        "min_temperatures": [20.0] * 48,  # Constant 20°C minimum
        "max_temperatures": [26.0] * 48,  # Constant 26°C maximum
        "u_value": 0.45,
        "envelope_area": 320.0,
        "ventilation_rate": 0.5,
        "heated_volume": 200.0
      }
    },
    {}
  ]
}
```

In this case:
- Load 0: Regular deferrable load (e.g., EV charger)
- Load 1: Thermal battery with heat pump
- Load 2: Another regular deferrable load (e.g., washing machine)

### Example 5: Hot water tank

A 200-liter hot water tank heated by a heat pump, with a daily shower profile.

```python
{
  "def_load_config": [
    {
      "thermal_battery": {
        "supply_temperature": 45.0,
        "volume": 0.2,
        "density": 997,
        "heat_capacity": 4.184,
        "thermal_loss": 0.035,
        "start_temperature": 50.0,
        "min_temperatures": [40.0] * 48,
        "max_temperatures": [60.0] * 48,
        "carnot_efficiency": 0.40,
        "draw_off_demand": [0,0,0,0,0,0, 0,0,0,0,0,0,
                            0.5,0.3,0,0,0,0, 0,0,0,0,0,0,
                            0,0,0,0,0,0, 0,0,0,0,0,0,
                            0.8,0.5,0.3,0,0,0, 0,0,0,0,0,0]
      }
    }
  ]
}
```

This configuration:
- Models a 200-liter (0.2 m³) hot water tank with water physics (density=997 kg/m³, heat_capacity=4.184 kJ/(kg·°C))
- Uses 45°C supply temperature (typical for domestic hot water)
- Has a constant 0.035 kW standby loss (well-insulated tank)
- Defines a daily draw-off profile (48 half-hour timesteps): morning shower at 06:00 (0.5 + 0.3 kWh) and evening shower at 18:00 (0.8 + 0.5 + 0.3 kWh)
- The profile repeats automatically if the optimization horizon exceeds 24 hours
- Maintains tank temperature between 40-60°C

### Example 6: Hot water tank

A 200-liter hot water tank heated by a heat pump, without any `draw_off_demand`.

```python
{
  "def_load_config": [
    {
      "thermal_battery": {
        "supply_temperature": 45.0,
        "volume": 0.2,
        "density": 997,
        "heat_capacity": 4.184,
        "thermal_loss": 0.035,
        "start_temperature": 50.0,
        "min_temperatures": [40.0] * 48,
        "max_temperatures": [60.0] * 48,
        "carnot_efficiency": 0.40,
        "specific_heating_demand": 0.0,
        "area": 1.0
      }
    }
  ]
}
```

This configuration:
- Models a 200-liter (0.2 m³) hot water tank with water physics (density=997 kg/m³, heat_capacity=4.184 kJ/(kg·°C))
- Uses 45°C supply temperature (typical for domestic hot water)
- Has a constant 0.035 kW standby loss (well-insulated tank)
- This config does not define a demand profile, to ensure backward compatibility the parameter `specific_heating_demand` and `area` must be present.
- Maintains tank temperature between 40-60°C

### Example 7: Hot water tank with soft constraints

Same hot water tank but with soft constraints to target 50°C while allowing deviations when electricity is expensive.

```python
{
  "def_load_config": [
    {
      "thermal_battery": {
        "supply_temperature": 45.0,
        "volume": 0.2,
        "density": 997,
        "heat_capacity": 4.184,
        "thermal_loss": 0.035,
        "start_temperature": 50.0,
        "min_temperatures": [40.0] * 48,
        "max_temperatures": [60.0] * 48,
        "desired_temperatures": [50.0] * 48,
        "overshoot_temperature": 55.0,
        "penalty_factor": 10,
        "sense": "heat",
        "carnot_efficiency": 0.40,
        "draw_off_demand": [0,0,0,0,0,0, 0,0,0,0,0,0,
                            0.5,0.3,0,0,0,0, 0,0,0,0,0,0,
                            0,0,0,0,0,0, 0,0,0,0,0,0,
                            0.8,0.5,0.3,0,0,0, 0,0,0,0,0,0]
      }
    }
  ]
}
```

This configuration:
- Targets 50°C (desired) but allows the optimizer to let it drop toward 40°C (min) when electricity is expensive
- Suppresses heating when temperature exceeds 55°C (overshoot)
- The penalty factor (10) balances comfort vs cost — increase for tighter temperature control

### Example 8: Heat pump group (underfloor heating + hot water tank)

A single heat pump serving both underfloor heating and a hot water tank. The optimizer ensures only one is active at a time.

```python
{
  "num_def_loads": 2,
  "nominal_power_of_deferrable_loads": [1000, 2000],
  "treat_deferrable_load_as_semi_cont": [true, false],
  "def_load_config": [
    {
      "thermal_battery": {
        "heatpump_group": "hp1",
        "indoor_target_temperature": 22,
        "volume": 8,
        "u_value": 0.3,
        "envelope_area": 400.0,
        "ventilation_rate": 0.5,
        "heated_volume": 450.0,
        "carnot_efficiency": 0.32,
        "supply_temperature": 30.0,
        "min_temperatures": [20.0] * 48,
        "max_temperatures": [22.0] * 48,
        "start_temperature": 20.0
      }
    },
    {
      "thermal_battery": {
        "heatpump_group": "hp1",
        "supply_temperature": 45.0,
        "volume": 0.2,
        "density": 997,
        "heat_capacity": 4.184,
        "thermal_loss": 0.035,
        "carnot_efficiency": 0.32,
        "start_temperature": 50.0,
        "min_temperatures": [40.0] * 48,
        "max_temperatures": [60.0] * 48,
        "carnot_efficiency": 0.40,
        "draw_off_demand": [0,0,0,0,0,0, 0,0,0,0,0,0,
                            0.5,0.3,0,0,0,0, 0,0,0,0,0,0,
                            0,0,0,0,0,0, 0,0,0,0,0,0,
                            0.8,0.5,0.3,0,0,0, 0,0,0,0,0,0]
      }
    }
  ]
}
```

This configuration:
- Load 0: Underfloor heating (semi-continuous, modulates 0-1000W) using `thermal_battery`
- Load 1: Hot water tank (non-semi-continuous, fixed 2000W) using `thermal_battery`
- Both share `"heatpump_group": "hp1"` — the optimizer ensures at most one is active per timestep
- For the semi-continuous load, the existing on/off binary is reused
- For the non-semi-continuous hot water tank, a new `hp_active` binary is created automatically

## How the optimization works

The thermal battery optimization uses a physics-based model with three main components:

### 1. Heat pump COP (Coefficient of Performance)

The heat pump COP varies with outdoor temperature. EMHASS calculates this using a Carnot-based formula:

```
COP = carnot_efficiency × T_supply_K / (T_supply_K - T_outdoor_K)
```

Where temperatures are in Kelvin (K = °C + 273.15).

Example: With 35°C supply, 5°C outdoor, and 0.4 Carnot efficiency:
- COP = 0.4 × 308.15 / 30 = 4.1
- Meaning: for every 1 kWh of electricity, you get 4.1 kWh of heat

Key insights:
- COP is higher when outdoor temperature is closer to supply temperature
- Lower supply temperatures (underfloor heating) give better COP than high temperatures (radiators)
- The `carnot_efficiency` parameter lets you match your specific heat pump's performance

### 2. Thermal losses

The thermal storage gradually loses heat to the environment. The loss model depends on the mode:

**Building heating mode** (no `draw_off_demand`): Uses the methodology from Langer & Volling (2020):

```
Loss = thermal_loss × (1 - 2 × Hot)
```

Where `Hot = 1` if outdoor temp ≥ indoor temp, else `0`. This means losses are positive when it's cold outside (heat escapes) and negative when warm (passive heat gain).

**Hot water tank mode** (with `draw_off_demand`): Uses a constant standby loss:

```
Loss = thermal_loss  (constant, not dependent on outdoor temperature)
```

This is appropriate because a tank sits indoors at roughly constant ambient temperature.

### 3. Heating demand / draw-off demand

**Building heating mode**: The building requires heat to maintain comfort. Calculated from:
- Physics-based: transmission losses + ventilation losses - solar gains
- HDD-based: historical consumption scaled by current weather

**Hot water tank mode**: The draw-off demand profile replaces the building heating demand. It represents energy withdrawn by hot water consumption (showers, taps). The daily profile is tiled to fill the optimization horizon.

### 4. Thermal balance

At each timestep, the storage temperature changes based on:

```
conversion = 3600 / (density × heat_capacity × volume)

predicted_temp[t+1] = predicted_temp[t]
    + conversion × (cop[t] × P[t] / 1000 × dt - demand[t] - loss[t])
```

Where `demand[t]` is either building heating demand or draw-off demand, and `loss[t]` is either outdoor-temperature-dependent or constant, depending on the mode.

The optimizer decides when to run the heat pump to:
- Minimize electricity costs (plus comfort penalty if soft constraints are configured)
- Keep storage temperature within min/max bounds (hard constraints)
- Approach desired temperatures if configured (soft constraints)
- Respect mutual exclusivity if in a heat pump group

### 5. Heat pump group coupling (optional)

When multiple loads share the same `heatpump_group`, the optimizer adds a mutual exclusivity constraint:

```
sum(activity_binary[k][t] for k in group) <= 1,  for all t
```

This ensures at most one load is active per timestep. Both loads can be off simultaneously. The optimizer decides the optimal time allocation between loads to minimize total cost while satisfying all temperature constraints.

### 5. Thermal inertia filter (optional)

When `thermal_inertia_time_constant` is set to a value greater than 0, the raw heat pump output passes through a first-order low-pass filter before affecting the storage temperature. This models the physical delay in heat transfer (e.g., water circulating through pipes, concrete warming up).

The filter introduces a new state variable `Q_input` that tracks the effective heat delivery:

```
Q_input[t+1] = Q_input[t] + (Δt/τ) × (raw_heat[t] - Q_input[t])
```

The temperature equation then uses `Q_input` instead of the raw heat, resulting in a smoother, delayed temperature response that better matches real-world behavior. This is particularly beneficial for MPC mode where short prediction horizons can lead to suboptimal scheduling without accounting for the thermal lag.

## Calibrating your thermal battery parameters

### Step 1: Measure your building

For physics-based approach:

1. **U-value**: Check your energy performance certificate, or use typical values based on your building age
2. **Envelope area**: Measure or calculate from building plans (walls + roof + floor)
3. **Ventilation rate**: Check your ventilation system specs, or use 0.4-0.5 for typical homes
4. **Heated volume**: Floor area × ceiling height

For HDD approach:

1. **Specific heating demand**: Check energy bills (annual heating kWh / floor area m²)
2. **Area**: Measure your heated floor area

### Step 2: Determine thermal mass volume

For underfloor heating:
- Measure floor area that has heating pipes
- Measure screed thickness (typically 15-25 cm)
- Volume = area × thickness (in meters)
- Example: 75 m² floor with 20 cm screed = 75 × 0.2 = 15 m³

For radiator systems with thermal mass:
- Estimate the concrete/masonry volume that stores heat
- Include floor slabs, internal walls in heated areas
- Be conservative (underestimate rather than overestimate)

### Step 3: Find your heat pump supply temperature

Check your heat pump controller or settings:
- For underfloor: usually shown as "flow temperature" or "supply temp"
- Typical range: 30-40°C depending on outdoor temp (heat curve)
- For the optimizer, use a typical mid-season value (e.g., 35°C)

### Step 4: Calibrate Carnot efficiency

Method 1 - From manufacturer specs:
- Find your heat pump's rated COP at a specific test condition (e.g., A7/W35)
- A7/W35 means: 7°C outdoor air, 35°C water output
- Calculate: `carnot_eff = COP_rated × (T_supply_K - T_outdoor_K) / T_supply_K`
- Example: COP = 4.5 at A7/W35
  - ΔT = 35 - 7 = 28 K
  - carnot_eff = 4.5 × 28 / 308.15 = 0.409 ≈ **0.41**

Method 2 - From real data:
- Monitor your heat pump for a few days
- Record: electricity consumed, heat delivered (if your heat pump shows this), outdoor temp, supply temp
- Calculate: actual COP = heat delivered / electricity consumed
- Then: carnot_eff = COP_actual × (T_supply_K - T_outdoor_K) / T_supply_K

Typical values:
- Modern inverter heat pump: 0.40-0.50
- Standard on/off heat pump: 0.35-0.42
- Older heat pump: 0.30-0.38

### Step 5: Validate and tune

Run the optimization for a past week and compare:
- Predicted vs actual heat pump runtime
- Predicted vs actual energy consumption
- Predicted vs actual floor/storage temperatures

If there's more than 20% difference:
- Adjust U-value if heating demand is consistently wrong
- Adjust carnot_efficiency if COP seems wrong
- Adjust thermal mass volume if temperature changes are too fast/slow

## Runtime parameters for optimization

When calling the optimization API (day-ahead or MPC), you need to provide the thermal battery configuration. Here's a complete example:

```yaml
rest_command:
  emhass_thermal_battery_optim:
    url: http://localhost:5000/action/naive-mpc-optim
    method: post
    headers:
      content-type: application/json
    payload: >
      {
        "prediction_horizon": 24,
        "load_cost_forecast": {{ (state_attr('sensor.electricity_price', 'forecasts') | map(attribute='price') | list)[:24] | tojson }},
        "outdoor_temperature_forecast": {{ (state_attr('sensor.weather_forecast', 'forecast') | map(attribute='temperature') | list)[:24] | tojson }},
        "def_load_config": [
          {
            "thermal_battery": {
              "supply_temperature": 35.0,
              "volume": 18.0,
              "start_temperature": {{ states('sensor.floor_temperature') | float }},
              "min_temperatures": {{ [20.0] * 24 | tojson }},
              "max_temperatures": {{ [28.0] * 24 | tojson }},
              "carnot_efficiency": 0.42,
              "u_value": 0.35,
              "envelope_area": 380.0,
              "ventilation_rate": 0.4,
              "heated_volume": 240.0,
              "window_area": 28.0,
              "shgc": 0.6
            }
          }
        ]
      }
```

Important notes:
- `outdoor_temperature_forecast` is required for thermal battery optimization (building heating mode)
- `start_temperature` should ideally come from a real sensor (floor temp for underfloor heating, tank sensor for hot water)
- If using solar gains, ensure your forecast data includes `ghi` (global horizontal irradiance)
- Add `"thermal_inertia_time_constant": 2.0` to the `thermal_battery` dict to enable the thermal inertia filter
- In MPC mode, `Q_input` auto-persists between solves; use `"q_input_initial": 0.5` to manually override
- For hot water tanks, add `"density": 997, "heat_capacity": 4.184` and a `"draw_off_demand"` profile or `"specific_heating_demand": 0.0, "area":1.0`
- For heat pump groups, add `"heatpump_group": "hp1"` to each coupled load

## Published sensors

After running optimization and publishing results, EMHASS creates these sensors in Home Assistant:

For each thermal battery (where `k` is the load index, starting from 0):

1. **sensor.p_deferrable{k}** - Heat pump power schedule (W)
2. **sensor.heating_demand{k}** - Heating energy demand per timestep (kWh)
3. **sensor.temp_predicted{k}** - Predicted thermal storage temperature (°C)
4. **sensor.q_input_heater{k}** - Filtered heat input (kWh per timestep) — only when `thermal_inertia_time_constant > 0`

You can customize these sensor names:

```json
{
  "custom_deferrable_forecast_id": [
    {
      "entity_id": "sensor.heatpump_power_schedule",
      "friendly_name": "Heat Pump Power Schedule"
    }
  ],
  "custom_heating_demand_id": [
    {
      "entity_id": "sensor.heating_demand",
      "friendly_name": "Heating Demand"
    }
  ],
  "custom_predicted_temperature_id": [
    {
      "entity_id": "sensor.floor_temperature_predicted",
      "friendly_name": "Floor Temperature (Predicted)"
    }
  ]
}
```

## Using the optimization results

### Controlling your heat pump

Create an automation that follows the optimized schedule:

```yaml
automation:
  - alias: "Heat Pump - EMHASS Optimal Control"
    trigger:
      - platform: state
        entity_id: sensor.heatpump_power_schedule
    action:
      - choose:
          - conditions:
              - condition: template
                value_template: "{{ states('sensor.heatpump_power_schedule') | float > 100 }}"
            sequence:
              - service: climate.turn_on
                target:
                  entity_id: climate.heat_pump
          - conditions:
              - condition: template
                value_template: "{{ states('sensor.heatpump_power_schedule') | float <= 100 }}"
            sequence:
              - service: climate.turn_off
                target:
                  entity_id: climate.heat_pump
```

### Monitoring heating costs

Track your heating expenses:

```yaml
template:
  - sensor:
      - name: "Heating Cost Today"
        unit_of_measurement: "€"
        device_class: monetary
        state: >
          {% set demand = states('sensor.heating_demand') | float %}
          {% set price = states('sensor.electricity_price') | float %}
          {{ (demand * price) | round(2) }}
```

### Energy dashboard integration

Add the heating demand sensor to Home Assistant's energy dashboard to track heating energy consumption over time.

## Troubleshooting

### Optimization returns "Infeasible"

**Possible causes:**
- Temperature constraints too tight (try widening min/max range)
- Thermal mass volume too small for the heating demand
- Heat pump power rating too low
- Heat pump group: loads in the same group compete for time — not enough timesteps to satisfy both

**Solutions:**
- Increase the gap between min and max temperatures
- Verify your volume calculation is correct
- Check that `nominal_power_of_deferrable_loads` is set correctly for your heat pump
- For heat pump groups: widen temperature bounds on one or both loads, or increase the prediction horizon

### COP values seem wrong

Check these:
- Supply temperature is correct for your system (30-40°C underfloor, 50-70°C radiators)
- Outdoor temperature forecast is in Celsius (not Fahrenheit!)
- Carnot efficiency is reasonable (0.35-0.50)

Expected COP ranges:
- Underfloor (35°C) at 5°C outdoor: COP ≈ 4.1
- Radiators (50°C) at 5°C outdoor: COP ≈ 2.9
- Very cold weather (35°C, -10°C): COP ≈ 2.7

### Heating demand too high or too low

**For physics-based method:**
- Check U-value is realistic for your building
- Verify envelope area includes all exterior surfaces
- Ensure ventilation rate matches your actual system

**For HDD method:**
- Verify specific_heating_demand matches your energy bills
- Check that area is heated floor area (not total building area)

### Floor gets too hot or too cold

The optimizer is keeping storage temperature within bounds, but:
- If too hot: lower the values in `max_temperatures`
- If too cold: raise the values in `min_temperatures`
- If temperature swings are too large: check your thermal mass `volume` is accurate
- Consider using variable temperature limits (night setback) by adjusting the list values

### Thermal inertia not having an effect

If you set `thermal_inertia_time_constant` but don't see a difference:
- Check the value is greater than 0 (τ=0 disables the filter)
- The effect is most visible in MPC with short prediction horizons
- Compare `q_input_heater0` with `P_deferrable0` — if they look identical, τ may be too small relative to your time step
- If τ < time_step (e.g., τ=0.3h with 30min steps), the filter coefficient is clamped to 1.0, effectively bypassing the filter

### Solar gains not working

Requirements for solar gains:
1. Define `window_area` in your config
2. Optimization data must include `ghi` (global horizontal irradiance) column
3. Must use physics-based method (not HDD method)

Check the logs for: "Using physics-based heating demand with solar gains"

### Hot water tank temperature drops too fast

If the tank temperature drops faster than expected:
- Check `thermal_loss` is realistic for your tank insulation (typical: 0.02-0.04 kW)
- Verify `draw_off_demand` values are in kWh per timestep (not total daily)
- Check `density` (997 for water) and `heat_capacity` (4.184 for water) are correct
- Verify `volume` matches your actual tank size in m³ (200 liters = 0.2 m³)

### Heat pump group not enforcing mutual exclusivity

If both loads in a group seem to run simultaneously:
- Verify both loads have the same `heatpump_group` value (string match is exact)
- Check the logs for "mutual exclusivity constraint added"
- If a group has only 1 load, the constraint is skipped (need at least 2 loads)

### Soft constraints not working

If `desired_temperatures` doesn't seem to affect the optimization:
- Ensure `desired_temperatures` is set (not just `min_temperatures`/`max_temperatures`)
- Try increasing `penalty_factor` (higher = tighter tracking)
- Check `sense` is correct: `"heat"` for heating, `"cool"` for cooling
- If `overshoot_temperature` is too close to `max_temperatures`, the optimizer may have no room to maneuver

## Tips for best results

1. **Start simple**: Use the HDD method first to get familiar, then switch to physics-based for better accuracy

2. **Measure accurately**: The quality of your optimization depends on accurate parameters

3. **Monitor and adjust**: Run the optimizer for a week, compare actual vs predicted, then tune parameters

4. **Give the optimizer flexibility**: Use a reasonable temperature range (e.g., 20-28°C for underfloor) rather than a tight range (23-24°C). This allows the optimizer to find the most cost-effective solution.

5. **Update start temperature**: For MPC, always use the actual current temperature from a sensor

6. **Consider weather**: The optimizer works best when outdoor temperature forecasts are accurate

7. **Size your thermal mass correctly**: For underfloor heating, measure the actual screed volume with heating pipes. Don't include areas without heating.

8. **Model DHW explicitly**: If your heat pump provides domestic hot water, model the tank as a separate `thermal_battery` with water physics (`density: 997`, `heat_capacity: 4.184`) and a `draw_off_demand` profile. Use `heatpump_group` to couple it with your space heating load.

9. **Solar gains matter**: If you have significant south-facing windows, modeling solar gains can improve optimization accuracy by 10-20%

10. **Validate regularly**: Compare predicted vs actual energy consumption weekly and adjust if needed

11. **Use thermal inertia for underfloor heating**: If you notice the optimizer schedules heating too late for short pre-heating windows, try setting `thermal_inertia_time_constant` to 1.0-3.0 hours. This models the delay between heat pump operation and measurable temperature change in the slab.

## References

- Langer, T., & Volling, T. (2020). "An optimal home energy management system for modulating heat pumps and photovoltaic systems". *Applied Energy*, Vol. 278. [https://doi.org/10.1016/j.apenergy.2020.115661](https://doi.org/10.1016/j.apenergy.2020.115661)

For more information:
- [Thermal Model (thermal_config)](thermal_model.md) - For direct heater/AC control without thermal mass
- [Configuration Parameters](config.md) - Complete EMHASS parameter reference
