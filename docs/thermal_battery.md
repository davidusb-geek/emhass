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

### Advanced parameters

* **thermal_loss_coefficient**: Base thermal loss coefficient from storage to environment in kW (default: 0.045).
    * Only adjust this if you have measured data showing different loss rates
    * Example: `0.045`

## Example configurations

### Example 1: Modern home with underfloor heating and solar gains

This example shows a well-insulated modern home with underfloor heating, using the physics-based heating demand calculation including solar gains through windows.

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
        "shgc": 0.6
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

### Example 3: Multiple deferrable loads

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

The thermal storage gradually loses heat to the environment. EMHASS uses the methodology from Langer & Volling (2020):

```
Loss = thermal_loss_coefficient × (1 - 2 × Hot)
```

Where `Hot = 1` if outdoor temp ≥ indoor temp, else `0`.

This means:
- When it's cold outside: positive losses (heat escapes)
- When it's warm outside: negative losses (passive heat gain)

### 3. Heating demand

The building requires a certain amount of heat to maintain comfort. This is calculated either from:
- Physics-based: transmission losses + ventilation losses - solar gains
- HDD-based: historical consumption scaled by current weather

### 4. Thermal balance

At each timestep, the storage temperature changes based on:
- Heat added by the heat pump (at its COP efficiency)
- Heat removed by the building heating demand
- Thermal losses (or gains) from the environment

The optimizer decides when to run the heat pump to:
- Minimize electricity costs
- Keep storage temperature within min/max bounds
- Satisfy heating requirements

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
- `outdoor_temperature_forecast` is required for thermal battery optimization
- `start_temperature` should ideally come from a real sensor (floor temp for underfloor heating)
- If using solar gains, ensure your forecast data includes `ghi` (global horizontal irradiance)

## Published sensors

After running optimization and publishing results, EMHASS creates these sensors in Home Assistant:

For each thermal battery (where `k` is the load index, starting from 0):

1. **sensor.p_deferrable{k}** - Heat pump power schedule (W)
2. **sensor.heating_demand{k}** - Heating energy demand per timestep (kWh)
3. **sensor.temp_predicted{k}** - Predicted thermal storage temperature (°C)

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

**Solutions:**
- Increase the gap between min and max temperatures
- Verify your volume calculation is correct
- Check that `nominal_power_of_deferrable_loads` is set correctly for your heat pump

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

### Solar gains not working

Requirements for solar gains:
1. Define `window_area` in your config
2. Optimization data must include `ghi` (global horizontal irradiance) column
3. Must use physics-based method (not HDD method)

Check the logs for: "Using physics-based heating demand with solar gains"

## Tips for best results

1. **Start simple**: Use the HDD method first to get familiar, then switch to physics-based for better accuracy

2. **Measure accurately**: The quality of your optimization depends on accurate parameters

3. **Monitor and adjust**: Run the optimizer for a week, compare actual vs predicted, then tune parameters

4. **Give the optimizer flexibility**: Use a reasonable temperature range (e.g., 20-28°C for underfloor) rather than a tight range (23-24°C). This allows the optimizer to find the most cost-effective solution.

5. **Update start temperature**: For MPC, always use the actual current temperature from a sensor

6. **Consider weather**: The optimizer works best when outdoor temperature forecasts are accurate

7. **Size your thermal mass correctly**: For underfloor heating, measure the actual screed volume with heating pipes. Don't include areas without heating.

8. **Account for DHW**: If your heat pump also provides domestic hot water, you may need to adjust your parameters or run a separate thermal battery for DHW

9. **Solar gains matter**: If you have significant south-facing windows, modeling solar gains can improve optimization accuracy by 10-20%

10. **Validate regularly**: Compare predicted vs actual energy consumption weekly and adjust if needed

## References

- Langer, T., & Volling, T. (2020). "An optimal home energy management system for modular battery electric vehicles (MBEV) and the power grid". *Energies*, 13(20), 5279. [DOI: 10.3390/en13205279](https://doi.org/10.3390/en13205279)

For more information:
- [Thermal Model (thermal_config)](thermal_model.md) - For direct heater/AC control without thermal mass
- [Configuration Parameters](config.md) - Complete EMHASS parameter reference