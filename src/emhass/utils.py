from __future__ import annotations

import ast
import copy
import csv
import logging
import os
import pathlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd
import plotly.express as px
import pytz
import yaml

if TYPE_CHECKING:
    from emhass.machine_learning_forecaster import MLForecaster

pd.options.plotting.backend = "plotly"

# Unit conversion constants
W_TO_KW = 1000  # Watts to kilowatts conversion factor


def get_root(file: str, num_parent: int = 3) -> str:
    """
    Get the root absolute path of the working directory.

    :param file: The passed file path with __file__
    :return: The root path
    :param num_parent: The number of parents levels up to desired root folder
    :type num_parent: int, optional
    :rtype: str

    """
    if num_parent == 3:
        root = pathlib.Path(file).resolve().parent.parent.parent
    elif num_parent == 2:
        root = pathlib.Path(file).resolve().parent.parent
    elif num_parent == 1:
        root = pathlib.Path(file).resolve().parent
    else:
        raise ValueError("num_parent value not valid, must be between 1 and 3")
    return root


def get_logger(
    fun_name: str,
    emhass_conf: dict[str, pathlib.Path],
    save_to_file: bool = True,
    logging_level: str = "DEBUG",
) -> tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.

    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param save_to_file: Write log to a file, defaults to True
    :type save_to_file: bool, optional
    :return: The logger object and the handler
    :rtype: object

    """
    # create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.fileSetting = save_to_file
    if save_to_file:
        if os.path.isdir(emhass_conf["data_path"]):
            ch = logging.FileHandler(emhass_conf["data_path"] / "logger_emhass.log")
        else:
            raise Exception("Unable to access data_path: " + emhass_conf["data_path"])
    else:
        ch = logging.StreamHandler()
    if logging_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch


def _get_now() -> datetime:
    """Helper function to get the current time, for easier mocking."""
    return datetime.now()


def get_forecast_dates(
    freq: int,
    delta_forecast: int,
    time_zone: datetime.tzinfo,
    timedelta_days: int | None = 0,
) -> pd.core.indexes.datetimes.DatetimeIndex:
    """
    Get the date_range list of the needed future dates using the delta_forecast parameter.

    :param freq: Optimization time step.
    :type freq: int
    :param delta_forecast: Number of days to forecast in the future to be used for the optimization.
    :type delta_forecast: int
    :param timedelta_days: Number of truncated days needed for each optimization iteration, defaults to 0
    :type timedelta_days: Optional[int], optional
    :return: A list of future forecast dates.
    :rtype: pd.core.indexes.datetimes.DatetimeIndex

    """
    freq = pd.to_timedelta(freq, "minutes")
    start_time = _get_now()

    start_forecast = pd.Timestamp(start_time, tz=time_zone).replace(microsecond=0).floor(freq=freq)
    end_forecast = start_forecast + pd.tseries.offsets.DateOffset(days=delta_forecast)
    final_end_date = end_forecast + pd.tseries.offsets.DateOffset(days=timedelta_days) - freq

    forecast_dates = pd.date_range(
        start=start_forecast,
        end=final_end_date,
        freq=freq,
        tz=time_zone,
    )

    return [ts.isoformat() for ts in forecast_dates]


def calculate_cop_heatpump(
    supply_temperature: float,
    carnot_efficiency: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
) -> np.ndarray:
    r"""
    Calculate heat pump Coefficient of Performance (COP) for each timestep in the prediction horizon.

    The COP is calculated using a Carnot-based formula:

    .. math::
        COP(h) = \eta_{carnot} \times \frac{T_{supply\_K}}{|T_{supply\_K} - T_{outdoor\_K}(h)|}

    Where temperatures are converted to Kelvin (K = °C + 273.15).

    This formula models real heat pump behavior where COP decreases as the temperature lift
    (difference between supply and outdoor temperature) increases. The carnot_efficiency factor
    represents the real-world efficiency as a fraction of the ideal Carnot cycle efficiency.

    :param supply_temperature: The heat pump supply temperature in degrees Celsius (constant value). \
        Typical values: 30-40°C for underfloor heating, 50-70°C for radiator systems.
    :type supply_temperature: float
    :param carnot_efficiency: Real-world efficiency factor as fraction of ideal Carnot cycle. \
        Typical range: 0.35-0.50 (35-50%). Default in thermal battery config: 0.4 (40%). \
        Higher values represent more efficient heat pumps.
    :type carnot_efficiency: float
    :param outdoor_temperature_forecast: Array of outdoor temperature forecasts in degrees Celsius, \
        one value per timestep in the prediction horizon.
    :type outdoor_temperature_forecast: np.ndarray or pd.Series
    :return: Array of COP values for each timestep, same length as outdoor_temperature_forecast. \
        Typical COP range: 2-6 for normal operating conditions.
    :rtype: np.ndarray

    Example:
        >>> supply_temp = 35.0  # °C, underfloor heating
        >>> carnot_eff = 0.4  # 40% of ideal Carnot efficiency
        >>> outdoor_temps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        >>> cops = calculate_cop_heatpump(supply_temp, carnot_eff, outdoor_temps)
        >>> cops
        array([3.521..., 4.108..., 4.926..., 6.163..., 8.217...])
        >>> # At 5°C outdoor: COP = 0.4 × 308.15K / 30K = 4.11

    """
    # Convert to numpy array if pandas Series
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor_temps = outdoor_temperature_forecast.values
    else:
        outdoor_temps = np.asarray(outdoor_temperature_forecast)

    # Convert temperatures from Celsius to Kelvin for Carnot formula
    supply_temperature_kelvin = supply_temperature + 273.15
    outdoor_temperature_kelvin = outdoor_temps + 273.15

    # Calculate temperature difference (supply - outdoor)
    # For heating, supply temperature should be higher than outdoor temperature
    temperature_diff = supply_temperature_kelvin - outdoor_temperature_kelvin

    # Check for non-physical scenarios where outdoor temp >= supply temp
    # This indicates cooling mode or invalid configuration for heating
    if np.any(temperature_diff <= 0):
        # Log warning about non-physical temperature scenario
        logger = logging.getLogger(__name__)
        num_invalid = np.sum(temperature_diff <= 0)
        invalid_indices = np.nonzero(temperature_diff <= 0)[0]
        logger.warning(
            f"COP calculation: {num_invalid} timestep(s) have outdoor temperature >= supply temperature. "
            f"This is non-physical for heating mode. Indices: {invalid_indices.tolist()[:5]}{'...' if len(invalid_indices) > 5 else ''}. "
            f"Supply temp: {supply_temperature:.1f}°C. Setting COP to 1.0 (direct electric heating) for these periods."
        )

    # Vectorized Carnot-based COP calculation
    # COP = carnot_efficiency × T_supply / (T_supply - T_outdoor)
    # For non-physical cases (outdoor >= supply), we use a neutral COP of 1.0
    # This prevents the optimizer from exploiting unrealistic high COP values

    # Avoid division by zero: use a mask to only calculate for valid cases
    cop_values = np.ones_like(outdoor_temperature_kelvin)  # Default to 1.0 everywhere
    valid_mask = temperature_diff > 0
    if np.any(valid_mask):
        cop_values[valid_mask] = (
            carnot_efficiency * supply_temperature_kelvin / temperature_diff[valid_mask]
        )

    # Apply realistic bounds: minimum 1.0, maximum 8.0
    # - Lower bound: 1.0 means direct electric heating (no efficiency gain)
    # - Upper bound: 8.0 is an optimistic but reasonable maximum for modern heat pumps
    #   (prevents numerical instability from very small temperature differences)
    cop_values = np.clip(cop_values, 1.0, 8.0)

    return cop_values


def calculate_thermal_loss_signed(
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    indoor_temperature: float,
    base_loss: float,
) -> np.ndarray:
    r"""
    Calculate signed thermal loss factor based on indoor/outdoor temperature difference.

    **SIGN CONVENTION:**
    - **Positive** (+loss): outdoor < indoor → heat loss, building cools, heating required
    - **Negative** (-loss): outdoor ≥ indoor → heat gain, building warms passively

    Formula: loss * (1 - 2 * Hot(h)), where Hot(h) = 1 if outdoor ≥ indoor, else 0.
    Based on Langer & Volling (2020) Equation B.13.

    :param outdoor_temperature_forecast: Outdoor temperature forecast (°C)
    :type outdoor_temperature_forecast: np.ndarray or pd.Series
    :param indoor_temperature: Indoor/target temperature threshold (°C)
    :type indoor_temperature: float
    :param base_loss: Base thermal loss coefficient in kW
    :type base_loss: float
    :return: Signed loss array (positive = heat loss, negative = heat gain)
    :rtype: np.ndarray

    """
    # Convert to numpy array if pandas Series
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor_temps = outdoor_temperature_forecast.values
    else:
        outdoor_temps = np.asarray(outdoor_temperature_forecast)

    # Create binary hot indicator: 1 if outdoor temp >= indoor temp, 0 otherwise
    hot_indicator = (outdoor_temps >= indoor_temperature).astype(float)

    return base_loss * (1.0 - 2.0 * hot_indicator)


def calculate_heating_demand(
    specific_heating_demand: float,
    floor_area: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    base_temperature: float = 18.0,
    annual_reference_hdd: float = 3000.0,
    optimization_time_step: int | None = None,
) -> np.ndarray:
    """
    Calculate heating demand per timestep based on heating degree days method.

    Uses heating degree days (HDD) to calculate heating demand based on outdoor temperature
    forecast, specific heating demand, and floor area. The specific heating demand should be
    calibrated to the annual reference HDD value.

    :param specific_heating_demand: Specific heating demand in kWh/m²/year (calibrated to annual_reference_hdd)
    :type specific_heating_demand: float
    :param floor_area: Floor area in m²
    :type floor_area: float
    :param outdoor_temperature_forecast: Outdoor temperature forecast in °C for each timestep
    :type outdoor_temperature_forecast: np.ndarray | pd.Series
    :param base_temperature: Base temperature for HDD calculation in °C, defaults to 18.0 (European standard)
    :type base_temperature: float, optional
    :param annual_reference_hdd: Annual reference HDD value for normalization, defaults to 3000.0 (Central Europe)
    :type annual_reference_hdd: float, optional
    :param optimization_time_step: Optimization time step in minutes. If None, automatically infers from
        pandas Series DatetimeIndex frequency. Falls back to 30 minutes if not inferrable.
    :type optimization_time_step: int | None, optional
    :return: Array of heating demand values (kWh) per timestep
    :rtype: np.ndarray

    """

    # Convert outdoor temperature forecast to numpy array if pandas Series
    outdoor_temps = (
        outdoor_temperature_forecast.values
        if isinstance(outdoor_temperature_forecast, pd.Series)
        else np.asarray(outdoor_temperature_forecast)
    )

    # Calculate heating degree days per timestep
    # HDD = max(base_temperature - outdoor_temperature, 0)
    hdd_per_timestep = np.maximum(base_temperature - outdoor_temps, 0.0)

    # Determine timestep duration in hours
    if optimization_time_step is None:
        # Try to infer from pandas Series DatetimeIndex
        if isinstance(outdoor_temperature_forecast, pd.Series) and isinstance(
            outdoor_temperature_forecast.index, pd.DatetimeIndex
        ):
            if len(outdoor_temperature_forecast.index) > 1:
                freq_minutes = (
                    outdoor_temperature_forecast.index[1] - outdoor_temperature_forecast.index[0]
                ).total_seconds() / 60.0
                hours_per_timestep = freq_minutes / 60.0
            else:
                # Single datapoint, fallback to default 30 min
                hours_per_timestep = 0.5
        else:
            # Cannot infer, use default 30 minutes
            hours_per_timestep = 0.5
    else:
        # Convert minutes to hours
        hours_per_timestep = optimization_time_step / 60.0

    # Scale HDD to timestep duration (standard HDD is per 24 hours)
    hdd_per_timestep_scaled = hdd_per_timestep * (hours_per_timestep / 24.0)

    return specific_heating_demand * floor_area * (hdd_per_timestep_scaled / annual_reference_hdd)


def calculate_heating_demand_physics(
    u_value: float,
    envelope_area: float,
    ventilation_rate: float,
    heated_volume: float,
    indoor_target_temperature: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    optimization_time_step: int,
    solar_irradiance_forecast: np.ndarray | pd.Series | None = None,
    window_area: float | None = None,
    shgc: float = 0.6,
    internal_gains_forecast: np.ndarray | pd.Series | None = None,
    internal_gains_factor: float = 0.0,
) -> np.ndarray:
    """
    Calculate heating demand per timestep based on building physics heat loss model.

    More accurate than HDD method as it directly calculates transmission and ventilation
    losses based on building thermal properties. Optionally accounts for solar gains
    through windows to reduce heating demand.

    :param u_value: Overall thermal transmittance (U-value) in W/(m²·K). Typical values:
        - 0.2-0.3: Well-insulated modern building
        - 0.4-0.6: Average insulation
        - 0.8-1.2: Poor insulation / old building
    :type u_value: float
    :param envelope_area: Total building envelope area (walls + roof + floor + windows) in m²
    :type envelope_area: float
    :param ventilation_rate: Air changes per hour (ACH). Typical values:
        - 0.3-0.5: Well-sealed modern building with controlled ventilation
        - 0.5-1.0: Average building
        - 1.0-2.0: Leaky old building
    :type ventilation_rate: float
    :param heated_volume: Total heated volume in m³
    :type heated_volume: float
    :param indoor_target_temperature: Target indoor temperature in °C
    :type indoor_target_temperature: float
    :param outdoor_temperature_forecast: Outdoor temperature forecast in °C for each timestep
    :type outdoor_temperature_forecast: np.ndarray | pd.Series
    :param optimization_time_step: Optimization time step in minutes
    :type optimization_time_step: int
    :param solar_irradiance_forecast: Global Horizontal Irradiance (GHI) in W/m² for each timestep.
        If provided along with window_area, solar gains will be subtracted from heating demand.
    :type solar_irradiance_forecast: np.ndarray | pd.Series | None, optional
    :param window_area: Total window area in m². If provided along with solar_irradiance_forecast,
        solar gains will reduce heating demand. Typical values: 15-25% of floor area.
    :type window_area: float | None, optional
    :param shgc: Solar Heat Gain Coefficient (dimensionless, 0-1). Fraction of solar radiation
        that becomes heat inside the building. Typical values:
        - 0.5-0.6: Modern low-e double-glazed windows
        - 0.6-0.7: Standard double-glazed windows
        - 0.7-0.8: Single-glazed windows
        Default: 0.6
    :type shgc: float, optional
    :param internal_gains_forecast: Electrical load power forecast in W for each timestep.
        If provided along with internal_gains_factor > 0, internal gains from electrical
        appliances will be subtracted from heating demand.
    :type internal_gains_forecast: np.ndarray | pd.Series | None, optional
    :param internal_gains_factor: Factor (0-1) representing what fraction of electrical load
        becomes useful internal heat gains. Typical values:
        - 0.0: No internal gains considered (default, backwards compatible)
        - 0.5-0.7: Conservative estimate (some heat lost to ventilation/drains)
        - 0.8-0.9: Most electrical energy becomes heat (well-insulated building)
        - 1.0: All electrical energy becomes internal heat (theoretical maximum)
        Default: 0.0
    :type internal_gains_factor: float, optional
    :return: Array of heating demand values (kWh) per timestep
    :rtype: np.ndarray

    Example:
        >>> outdoor_temps = np.array([5, 8, 12, 15])
        >>> ghi = np.array([0, 100, 400, 600])  # W/m²
        >>> demand = calculate_heating_demand_physics(
        ...     u_value=0.3,
        ...     envelope_area=400,
        ...     ventilation_rate=0.5,
        ...     heated_volume=250,
        ...     indoor_target_temperature=20,
        ...     outdoor_temperature_forecast=outdoor_temps,
        ...     optimization_time_step=30,
        ...     solar_irradiance_forecast=ghi,
        ...     window_area=50,
        ...     shgc=0.6
        ... )
    """

    # Convert outdoor temperature forecast to numpy array if pandas Series
    outdoor_temps = (
        outdoor_temperature_forecast.values
        if isinstance(outdoor_temperature_forecast, pd.Series)
        else np.asarray(outdoor_temperature_forecast)
    )

    # Calculate temperature difference (only heat when outdoor < indoor)
    temp_diff = indoor_target_temperature - outdoor_temps
    temp_diff = np.maximum(temp_diff, 0.0)

    # Transmission losses: Q_trans = U * A * ΔT (W to kW)
    transmission_loss_kw = u_value * envelope_area * temp_diff / 1000.0

    # Ventilation losses: Q_vent = V * ρ * c * n * ΔT / 3600
    # ρ = air density (kg/m³), c = specific heat capacity (kJ/(kg·K)), n = ACH
    air_density = 1.2  # kg/m³ at 20°C
    air_heat_capacity = 1.005  # kJ/(kg·K)
    ventilation_loss_kw = (
        ventilation_rate * heated_volume * air_density * air_heat_capacity * temp_diff / 3600.0
    )

    # Total heat loss in kW
    total_loss_kw = transmission_loss_kw + ventilation_loss_kw

    # Calculate solar gains if irradiance and window area are provided
    if solar_irradiance_forecast is not None and window_area is not None:
        # Convert solar irradiance to numpy array if pandas Series
        solar_irradiance = (
            solar_irradiance_forecast.values
            if isinstance(solar_irradiance_forecast, pd.Series)
            else np.asarray(solar_irradiance_forecast)
        )

        # Solar gains: Q_solar = window_area * SHGC * GHI (W to kW)
        # GHI is in W/m², so multiply by window_area (m²) gives W, then divide by 1000 for kW
        solar_gains_kw = window_area * shgc * solar_irradiance / 1000.0

        # Subtract solar gains from heat loss (but never go negative)
        total_loss_kw = np.maximum(total_loss_kw - solar_gains_kw, 0.0)

    # Validate internal_gains_factor is in expected range [0, 1]
    if internal_gains_factor < 0 or internal_gains_factor > 1:
        raise ValueError(
            f"internal_gains_factor must be between 0 and 1, got {internal_gains_factor}"
        )

    # Calculate internal gains from electrical load if provided and applicable
    if internal_gains_forecast is not None and internal_gains_factor > 0:
        # Convert internal gains forecast to numpy array and normalize to 1D
        # to align with other forecast inputs and avoid broadcast surprises
        internal_gains = (
            internal_gains_forecast.values
            if isinstance(internal_gains_forecast, pd.Series)
            else internal_gains_forecast
        )
        internal_gains = np.asarray(internal_gains).reshape(-1)

        # Validate that internal gains forecast length matches outdoor temperature forecast
        if len(internal_gains) != len(outdoor_temps):
            raise ValueError(
                f"internal_gains_forecast length ({len(internal_gains)}) must match "
                f"outdoor_temperature_forecast length ({len(outdoor_temps)})"
            )

        # Warn if values seem like they might be in kW instead of W
        # Typical household load is 100-10000W; values below 10 suggest kW was passed
        max_load = np.max(internal_gains)
        if max_load > 0 and max_load < 10:
            import warnings

            warnings.warn(
                f"internal_gains_forecast max value ({max_load:.2f}) is very low. "
                "Expected values in W (e.g., 500-5000), but received values that "
                "look like kW. Please ensure you're passing Watts, not kilowatts.",
                UserWarning,
                stacklevel=2,
            )

        # Internal gains: Q_internal = load_power * factor
        # load_power is in W, convert to kW; factor is dimensionless (0-1)
        internal_gains_kw = internal_gains * internal_gains_factor / W_TO_KW

        # Subtract internal gains from heat loss (but never go negative)
        total_loss_kw = np.maximum(total_loss_kw - internal_gains_kw, 0.0)

    # Convert to kWh for the timestep
    hours_per_timestep = optimization_time_step / 60.0
    return total_loss_kw * hours_per_timestep


def update_params_with_ha_config(
    params: str,
    ha_config: dict,
) -> dict:
    """
    Update the params with the Home Assistant configuration.

    Parameters
    ----------
    params : str
        The serialized params.
    ha_config : dict
        The Home Assistant configuration.

    Returns
    -------
    dict
        The updated params.
    """
    # Load serialized params
    params = orjson.loads(params)
    # Update params
    currency_to_symbol = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
        "YEN": "¥",
        "JPY": "¥",
        "AUD": "A$",
        "CAD": "C$",
        "CHF": "CHF",  # Swiss Franc has no special symbol
        "CNY": "¥",
        "INR": "₹",
        "CZK": "Kč",
        "BGN": "лв",
        "DKK": "kr",
        "HUF": "Ft",
        "PLN": "zł",
        "RON": "Leu",
        "SEK": "kr",
        "TRY": "Lira",
        "VEF": "Bolivar",
        "VND": "Dong",
        "THB": "Baht",
        "SGD": "S$",
        "IDR": "Roepia",
        "ZAR": "Rand",
        # Add more as needed
    }
    if "currency" in ha_config.keys():
        ha_config["currency"] = currency_to_symbol.get(ha_config["currency"], "Unknown")
    else:
        ha_config["currency"] = "€"

    updated_passed_dict = {
        "custom_cost_fun_id": {
            "unit_of_measurement": ha_config["currency"],
        },
        "custom_unit_load_cost_id": {
            "unit_of_measurement": f"{ha_config['currency']}/kWh",
        },
        "custom_unit_prod_price_id": {
            "unit_of_measurement": f"{ha_config['currency']}/kWh",
        },
    }
    for key, value in updated_passed_dict.items():
        params["passed_data"][key]["unit_of_measurement"] = value["unit_of_measurement"]
    # Serialize the final params
    params = orjson.dumps(params, default=str).decode("utf-8")
    return params


async def treat_runtimeparams(
    runtimeparams: str,
    params: dict[str, dict],
    retrieve_hass_conf: dict[str, str],
    optim_conf: dict[str, str],
    plant_conf: dict[str, str],
    set_type: str,
    logger: logging.Logger,
    emhass_conf: dict[str, pathlib.Path],
) -> tuple[str, dict[str, dict]]:
    """
    Treat the passed optimization runtime parameters.

    :param runtimeparams: Json string containing the runtime parameters dict.
    :type runtimeparams: str
    :param params: Built configuration parameters
    :type params: str
    :param retrieve_hass_conf: Config dictionary for data retrieving parameters.
    :type retrieve_hass_conf: dict
    :param optim_conf: Config dictionary for optimization parameters.
    :type optim_conf: dict
    :param plant_conf: Config dictionary for technical plant parameters.
    :type plant_conf: dict
    :param set_type: The type of action to be performed.
    :type set_type: str
    :param logger: The logger object.
    :type logger: logging.Logger
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :return: Returning the params and optimization parameter container.
    :rtype: Tuple[str, dict]

    """
    # Check if passed params is a dict
    if (params is not None) and (params != "null"):
        if type(params) is str:
            params = orjson.loads(params)
    else:
        params = {}

    # Merge current config categories to params
    params["retrieve_hass_conf"].update(retrieve_hass_conf)
    params["optim_conf"].update(optim_conf)
    params["plant_conf"].update(plant_conf)

    # Check defaults on HA retrieved config
    default_currency_unit = "€"
    default_temperature_unit = "°C"

    # Some default data needed
    custom_deferrable_forecast_id = []
    custom_predicted_temperature_id = []
    custom_heating_demand_id = []
    for k in range(params["optim_conf"]["number_of_deferrable_loads"]):
        custom_deferrable_forecast_id.append(
            {
                "entity_id": f"sensor.p_deferrable{k}",
                "device_class": "power",
                "unit_of_measurement": "W",
                "friendly_name": f"Deferrable Load {k}",
            }
        )
        custom_predicted_temperature_id.append(
            {
                "entity_id": f"sensor.temp_predicted{k}",
                "device_class": "temperature",
                "unit_of_measurement": default_temperature_unit,
                "friendly_name": f"Predicted temperature {k}",
            }
        )
        custom_heating_demand_id.append(
            {
                "entity_id": f"sensor.heating_demand{k}",
                "device_class": "energy",
                "unit_of_measurement": "kWh",
                "friendly_name": f"Heating demand {k}",
            }
        )
    default_passed_dict = {
        "custom_pv_forecast_id": {
            "entity_id": "sensor.p_pv_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Forecast",
        },
        "custom_load_forecast_id": {
            "entity_id": "sensor.p_load_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Load Power Forecast",
        },
        "custom_pv_curtailment_id": {
            "entity_id": "sensor.p_pv_curtailment",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Curtailment",
        },
        "custom_hybrid_inverter_id": {
            "entity_id": "sensor.p_hybrid_inverter",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Hybrid Inverter",
        },
        "custom_batt_forecast_id": {
            "entity_id": "sensor.p_batt_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Battery Power Forecast",
        },
        "custom_batt_soc_forecast_id": {
            "entity_id": "sensor.soc_batt_forecast",
            "device_class": "battery",
            "unit_of_measurement": "%",
            "friendly_name": "Battery SOC Forecast",
        },
        "custom_grid_forecast_id": {
            "entity_id": "sensor.p_grid_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Grid Power Forecast",
        },
        "custom_cost_fun_id": {
            "entity_id": "sensor.total_cost_fun_value",
            "device_class": "monetary",
            "unit_of_measurement": default_currency_unit,
            "friendly_name": "Total cost function value",
        },
        "custom_optim_status_id": {
            "entity_id": "sensor.optim_status",
            "device_class": "",
            "unit_of_measurement": "",
            "friendly_name": "EMHASS optimization status",
        },
        "custom_unit_load_cost_id": {
            "entity_id": "sensor.unit_load_cost",
            "device_class": "monetary",
            "unit_of_measurement": f"{default_currency_unit}/kWh",
            "friendly_name": "Unit Load Cost",
        },
        "custom_unit_prod_price_id": {
            "entity_id": "sensor.unit_prod_price",
            "device_class": "monetary",
            "unit_of_measurement": f"{default_currency_unit}/kWh",
            "friendly_name": "Unit Prod Price",
        },
        "custom_deferrable_forecast_id": custom_deferrable_forecast_id,
        "custom_predicted_temperature_id": custom_predicted_temperature_id,
        "custom_heating_demand_id": custom_heating_demand_id,
        "publish_prefix": "",
    }
    if "passed_data" in params.keys():
        for key, value in default_passed_dict.items():
            params["passed_data"][key] = value
    else:
        params["passed_data"] = default_passed_dict

    # Capture defaults for power limits before association loop
    power_limit_defaults = {
        "maximum_power_from_grid": params["plant_conf"].get("maximum_power_from_grid"),
        "maximum_power_to_grid": params["plant_conf"].get("maximum_power_to_grid"),
    }

    # If any runtime parameters where passed in action call
    if runtimeparams is not None:
        if type(runtimeparams) is str:
            runtimeparams = orjson.loads(runtimeparams)

        # Loop though parameters stored in association file, Check to see if any stored in runtime
        # If true, set runtime parameter to params
        if emhass_conf["associations_path"].exists():
            async with aiofiles.open(emhass_conf["associations_path"]) as data:
                content = await data.read()
                associations = list(csv.reader(content.splitlines(), delimiter=","))
                # Association file key reference
                # association[0] = config categories
                # association[1] = legacy parameter name
                # association[2] = parameter (config.json/config_defaults.json)
                # association[3] = parameter list name if exists (not used, from legacy options.json)
                for association in associations:
                    # Check parameter name exists in runtime
                    if runtimeparams.get(association[2], None) is not None:
                        params[association[0]][association[2]] = runtimeparams[association[2]]
                    # Check Legacy parameter name runtime
                    elif runtimeparams.get(association[1], None) is not None:
                        params[association[0]][association[2]] = runtimeparams[association[1]]
        else:
            logger.warning(
                "Cant find associations file (associations.csv) in: "
                + str(emhass_conf["associations_path"])
            )

        # Special handling for power limit parameters - they can be vectors (Tier 1a)
        def _parse_power_limit(key: str) -> None:
            """Helper to parse list/scalar power limits safely."""
            if key in runtimeparams:
                value = runtimeparams[key]
                try:
                    # If it's a string representation of a list, parse it
                    if isinstance(value, str):
                        parsed = ast.literal_eval(value)
                        params["plant_conf"][key] = parsed
                    # If already a list/array, use it directly
                    # Ruff preferred bitwise OR '|' for union types
                    elif isinstance(value, list | tuple):
                        params["plant_conf"][key] = list(value)
                    # If scalar, use as-is
                    else:
                        params["plant_conf"][key] = value
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse {key}: {e}. Using default.")
                    if power_limit_defaults.get(key) is not None:
                        params["plant_conf"][key] = power_limit_defaults[key]

        # Apply the helper
        _parse_power_limit("maximum_power_from_grid")
        _parse_power_limit("maximum_power_to_grid")

        # Generate forecast_dates
        # Force update optimization_time_step if present in runtimeparams
        if "optimization_time_step" in runtimeparams:
            optimization_time_step = int(runtimeparams["optimization_time_step"])
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                optimization_time_step, "minutes"
            )
        elif "freq" in runtimeparams:
            optimization_time_step = int(runtimeparams["freq"])
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                optimization_time_step, "minutes"
            )
        else:
            optimization_time_step = int(
                params["retrieve_hass_conf"]["optimization_time_step"].seconds / 60.0
            )

        if (
            runtimeparams.get("delta_forecast_daily", None) is not None
            or runtimeparams.get("delta_forecast", None) is not None
        ):
            # Use old param name delta_forecast (if provided) for backwards compatibility
            delta_forecast = runtimeparams.get("delta_forecast", None)
            # Prefer new param name delta_forecast_daily
            delta_forecast = runtimeparams.get("delta_forecast_daily", delta_forecast)
            # Ensure delta_forecast is numeric and at least 1 day
            if delta_forecast is None:
                logger.warning("delta_forecast_daily is missing so defaulting to 1 day")
                delta_forecast = 1
            else:
                try:
                    delta_forecast = int(delta_forecast)
                except ValueError:
                    logger.warning(
                        "Invalid delta_forecast_daily value (%s) so defaulting to 1 day",
                        delta_forecast,
                    )
                    delta_forecast = 1
            if delta_forecast <= 0:
                logger.warning(
                    "delta_forecast_daily is too low (%s) so defaulting to 1 day",
                    delta_forecast,
                )
                delta_forecast = 1
            params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(days=delta_forecast)
        else:
            delta_forecast = int(params["optim_conf"]["delta_forecast_daily"].days)
        if runtimeparams.get("time_zone", None) is not None:
            time_zone = pytz.timezone(params["retrieve_hass_conf"]["time_zone"])
            params["retrieve_hass_conf"]["time_zone"] = time_zone
        else:
            time_zone = params["retrieve_hass_conf"]["time_zone"]

        forecast_dates = get_forecast_dates(optimization_time_step, delta_forecast, time_zone)

        # Add runtime exclusive (not in config) parameters to params
        # regressor-model-fit
        if set_type == "regressor-model-fit":
            if "csv_file" in runtimeparams:
                csv_file = runtimeparams["csv_file"]
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target
            if "timestamp" not in runtimeparams:
                params["passed_data"]["timestamp"] = None
            else:
                timestamp = runtimeparams["timestamp"]
                params["passed_data"]["timestamp"] = timestamp
            if "date_features" not in runtimeparams:
                params["passed_data"]["date_features"] = []
            else:
                date_features = runtimeparams["date_features"]
                params["passed_data"]["date_features"] = date_features

        # regressor-model-predict
        if set_type == "regressor-model-predict":
            if "new_values" in runtimeparams:
                new_values = runtimeparams["new_values"]
                params["passed_data"]["new_values"] = new_values
            if "csv_file" in runtimeparams:
                csv_file = runtimeparams["csv_file"]
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target

        # export-influxdb-to-csv
        if set_type == "export-influxdb-to-csv":
            # Use dictionary comprehension to simplify parameter assignment
            export_keys = {
                k: runtimeparams[k]
                for k in (
                    "sensor_list",
                    "csv_filename",
                    "start_time",
                    "end_time",
                    "resample_freq",
                    "timestamp_col_name",
                    "decimal_places",
                    "handle_nan",
                )
                if k in runtimeparams
            }
            params["passed_data"].update(export_keys)

        # MPC control case
        if set_type == "naive-mpc-optim":
            if "prediction_horizon" not in runtimeparams.keys():
                prediction_horizon = 10  # 10 time steps by default
            else:
                prediction_horizon = runtimeparams["prediction_horizon"]
            params["passed_data"]["prediction_horizon"] = prediction_horizon
            if "soc_init" not in runtimeparams.keys():
                soc_init = params["plant_conf"]["battery_target_state_of_charge"]
            else:
                soc_init = runtimeparams["soc_init"]
            if soc_init < params["plant_conf"]["battery_minimum_state_of_charge"]:
                logger.warning(
                    f"Passed soc_init={soc_init} is lower than soc_min={params['plant_conf']['battery_minimum_state_of_charge']}, setting soc_init=soc_min"
                )
                soc_init = params["plant_conf"]["battery_minimum_state_of_charge"]
            if soc_init > params["plant_conf"]["battery_maximum_state_of_charge"]:
                logger.warning(
                    f"Passed soc_init={soc_init} is greater than soc_max={params['plant_conf']['battery_maximum_state_of_charge']}, setting soc_init=soc_max"
                )
                soc_init = params["plant_conf"]["battery_maximum_state_of_charge"]
            params["passed_data"]["soc_init"] = soc_init
            if "soc_final" not in runtimeparams.keys():
                soc_final = params["plant_conf"]["battery_target_state_of_charge"]
            else:
                soc_final = runtimeparams["soc_final"]
            if soc_final < params["plant_conf"]["battery_minimum_state_of_charge"]:
                logger.warning(
                    f"Passed soc_final={soc_final} is lower than soc_min={params['plant_conf']['battery_minimum_state_of_charge']}, setting soc_final=soc_min"
                )
                soc_final = params["plant_conf"]["battery_minimum_state_of_charge"]
            if soc_final > params["plant_conf"]["battery_maximum_state_of_charge"]:
                logger.warning(
                    f"Passed soc_final={soc_final} is greater than soc_max={params['plant_conf']['battery_maximum_state_of_charge']}, setting soc_final=soc_max"
                )
                soc_final = params["plant_conf"]["battery_maximum_state_of_charge"]
            params["passed_data"]["soc_final"] = soc_final
            if "operating_timesteps_of_each_deferrable_load" in runtimeparams.keys():
                params["passed_data"]["operating_timesteps_of_each_deferrable_load"] = (
                    runtimeparams["operating_timesteps_of_each_deferrable_load"]
                )
                params["optim_conf"]["operating_timesteps_of_each_deferrable_load"] = runtimeparams[
                    "operating_timesteps_of_each_deferrable_load"
                ]
            if "operating_hours_of_each_deferrable_load" in params["optim_conf"].keys():
                params["passed_data"]["operating_hours_of_each_deferrable_load"] = params[
                    "optim_conf"
                ]["operating_hours_of_each_deferrable_load"]
            params["passed_data"]["start_timesteps_of_each_deferrable_load"] = params[
                "optim_conf"
            ].get("start_timesteps_of_each_deferrable_load", None)
            params["passed_data"]["end_timesteps_of_each_deferrable_load"] = params[
                "optim_conf"
            ].get("end_timesteps_of_each_deferrable_load", None)

            forecast_dates = copy.deepcopy(forecast_dates)[0:prediction_horizon]
        else:
            params["passed_data"]["prediction_horizon"] = None
            params["passed_data"]["soc_init"] = None
            params["passed_data"]["soc_final"] = None

        # Parsing the thermal model parameters
        # Load the default config
        if "def_load_config" in runtimeparams:
            params["optim_conf"]["def_load_config"] = runtimeparams["def_load_config"]
            params["optim_conf"]["number_of_deferrable_loads"] = len(
                runtimeparams["def_load_config"]
            )
        if "def_load_config" in params["optim_conf"]:
            for k in range(len(params["optim_conf"]["def_load_config"])):
                if "thermal_config" in params["optim_conf"]["def_load_config"][k]:
                    if (
                        "heater_desired_temperatures" in runtimeparams
                        and len(runtimeparams["heater_desired_temperatures"]) > k
                    ):
                        params["optim_conf"]["def_load_config"][k]["thermal_config"][
                            "desired_temperatures"
                        ] = runtimeparams["heater_desired_temperatures"][k]
                    if (
                        "heater_start_temperatures" in runtimeparams
                        and len(runtimeparams["heater_start_temperatures"]) > k
                    ):
                        params["optim_conf"]["def_load_config"][k]["thermal_config"][
                            "start_temperature"
                        ] = runtimeparams["heater_start_temperatures"][k]

        # Treat passed forecast data lists
        list_forecast_key = [
            "pv_power_forecast",
            "load_power_forecast",
            "load_cost_forecast",
            "prod_price_forecast",
            "outdoor_temperature_forecast",
        ]
        forecast_methods = [
            "weather_forecast_method",
            "load_forecast_method",
            "load_cost_forecast_method",
            "production_price_forecast_method",
            "outdoor_temperature_forecast_method",
        ]

        # Loop forecasts, check if value is a list and greater than or equal to forecast_dates
        for method, forecast_key in enumerate(list_forecast_key):
            if forecast_key in runtimeparams.keys():
                forecast_input = runtimeparams[forecast_key]
                if isinstance(forecast_input, dict):
                    forecast_data_df = pd.DataFrame.from_dict(
                        forecast_input, orient="index"
                    ).reset_index()
                    forecast_data_df.columns = ["time", "value"]
                    forecast_data_df["time"] = pd.to_datetime(
                        forecast_data_df["time"], format="ISO8601", utc=True
                    ).dt.tz_convert(time_zone)

                    # align index with forecast_dates
                    forecast_data_df = (
                        forecast_data_df.resample(
                            pd.to_timedelta(optimization_time_step, "minutes"),
                            on="time",
                        )
                        .aggregate({"value": "mean"})
                        .reindex(forecast_dates, method="nearest")
                    )
                    forecast_data_df["value"] = forecast_data_df["value"].ffill().bfill()
                    forecast_input = forecast_data_df["value"].tolist()
                if isinstance(forecast_input, list) and len(forecast_input) >= len(forecast_dates):
                    params["passed_data"][forecast_key] = forecast_input
                    params["optim_conf"][forecast_methods[method]] = "list"
                else:
                    logger.error(
                        f"ERROR: The passed data is either the wrong type or the length is not correct, length should be {str(len(forecast_dates))}"
                    )
                    logger.error(
                        f"Passed type is {str(type(runtimeparams[forecast_key]))} and length is {str(len(runtimeparams[forecast_key]))}"
                    )
                # Check if string contains list, if so extract
                if isinstance(forecast_input, str) and isinstance(
                    ast.literal_eval(forecast_input), list
                ):
                    forecast_input = ast.literal_eval(forecast_input)
                    runtimeparams[forecast_key] = forecast_input
                list_non_digits = [
                    x for x in forecast_input if not (isinstance(x, int) or isinstance(x, float))
                ]
                if len(list_non_digits) > 0:
                    logger.warning(
                        f"There are non numeric values on the passed data for {forecast_key}, check for missing values (nans, null, etc)"
                    )
                    for x in list_non_digits:
                        logger.warning(
                            f"This value in {forecast_key} was detected as non digits: {str(x)}"
                        )
            else:
                params["passed_data"][forecast_key] = None

        # Explicitly handle historic_days_to_retrieve from runtimeparams BEFORE validation
        if "historic_days_to_retrieve" in runtimeparams:
            params["retrieve_hass_conf"]["historic_days_to_retrieve"] = int(
                runtimeparams["historic_days_to_retrieve"]
            )

        # Treat passed data for forecast model fit/predict/tune at runtime
        if (
            params["passed_data"].get("historic_days_to_retrieve", None) is not None
            and params["passed_data"]["historic_days_to_retrieve"] < 9
        ):
            logger.warning(
                "warning `days_to_retrieve` is set to a value less than 9, this could cause an error with the fit"
            )
            logger.warning("setting`passed_data:days_to_retrieve` to 9 for fit/predict/tune")
            params["passed_data"]["historic_days_to_retrieve"] = 9
        else:
            if params["retrieve_hass_conf"].get("historic_days_to_retrieve", 0) < 9:
                logger.debug("setting`passed_data:days_to_retrieve` to 9 for fit/predict/tune")
                params["passed_data"]["historic_days_to_retrieve"] = 9
            else:
                params["passed_data"]["historic_days_to_retrieve"] = params["retrieve_hass_conf"][
                    "historic_days_to_retrieve"
                ]

        # UPDATED ML PARAMETER HANDLING
        # Define Helper Functions
        def _cast_bool(value):
            """Helper to cast string inputs to boolean safely."""
            try:
                return ast.literal_eval(str(value).capitalize())
            except (ValueError, SyntaxError):
                return False

        def _get_ml_param(name, params, runtimeparams, default=None, cast=None):
            """
            Prioritize Runtime Params -> Config Params (optim_conf) -> Default.
            """
            if name in runtimeparams:
                value = runtimeparams[name]
            else:
                value = params["optim_conf"].get(name, default)

            if cast is not None and value is not None:
                try:
                    value = cast(value)
                except Exception:
                    pass
            return value

        # Compute dynamic defaults
        # Default for var_model falls back to the configured load sensor
        default_var_model = params["retrieve_hass_conf"].get(
            "sensor_power_load_no_var_loads", "sensor.power_load_no_var_loads"
        )

        # Define Configuration Table
        # Format: (parameter_name, default_value, cast_function)
        ml_param_defs = [
            ("model_type", "long_train_data", None),
            ("var_model", default_var_model, None),
            ("sklearn_model", "KNeighborsRegressor", None),
            ("regression_model", "AdaBoostRegression", None),
            ("num_lags", 48, None),
            ("split_date_delta", "48h", None),
            ("n_trials", 10, int),
            ("perform_backtest", False, _cast_bool),
        ]

        # Apply Configuration
        for name, default, caster in ml_param_defs:
            params["passed_data"][name] = _get_ml_param(
                name=name,
                params=params,
                runtimeparams=runtimeparams,
                default=default,
                cast=caster,
            )

        # Other non-dynamic options
        if "model_predict_publish" not in runtimeparams.keys():
            model_predict_publish = False
        else:
            model_predict_publish = ast.literal_eval(
                str(runtimeparams["model_predict_publish"]).capitalize()
            )
        params["passed_data"]["model_predict_publish"] = model_predict_publish
        if "model_predict_entity_id" not in runtimeparams.keys():
            model_predict_entity_id = "sensor.p_load_forecast_custom_model"
        else:
            model_predict_entity_id = runtimeparams["model_predict_entity_id"]
        params["passed_data"]["model_predict_entity_id"] = model_predict_entity_id
        if "model_predict_device_class" not in runtimeparams.keys():
            model_predict_device_class = "power"
        else:
            model_predict_device_class = runtimeparams["model_predict_device_class"]
        params["passed_data"]["model_predict_device_class"] = model_predict_device_class
        if "model_predict_unit_of_measurement" not in runtimeparams.keys():
            model_predict_unit_of_measurement = "W"
        else:
            model_predict_unit_of_measurement = runtimeparams["model_predict_unit_of_measurement"]
        params["passed_data"]["model_predict_unit_of_measurement"] = (
            model_predict_unit_of_measurement
        )
        if "model_predict_friendly_name" not in runtimeparams.keys():
            model_predict_friendly_name = "Load Power Forecast custom ML model"
        else:
            model_predict_friendly_name = runtimeparams["model_predict_friendly_name"]
        params["passed_data"]["model_predict_friendly_name"] = model_predict_friendly_name
        if "mlr_predict_entity_id" not in runtimeparams.keys():
            mlr_predict_entity_id = "sensor.mlr_predict"
        else:
            mlr_predict_entity_id = runtimeparams["mlr_predict_entity_id"]
        params["passed_data"]["mlr_predict_entity_id"] = mlr_predict_entity_id
        if "mlr_predict_device_class" not in runtimeparams.keys():
            mlr_predict_device_class = "power"
        else:
            mlr_predict_device_class = runtimeparams["mlr_predict_device_class"]
        params["passed_data"]["mlr_predict_device_class"] = mlr_predict_device_class
        if "mlr_predict_unit_of_measurement" not in runtimeparams.keys():
            mlr_predict_unit_of_measurement = None
        else:
            mlr_predict_unit_of_measurement = runtimeparams["mlr_predict_unit_of_measurement"]
        params["passed_data"]["mlr_predict_unit_of_measurement"] = mlr_predict_unit_of_measurement
        if "mlr_predict_friendly_name" not in runtimeparams.keys():
            mlr_predict_friendly_name = "mlr predictor"
        else:
            mlr_predict_friendly_name = runtimeparams["mlr_predict_friendly_name"]
        params["passed_data"]["mlr_predict_friendly_name"] = mlr_predict_friendly_name

        # Treat passed data for other parameters
        if "alpha" not in runtimeparams.keys():
            alpha = 0.5
        else:
            alpha = runtimeparams["alpha"]
        params["passed_data"]["alpha"] = alpha
        if "beta" not in runtimeparams.keys():
            beta = 0.5
        else:
            beta = runtimeparams["beta"]
        params["passed_data"]["beta"] = beta

        # Param to save forecast cache (i.e. Solcast)
        if "weather_forecast_cache" not in runtimeparams.keys():
            weather_forecast_cache = False
        else:
            weather_forecast_cache = runtimeparams["weather_forecast_cache"]
        params["passed_data"]["weather_forecast_cache"] = weather_forecast_cache

        # Param to make sure optimization only uses cached data. (else produce error)
        if "weather_forecast_cache_only" not in runtimeparams.keys():
            weather_forecast_cache_only = False
        else:
            weather_forecast_cache_only = runtimeparams["weather_forecast_cache_only"]
        params["passed_data"]["weather_forecast_cache_only"] = weather_forecast_cache_only

        # A condition to manually save entity data under data_path/entities after optimization
        if "entity_save" not in runtimeparams.keys():
            entity_save = ""
        else:
            entity_save = runtimeparams["entity_save"]
        params["passed_data"]["entity_save"] = entity_save

        # A condition to put a prefix on all published data, or check for saved data under prefix name
        if "publish_prefix" not in runtimeparams.keys():
            publish_prefix = ""
        else:
            publish_prefix = runtimeparams["publish_prefix"]
        params["passed_data"]["publish_prefix"] = publish_prefix

        # Treat optimization (optim_conf) configuration parameters passed at runtime
        if "def_current_state" in runtimeparams.keys():
            params["optim_conf"]["def_current_state"] = [
                bool(s) for s in runtimeparams["def_current_state"]
            ]

        # Treat retrieve data from Home Assistant (retrieve_hass_conf) configuration parameters passed at runtime
        # Secrets passed at runtime
        if "solcast_api_key" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solcast_api_key"] = runtimeparams["solcast_api_key"]
        if "solcast_rooftop_id" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = runtimeparams["solcast_rooftop_id"]
        if "solar_forecast_kwp" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = runtimeparams["solar_forecast_kwp"]
        # Treat custom entities id's and friendly names for variables
        if "custom_pv_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_forecast_id"] = runtimeparams["custom_pv_forecast_id"]
        if "custom_load_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_load_forecast_id"] = runtimeparams[
                "custom_load_forecast_id"
            ]
        if "custom_pv_curtailment_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_curtailment_id"] = runtimeparams[
                "custom_pv_curtailment_id"
            ]
        if "custom_hybrid_inverter_id" in runtimeparams.keys():
            params["passed_data"]["custom_hybrid_inverter_id"] = runtimeparams[
                "custom_hybrid_inverter_id"
            ]
        if "custom_batt_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_forecast_id"] = runtimeparams[
                "custom_batt_forecast_id"
            ]
        if "custom_batt_soc_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_soc_forecast_id"] = runtimeparams[
                "custom_batt_soc_forecast_id"
            ]
        if "custom_grid_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_grid_forecast_id"] = runtimeparams[
                "custom_grid_forecast_id"
            ]
        if "custom_cost_fun_id" in runtimeparams.keys():
            params["passed_data"]["custom_cost_fun_id"] = runtimeparams["custom_cost_fun_id"]
        if "custom_optim_status_id" in runtimeparams.keys():
            params["passed_data"]["custom_optim_status_id"] = runtimeparams[
                "custom_optim_status_id"
            ]
        if "custom_unit_load_cost_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_load_cost_id"] = runtimeparams[
                "custom_unit_load_cost_id"
            ]
        if "custom_unit_prod_price_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_prod_price_id"] = runtimeparams[
                "custom_unit_prod_price_id"
            ]
        if "custom_deferrable_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_deferrable_forecast_id"] = runtimeparams[
                "custom_deferrable_forecast_id"
            ]
        if "custom_predicted_temperature_id" in runtimeparams.keys():
            params["passed_data"]["custom_predicted_temperature_id"] = runtimeparams[
                "custom_predicted_temperature_id"
            ]
        if "custom_heating_demand_id" in runtimeparams.keys():
            params["passed_data"]["custom_heating_demand_id"] = runtimeparams[
                "custom_heating_demand_id"
            ]

    # split config categories from params
    retrieve_hass_conf = params["retrieve_hass_conf"]
    optim_conf = params["optim_conf"]
    plant_conf = params["plant_conf"]

    # Serialize the final params
    params = orjson.dumps(params, default=str).decode()
    return params, retrieve_hass_conf, optim_conf, plant_conf


def get_yaml_parse(params: str | dict, logger: logging.Logger) -> tuple[dict, dict, dict]:
    """
    Perform parsing of the params into the configuration catagories

    :param params: Built configuration parameters
    :type params: str or dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    if params:
        if type(params) is str:
            input_conf = orjson.loads(params)
        else:
            input_conf = params
    else:
        input_conf = {}
        logger.error("No params have been detected for get_yaml_parse")
        return False, False, False

    optim_conf = input_conf.get("optim_conf", {})
    retrieve_hass_conf = input_conf.get("retrieve_hass_conf", {})
    plant_conf = input_conf.get("plant_conf", {})

    # Format time parameters
    if optim_conf.get("delta_forecast_daily", None) is not None:
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=optim_conf["delta_forecast_daily"])
    if retrieve_hass_conf.get("optimization_time_step", None) is not None:
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(
            retrieve_hass_conf["optimization_time_step"], "minutes"
        )
    if retrieve_hass_conf.get("time_zone", None) is not None:
        retrieve_hass_conf["time_zone"] = pytz.timezone(retrieve_hass_conf["time_zone"])

    return retrieve_hass_conf, optim_conf, plant_conf


def get_injection_dict(df: pd.DataFrame, plot_size: int | None = 1366) -> dict:
    """
    Build a dictionary with graphs and tables for the webui.

    :param df: The optimization result DataFrame
    :type df: pd.DataFrame
    :param plot_size: Size of the plot figure in pixels, defaults to 1366
    :type plot_size: Optional[int], optional
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict

    """
    cols_p = [i for i in df.columns.to_list() if "P_" in i]
    # Let's round the data in the DF
    if "optim_status" in df.columns:
        optim_status = df["optim_status"].iloc[0]
    else:
        optim_status = "Status not available"
    df.drop("optim_status", axis=1, inplace=True)
    cols_else = [i for i in df.columns.to_list() if "P_" not in i]
    df = df.apply(pd.to_numeric)
    df[cols_p] = df[cols_p].astype(int)
    df[cols_else] = df[cols_else].round(3)
    # Create plots
    # Figure 0: Systems Powers
    n_colors = len(cols_p)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
    )
    fig_0 = px.line(
        df[cols_p],
        title="Systems powers schedule after optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
        render_mode="svg",
    )
    fig_0.update_layout(xaxis_title="Timestamp", yaxis_title="System powers (W)")
    image_path_0 = fig_0.to_html(full_html=False, default_width="75%")
    # Figure 1: Battery SOC (Optional)
    image_path_1 = None
    if "SOC_opt" in df.columns.to_list():
        fig_1 = px.line(
            df["SOC_opt"],
            title="Battery state of charge schedule after optimization results",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
            render_mode="svg",
        )
        fig_1.update_layout(xaxis_title="Timestamp", yaxis_title="Battery SOC (%)")
        image_path_1 = fig_1.to_html(full_html=False, default_width="75%")
    # Figure Thermal: Temperatures (Optional)
    # Detect columns for predicted, target, min, or max temperatures
    cols_temp = [
        i
        for i in df.columns.to_list()
        if "predicted_temp_heater" in i
        or "target_temp_heater" in i
        or "min_temp_heater" in i
        or "max_temp_heater" in i
    ]
    image_path_temp = None
    if len(cols_temp) > 0:
        n_colors = len(cols_temp)
        colors = px.colors.sample_colorscale(
            "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
        )
        fig_temp = px.line(
            df[cols_temp],
            title="Thermal loads temperature schedule",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
            render_mode="svg",
        )
        fig_temp.update_layout(xaxis_title="Timestamp", yaxis_title="Temperature (&deg;C)")
        image_path_temp = fig_temp.to_html(full_html=False, default_width="75%")
    # Figure 2: Costs
    cols_cost = [i for i in df.columns.to_list() if "cost_" in i or "unit_" in i]
    n_colors = len(cols_cost)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
    )
    fig_2 = px.line(
        df[cols_cost],
        title="Systems costs obtained from optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
        render_mode="svg",
    )
    fig_2.update_layout(xaxis_title="Timestamp", yaxis_title="System costs (currency)")
    image_path_2 = fig_2.to_html(full_html=False, default_width="75%")
    # Tables
    table1 = df.reset_index().to_html(classes="mystyle", index=False)
    cost_cols = [i for i in df.columns if "cost_" in i]
    table2 = df[cost_cols].reset_index().sum(numeric_only=True)
    table2["optim_status"] = optim_status
    table2 = (
        table2.to_frame(name="Value")
        .reset_index(names="Variable")
        .to_html(classes="mystyle", index=False)
    )
    # Construct Injection Dict
    injection_dict = {}
    injection_dict["title"] = "<h2>EMHASS optimization results</h2>"
    injection_dict["subsubtitle0"] = "<h4>Plotting latest optimization results</h4>"
    # Add Powers
    injection_dict["figure_0"] = image_path_0
    # Add Thermal
    if image_path_temp is not None:
        injection_dict["figure_thermal"] = image_path_temp
    # Add SOC
    if image_path_1 is not None:
        injection_dict["figure_1"] = image_path_1
    # Add Costs
    injection_dict["figure_2"] = image_path_2
    injection_dict["subsubtitle1"] = "<h4>Last run optimization results table</h4>"
    injection_dict["table1"] = table1
    injection_dict["subsubtitle2"] = "<h4>Summary table for latest optimization results</h4>"
    injection_dict["table2"] = table2
    return injection_dict


def get_injection_dict_forecast_model_fit(df_fit_pred: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF fit case.

    :param df_fit_pred: The fit result DataFrame
    :type df_fit_pred: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_fit_pred.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = fig.to_html(full_html=False, default_width="75%")
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model fit</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Plotting train/test forecast model results for "
        + mlf.model_type
        + "<br>"
        + "Forecasting variable "
        + mlf.var_model
        + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


def get_injection_dict_forecast_model_tune(df_pred_optim: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF tune case.

    :param df_pred_optim: The tune result DataFrame
    :type df_pred_optim: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_pred_optim.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = fig.to_html(full_html=False, default_width="75%")
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model tune</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Performed a tuning routine using bayesian optimization for "
        + mlf.model_type
        + "<br>"
        + "Forecasting variable "
        + mlf.var_model
        + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


async def build_config(
    emhass_conf: dict,
    logger: logging.Logger,
    defaults_path: str,
    config_path: str | None = None,
    legacy_config_path: str | None = None,
) -> dict:
    """
    Retrieve parameters from configuration files.
    priority order (low - high) = defaults_path, config_path legacy_config_path

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param defaults_path: path to config file for parameter defaults (config_defaults.json)
    :type defaults_path: str
    :param config_path: path to the main configuration file (config.json)
    :type config_path: str
    :param legacy_config_path: path to legacy config file (config_emhass.yaml)
    :type legacy_config_path: str
    :return: The built config dictionary
    :rtype: dict
    """

    # Read default parameters (default root_path/data/config_defaults.json)
    if defaults_path and pathlib.Path(defaults_path).is_file():
        async with aiofiles.open(defaults_path) as data:
            content = await data.read()
            config = orjson.loads(content)
    else:
        logger.error("config_defaults.json. does not exist ")
        return False

    # Read user config parameters if provided (default /share/config.json)
    if config_path and pathlib.Path(config_path).is_file():
        async with aiofiles.open(config_path) as data:
            content = await data.read()
            # Set override default parameters (config_defaults) with user given parameters (config.json)
            logger.info("Obtaining parameters from config.json:")
            config.update(orjson.loads(content))
    else:
        logger.info(
            "config.json does not exist, or has not been passed. config parameters may default to config_defaults.json"
        )
        logger.info("you may like to generate the config.json file on the configuration page")

    # Check to see if legacy config_emhass.yaml was provided (default /app/config_emhass.yaml)
    # Convert legacy parameter definitions/format to match config.json
    if legacy_config_path and pathlib.Path(legacy_config_path).is_file():
        async with aiofiles.open(legacy_config_path) as data:
            content = await data.read()
            legacy_config = yaml.safe_load(content)
            legacy_config_parameters = await build_legacy_config_params(
                emhass_conf, legacy_config, logger
            )
            if type(legacy_config_parameters) is not bool:
                logger.info(
                    "Obtaining parameters from config_emhass.yaml: (will overwrite config parameters)"
                )
                config.update(legacy_config_parameters)

    return config


async def build_legacy_config_params(
    emhass_conf: dict[str, pathlib.Path],
    legacy_config: dict[str, str],
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Build a config dictionary with legacy config_emhass.yaml file.
    Uses the associations file to convert parameter naming conventions (to config.json/config_defaults.json).
    Extracts the parameter values and formats to match config.json.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param legacy_config: The legacy config dictionary
    :type legacy_config: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict
    """

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json)

    # Check each config catagories exists, else create blank dict for categories (avoid errors)
    legacy_config["retrieve_hass_conf"] = legacy_config.get("retrieve_hass_conf", {})
    legacy_config["optim_conf"] = legacy_config.get("optim_conf", {})
    legacy_config["plant_conf"] = legacy_config.get("plant_conf", {})
    config = {}

    # Use associations list to map legacy parameter name with config.json parameter name
    if emhass_conf["associations_path"].exists():
        async with aiofiles.open(emhass_conf["associations_path"]) as data:
            content = await data.read()
            associations = list(csv.reader(content.splitlines(), delimiter=","))
    else:
        logger.error(
            "Cant find associations file (associations.csv) in: "
            + str(emhass_conf["associations_path"])
        )
        return False

    # Loop through all parameters in association file
    # Append config with existing legacy config parameters (converting alternative parameter naming conventions with associations list)
    for association in associations:
        # if legacy config catagories exists and if legacy parameter exists in config catagories
        if (
            legacy_config.get(association[0]) is not None
            and legacy_config[association[0]].get(association[1], None) is not None
        ):
            config[association[2]] = legacy_config[association[0]][association[1]]

            # If config now has load_peak_hour_periods, extract from list of dict
            if association[2] == "load_peak_hour_periods" and type(config[association[2]]) is list:
                config[association[2]] = {key: d[key] for d in config[association[2]] for key in d}

    return config


def get_keys_to_mask() -> list[str]:
    """
    Return a list of sensitive configuration keys that should be masked in logs
    or treated specially in the UI (e.g., secrets).
    """
    return [
        "influxdb_username",
        "influxdb_password",
        "solcast_api_key",
        "solcast_rooftop_id",
        "long_lived_token",
        "time_zone",
        "Latitude",
        "Longitude",
        "Altitude",
        "hass_url",  # Ensure this is included if you want it masked everywhere
        "solar_forecast_kwp",  # Ensure this is included if you want it masked everywhere
    ]


def param_to_config(param: dict[str, dict], logger: logging.Logger) -> dict[str, str]:
    """
    A function that extracts the parameters from param back to the config.json format.
    Extracts parameters from config catagories.
    Attempts to exclude secrets hosed in retrieve_hass_conf.

    :param params: Built configuration parameters
    :type param: dict[str, dict]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict[str, str]
    """
    logger.debug("Converting param to config")

    return_config = {}

    config_categories = ["retrieve_hass_conf", "optim_conf", "plant_conf"]
    secret_params = get_keys_to_mask()

    # Loop through config catagories that contain config params, and extract
    for config in config_categories:
        for parameter in param[config]:
            # If parameter is not a secret, append to return_config
            if parameter not in secret_params:
                return_config[str(parameter)] = param[config][parameter]

    return return_config


async def build_secrets(
    emhass_conf: dict[str, pathlib.Path],
    logger: logging.Logger,
    argument: dict[str, str] | None = None,
    options_path: str | None = None,
    secrets_path: str | None = None,
    no_response: bool = False,
) -> tuple[dict[str, pathlib.Path], dict[str, str | float]]:
    """
    Retrieve and build parameters from secrets locations (ENV, ARG, Secrets file (secrets_emhass.yaml/options.json) and/or Home Assistant (via API))
    priority order (lwo to high) = Defaults (written in function), ENV, Options json file, Home Assistant API,  Secrets yaml file, Arguments

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param argument: dictionary of secrets arguments passed (url,key)
    :type argument: dict
    :param options_path: path to the options file (options.json) (usually provided by EMHASS-Add-on)
    :type options_path: str
    :param secrets_path: path to secrets file (secrets_emhass.yaml)
    :type secrets_path: str
    :param no_response: bypass get request to Home Assistant (json response errors)
    :type no_response: bool
    :return: Updated emhass_conf, the built secrets dictionary
    :rtype: Tuple[dict, dict]:
    """
    # Set defaults to be overwritten
    if argument is None:
        argument = {}
    params_secrets = {
        "hass_url": "https://myhass.duckdns.org/",
        "long_lived_token": "thatverylongtokenhere",
        "time_zone": "Europe/Paris",
        "Latitude": 45.83,
        "Longitude": 6.86,
        "Altitude": 4807.8,
        "solcast_api_key": "yoursecretsolcastapikey",
        "solcast_rooftop_id": "yourrooftopid",
        "solar_forecast_kwp": 5,
        "influxdb_username": "yourinfluxdbusername",
        "influxdb_password": "yourinfluxdbpassword",
    }

    # Obtain Secrets from ENV?
    params_secrets["hass_url"] = os.getenv("EMHASS_URL", params_secrets["hass_url"])
    params_secrets["long_lived_token"] = os.getenv(
        "SUPERVISOR_TOKEN", params_secrets["long_lived_token"]
    )
    params_secrets["time_zone"] = os.getenv("TIME_ZONE", params_secrets["time_zone"])
    params_secrets["Latitude"] = float(os.getenv("LAT", params_secrets["Latitude"]))
    params_secrets["Longitude"] = float(os.getenv("LON", params_secrets["Longitude"]))
    params_secrets["Altitude"] = float(os.getenv("ALT", params_secrets["Altitude"]))

    # Obtain secrets from options.json (Generated from EMHASS-Add-on, Home Assistant addon Configuration page) or Home Assistant API (from local Supervisor API)?
    # Use local supervisor API to obtain secrets from Home Assistant if hass_url in options.json is empty and SUPERVISOR_TOKEN ENV exists (provided by Home Assistant when running the container as addon)
    options = {}
    if options_path and pathlib.Path(options_path).is_file():
        async with aiofiles.open(options_path) as data:
            content = await data.read()
            options = orjson.loads(content)

            # Obtain secrets from Home Assistant?
            url_from_options = options.get("hass_url", "empty")
            key_from_options = options.get("long_lived_token", "empty")

            # If data path specified by options.json, overwrite emhass_conf['data_path']
            data_path_value = options.get("data_path", None)
            if data_path_value is not None and data_path_value != "" and data_path_value != "default":
                # Try to create directory if it doesn't exist. if successful set data_path in emhass_conf
                try: 
                    data_path = pathlib.Path(data_path_value)
                    # Use parents=True to create nested directories
                    data_path.mkdir(parents=True, exist_ok=True)
                    emhass_conf["data_path"] = data_path
                    logger.info(f"Using custom data_path: {data_path}")
                except Exception as e:
                    logger.warning(
                        f"Cannot create data_path directory '{data_path_value}' provided via options. Keeping default. Error: {e}"
                    )

           # If config path specified by options.json, overwrite emhass_conf['config_path']
            config_path_value = options.get("config_path", None)
            if config_path_value is not None and config_path_value != "" and config_path_value != "default":
                try: 
                    config_path = pathlib.Path(config_path_value)
                    # Validate that the config file or its parent directory path is valid
                    if config_path.exists():
                        # File exists - use it
                        emhass_conf["config_path"] = config_path
                        logger.info(f"Using custom config_path from addon settings: {config_path}")
                        print(f"Using custom config_path from addon settings: {config_path}")
                    elif config_path.parent.exists():
                        # Parent directory exists but file doesn't - set path anyway (file may be created later)
                        emhass_conf["config_path"] = config_path
                        logger.warning(f"Config file does not exist yet: {config_path} (will use defaults until created)")
                    else:
                        # Neither file nor parent directory exists - this is likely an error
                        logger.error(f"Invalid config_path '{config_path_value}': parent directory does not exist. Keeping default config_path.")
                except Exception as e:
                    logger.warning(
                        f"Cannot set config_path '{config_path_value}' provided via options. Keeping default. Error: {e}"
                    )
            elif config_path_value == "default":
                logger.info("set config_path to addon-mode default /config/config.json.")
                print("set config_path to addon-mode default /config/config.json.")
                emhass_conf["config_path"] = pathlib.Path("/config/config.json")
            else:
                logger.info("No config_path provided via options.json, checking legacy path /share/config.json or using addon-mode default /config/config.json.")
                print("No config_path provided via options.json, checking legacy path /share/config.json or using addon-mode default /config/config.json.")
                    # Check if legacy config path exists, if yes use it, otherwise use addon-mode default
                legacy_config_path = pathlib.Path("/share/config.json")
                if legacy_config_path.is_file():
                    logger.info("Found legacy config.json in /share, using this path for config_path.")
                    print("Found legacy config.json in /share, using this path for config_path.")
                    emhass_conf["config_path"] = legacy_config_path
                else:
                    logger.info("No legacy config.json found in /share, using addon-mode default /config/config.json for config_path.")
                    print("No legacy config.json found in /share, using addon-mode default /config/config.json for config_path.")
                    emhass_conf["config_path"] = pathlib.Path("/config/config.json")


            # Check to use Home Assistant local API
            if not no_response and os.getenv("SUPERVISOR_TOKEN", None) is not None:
                params_secrets["long_lived_token"] = os.getenv("SUPERVISOR_TOKEN", None)
                # Use hass_url from options.json if available, otherwise use supervisor API for addon
                if url_from_options != "empty" and url_from_options != "":
                    params_secrets["hass_url"] = url_from_options
                else:
                    # For addons, use supervisor API for both REST and WebSocket access
                    params_secrets["hass_url"] = "http://supervisor/core/api"
                headers = {
                    "Authorization": "Bearer " + params_secrets["long_lived_token"],
                    "content-type": "application/json",
                }
                # Obtain secrets from Home Assistant via API
                logger.debug("Obtaining secrets from Home Assistant Supervisor API")
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        params_secrets["hass_url"] + "/config", headers=headers
                    ) as response:
                        if response.status < 400:
                            config_hass = await response.json()
                            params_secrets.update(
                                {
                                    "hass_url": params_secrets["hass_url"],
                                    "long_lived_token": params_secrets["long_lived_token"],
                                    "time_zone": config_hass["time_zone"],
                                    "Latitude": config_hass["latitude"],
                                    "Longitude": config_hass["longitude"],
                                    "Altitude": config_hass["elevation"],
                                    # If defined in HA config, use them, otherwise keep defaults
                                    "solcast_api_key": config_hass.get(
                                        "solcast_api_key", params_secrets["solcast_api_key"]
                                    ),
                                    "solcast_rooftop_id": config_hass.get(
                                        "solcast_rooftop_id", params_secrets["solcast_rooftop_id"]
                                    ),
                                    "solar_forecast_kwp": config_hass.get(
                                        "solar_forecast_kwp", params_secrets["solar_forecast_kwp"]
                                    ),
                                    "influxdb_username": config_hass.get(
                                        "influxdb_username", params_secrets.get("influxdb_username")
                                    ),
                                    "influxdb_password": config_hass.get(
                                        "influxdb_password", params_secrets.get("influxdb_password")
                                    ),
                                }
                            )
                        else:
                            # Obtain the url and key secrets if any from options.json (default /app/options.json)
                            logger.warning(
                                "Error obtaining secrets from Home Assistant Supervisor API"
                            )
                            logger.debug("Obtaining url and key secrets from options.json")
                            if url_from_options != "empty" and url_from_options != "":
                                params_secrets["hass_url"] = url_from_options
                            if key_from_options != "empty" and key_from_options != "":
                                params_secrets["long_lived_token"] = key_from_options
                            if (
                                options.get("time_zone", "empty") != "empty"
                                and options["time_zone"] != ""
                            ):
                                params_secrets["time_zone"] = options["time_zone"]
                            if options.get("Latitude", None) is not None and bool(
                                options["Latitude"]
                            ):
                                params_secrets["Latitude"] = options["Latitude"]
                            if options.get("Longitude", None) is not None and bool(
                                options["Longitude"]
                            ):
                                params_secrets["Longitude"] = options["Longitude"]
                            if options.get("Altitude", None) is not None and bool(
                                options["Altitude"]
                            ):
                                params_secrets["Altitude"] = options["Altitude"]

            # Obtain the forecast secrets (if any) from options.json (default /app/options.json)
            # This logic runs regardless of whether HA API call above succeeded or failed,
            # so we removed the duplicate logic from the 'else' block above.
            forecast_secrets = [
                "solcast_api_key",
                "solcast_rooftop_id",
                "solar_forecast_kwp",
            ]
            if any(x in forecast_secrets for x in list(options.keys())):
                logger.debug("Obtaining forecast secrets from options.json")
                if (
                    options.get("solcast_api_key", "empty") != "empty"
                    and options["solcast_api_key"] != ""
                ):
                    params_secrets["solcast_api_key"] = options["solcast_api_key"]
                if (
                    options.get("solcast_rooftop_id", "empty") != "empty"
                    and options["solcast_rooftop_id"] != ""
                ):
                    params_secrets["solcast_rooftop_id"] = options["solcast_rooftop_id"]
                if options.get("solar_forecast_kwp", None) and bool(options["solar_forecast_kwp"]):
                    params_secrets["solar_forecast_kwp"] = options["solar_forecast_kwp"]

            # Obtain InfluxDB secrets from options.json
            influx_secrets = ["influxdb_username", "influxdb_password"]
            if any(x in influx_secrets for x in list(options.keys())):
                logger.debug("Obtaining InfluxDB secrets from options.json")
                if (
                    options.get("influxdb_username", "empty") != "empty"
                    and options["influxdb_username"] != ""
                ):
                    params_secrets["influxdb_username"] = options["influxdb_username"]
                if (
                    options.get("influxdb_password", "empty") != "empty"
                    and options["influxdb_password"] != ""
                ):
                    params_secrets["influxdb_password"] = options["influxdb_password"]

    # Obtain secrets from secrets_emhass.yaml? (default /app/secrets_emhass.yaml)
    if secrets_path and pathlib.Path(secrets_path).is_file():
        logger.debug("Obtaining secrets from secrets file")
        async with aiofiles.open(pathlib.Path(secrets_path)) as file:
            content = await file.read()
            params_secrets.update(yaml.safe_load(content))

    # Receive key and url from ARG/arguments?
    if argument.get("url") is not None:
        params_secrets["hass_url"] = argument["url"]
        logger.debug("Obtaining url from passed argument")
    if argument.get("key") is not None:
        params_secrets["long_lived_token"] = argument["key"]
        logger.debug("Obtaining long_lived_token from passed argument")

    return emhass_conf, params_secrets


async def build_params(
    emhass_conf: dict[str, pathlib.Path],
    params_secrets: dict[str, str | float],
    config: dict[str, str],
    logger: logging.Logger,
) -> dict[str, dict]:
    """
    Build the main params dictionary from the config and secrets
    Appends configuration catagories used by emhass to the parameters. (with use of the associations file as a reference)

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict[str, pathlib.Path]
    :param params_secrets: The dictionary containing the built secret variables
    :type params_secrets: dict[str, str | float]
    :param config: The dictionary of built config parameters
    :type config: dict[str, str]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built param dictionary
    :rtype: dict[str, dict]
    """
    if not isinstance(params_secrets, dict):
        params_secrets = {}

    params = {}
    # Start with blank config catagories
    params["retrieve_hass_conf"] = {}
    params["params_secrets"] = {}
    params["optim_conf"] = {}
    params["plant_conf"] = {}

    # Obtain associations to categorize parameters to their corresponding config catagories
    if emhass_conf.get(
        "associations_path", get_root(__file__, num_parent=2) / "data/associations.csv"
    ).exists():
        async with aiofiles.open(emhass_conf["associations_path"]) as data:
            content = await data.read()
            associations = list(csv.reader(content.splitlines(), delimiter=","))
    else:
        logger.error(
            "Unable to obtain the associations file (associations.csv) in: "
            + str(emhass_conf["associations_path"])
        )
        return False

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json)
    # Use association list to append parameters from config into params (with corresponding config catagories)
    for association in associations:
        # If parameter has list_ name and parameter in config is presented with its list name
        # (ie, config parameter is in legacy options.json format)
        if len(association) == 4 and config.get(association[3]) is not None:
            # Extract lists of dictionaries
            if config[association[3]] and type(config[association[3]][0]) is dict:
                params[association[0]][association[2]] = [
                    i[association[2]] for i in config[association[3]]
                ]
            else:
                params[association[0]][association[2]] = config[association[3]]
        # Else, directly set value of config parameter to param
        elif config.get(association[2]) is not None:
            params[association[0]][association[2]] = config[association[2]]

    # Check if we need to create `list_hp_periods` from config (ie. legacy options.json format)
    if (
        params.get("optim_conf") is not None
        and config.get("list_peak_hours_periods_start_hours") is not None
        and config.get("list_peak_hours_periods_end_hours") is not None
    ):
        start_hours_list = [
            i["peak_hours_periods_start_hours"]
            for i in config["list_peak_hours_periods_start_hours"]
        ]
        end_hours_list = [
            i["peak_hours_periods_end_hours"] for i in config["list_peak_hours_periods_end_hours"]
        ]
        num_peak_hours = len(start_hours_list)
        list_hp_periods_list = {
            "period_hp_" + str(i + 1): [
                {"start": start_hours_list[i]},
                {"end": end_hours_list[i]},
            ]
            for i in range(num_peak_hours)
        }
        params["optim_conf"]["load_peak_hour_periods"] = list_hp_periods_list
    else:
        # Else, check param already contains load_peak_hour_periods from config
        if params["optim_conf"].get("load_peak_hour_periods", None) is None:
            logger.warning("Unable to detect or create load_peak_hour_periods parameter")

    # Format load_peak_hour_periods list to dict if necessary
    if params["optim_conf"].get("load_peak_hour_periods", None) is not None and isinstance(
        params["optim_conf"]["load_peak_hour_periods"], list
    ):
        params["optim_conf"]["load_peak_hour_periods"] = {
            key: d[key] for d in params["optim_conf"]["load_peak_hour_periods"] for key in d
        }

    # Call function to check parameter lists that require the same length as deferrable loads
    # If not, set defaults it fill in gaps
    if params["optim_conf"].get("number_of_deferrable_loads", None) is not None:
        num_def_loads = params["optim_conf"]["number_of_deferrable_loads"]
        params["optim_conf"]["start_timesteps_of_each_deferrable_load"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            0,
            "start_timesteps_of_each_deferrable_load",
            logger,
        )
        params["optim_conf"]["end_timesteps_of_each_deferrable_load"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            0,
            "end_timesteps_of_each_deferrable_load",
            logger,
        )
        params["optim_conf"]["set_deferrable_load_single_constant"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            False,
            "set_deferrable_load_single_constant",
            logger,
        )
        params["optim_conf"]["treat_deferrable_load_as_semi_cont"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            True,
            "treat_deferrable_load_as_semi_cont",
            logger,
        )
        params["optim_conf"]["set_deferrable_startup_penalty"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            0.0,
            "set_deferrable_startup_penalty",
            logger,
        )
        params["optim_conf"]["operating_hours_of_each_deferrable_load"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            0,
            "operating_hours_of_each_deferrable_load",
            logger,
        )
        params["optim_conf"]["nominal_power_of_deferrable_loads"] = check_def_loads(
            num_def_loads,
            params["optim_conf"],
            0,
            "nominal_power_of_deferrable_loads",
            logger,
        )
    else:
        logger.warning("unable to obtain parameter: number_of_deferrable_loads")
    # historic_days_to_retrieve should be no less then 2
    if params["retrieve_hass_conf"].get("historic_days_to_retrieve", None) is not None:
        if params["retrieve_hass_conf"]["historic_days_to_retrieve"] < 2:
            params["retrieve_hass_conf"]["historic_days_to_retrieve"] = 2
            logger.warning(
                "days_to_retrieve should not be lower then 2, setting days_to_retrieve to 2. Make sure your sensors also have at least 2 days of history"
            )
    else:
        logger.warning("unable to obtain parameter: historic_days_to_retrieve")

    # Configure secrets, set params to correct config categorie
    # retrieve_hass_conf
    params["retrieve_hass_conf"]["hass_url"] = params_secrets.get("hass_url")
    params["retrieve_hass_conf"]["long_lived_token"] = params_secrets.get("long_lived_token")
    params["retrieve_hass_conf"]["time_zone"] = params_secrets.get("time_zone")
    params["retrieve_hass_conf"]["Latitude"] = params_secrets.get("Latitude")
    params["retrieve_hass_conf"]["Longitude"] = params_secrets.get("Longitude")
    params["retrieve_hass_conf"]["Altitude"] = params_secrets.get("Altitude")
    if params_secrets.get("influxdb_username") is not None:
        params["retrieve_hass_conf"]["influxdb_username"] = params_secrets.get("influxdb_username")
        params["params_secrets"]["influxdb_username"] = params_secrets.get("influxdb_username")
    if params_secrets.get("influxdb_password") is not None:
        params["retrieve_hass_conf"]["influxdb_password"] = params_secrets.get("influxdb_password")
        params["params_secrets"]["influxdb_password"] = params_secrets.get("influxdb_password")
    # Update optional param secrets
    if params["optim_conf"].get("weather_forecast_method", None) is not None:
        if params["optim_conf"]["weather_forecast_method"] == "solcast":
            params["retrieve_hass_conf"]["solcast_api_key"] = params_secrets.get(
                "solcast_api_key", "123456"
            )
            params["params_secrets"]["solcast_api_key"] = params_secrets.get(
                "solcast_api_key", "123456"
            )
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = params_secrets.get(
                "solcast_rooftop_id", "123456"
            )
            params["params_secrets"]["solcast_rooftop_id"] = params_secrets.get(
                "solcast_rooftop_id", "123456"
            )
        elif params["optim_conf"]["weather_forecast_method"] == "solar.forecast":
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = params_secrets.get(
                "solar_forecast_kwp", 5
            )
            params["params_secrets"]["solar_forecast_kwp"] = params_secrets.get(
                "solar_forecast_kwp", 5
            )
    else:
        logger.warning("Unable to detect weather_forecast_method parameter")
    #  Check if secrets parameters still defaults values
    secret_params = [
        "https://myhass.duckdns.org/",
        "thatverylongtokenhere",
        45.83,
        6.86,
        4807.8,
    ]
    if any(x in secret_params for x in params["retrieve_hass_conf"].values()):
        logger.warning("Some secret parameters values are still matching their defaults")

    # Set empty dict objects for params passed_data
    # To be latter populated with runtime parameters (treat_runtimeparams)
    params["passed_data"] = {
        "pv_power_forecast": None,
        "load_power_forecast": None,
        "load_cost_forecast": None,
        "prod_price_forecast": None,
        "prediction_horizon": None,
        "soc_init": None,
        "soc_final": None,
        "operating_hours_of_each_deferrable_load": None,
        "start_timesteps_of_each_deferrable_load": None,
        "end_timesteps_of_each_deferrable_load": None,
        "alpha": None,
        "beta": None,
    }

    return params


def check_def_loads(
    num_def_loads: int,
    parameter: list[dict],
    default: str | float,
    parameter_name: str,
    logger: logging.Logger,
) -> list[dict]:
    """
    Check parameter lists with deferrable loads number, if they do not match, enlarge to fit.

    :param num_def_loads: Total number deferrable loads
    :type num_def_loads: int
    :param parameter: parameter config dict containing paramater
    :type parameter: list[dict]
    :param default: default value for parameter to pad missing
    :type default: str | int | float
    :param parameter_name: name of parameter
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: parameter list
    :rtype: list[dict]
    """
    if (
        parameter.get(parameter_name, None) is not None
        and type(parameter[parameter_name]) is list
        and num_def_loads > len(parameter[parameter_name])
    ):
        logger.warning(
            parameter_name
            + " does not match number in num_def_loads, adding default values ("
            + str(default)
            + ") to parameter"
        )
        for _x in range(len(parameter[parameter_name]), num_def_loads):
            parameter[parameter_name].append(default)
    return parameter[parameter_name]


def get_days_list(days_to_retrieve: int) -> pd.DatetimeIndex:
    """
    Get list of past days from today to days_to_retrieve.

    :param days_to_retrieve: Total number of days to retrieve from the past
    :type days_to_retrieve: int
    :return: The list of days
    :rtype: pd.DatetimeIndex

    """
    today = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    d = (today - timedelta(days=days_to_retrieve)).isoformat()
    days_list = pd.date_range(start=d, end=today.isoformat(), freq="D").normalize()
    return days_list


def add_date_features(
    data: pd.DataFrame,
    timestamp: str | None = None,
    date_features: list[str] | None = None,
) -> pd.DataFrame:
    """Add date-related features from a DateTimeIndex or a timestamp column.

    :param data: The input DataFrame.
    :type data: pd.DataFrame
    :param timestamp: The column containing the timestamp (optional if DataFrame has a DateTimeIndex).
    :type timestamp: Optional[str]
    :param date_features: List of date features to extract (default: all).
    :type date_features: Optional[List[str]]
    :return: The DataFrame with added date features.
    :rtype: pd.DataFrame
    """

    df = copy.deepcopy(data)  # Avoid modifying the original DataFrame

    # If no specific features are requested, extract all by default
    default_features = ["year", "month", "day_of_week", "day_of_year", "day", "hour"]
    date_features = date_features or default_features

    # Determine whether to use index or a timestamp column
    if timestamp:
        df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
        source = df[timestamp].dt
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DateTimeIndex or a valid timestamp column.")
        source = df.index

    # Extract date features
    if "year" in date_features:
        df["year"] = source.year
    if "month" in date_features:
        df["month"] = source.month
    if "day_of_week" in date_features:
        df["day_of_week"] = source.dayofweek
    if "day_of_year" in date_features:
        df["day_of_year"] = source.dayofyear
    if "day" in date_features:
        df["day"] = source.day
    if "hour" in date_features:
        df["hour"] = source.hour

    return df


def set_df_index_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the freq of a DataFrame DateTimeIndex.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Input DataFrame with freq defined
    :rtype: pd.DataFrame

    """
    idx_diff = np.diff(df.index)
    # Sometimes there are zero values in this list.
    idx_diff = idx_diff[np.nonzero(idx_diff)]
    sampling = pd.to_timedelta(np.median(idx_diff))
    df = df[~df.index.duplicated()]
    return df.asfreq(sampling)


def parse_export_time_range(
    start_time: str,
    end_time: str | None,
    time_zone: pd.Timestamp.tz,
    logger: logging.Logger,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[bool, bool]:
    """
    Parse and validate start_time and end_time for export operations.

    :param start_time: Start time string in ISO format
    :type start_time: str
    :param end_time: End time string in ISO format (optional)
    :type end_time: str | None
    :param time_zone: Timezone for localization
    :type time_zone: pd.Timestamp.tz
    :param logger: Logger object
    :type logger: logging.Logger
    :return: Tuple of (start_dt, end_dt) or (False, False) on error
    :rtype: tuple[pd.Timestamp, pd.Timestamp] | tuple[bool, bool]
    """
    try:
        start_dt = pd.to_datetime(start_time)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(time_zone)
    except Exception as e:
        logger.error(f"Invalid start_time format: {start_time}. Error: {e}")
        logger.error("Use format like '2024-01-01' or '2024-01-01 00:00:00'")
        return False, False

    if end_time:
        try:
            end_dt = pd.to_datetime(end_time)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize(time_zone)
        except Exception as e:
            logger.error(f"Invalid end_time format: {end_time}. Error: {e}")
            return False, False
    else:
        end_dt = pd.Timestamp.now(tz=time_zone)
        logger.info(f"No end_time specified, using current time: {end_dt}")

    return start_dt, end_dt


def clean_sensor_column_names(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Clean sensor column names by removing 'sensor.' prefix.

    :param df: Input DataFrame with sensor columns
    :type df: pd.DataFrame
    :param timestamp_col: Name of timestamp column to preserve
    :type timestamp_col: str
    :return: DataFrame with cleaned column names
    :rtype: pd.DataFrame
    """
    column_mapping = {}
    for col in df.columns:
        if col != timestamp_col and col.startswith("sensor."):
            column_mapping[col] = col.replace("sensor.", "")
    return df.rename(columns=column_mapping)


def handle_nan_values(
    df: pd.DataFrame,
    handle_nan: str,
    timestamp_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Handle NaN values in DataFrame according to specified strategy.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param handle_nan: Strategy for handling NaN values
    :type handle_nan: str
    :param timestamp_col: Name of timestamp column to exclude from processing
    :type timestamp_col: str
    :param logger: Logger object
    :type logger: logging.Logger
    :return: DataFrame with NaN values handled
    :rtype: pd.DataFrame
    """
    nan_count_before = df.isna().sum().sum()
    if nan_count_before == 0:
        return df

    logger.info(f"Found {nan_count_before} NaN values, applying handle_nan method: {handle_nan}")

    if handle_nan == "drop":
        df = df.dropna()
        logger.info(f"Dropped rows with NaN. Remaining rows: {len(df)}")
    elif handle_nan == "fill_zero":
        # Exclude timestamp_col from fillna to avoid unintended changes
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].fillna(0)
        logger.info("Filled NaN values with 0 (excluding timestamp)")
    elif handle_nan == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude timestamp_col from interpolation
        interp_cols = [col for col in numeric_cols if col != timestamp_col]
        df[interp_cols] = df[interp_cols].interpolate(method="linear", limit_direction="both")
        df[interp_cols] = df[interp_cols].ffill().bfill()
        logger.info("Interpolated NaN values (excluding timestamp)")
    elif handle_nan == "forward_fill":
        # Exclude timestamp_col from forward fill
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].ffill()
        logger.info("Forward filled NaN values (excluding timestamp)")
    elif handle_nan == "backward_fill":
        # Exclude timestamp_col from backward fill
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].bfill()
        logger.info("Backward filled NaN values (excluding timestamp)")
    elif handle_nan == "keep":
        logger.info("Keeping NaN values as-is")
    else:
        logger.warning(f"Unknown handle_nan option '{handle_nan}', keeping NaN values")

    return df


def resample_and_filter_data(
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    resample_freq: str,
    logger: logging.Logger,
) -> pd.DataFrame | bool:
    """
    Filter DataFrame to time range and resample to specified frequency.

    :param df: Input DataFrame with datetime index
    :type df: pd.DataFrame
    :param start_dt: Start datetime for filtering
    :type start_dt: pd.Timestamp
    :param end_dt: End datetime for filtering
    :type end_dt: pd.Timestamp
    :param resample_freq: Resampling frequency string (e.g., '1h', '30min')
    :type resample_freq: str
    :param logger: Logger object
    :type logger: logging.Logger
    :return: Resampled DataFrame or False on error
    :rtype: pd.DataFrame | bool
    """
    # Validate that DataFrame index is datetime and properly localized
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}")
        return False

    # Check if timezone aware and matches expected timezone
    if df.index.tz is None:
        logger.warning("DataFrame index is timezone-naive, localizing to match start/end times")
        df = df.copy()
        df.index = df.index.tz_localize(start_dt.tz)
    elif df.index.tz != start_dt.tz:
        logger.warning(
            f"DataFrame timezone ({df.index.tz}) differs from filter timezone ({start_dt.tz}), converting"
        )
        df = df.copy()
        df.index = df.index.tz_convert(start_dt.tz)

    # Filter to exact time range
    df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]

    if df_filtered.empty:
        logger.error("No data in the specified time range after filtering")
        return False

    logger.info(f"Retrieved {len(df_filtered)} data points")

    # Resample to specified frequency
    logger.info(f"Resampling data to frequency: {resample_freq}")
    try:
        df_resampled = df_filtered.resample(resample_freq).mean()
        df_resampled = df_resampled.dropna(how="all")

        if df_resampled.empty:
            logger.error("No data after resampling. Check frequency and data availability.")
            return False

        logger.info(f"After resampling: {len(df_resampled)} data points")
        return df_resampled

    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        return False
