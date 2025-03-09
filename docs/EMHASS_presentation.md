---
theme: gaia # default, gaia, uncover,
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
marp: true

---

![bg left:40% 80%](https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/emhass_logo.svg)

# **EMHASS**

Energy Management for Home Assistant

Presented by David HERNANDEZ TORRES
April 2025

https://github.com/davidusb-geek/emhass

---
# Introduction

**EMHASS is a Python module designed to optimize your home energy**
**interfacing with Home Assistant**

Main features :rocket: :
- Real optimization algorithms for home energy management
- Fully integrated to Home Assistant, but standalone usage also possible
- Integrated forecast methods 
- Built-in machine learning
- Solar PV, batteries, thermal loads, and more...

---
# Introduction

## Main principles

- EMHASS is a powerful energy management tool that generates an optimization plan based on variables such as solar power production, energy usage, and energy costs. 

- The plan provides valuable insights into how energy can be better managed and utilized in the household. 

- Even if households do not have all the necessary equipment, such as solar panels or batteries, EMHASS can still provide a minimal use case solution to optimize energy usage for controllable/deferrable loads.

---
<!-- paginate: true -->
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# Introduction

Already 4 years of development :tada:
Currently on v0.12.8!

## Star History

[![w:500 center Star History Chart](https://api.star-history.com/svg?repos=davidusb-geek/emhass,davidusb-geek/emhass-add-on&type=Date)](https://star-history.com/#davidusb-geek/emhass&davidusb-geek/emhass-add-on&Date)

---
<!-- paginate: true -->
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# Introduction

- Package principle

![w:900 center](https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/ems_schema.svg)

---

# An EMS based on Linear Programming

- The main profit maximize cost function

$$
\sum_{i=1}^{\Delta_{opt}/\Delta_t} -0.001*\Delta_t*(unit_{LoadCost}[i]*P_{gridPos}[i] + prod_{SellPrice}*P_{gridNeg}[i])
$$

- The main contraint power balance

$$
P_{PV_i}-P_{defSum_i}-P_{load_i}+P_{gridNeg_i}+P_{gridPos_i}+P_{stoPos_i}+P_{stoNeg_i}=0
$$

- Many other constraints: deferrables, battery, thermal loads, hybrid inverter, PV curtailement, etc.

---

# An EMS based on Linear Programming

## The EMHASS optimizations

There are 3 different optimization types that are implemented in EMHASS.

- A perfect forecast optimization.

- A day-ahead optimization.

- A Model Predictive Control optimization.

---
<!-- paginate: true -->
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# An EMS based on Linear Programming

## The EMHASS optimizations

![w:800 center](https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/optimization_graphics.svg)

---

# Forecasting utilities

## Taming uncertainties

EMHASS will need 4 forecasts to work properly:

- PV power production forecast
    - Detailed weather forecast with `open-meteo` (15 min/1 km) and PV modeling using NREL PVLib module 
- Load power forecast
    - Custom machine learning autorgressive models
- Load cost forecast

- PV production selling price forecast

---
<!-- paginate: true -->
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# The machine learning forecaster

- An efficient way to forecast the power load consumption

- Based on the `skforecast` module that uses `scikit-learn` regression models considering auto-regression lags as features

- The hyperparameter optimization is proposed using **Bayesian optimization** from the `optuna` 

---

# The machine learning forecaster

![w:600 center](https://emhass.readthedocs.io/en/latest/_images/load_forecast_knn_optimized.svg)

---

# Deferrable load thermal model

- EMHASS supports defining a deferrable load as a thermal model.
This is useful to control thermal equipment such as heaters, heat pumps, air conditioners, etc.
- The advantage of using this approach is that you will be able to define your desired room temperature just as you will do with your real equipment thermostat.
- Then EMHASS will deliver the operating schedule to maintain that desired temperature while minimizing the energy bill and taking into account the forecasted outdoor temperature.

---

# Deferrable load thermal model

The thermal model implemented in EMHASS is a linear model represented by the following equation:

$$
    T_{in}^{pred}[k+1] = T_{in}^{pred}[k] + P_{def}[k]\frac{\alpha_h\Delta t}{P_{def}^{nom}}-(\gamma_c(T_{in}^{pred}[k] - T_{out}^{fcst}[k]))
$$

where $k$ is each time instant, $T_{in}^{pred}$ is the indoor predicted temperature, $T_{out}^{fcst}$ is the outdoor forecasted temperature and $P_{def}$ is the deferrable load power.

In this model we can see two main configuration parameters:
- The heating rate $\alpha_h$ in degrees per hour.
- The cooling constant $\gamma_c$ in degrees per hour per degree of cooling.

---

# Deferrable load thermal model

A graphic representation might be better right? :wink: 

![w:800 center](https://emhass.readthedocs.io/en/latest/_images/thermal_load_diagram.svg)

--- 

# Current and future work

- Detailed PV forecast adjusted to local weather conditions using machine learning
- Support V2x natively
- Improved on-board experience
- Code quamity, security, maintainability, etc.

---

# Thank you!