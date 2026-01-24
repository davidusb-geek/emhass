# Main Core Concepts: The Basics

EMHASS is the "brain" of your energy system. While Home Assistant sees what is happening *now*, EMHASS looks at the future (forecasts) to decide what *should* happen to save you the most money.

You do not need to understand complex mathematics to use EMHASS. You simply need to understand three things:
1. **The Goal** (Cost Functions)
2. **The Timing** (Optimization Types)
3. **The Devices** (Deferrable Loads, Solar PV Panels & Batteries)

---

## 1. The Goal: Choosing a Cost Function

The most important setting in EMHASS is the `costfun` (Cost Function). This tells the system what your priority is. 

There are three modes. **For 95% of users, "Profit" is the correct choice.**

### 🟢 Profit (Recommended)
**Objective:** Maximize your total wallet balance.
> Formula logic: (Money Earned from Export) - (Money Spent on Import)

This is the smartest mode. It considers the value of energy in both directions.
* **How it behaves:** It calculates opportunity costs. For example, it might decide *not* to charge your battery from solar right now because export prices are high. Instead, it sells the solar now and charges the battery from the grid at 2 AM when power is dirt cheap.
* **Best for:**
    * Users with **Dynamic Tariffs** (e.g., Octopus Agile, Tibber, Amber).
    * Users with **Batteries**.
    * Users who get paid for **Export** (Feed-in Tariffs).

### 🔵 Cost
**Objective:** Minimize the money you pay to the grid.
> Formula logic: Minimize (Money Spent on Import)

This is very similar to *Profit*, but it effectively treats export revenue as $0. It focuses purely on self-sufficiency to save money.
* **How it behaves:** It will aggressively prevent you from buying from the grid. However, it might miss arbitrage opportunities (e.g., buying cheap grid power to sell back later for a profit) because it doesn't value the "sale" aspect.
* **Best for:**
    * Users with Time-of-Use tariffs (cheap night rates) but **NO** export payments.
    * Users who strictly want to lower their bill, ignoring potential export income.

### 🟠 Self-Consumption
**Objective:** Minimize grid interaction entirely.
> Formula logic: Minimize (Grid Import) + (Grid Export)

This function ignores prices entirely. It treats the grid as "lava."
* **How it behaves:** It matches your consumption curve exactly to your solar generation. It will run devices only when the sun is shining, even if electricity is free at night. It will hold battery charge strictly to avoid grid usage, even if it makes financial sense to export.
* **Best for:**
    * Users with **Zero Export** limitations (physically not allowed to send power to grid).
    * Off-grid setups.
    * "Green" idealists who want to be self-sufficient regardless of the financial cost.

---

## 2. The Timing: Optimization Types

EMHASS generates a plan (a schedule) for your devices. There are two main ways to generate this plan.

### 📅 Day-Ahead Optimization
* **What it is:** The "Big Plan" for the next 24 hours.
* **When to run it:** Once a day (e.g., at 5:30 AM or when your new energy prices are published).
* **What it does:** It looks at the weather forecast and price forecast for the whole day and generates the optimal schedule for your battery and appliances.

### ⏱️ Model Predictive Control (MPC)
* **What it is:** The "Course Correction."
* **When to run it:** Frequently (e.g., every 5, 10, or 15 minutes).
* **What it does:** Forecasts are never perfect. Clouds move, and you might turn on the kettle unexpectedly. MPC updates the plan based on what is *actually* happening right now. It keeps the long-term goal in mind but adjusts the immediate actions to handle reality.

> **Analogy:** 
> * **Day-Ahead** is checking Google Maps before you leave your house to see the route.
> * **MPC** is the GPS re-routing you while you drive because of a traffic jam.


---

## 3. The Devices

### ☀️ Solar PV Panels
EMHASS needs to know how much power your solar panels will produce to make smart decisions. We have simplified this setup significantly:

* **Easiest Method (External APIs):** You can simply connect EMHASS to forecast providers like **Solcast** or **Forecast.Solar**. EMHASS will automatically pull the production data it needs.
* **Internal Method (PVLib):** If you prefer to calculate forecasts locally, you no longer need to hunt for obscure technical model names in a database. Just enter the **nominal power** (in Watts) of your PV panels and your Inverter. EMHASS will automatically search its extensive database to find a matching device profile for you.

### 🔌 Deferrable Loads
These are devices that need to run for a set amount of time, but it doesn't matter *when* (within a window).
* **Examples:** Washing machines, dishwashers, pool pumps, EV chargers.
* **How EMHASS handles them:** You tell EMHASS: "This needs to run for 3 hours before 8 PM." EMHASS finds the 3 cheapest/sunniest hours in that window and gives you a schedule.

### 🔋 Batteries
Batteries are the most complex device to manage manually, but the easiest for EMHASS.
* **How EMHASS handles them:** You don't need to write rules like "Charge if solar > 2000W." EMHASS simply simulates the battery's state of charge for the next 24 hours and finds the mathematically optimal times to charge or discharge to hit the "Profit" goal.

```{note} 

If you don't have any of these devices, they can be easily skipped in the Configurator editor on the Add-on user interface
```

---

## Technical Reference

For the mathematical definitions, linear programming matrix formulations, and academic references, please see the [Advanced Mathematical Model](./advanced_math_model.md) page.