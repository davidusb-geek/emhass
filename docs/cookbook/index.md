# 🧑‍🍳 Cookbook

Short, standalone, copy-pasteable recipes for common EMHASS patterns. Each recipe follows a fixed template: Goal / Prerequisites / Config / Snippet / Caveats / Credits.

> If you need a longer narrative walkthrough, see [Study Cases](../study_cases/index.md). The Cookbook is the [Diátaxis](https://diataxis.fr/) **how-to-guide** quadrant — short, task-oriented, scannable.

## How to contribute

1. Copy `_template.md` to `<category>_<pattern>.md` (e.g. `ev_calendar_driven.md`).
2. Fill the 6 sections.
3. Link your file under the matching category below.
4. Open a PR. Contributor rules are inside the template.

## Recipes by category

### EV charging

- [EMHASS as planner, evcc as executor](ev_evcc_executor.md): EMHASS plans the EV charge as one deferrable in the whole-home optimization, and evcc executes it on the car (current modulation, live SOC, PV tracking). Works today on released evcc and EMHASS.

This is the opposite coupling to the evcc-as-frontend idea under discussion at [evcc-io/evcc#29815](https://github.com/evcc-io/evcc/discussions/29815) (evcc calling EMHASS as its backend optimizer). That recipe can land if and when #29815 resolves; the two coexist.

More HA-flavored EV recipes are welcome. See the [Discussion #824](https://github.com/davidusb-geek/emhass/discussions/824) thread for seed patterns (daily-commute, surplus-only, multi-day, calendar-driven, negative-price-aware, modulating-power).

### Domestic hot water (DHW)

No recipes yet. See `docs/study_cases/dhw_walkthrough.md` for the long-form walkthrough. Contributions welcome.

### Heat pump

No recipes yet. See `docs/study_cases/heat_pump_walkthrough.md` for the long-form walkthrough. Contributions welcome.

### Battery

- [Battery-aware runtime params](battery_aware_runtime_params.md) — feed live SOC into MPC; avoids the percent/fraction gotcha.

Additional battery recipes welcome (charging-from-grid strategies, calendar-aware reservation, etc.) — see [Discussion #823](https://github.com/davidusb-geek/emhass/discussions/823) for good-practices crowdsourcing.

### Forecast

No recipes yet. Topics that would fit: ML vs naive load forecaster selection, custom forecast injection via runtime params, dealing with forecast outages. Contributions welcome.

### Tariff

No recipes yet. Topics that would fit: dynamic-price (EPEX, Tibber, etc.) injection, multi-tier tariffs, sell-vs-self-consume thresholds. Contributions welcome.

### Transport / integration

- [MPC orchestration via Node-RED](transport_nodered_mpc_orchestration.md) — generic Node-RED → EMHASS pattern, transport-agnostic on inputs.

Additional transport recipes welcome: Home Assistant `rest_command` (HA users — see [Discussion #824](https://github.com/davidusb-geek/emhass/discussions/824) for community patterns), AppDaemon, EVCC API integration (pending #29815), other smart-home-native integrations, etc.
