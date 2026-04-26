# Study Cases

End-to-end walkthroughs for common EMHASS configurations using real data.
Each page is tagged by Diátaxis mode so you can pick the entry point that
fits your need.

## Tutorials — start here if you're new

| Configuration | Page |
|---------------|------|
| No PV, two deferrable loads | [basic_no_pv](basic_no_pv.md) |
| PV + two deferrable loads | [basic_pv](basic_pv.md) |
| PV + battery + deferrable loads | [basic_pv_battery](basic_pv_battery.md) |

## How-to guides — task-oriented

| Goal | Page |
|------|------|
| Run rolling-horizon control with naive-mpc-optim | [mpc](mpc.md) |
| End-to-end heat-pump scenario combining PV/Batt/thermal_battery | [heat_pump_walkthrough](heat_pump_walkthrough.md) |
| EV charging as a deferrable load | [ev](ev.md) |

## Reference

| Lookup | Page |
|--------|------|
| Config blueprints for common archetypes | [reference_configs](reference_configs.md) |
| Legacy CLI command equivalents | [legacy_cli](legacy_cli.md) |

## Explanation

| Topic | Page |
|-------|------|
| Hard-learned wisdom: forecast quality, SOC semantics, infeasibility triage | [good_practices](good_practices.md) |

```{toctree}
:maxdepth: 1
:hidden:

basic_no_pv
basic_pv
basic_pv_battery
mpc
heat_pump_walkthrough
ev
good_practices
reference_configs
legacy_cli
```
