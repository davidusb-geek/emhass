# Study Cases

End-to-end walkthroughs for common EMHASS configurations using real data.
Each page is tagged by Diátaxis mode so you can pick the entry point that
fits your need.

## Tutorials — start here if you're new

| Configuration | Page |
|---------------|------|
| No PV, two deferrable loads | [Basic — no PV](basic_no_pv.md) |
| PV + two deferrable loads | [Basic — PV](basic_pv.md) |
| PV + battery + deferrable loads | [Basic — PV + Battery](basic_pv_battery.md) |

## How-to guides — task-oriented

| Goal | Page |
|------|------|
| Run rolling-horizon control with naive-mpc-optim | [MPC walkthrough](mpc.md) |
| End-to-end heat-pump scenario combining PV/Batt/thermal_battery | [Heat-pump walkthrough](heat_pump_walkthrough.md) |
| EV charging as a deferrable load | [EV as deferrable](ev.md) |

## Reference

| Lookup | Page |
|--------|------|
| Config blueprints for common archetypes | [Reference Configurations](reference_configs.md) |
| Legacy CLI command equivalents | [Legacy CLI commands](legacy_cli.md) |

## Explanation

| Topic | Page |
|-------|------|
| Hard-learned wisdom: forecast quality, SOC semantics, infeasibility triage | [Good Practices](good_practices.md) |

```{toctree}
:maxdepth: 1
:hidden: true

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
