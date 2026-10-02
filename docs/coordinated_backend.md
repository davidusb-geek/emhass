# Coordinated backend (opt-in)

By default EMHASS plans every device in one MILP (`optimization_backend: cvxpy`). The coordinated backend plans each device as its own subproblem instead, with EMHASS's own model of that device, and a coordinator makes the plans agree at the meter. The plan comes back in the same `opt_res` columns, so publishing and automations are unchanged.

It is useful when devices should stay separate: so that each device's share of the saving can be settled, or so that one device can be planned by a different solver than the rest. On a home whose devices are all modelled in EMHASS, the default MILP is faster.

## Enable it

Install the optional extra, then set the backend:

```bash
pip install emhass[federated]
```

```json
"optimization_backend": "dantzig_wolfe"
```

Every device (`battery`, `deferrable0`, `deferrable1`, ...) is then its own participant, solved by EMHASS. `admm` is reserved for a later release.

## Participants

A participant is one device, or a group of devices solved together by one solver. `participants` sets the groups; a device it leaves out stays its own participant, solved by EMHASS.

```json
"participants": [
  {"devices": ["battery"], "solver": "home_energy_optimizer"},
  {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"}
]
```

| solver | what plans the group |
| --- | --- |
| `emhass` | EMHASS's own model, with only the group's devices enabled |
| `home_energy_optimizer` | the package's solver for that device type (`battery`; also `water_heater` and `hvac`, given a `config`) |

A battery planned by `home_energy_optimizer` values the energy it leaves at the end at the average import price, instead of being held to `soc_final`.

## What falls back

Some options tie a device to PV or to another device, so they cannot be split per device yet. With any of them, EMHASS runs its default MILP and logs one line naming the option:

- `costfun: self-consumption`, `set_total_pv_sell`
- `set_nocharge_from_grid`, `set_battery_first_priority`, a hybrid inverter, more than one battery
- `heat_topology`, shared thermal tanks, `deferrable_load_groups`
- `cost_forecast_per_deferrable_load`, `set_deferrable_startup_penalty`, `deferrable_load_max_cost`
- capacity charges, and the runtime `soc_target`

`set_nodischarge_to_grid` is supported: the coordinator caps export at the PV surplus, as EMHASS does.

If the coordinator fails, or the package is not installed, EMHASS also falls back to the default MILP, so a plan is always published.

## Outputs

The usual `opt_res` columns, plus:

| column | meaning |
| --- | --- |
| `fed_meter_price` | the coordinator's price at the meter, per timestep (currency/kWh) |
| `fed_lower_bound` | a lower bound on the plan's cost |
| `fed_gap` | the plan's cost minus that bound |
| `fed_share_<player>` | that player's share of the saving over the horizon (currency), the same in every row; one column for `solar` and one per participant |

With EMHASS's devices answered as black boxes the bound is often loose, so a large `fed_gap` does not by itself mean a poor plan.

## Sharing the saving

Each participant is first reimbursed its change in private cost (comfort, stored energy, a missed goal) against its plan without coordination. The rest of the saving is split between solar and the participants by averaging two orders of arrival, solar first and participants first (an Owen value), and among the participants by pricing each kWh a participant shifts at the average tariff the house meets as they all move together (Aumann-Shapley). This needs one more plan, the same devices with no PV, so it roughly doubles the backend's solve time. If the split cannot be computed, the plan is still published, with a warning.

## Dry run

Any optimisation action accepts the runtime parameter `dry_run`. With it set, EMHASS solves as usual, returns the plan in the HTTP response (the same records `/api/v1/plan` serves), and changes nothing the next live run sees:

- no files: no results CSV (`opt_res_latest.csv` or the dated one), no entity files for continual publish, no plan store, no last-run record, no web-page plot, no publish;
- no in-memory state: a dry run builds its own problem and never reads or writes the optimisation cache, so the next live solve warm-starts from the last live solution, with the live configuration, as if the dry run had not happened.

The only cost is that a dry run itself starts cold. EMHASS's action log still records it. A coordinator outside EMHASS can then ask EMHASS for its plan at trial prices without replacing or disturbing the live plan. It works with either backend.

```bash
curl -X POST http://localhost:5000/action/naive-mpc-optim -H 'Content-Type: application/json' \
  -d '{"load_cost_forecast": [...], "prod_price_forecast": [...], "dry_run": true}'
```

```json
{"dry_run": true, "plan": [{"timestamp": "...", "P_grid": 512.0, "P_batt": -300.0, ...}, ...]}
```

## Checking it against the default solver

`scripts/federated_benchmark.py` solves EMHASS's study cases and a set of synthetic days with both backends and prints each cost (EMHASS's own objective) side by side. The default MILP is solved exactly unless `--mip-gap` is given.

```bash
python scripts/federated_benchmark.py
```

## How it works

Each participant answers one question: its cheapest plan when its energy is charged at a given price. The coordinator (Dantzig-Wolfe decomposition) asks every participant at the current meter price, combines the plans they offer in a small linear programme that holds the meter (tariff, grid limits), and updates the price until the plans stop improving. A participant whose devices are on/off is then fixed to one plan, and the others re-plan around it. The interface and the coordinator live in [home-energy-optimizer](https://github.com/ameetdesh/home-energy-optimizer).
