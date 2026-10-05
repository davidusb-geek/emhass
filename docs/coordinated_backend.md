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

Every device (`battery`, `deferrable0`, `deferrable1`, ...) is then its own participant, solved by EMHASS.

## Participants

A participant is one device, or a group of devices solved together by one solver. `participants` sets the groups, as a list of objects (its JSON schema is in `openapi.json`, under `Config`); a device it leaves out stays its own participant, solved by EMHASS.

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

## Electrical topology

Devices do not always reach the meter directly: the PV and a battery may sit on a hybrid inverter's DC bus, a heat pump and an EV charger behind one breaker, a sub-panel behind another. `electrical_topology` describes that tree, and the coordinator plans within every limit on it.

```json
"electrical_topology": {
  "nodes": [
    {"id": "inverter", "type": "hybrid_inverter", "max_import": 4000, "max_export": 4000,
     "efficiency_from_parent": 0.97, "efficiency_to_parent": 0.97},
    {"id": "garage", "type": "panel", "max_import": 7400, "max_export": 0},
    {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500}
  ],
  "devices": {"pv": "inverter", "battery": "inverter",
              "deferrable0": "garage", "deferrable1": "garage",
              "water_heater": "heat", "hvac": "heat"},
  "constraints": [
    {"name": "l1", "devices": ["battery", "hvac"], "max_import": 5000}
  ]
}
```

- **`nodes`** are connection points behind the main meter (`grid`): a hybrid inverter, a panel, a breaker, a meter. Each has a `parent` (another node, or `grid`, the default), so they nest to any depth. `max_import` and `max_export` (W) bound its connection to the parent, each way: `max_export: 0` means no backfeed. `efficiency_from_parent` and `efficiency_to_parent` make the connection a converter (1 by default: lossless). `type` names what it is.
- **`devices`** puts each device (`battery`, `water_heater`, `hvac`, `deferrableN`) and the PV (`pv`) on a node, or on `grid`; anything not listed is on `grid`.
- **`constraints`** limit what a set of devices draw together, wherever they are in the tree - a phase, a shared cable, a contract: `max_import` and/or `max_export` (W).

The coordinator sees an EMHASS participant group only as its total, so a group's devices sit on one node, and a constraint holds all of a group's devices or none of them.

A node of type `hybrid_inverter` on the main meter that holds the PV and the battery (and nothing else) is EMHASS's own inverter too: its ratings and efficiencies fill `inverter_is_hybrid`, `inverter_ac_output_max`, `inverter_ac_input_max` and both `inverter_efficiency_*` keys (the topology wins where they disagree, with a warning), so the default solver plans the same inverter. Every other node and every constraint is the coordinator's: the default solver does not hold them, and when the coordinator falls back the log says so. For deferrable loads alone, `deferrable_load_groups` is held either way.

**`deferrable_load_groups`**, EMHASS's own key for deferrable loads (`max_power`, `mutual_exclusion`), still works: when all of a group's loads are in one participant group solved by EMHASS, that participant's own model holds it, exactly as the default solver would; otherwise the coordinator holds its `max_power` as a constraint (`mutual_exclusion` across participants falls back).

Without `electrical_topology`, EMHASS's inverter keys describe a hybrid inverter as before.

**Prices.** The coordinator holds each node as a balance of its own, so the devices on it are priced at the node's own price (`fed_local_price_<id>`): the parent's price through the connection's efficiency while it has headroom, apart from it while it is at a rating - a hybrid inverter's falls to zero while its PV is being clipped. A constraint adds its own premium while it binds (`fed_limit_price_<name>`). The topology needs home-energy-optimizer 0.2.7 or later.

## What falls back

Some options tie a device to PV or to another device, so they cannot be split per device yet. With any of them, EMHASS runs its default MILP and logs one line naming the option:

- `costfun: self-consumption`, `set_total_pv_sell`
- `set_nocharge_from_grid`, `set_battery_first_priority`, more than one battery
- a hybrid inverter (or the battery and the PV on one node) with `set_nodischarge_to_grid`, `inverter_stress_cost`, an inverter rated only by `pv_inverter_model` name, or the battery in a participant group with other devices
- `heat_topology`, shared thermal tanks
- `deferrable_load_groups` with `mutual_exclusion` across participants, and a node or constraint that splits an EMHASS participant group
- `cost_forecast_per_deferrable_load`, `set_deferrable_startup_penalty`, `deferrable_load_max_cost`
- capacity charges, and the runtime `soc_target`

`set_nodischarge_to_grid` is supported (without a hybrid inverter): the coordinator caps export at the PV surplus, as EMHASS does.

A hybrid inverter (`inverter_is_hybrid`) is supported: the coordinator holds the inverter itself, with the PV and the battery on its DC bus, its AC ratings (`inverter_ac_output_max`, `inverter_ac_input_max`) and its efficiencies each way. The battery is then priced at the DC bus's own price, which falls to zero while PV is being clipped at the rating, so the battery stores PV the inverter cannot pass. The plan carries `P_hybrid_inverter` (+ DC to AC), as EMHASS's does. Needs home-energy-optimizer 0.2.5 or later.

If the coordinator fails, or the package is not installed, EMHASS also falls back to the default MILP, so a plan is always published. The plan then says so: see `backend_used` below.

## Outputs

The usual `opt_res` columns, plus:

| column | meaning |
| --- | --- |
| `fed_meter_price` | the coordinator's price at the meter, per timestep (currency/kWh) |
| `fed_lower_bound` | a lower bound on the plan's cost |
| `fed_gap` | the plan's cost minus that bound |
| `fed_stop_reason`, `fed_iterations` | why the coordinator stopped (`converged`, `stalled`, `no new proposals`, `iteration cap`) and after how many rounds |
| `fed_share_<player>` | that player's share of the saving over the horizon (currency), the same in every row; one column for `solar` and one per participant |
| `fed_local_price_<id>` | the price of one more kWh on a node of the topology (or the hybrid inverter EMHASS's keys describe, `inverter`), per timestep (currency/kWh) |
| `fed_node_power_<id>` | that node's power to its parent, per timestep (W, + = up the tree) |
| `fed_limit_price_<name>` | a constraint's premium while it binds, per timestep (currency/kWh): a topology constraint's name, or a `deferrable_load_groups` group's loads (`deferrable0+deferrable1`) |

With EMHASS's devices answered as black boxes the bound is often loose, so a large `fed_gap` does not by itself mean a poor plan.

`optim_status` is `Optimal` only when the plan is proven within 0.1% of that bound; otherwise it is `Optimal_Inaccurate`: a runnable plan, published as usual, without the proof.

With a coordinator backend set, every plan, coordinated or not, also says which backend made it (a default configuration's plan has none of these columns):

| column | meaning |
| --- | --- |
| `backend_requested` | `optimization_backend` as configured |
| `backend_used` | the backend that made this plan: the one requested, or `cvxpy` after a fallback |
| `backend_fallback_reason` | why it fell back (the option, the error, or the missing package); empty when it did not |

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

## Future: ADMM

A second coordinator, ADMM, is planned but not offered yet: `optimization_backend` accepts only `cvxpy` and `dantzig_wolfe`, and `admm` set by hand logs that it is not available and runs the default MILP. It needs EMHASS's model to accept a pull towards a target plan (a linear, absolute-value term, so the problem stays a MILP HiGHS can solve), and it gives no bound on how far its plan is from the best, so it would report no `fed_gap`.

## How it works

Each participant answers one question: its cheapest plan when its energy is charged at a given price. The coordinator (Dantzig-Wolfe decomposition) asks every participant at the current meter price, combines the plans they offer in a small linear programme that holds the meter (tariff, grid limits), and updates the price until the plans stop improving. A participant whose devices are on/off is then fixed to one plan, and the others re-plan around it. The interface and the coordinator live in [home-energy-optimizer](https://github.com/ameetdesh/home-energy-optimizer).
