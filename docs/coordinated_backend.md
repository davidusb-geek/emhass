# Coordinated backend (opt-in)

By default EMHASS plans every device in one MILP (`optimization_backend: cvxpy`). The coordinated backend plans each device as its own subproblem instead, with EMHASS's own model of that device, and a coordinator makes the plans agree at the meter. The plan comes back in the same `opt_res` columns, so publishing and automations are unchanged.

It is useful when you want each device's share of the saving, or when the house has limits behind the meter - a sub-panel, a breaker, a shared cable - that the default MILP does not hold. When you need neither, the default MILP is faster.

## Enable it

It is part of EMHASS, with no extra to install. Set the backend:

```json
"optimization_backend": "dantzig_wolfe"
```

Every device (the battery, `deferrable0`, `deferrable1`, ...) is then planned on its own by EMHASS's model, on the main meter.

## The site

`site` describes the house in one list: the main meter, the nodes behind it, limits on sets of devices, and the devices - where each is wired, and which are planned together. Each element is an object with an `id`, and every one but the main meter and the limits names its `parent`: `grid`, the main meter, or a node (with one tariff, everything is wired behind the one meter). Its JSON schema is in `openapi.json`, under `Config`.

```json
"site": [
  {"id": "grid", "max_import": 9000, "max_export": 5000},
  {"id": "inverter", "parent": "grid", "type": "hybrid_inverter", "max_import": 4000, "max_export": 4000,
   "efficiency_import": 0.97, "efficiency_export": 0.97},
  {"id": "garage", "parent": "grid", "type": "panel", "max_import": 7400, "max_export": 0},
  {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500},
  {"id": "l1", "type": "limit", "max_import": 5000},
  {"id": "pv", "parent": "inverter"},
  {"id": "battery0", "parent": "inverter", "limits": ["l1"]},
  {"id": "deferrable0", "parent": "garage", "group": "loads"},
  {"id": "deferrable1", "parent": "garage", "group": "loads"},
  {"id": "deferrable2", "parent": "heat", "limits": ["l1"]}
]
```

| element | recognised by | what it holds |
| --- | --- | --- |
| the main meter | `id: "grid"` | `max_import`, `max_export` (W): they set `maximum_power_from_grid` / `maximum_power_to_grid` |
| a node | any other id | an inverter, a panel, a breaker or a meter behind the main meter: `parent` (another node, or `grid`), so nodes nest to any depth; `max_import` / `max_export` (W) on its connection to the parent (`max_export: 0`: no backfeed); `efficiency_import` / `efficiency_export` for a converter (1 by default); `type` (`hybrid_inverter`, `inverter`, `panel`, `breaker`, `meter`) |
| a limit | `type: "limit"` | `max_import` and/or `max_export` (W) on what the devices tagged with it draw together, wherever they are - a phase, a shared cable, a contract |
| a device | a device name: `pv`, `battery0` (or `battery`), `deferrable0`, ... | `parent`; `group`; `limits` (the limits it counts against) |


**Groups.** Each device is planned by EMHASS's own model, with only that device, or its group, enabled. Devices that share a `group` are planned together, so a `deferrable_load_groups` budget among them is held exactly; each device has its own share of the saving otherwise, the group one. A device `site` leaves out is planned on its own, on the main meter. The PV is a forecast: it has a `parent` and nothing else. The battery's indexed name, `battery0`, follows the multi-battery convention (#1032); with one battery its columns stay `P_batt` and `fed_share_battery`, as EMHASS's own do.

The coordinator sees a group only as its total, so a group's devices share one parent, and a limit holds all of a group's devices or none of them.

**EMHASS's own keys.** The main meter's limits, and a node of type `hybrid_inverter` on the main meter that holds the PV and the battery (and nothing else), set EMHASS's own keys - `maximum_power_from_grid` / `maximum_power_to_grid`, `inverter_is_hybrid`, `inverter_ac_output_max`, `inverter_ac_input_max` and both `inverter_efficiency_*` - so the default solver plans the same meter and inverter (`site` wins where they disagree, with a warning). Every other node and every limit is the coordinator's: the default solver does not hold them, and when the coordinator falls back the log says so. For deferrable loads alone, `deferrable_load_groups` is held either way. With no node in `site`, EMHASS's inverter keys describe a hybrid inverter as before.

**`deferrable_load_groups`**, EMHASS's own key for deferrable loads (`max_power`, `mutual_exclusion`), still works: when all of a group's loads are in one `group`, that group's own model holds it, exactly as the default solver would; otherwise the coordinator holds its `max_power` as a limit (`mutual_exclusion` across groups falls back).

**Prices.** The coordinator holds each node as a balance of its own, so the devices on it are priced at the node's own price (`fed_local_price_<id>`): the parent's price through the connection's efficiency while it has headroom, apart from it while it is at a rating - a hybrid inverter's falls to zero while its PV is being clipped. A limit adds its own premium while it binds (`fed_limit_price_<id>`).

**From a request.** `site` may come with a request, like any runtime parameter; it is checked by shape before anything reads it, and a request's `site` that cannot be used leaves the configured one in place.

## What falls back

Some options tie a device to PV or to another device, so they cannot be split per device yet. With any of them, EMHASS runs its default MILP and logs one line naming the option:

- `costfun: self-consumption`, `set_total_pv_sell`
- `set_nocharge_from_grid`, `set_battery_first_priority`, more than one battery
- a hybrid inverter (or the battery and the PV on one node) with `set_nodischarge_to_grid`, `inverter_stress_cost`, an inverter rated only by `pv_inverter_model` name, or the battery in a group with other devices
- `heat_topology`, shared thermal tanks
- `deferrable_load_groups` with `mutual_exclusion` across groups, a group on two parents, a limit that holds part of a group
- `cost_forecast_per_deferrable_load`, `set_deferrable_startup_penalty`, `deferrable_load_max_cost`
- capacity charges, and the runtime `soc_target`
- more than one inverter (`number_of_inverters`), and the runtime `battery_availability`, `battery_discharge_energy_max`, `def_availability` and per-plant PV split (`p_pv_plants`): the coordinator does not model them yet

`set_nodischarge_to_grid` is supported (without a hybrid inverter): the coordinator caps export at the PV surplus, as EMHASS does.

A hybrid inverter (`inverter_is_hybrid`) is supported: the coordinator holds the inverter itself, with the PV and the battery on its DC bus, its AC ratings (`inverter_ac_output_max`, `inverter_ac_input_max`) and its efficiencies each way. The battery is then priced at the DC bus's own price, which falls to zero while PV is being clipped at the rating, so the battery stores PV the inverter cannot pass. The plan carries `P_hybrid_inverter` (+ DC to AC), as EMHASS's does.

If the coordinator fails, EMHASS also falls back to the default MILP, so a plan is always published. The plan then says so: see `backend_used` below.

## Outputs

The usual `opt_res` columns, plus:

| column | meaning |
| --- | --- |
| `fed_meter_price` | the coordinator's price at the meter, per timestep (currency/kWh) |
| `fed_lower_bound` | a lower bound on the plan's cost |
| `fed_gap` | the plan's cost minus that bound |
| `fed_stop_reason`, `fed_iterations` | why the coordinator stopped (`converged`, `stalled`, `no new proposals`, `iteration cap`) and after how many rounds |
| `fed_share_<player>` | that player's share of the saving over the horizon (currency), the same in every row; one column for `solar`, one per device and one per group (its id) |
| `fed_local_price_<id>` | the price of one more kWh on a node of `site` (or the hybrid inverter EMHASS's keys describe, `inverter`), per timestep (currency/kWh) |
| `fed_node_power_<id>` | that node's power to its parent, per timestep (W, + = up the tree) |
| `fed_limit_price_<id>` | a limit's premium while it binds, per timestep (currency/kWh): a `site` limit's id, or a `deferrable_load_groups` group's loads (`deferrable0+deferrable1`) |

With EMHASS's devices answered as black boxes the bound is often loose, so a large `fed_gap` does not by itself mean a poor plan.

`optim_status` is `Optimal` only when the plan is proven within 0.1% of that bound; otherwise it is `Optimal_Inaccurate`: a runnable plan, published as usual, without the proof.

With a coordinator backend set, every plan, coordinated or not, also says which backend made it (a default configuration's plan has none of these columns):

| column | meaning |
| --- | --- |
| `backend_requested` | `optimization_backend` as configured |
| `backend_used` | the backend that made this plan: the one requested, or `cvxpy` after a fallback |
| `backend_fallback_reason` | why it fell back (the option, or the error); empty when it did not |

## Sharing the saving

Each participant is first reimbursed its change in private cost (comfort, stored energy, a missed goal) against its plan without coordination. The rest of the saving is split between solar and the participants by averaging two orders of arrival, solar first and participants first (an Owen value), and among the participants by pricing each kWh a participant shifts at the average tariff the house meets as they all move together (Aumann-Shapley). This needs one more plan, the same devices with no PV, so it roughly doubles the backend's solve time. If the split cannot be computed, the plan is still published, with a warning.

## Dry run

A coordinator outside EMHASS can ask EMHASS for its plan at trial prices without replacing or disturbing the live plan: send the action's body to `/api/v1/dry-run/<action>`. It works with either backend. See [Dry run](dry_run.md).

```bash
curl -X POST http://localhost:5000/api/v1/dry-run/naive-mpc-optim -H 'Content-Type: application/json' \
  -d '{"load_cost_forecast": [...], "prod_price_forecast": [...]}'
```

## Checking it against the default solver

`scripts/federated_benchmark.py` solves EMHASS's study cases and a set of synthetic days with both backends and prints each cost (EMHASS's own objective) side by side. The default MILP is solved exactly unless `--mip-gap` is given.

```bash
python scripts/federated_benchmark.py
```

## Future

**Remote solvers.** A part of the house planned by a solver elsewhere - another EMHASS, in its own container - as one element of `site`, through a standard query-and-answer interface. Planned for a follow-up.

**ADMM.** A second coordinator, ADMM, is planned but not offered yet: `optimization_backend` accepts only `cvxpy` and `dantzig_wolfe`, and `admm` set by hand logs that it is not available and runs the default MILP. It needs EMHASS's model to accept a pull towards a target plan (a linear, absolute-value term, so the problem stays a MILP HiGHS can solve), and it gives no bound on how far its plan is from the best, so it would report no `fed_gap`.

## How it works

Each participant answers one question: its cheapest plan when its energy is charged at a given price. The coordinator (Dantzig-Wolfe decomposition) asks every participant at the current meter price, combines the plans they offer in a small linear programme that holds the meter (tariff, grid limits), and updates the price until the plans stop improving. A participant whose devices are on/off is then fixed to one plan, and the others re-plan around it. The code is in `src/emhass/coordinator/`.
