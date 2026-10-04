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

## Shared limits

Devices that share a breaker or a sub-panel can be held to a limit on their total draw. There are two ways to say it:

- **`deferrable_load_groups`**, EMHASS's own key, for deferrable loads (`max_power` and `mutual_exclusion`, as without the coordinator). When all of a group's loads are in one participant group solved by EMHASS, that participant's own model holds it, exactly as the default solver would. Otherwise the coordinator holds its `max_power`; `mutual_exclusion` across participants falls back.
- **`group_limits`**, for any devices, including those EMHASS does not model (`water_heater`, `hvac`):

```json
"group_limits": [
  {"name": "garage", "devices": ["water_heater", "hvac"], "max_power": 3500, "min_power": 0}
]
```

`max_power` (W) caps the devices' total draw; `min_power` (W, at most 0) caps what they may push back to the house, so 0 means no backfeed. Each needs a `name` (lower case, digits and `_`), and a device is under one limit at most. The coordinator sees a participant group only as a whole, so an EMHASS participant group must be wholly inside a limit or wholly outside it. The default solver does not hold `group_limits`; when the coordinator falls back, the log says so. For deferrable loads alone, `deferrable_load_groups` is held either way. Shared limits need home-energy-optimizer 0.2.6 or later.

The coordinator holds each limit as a balance of its own, so the devices behind it are priced at the limit's own price (`fed_local_price_<name>`): the meter's price while the limit has headroom, more while it binds. A hybrid inverter is the same kind of limit, with losses and the PV on its bus (`fed_local_price_inverter`).

## What falls back

Some options tie a device to PV or to another device, so they cannot be split per device yet. With any of them, EMHASS runs its default MILP and logs one line naming the option:

- `costfun: self-consumption`, `set_total_pv_sell`
- `set_nocharge_from_grid`, `set_battery_first_priority`, more than one battery
- a hybrid inverter with `set_nodischarge_to_grid`, `inverter_stress_cost`, an inverter rated only by `pv_inverter_model` name, or the battery in a participant group with other devices
- `heat_topology`, shared thermal tanks
- `deferrable_load_groups` with `mutual_exclusion` across participants, and a limit that splits an EMHASS participant group
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
| `fed_local_price_<name>` | the price of one more kWh behind a limit the coordinator holds, per timestep (currency/kWh): `inverter` for a hybrid inverter, a `group_limits` name, or a `deferrable_load_groups` group's loads (`deferrable0+deferrable1`) |

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
