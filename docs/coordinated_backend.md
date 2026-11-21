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
| a remote solver | `solver: {"url": ..., "token_secret": ...}` | another EMHASS, or any solver serving the participant API, as one participant: `parent`; `limits` (see "Remote solvers") |


**Groups.** Each device is planned by EMHASS's own model, with only that device, or its group, enabled. Devices that share a `group` are planned together, so a `deferrable_load_groups` budget among them is held exactly; each device has its own share of the saving otherwise, the group one. A device `site` leaves out is planned on its own, on the main meter. The PV is a forecast: it has a `parent` and nothing else. The battery's indexed name, `battery0`, follows the multi-battery convention (#1032); with one battery its columns stay `P_batt` and `fed_share_battery`, as EMHASS's own do.

The coordinator sees a group only as its total, so a group's devices share one parent, and a limit holds all of a group's devices or none of them.

**EMHASS's own keys.** The main meter's limits, and a node of type `hybrid_inverter` on the main meter that holds the PV and the battery (and nothing else), set EMHASS's own keys - `maximum_power_from_grid` / `maximum_power_to_grid`, `inverter_is_hybrid`, `inverter_ac_output_max`, `inverter_ac_input_max` and both `inverter_efficiency_*` - so the default solver plans the same meter and inverter (`site` wins where they disagree, with a warning). Every other node and every limit is the coordinator's: the default solver does not hold them, and when the coordinator falls back the log says so. For deferrable loads alone, `deferrable_load_groups` is held either way. With no node in `site`, EMHASS's inverter keys describe a hybrid inverter as before.

**`deferrable_load_groups`**, EMHASS's own key for deferrable loads (`max_power`, `mutual_exclusion`), still works: when all of a group's loads are in one `group`, that group's own model holds it, exactly as the default solver would; otherwise the coordinator holds its `max_power` as a limit (`mutual_exclusion` across groups falls back).

**Prices.** The coordinator holds each node as a balance of its own, so the devices on it are priced at the node's own price (`fed_local_price_<id>`): the parent's price through the connection's efficiency while it has headroom, apart from it while it is at a rating - a hybrid inverter's falls to zero while its PV is being clipped. A limit adds its own premium while it binds (`fed_limit_price_<id>`).

**From a request.** `site` may come with a request, like any runtime parameter; it is checked by shape before anything reads it, and a request's `site` that cannot be used leaves the configured one in place.

## Remote solvers

A part of the house can be planned by a solver elsewhere - another EMHASS, in its own container, planning the garage's EV and battery - as one element of `site`:

```json
{"id": "garage_emhass", "parent": "garage",
 "solver": {"url": "http://emhass-garage:5000/participant", "token_secret": "garage_token",
            "trusted_network": true}}
```

The coordinator asks it what it would do at given prices, and, once it has chosen, asks it to run its part (the participant API: [participant-api.v1.json](https://github.com/davidusb-geek/emhass/blob/master/src/emhass/static/participant-api/participant-api.v1.json), with the query and answer schemas beside it). The API is versioned: its major version is in every path (`/v1/`) and body, and within it the API only grows. A participant lists what it serves beyond the first draft in its description's `features`, and a coordinator uses a new feature only with a participant that lists it, so an older and a newer EMHASS still plan together. It is never sent the rest of the house's load: it is asked at its connection's marginal prices - what drawing costs and supplying is worth there - which carry the tariff and every rating. `token_secret` names the secret in `secrets_emhass.yaml` that holds the token it expects. It must be a secret of its own: one of EMHASS's own secrets (`long_lived_token`, `participant_token`, an API key, ...) is refused, so no saved `site` sends one of them to a remote's URL. A remote solver is accepted from the configuration only: a `site` sent with a request may not name one. If it does not answer, the house is planned without it and the plan says so (`fed_remote_status_<id>`). The coordinator asks the remotes to run their parts only once the whole plan is made, so nothing that follows can send this EMHASS to its default solver after a remote has committed; A remote answers a commit at once and runs it after; the coordinator then asks for its outcome until it is known, or for 30 s. A commit the remote refuses (nothing was published) is that remote's refusal. A commit that got no answer is asked about, and sent again only if the remote never had it: a remote never runs one commit twice. If the outcome is still not known, or the remote published part of the plan before failing, the remote is recorded as `uncertain` and its part kept in the plan, since it may be running. A remote that refuses is then planned without, as one that did not answer: the house is planned again, the remotes that committed held to the plans they took, and the remotes still to ask are asked for their part of the new plan. A token is sent as a bearer token, which anyone on the network between could read over plain `http://`. So a remote on another machine with a `token_secret` must be `https://`, or its `solver` must say the network between is trusted (`"trusted_network": true`, for example two EMHASS on port 5000 of a home LAN); otherwise it is not called, and the plan says why (`declined: ...`). `https://` is recommended.

**Serving this EMHASS** to a coordinator elsewhere: set `participant_api` and a `participant_token` secret.

```json
"participant_api": {"publish_prefix": "garage_"}
```

It then answers at `/participant/v1/describe`, `baseline`, `query` and `commit`, and `commit/<id>` for a commit's outcome, every call presenting `participant_token` as a bearer token - that token only: these routes do not ask for the server's own API token (`emhass_api_token`), so a coordinator never holds it, but Home Assistant's ingress port refuses them, as anything but the Supervisor's requests -. `participant_token` must be at least as long as the API token's minimum (16 characters), or the API answers 503. It is checked before the configuration or any body is read, so a caller without it learns nothing of the configuration (a body over 1 MiB is refused, 413, as is one that takes over 30 s to arrive, 408) (off, 404, without `participant_api`; 503 without the token, or with more than one web worker, `WEB_CONCURRENCY`, since a coordinator's answers live in one worker's memory). Every horizon it is asked about names its first slot (`start`), which must be the slot it plans now. Its answers come from its own model: its battery, starting from its `sensor_battery_state_of_charge`, and its deferrable loads, planned together within its own `maximum_power_from_grid` / `maximum_power_to_grid` - its connection's limits, a panel's rating, no backfeed. Queries have no effect. A commit is answered at once (202) and then publishes the chosen plan as a run would, holding the live-optimization slot until it has; its record (running, accepted, refused, or partial when publishing failed part way) is kept 48 hours in `participant_commits.json` in the data folder, across a restart, so a coordinator can ask for it again and a commit sent twice runs once. It publishes its sensors prefixed by `publish_prefix` so two EMHASS on one Home Assistant do not collide. While it is coordinated it should not run its own optimizations. If it does, a commit that arrives during one of them is refused (429, as a second live optimization is), and the coordinator records it as refused. A commit is also refused while `continual_publish` is on, since its loop would publish an earlier plan back. An EMHASS with a hybrid inverter is not served (its answers would not hold the inverter's AC limits): the coordinator records it as declined.

Each call to a remote has 30 s in all, its answer at most 1 MiB, and a redirect is not followed (it would carry the token elsewhere); an answer that is not a finite plan leaves that remote out of the plan. An EMHASS serving its devices declines (`declined: ...`) what the coordinator would decline in its own house - a non-electric load, a final-SoC reward and the rest of "What falls back" - since its answers would not be what the coordinator can coordinate.

Any solver that serves the participant API can be a remote, not only EMHASS: the API is the standard interface between optimizers, versioned (`version` in every body) and documented by those schemas.

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
- per-slot grid limits (`maximum_power_from_grid` / `maximum_power_to_grid` as a list), a thermal load with `internal_gains_factor` (its heat demand follows the house's load), and a further battery named in `site` (`battery1`, ...)
- a non-electric load (`is_electric_load: false`) and `battery_soc_final_reward_factor` (EMHASS prices each at the import tariff, which the coordinator varies), and a thermal load with `thermal_inertia_time_constant` but no `q_input_initial` (its heat input is carried from one run to the next)

The coordinator prices the main meter's limits and every node and limit of `site`, and does not publish a plan past any of them: when its on/off plans cannot fit, it falls back, and `backend_fallback_reason` gives the energy by which its plan exceeded them, summed over the main meter, every node and limit and every slot (up to 0.001 kWh in all is tolerated, as solver rounding). The default solver then holds only what EMHASS's own keys describe - the main meter's limits and a hybrid inverter on it - not `site`'s other nodes and limits; the log says so when it plans for a `site` that has them. A plan that must keep a panel or breaker rating is therefore the coordinator's, when it can find one.

The battery is held in the coordinator's own LP when its model is just its limits and efficiencies. With a penalty on its state of charge, a start outside its SoC window, or a negative import price (where only EMHASS's charge-or-discharge binary stops it charging and discharging at once), EMHASS's own model plans it instead, as a participant.

A load that is on/off at a level - semi-continuous, one block, a power sequence (a list in `nominal_power_of_deferrable_loads`), a minimum power or an overshoot temperature - is planned as one of its own plans, never a mix of them.

Each coordinator run stops iterating after EMHASS's `lp_solver_timeout` and keeps its best plan so far (`fed_stop_reason: time limit`). `site` has at most 64 elements and 16 nodes and limits together, which bounds what a `site` sent with a request can cost.

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
| `fed_remote_power_<id>` | a remote solver's planned power, per timestep (W, + = drawn) |
| `fed_remote_status_<id>` | what became of it: `committed`, `uncertain: ...` (its commit's outcome is not known, or it published part of the plan: it may be running, so its part stays in the plan), `refused: ...`, `unreachable`, `unauthorized` or `declined: ...` (planned without it; `declined` when it cannot plan this horizon, such as one starting on another slot), `planned (dry run)` |
| `fed_remote_last_commit_<id>` | how the remote's last commit ended, as it records it: `accepted`, `refused: ...`, `partial: ...` or `running` (the commit of an earlier run, unless this run's has just been made) |

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

A coordinator outside EMHASS can ask EMHASS for its plan at trial prices without replacing or disturbing the live plan: send the action's body to `/api/v1/dry-run/<action>`. It works with either backend, and a dry run never asks a remote solver to run its part. See [Dry run](dry_run.md).

```bash
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -X POST http://localhost:5000/api/v1/dry-run/naive-mpc-optim -H 'Content-Type: application/json' \
  -d '{"load_cost_forecast": [...], "prod_price_forecast": [...]}'
```

## Checking it against the default solver

`scripts/federated_benchmark.py` solves EMHASS's study cases and a set of synthetic days with both backends and prints each cost (EMHASS's own objective) side by side. The default MILP is solved exactly unless `--mip-gap` is given.

```bash
python scripts/federated_benchmark.py
```

## Future

**ADMM.** A second coordinator, ADMM, is planned but not offered yet: `optimization_backend` accepts only `cvxpy` and `dantzig_wolfe`, and `admm` set by hand logs that it is not available and runs the default MILP. It needs EMHASS's model to accept a pull towards a target plan (a linear, absolute-value term, so the problem stays a MILP HiGHS can solve), and it gives no bound on how far its plan is from the best, so it would report no `fed_gap`.

## How it works

Each participant answers one question: its cheapest plan when its energy is charged at a given price. The coordinator (Dantzig-Wolfe decomposition) asks every participant at the current meter price, combines the plans they offer in a small linear programme that holds the meter (tariff, grid limits), and updates the price until the plans stop improving. A participant whose devices are on/off is then fixed to one plan, and the others re-plan around it. The code is in `src/emhass/coordinator/`.
