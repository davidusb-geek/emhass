"""EMHASS's devices as participants of the coordinator.

EMHASS calls `optimize` from `Optimization.perform_optimization` when its
`optimization_backend` is not "cvxpy". Each participant - by default one per
device, or one per `group` in `site` - is answered by EMHASS's own model with
only its devices enabled (`EmhassParticipant`): its price response is
EMHASS's own problem with no PV and no load, priced at the coordinator's
price. The plan comes back as EMHASS's own `opt_res` columns, plus `fed_*`
columns, including each player's share of the saving (`fed_share_*`).

`optimize` returns None, after logging why, whenever the configuration uses
something it cannot split per device yet; EMHASS then runs its default solver.

EMHASS works in W, the coordinator in kW; EMHASS's P_batt is + when
discharging, the interface's plans are + when drawn from the meter.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

from emhass.coordinator.interface import Answer, Query
from emhass.coordinator.master import Column, DWCoordinator, DWResult, RunOptions
from emhass.coordinator.model import (
    MAIN,
    BatteryConfig,
    Forecasts,
    GridLimits,
    Horizon,
    SetLimit,
    SiteConfig,
    SubMeter,
    hybrid_inverter,
)
from emhass.coordinator.remote import (
    PlainHttpRefused,
    PlanWindow,
    RemoteError,
    RemoteParticipant,
)
from emhass.coordinator.settlement import ledger
from emhass.coordinator.site import _SITE_DEVICE, site_kind
from emhass.coordinator.submeter import site_meter

if TYPE_CHECKING:
    import pandas as pd


class EmhassOptimization(Protocol):
    """What this module uses of EMHASS's `Optimization` (emhass.optimization):
    the one `optimize` is handed, and the one each participant builds for its
    own devices - one list of what a change to Optimization must keep."""

    optim_conf: dict[str, Any]
    plant_conf: dict[str, Any]
    retrieve_hass_conf: dict[str, Any]
    emhass_conf: dict[str, Any]
    var_load_cost: str
    var_prod_price: str
    costfun: str  # "profit", "cost" or "self-consumption"
    time_step: float  # hours
    logger: logging.Logger
    optim_status: str
    # Set by this adapter on every call: why it declined, or None when it
    # planned. EMHASS reads it to say which solver actually made the plan.
    fed_fallback_reason: str | None
    prob: Any  # the solved problem; its .value is the objective
    _persist_q_input: Callable[..., None]

    def perform_optimization(
        self,
        data_opt: pd.DataFrame,
        p_pv: np.ndarray,
        p_load: np.ndarray,
        unit_load_cost: np.ndarray,
        unit_prod_price: np.ndarray,
        **kwargs: Any,
    ) -> pd.DataFrame: ...

    def _prepare_power_limit_array(self, value: Any, name: str, n: int) -> np.ndarray: ...


# DWCoordinator.run settings. A participant answered by a black-box model gives
# a weak lower bound, so the gap test alone would run to max_iter: stop once the
# master's value has stopped falling, then fix the on/off participants to one
# plan each and let the rest re-plan around them (the dive).
RUN_DEFAULTS: RunOptions = {"stall": 2, "smoothing": 0.0, "dive": 6}
# The statuses EMHASS publishes a plan for (perform_optimization)
OK_STATUSES = ("Optimal", "Optimal_Inaccurate", "Optimal (Relaxed)", "Optimal (Incumbent)")
# A plan is reported "Optimal" only when it is proven within this relative
# distance of the best possible (DWCoordinator.run's own gap_tol); otherwise
# "Optimal_Inaccurate": a runnable plan, published as usual (both are in
# EMHASS's OK statuses), but not a certified optimum - the coordinator stopped
# on a stall, its iteration cap, or no new proposals with the bound still open.
GAP_TOL = 1e-3
# Seconds a remote solver has for each call before its last plan stands in
REMOTE_TIMEOUT_S = 30.0
# How long a remote's commit outcome is asked for before it counts as not known
REMOTE_COMMIT_WAIT_S = 30.0


def plan_status(upper: float, lower: float) -> str:
    """EMHASS's optim_status for a coordinated plan with objective `upper`
    and lower bound `lower`."""
    return "Optimal" if upper - lower <= GAP_TOL * max(1.0, abs(upper)) else "Optimal_Inaccurate"


def _decline(opt: EmhassOptimization, reason: str) -> None:
    """Log why the coordinator will not plan and record it on `opt` for EMHASS;
    EMHASS's default solver then plans."""
    opt.logger.warning(f"optimization_backend: {reason}; using the default solver")
    opt.fed_fallback_reason = reason


# Runtime per-load lists perform_optimization takes; cut to a participant's loads.
RUNTIME_LOAD_LISTS = (
    "def_total_hours",
    "def_total_timestep",
    "def_start_timestep",
    "def_end_timestep",
    "def_init_temp",
    "min_power_of_deferrable_loads",
)
# Inputs a participant cannot split per device yet (perform_optimization arguments).
RUNTIME_UNSUPPORTED = (
    "soc_target",
    "soc_target_timestep",
    "current_period_peak",
    "capacity_charge_window",
    "capacity_charge_consideration",
    "capacity_charge_current_interval_history",
)
# Arguments that change nothing here. Any other argument that is set means a
# feature this adapter does not know, so EMHASS's own solver runs.
RUNTIME_IGNORED = ("debug", "stage_times")


def _per_load_keys() -> set[str]:
    """The optim_conf keys that hold one entry per deferrable load, so are cut
    to a participant's loads (EMHASS's own list, plus a few it handles apart)."""
    from emhass.utils import DEF_LOAD_ARRAY_PARAMS

    return set(DEF_LOAD_ARRAY_PARAMS) | {
        "minimum_power_of_deferrable_loads",
        "cost_forecast_per_deferrable_load",
        "is_electric_load",
        "def_load_config",
    }


def unsupported(
    optim_conf: dict[str, Any], plant_conf: dict[str, Any], costfun: str, runtime: dict[str, Any]
) -> str | None:
    """Why this configuration cannot be split per device yet, or None if it can.

    `optim_conf`, `plant_conf`: EMHASS's configuration dicts. `costfun`: EMHASS's
    cost function ("profit", "cost" or "self-consumption"). `runtime`: the
    remaining keyword arguments of perform_optimization. Returns a short reason
    for the log (the option that blocks it), or None.
    """
    oc, pc = optim_conf, plant_conf
    if costfun not in ("profit", "cost"):
        return f"costfun {costfun!r} (supported: profit, cost)"
    if oc.get("set_total_pv_sell"):
        return "set_total_pv_sell"
    for old in OLD_KEYS:
        if oc.get(old):
            # keys of an earlier draft of this backend; planning without what they
            # said would be worse than the default solver
            return f"{old} (replaced by site: one list of the meter, nodes, limits and devices)"
    elements = [e for e in oc.get("site") or [] if isinstance(e, dict)]
    # battery0 is the battery's indexed name (#1032); with one battery, EMHASS's P_batt
    homes = {
        ("battery" if e.get("id") == "battery0" else str(e.get("id"))): str(e.get("parent", MAIN))
        for e in elements
        if _kind(e) == "device"
    }
    if "battery" in homes and not oc.get("set_use_battery"):
        # EMHASS has no battery, so no state of charge or plant_conf to plan one from
        return "site lists the battery, but set_use_battery is off"
    if oc.get("set_use_battery") and oc.get("set_nocharge_from_grid"):
        return "set_nocharge_from_grid (ties the battery to PV)"
    if oc.get("set_battery_first_priority"):
        return "set_battery_first_priority"
    if pc.get("inverter_is_hybrid"):
        # The PV and the battery on the inverter's DC bus are a sub-meter
        # (types.hybrid_inverter), which the coordinator models exactly -
        # except EMHASS's options that tie it to the meter or price it
        if oc.get("set_nodischarge_to_grid"):
            return "set_nodischarge_to_grid with a hybrid inverter (no discharge while exporting)"
        if float(pc.get("inverter_stress_cost", 0) or 0) > 0:
            return "inverter_stress_cost"
        if (
            not any(_kind(e) == "node" for e in elements)
            and pc.get("inverter_ac_output_max") is None
            and not isinstance(pc.get("pv_inverter_model"), (int, float))
        ):
            return "a hybrid inverter rated by pv_inverter_model (set inverter_ac_output_max)"
    if (
        oc.get("set_nodischarge_to_grid")
        and homes.get("pv", MAIN) != MAIN
        and homes.get("pv") == homes.get("battery")
    ):
        # EMHASS ties a DC-coupled battery to the meter's direction then
        return "set_nodischarge_to_grid with the battery and the PV on one node (no discharge while exporting)"
    if int(pc.get("number_of_batteries", 1)) > 1:
        return "more than one battery"
    extra = sorted(d for d in homes if d.startswith("battery") and d != "battery")
    if extra:
        return f"site names {extra[0]!r}; EMHASS plans one battery here (battery0)"
    for key in ("maximum_power_from_grid", "maximum_power_to_grid"):
        if isinstance(pc.get(key), list | tuple):
            return f"{key} as a list (per-slot grid limits)"
    for cfg in oc.get("def_load_config") or []:
        thermal = (cfg or {}).get("thermal_battery") or {} if isinstance(cfg, dict) else {}
        if float(thermal.get("internal_gains_factor", 0) or 0) > 0:
            # its heat demand reads the house's load, which a price response sets to zero
            return "internal_gains_factor (a thermal load's gains follow the house's load)"
        if (
            float(thermal.get("thermal_inertia_time_constant", 0) or 0) > 0
            and "q_input_initial" not in thermal
        ):
            # the default solver carries its heat input from one run to the next
            # (a cached problem); a participant starts each run afresh
            return "thermal_inertia_time_constant without q_input_initial (heat input carried between runs)"
    if any(v is False for v in oc.get("is_electric_load") or []):
        # its commodity is priced at the queried price in EMHASS's model: a
        # query would change its own cost, not only the electricity's
        return "a non-electric load (is_electric_load: false)"
    if oc.get("set_use_battery") and float(oc.get("battery_soc_final_reward_factor", 0) or 0) > 0:
        # EMHASS prices the reward at the last import price: a query would
        # change the battery's own objective, not only its bill
        return "battery_soc_final_reward_factor (the reward follows the import price)"
    if oc.get("heat_topology") or oc.get("shared_thermal_tanks"):
        return "heat_topology or shared thermal tanks"
    if any(v is not None for v in (oc.get("cost_forecast_per_deferrable_load") or [])):
        return "cost_forecast_per_deferrable_load"
    if any(float(v or 0) > 0 for v in (oc.get("set_deferrable_startup_penalty") or [])):
        return "set_deferrable_startup_penalty (priced at the import tariff)"
    if any(float(v or 0) > 0 for v in (oc.get("deferrable_load_max_cost") or [])):
        return "deferrable_load_max_cost"
    cap = oc.get("capacity_cost_per_kw") or 0.0
    if any(float(c or 0) > 0 for c in cap) if isinstance(cap, list) else float(cap) > 0:
        return "capacity charges"
    for name in RUNTIME_UNSUPPORTED:
        if runtime.get(name) is not None:
            return name
    known = set(RUNTIME_LOAD_LISTS) | set(RUNTIME_UNSUPPORTED) | set(RUNTIME_IGNORED)
    for name, value in runtime.items():
        if name not in known and value is not None:
            return f"the argument {name!r}"
    return None


def _hold_q_input(k: int, params: dict[str, Any], hc: dict[str, Any]) -> None:
    """Stand-in for Optimization._persist_q_input in a participant's model: keep
    the heat input a thermal battery starts from (its configured
    `q_input_initial`, else the value it was built with) on every re-solve.
    `k`: the load's index; `params`: its thermal parameters; `hc`: its
    thermal_battery config."""
    if "q_input_initial" in hc:
        params["q_input_start"].value = float(hc.get("q_input_initial", 0.0) or 0.0)


# ------------------------------------------------------------------ participants
class EmhassParticipant:
    """One participant answered by EMHASS's own model, restricted to its devices.

    The model is built once and re-solved at each query's prices: EMHASS holds
    prices, PV and load in CVXPY parameters, so a re-solve does not rebuild it.

    `opt`: the calling EMHASS Optimization (its configuration is copied, then
    restricted). `key`: the participant's name in the coordinator. `battery`:
    whether the participant includes EMHASS's battery. `loads`: indices of its
    deferrable loads in EMHASS's configuration. `data_opt`: EMHASS's input
    DataFrame (its index sets the horizon). `soc_init`, `soc_final`: the
    battery's start and end-of-day state of charge (fractions, 0-1). `runtime`:
    perform_optimization's other keyword arguments; the per-load lists are cut
    to `loads`. `buy`: the import tariff per slot (currency/kWh), for its
    baseline. `reach_w`: grid limits (W) for its own model, sized above every
    flow the house can make; None keeps the configured ones (an EMHASS serving
    its devices to a coordinator elsewhere: they are its own connection's).
    """

    def __init__(
        self,
        opt: EmhassOptimization,
        key: str,
        battery: bool,
        loads: list[int],
        data_opt: pd.DataFrame,
        soc_init: float | None,
        soc_final: float | None,
        runtime: dict[str, Any],
        buy: np.ndarray,
        reach_w: float | None = 1e5,
        load_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the participant's own EMHASS model: a copy of `opt`'s
        configuration with only this participant's devices enabled (the
        parameters are described on the class). Solves nothing yet; sets
        `max_power_kw`, `modulating` and `onoff` for the coordinator."""
        from emhass.optimization import Optimization

        oc, pc = copy.deepcopy(opt.optim_conf), copy.deepcopy(opt.plant_conf)
        oc["optimization_backend"] = "cvxpy"  # its own solves run the default solver
        oc["set_use_battery"] = bool(battery)
        # PV stays enabled: a best response meters the participant under the rest
        # of the house, given as load and PV (zero in a price response)
        oc["set_use_pv"] = True
        oc["set_nodischarge_to_grid"] = False  # the coordinator holds the meter
        oc["number_of_deferrable_loads"] = len(loads)
        for name in _per_load_keys():
            if isinstance(oc.get(name), list):
                oc[name] = [oc[name][k] for k in loads if k < len(oc[name])]
        # deferrable_load_groups whose loads are all this participant's,
        # renumbered to its own loads (`_limits`); the others the coordinator holds
        oc["deferrable_load_groups"] = [
            {
                **g,
                "names": [
                    f"deferrable{loads.index(int(nm[len('deferrable') :]))}" for nm in g["names"]
                ],
            }
            for g in (load_groups or [])
        ]
        # The coordinator holds the meter, so the view's own limits must never
        # bind: `reach_w` exceeds every flow the house can make. Not 'infinite':
        # EMHASS uses these as big-M bounds, and a huge one breaks HiGHS.
        if reach_w is not None:
            pc["maximum_power_from_grid"] = float(reach_w)
            pc["maximum_power_to_grid"] = float(reach_w)
        pc["compute_curtailment"] = False
        # A hybrid inverter is the coordinator's too (a sub-meter in its master):
        # the participant's own model plans its devices on the house's AC bus.
        pc["inverter_is_hybrid"] = False
        self.key, self.battery, self.loads = key, bool(battery), list(loads)
        self.opt: EmhassOptimization = Optimization(
            opt.retrieve_hass_conf,
            oc,
            pc,
            opt.var_load_cost,
            opt.var_prod_price,
            "profit",
            opt.emhass_conf,
            opt.logger,
            num_timesteps=len(data_opt),
        )
        # Every query starts from the same state. On a re-solve EMHASS carries a
        # heat pump's heat input (thermal inertia) over from the last solve, as
        # MPC needs when the horizon moves on; here every solve is the same
        # horizon at other prices, so only a configured q_input_initial applies.
        self.opt._persist_q_input = _hold_q_input
        # EMHASS scales its terminal-SoC penalty with the highest import price
        # it is given: that is the battery's own objective, so it is pinned to
        # this participant's own tariff, not the query's - the plan then
        # minimises the objective its answer reports.
        self.opt.soc_final_reference_tariff = np.asarray(buy, dtype=float)
        self.data = data_opt
        self.n, self.dt = len(data_opt), float(opt.time_step)
        self.soc_init, self.soc_final = soc_init, soc_final
        self.runtime = {
            k: (None if v is None else [v[i] for i in loads if i < len(v)])
            for k, v in runtime.items()
            if k in RUNTIME_LOAD_LISTS
        }
        p = 0.0
        if battery:
            p += (
                max(float(pc["battery_charge_power_max"]), float(pc["battery_discharge_power_max"]))
                / 1000.0
            )
        for k in range(len(loads)):
            nominal = oc["nominal_power_of_deferrable_loads"][k]
            p += (max(nominal) if isinstance(nominal, list) else float(nominal)) / 1000.0
        self.max_power_kw = p
        self.cap_wh = float(pc.get("battery_nominal_energy_capacity", 0.0))
        self.buy = np.asarray(buy, dtype=float)
        # Convex: loads that run at any power, with no minimum on/off time, start
        # count or single block, and no battery (EMHASS's battery has a charge or
        # discharge binary per slot). A mix of such plans runs as it is, so the
        # coordinator may keep one (see blend).
        min_power = oc.get("minimum_power_of_deferrable_loads") or []
        cfgs = list(oc.get("def_load_config") or []) + [{}] * len(loads)
        runtime_min = (runtime.get("min_power_of_deferrable_loads") or []) if runtime else []

        def binary(k: int) -> bool:
            """Whether load k is on/off at a level: semi-continuous, one block, a
            power sequence (a list), or a minimum power - EMHASS gives each a
            binary, so a mix of its plans is no plan it can run."""
            local = loads[k] if k < len(loads) else k
            return bool(
                oc["treat_deferrable_load_as_semi_cont"][k]
                or oc["set_deferrable_load_single_constant"][k]
                or isinstance(oc["nominal_power_of_deferrable_loads"][k], list)
                or float((min_power[k] if k < len(min_power) else 0) or 0) > 0
                or float((runtime_min[local] if local < len(runtime_min) else 0) or 0) > 0
                or (cfgs[k] or {}).get("thermal_battery", {}).get("overshoot_temperature")
                is not None
            )

        self.modulating = (
            not battery
            and not any(g.get("mutual_exclusion") for g in oc["deferrable_load_groups"])
            and all(
                not binary(k)
                and not oc["set_deferrable_load_single_constant"][k]
                and not oc["def_minimum_on_time"][k]
                and not oc["def_minimum_off_time"][k]
                and not oc["set_deferrable_max_startups"][k]
                and not ((oc.get("def_load_config") or [{}] * len(loads))[k] or {})
                .get("thermal_config", {})
                .get("overshoot_temperature")
                for k in range(len(loads))
            )
        )
        # On/off scheduling (a load at fixed power, a single block, minimum on/off
        # times, a start limit): the coordinator fixes such a participant to one
        # plan before the others re-plan around it (DWCoordinator.run dive).
        self.onoff = any(
            binary(k)
            or oc["def_minimum_on_time"][k]
            or oc["def_minimum_off_time"][k]
            or oc["set_deferrable_max_startups"][k]
            for k in range(len(loads))
        )
        self._good: Answer | None = None  # the last answer that solved, a stand-in if one fails
        self.solves, self.solve_s = 0, 0.0
        self._last: tuple[tuple[Any, ...], Answer] | None = None  # (query key, answer): a
        # repeated query is not re-solved

    def _run(
        self, pv_w: np.ndarray, load_w: np.ndarray, buy: np.ndarray, sell: np.ndarray
    ) -> Answer:
        """Solve EMHASS's model of this participant once and read off its answer.

        `pv_w`, `load_w`: the PV and load (W per slot) it is metered with - zero
        in a price response, the rest of the house in a best response. `buy`,
        `sell`: the import and export prices (currency/kWh per slot). Returns the
        plan (kW, + = drawn), its private cost (EMHASS's objective less the
        bill) and its state: stored energy (kWh) with a battery, else energy used
        so far. If the solve fails, returns the last good answer with status
        "fallback"; raises RuntimeError if there has been none.
        """
        t0 = time.perf_counter()
        res = self.opt.perform_optimization(
            self.data,
            pv_w,
            load_w,
            np.asarray(buy, dtype=float),
            np.asarray(sell, dtype=float),
            soc_init=self.soc_init,
            soc_final=self.soc_final,
            **self.runtime,
        )
        self.solves += 1
        self.solve_s += time.perf_counter() - t0
        if self.opt.optim_status not in OK_STATUSES or (self.battery and "P_batt" not in res):
            if self._good is None:
                raise RuntimeError(
                    f"EMHASS could not plan participant {self.key}: {self.opt.optim_status}"
                )
            g = self._good
            return Answer(
                g.plan_kw, g.private_cost, g.trajectory, status="fallback", detail=g.detail
            )
        plan_w = np.zeros(self.n)
        electric = self.opt.optim_conf.get("is_electric_load") or []
        for k in range(len(self.loads)):
            # A load EMHASS keeps off the electrical balance (gas, say) is not
            # drawn from the meter: its cost stays in the private cost
            if k < len(electric) and electric[k] is False:
                continue
            plan_w += res[f"P_deferrable{k}"].values
        if self.battery:
            plan_w -= res["P_batt"].values
        plan_kw = plan_w / 1000.0
        # EMHASS maximises -(bill + private); the bill is on the metered flow
        scale = 0.001 * self.dt
        bill = scale * float(
            np.sum(
                np.asarray(buy) * res["P_grid_pos"].values
                + np.asarray(sell) * res["P_grid_neg"].values
            )
        )
        # The objective of the solve that was published: a relaxed retry or a DP
        # re-solve is not self.opt.prob (#1048)
        value = getattr(self.opt, "solved_value", None)
        value = self.opt.prob.value if value is None else value
        if value is None or not np.isfinite(float(value)):
            raise RuntimeError(f"participant {self.key}: no finite objective ({value})")
        private = -float(value) - bill
        if self.battery:
            # the start the solve used: as perform_optimization fills it, the
            # final SoC when only that is given, else the target
            if self.soc_init is not None:
                s0 = self.soc_init
            elif self.soc_final is not None:
                s0 = self.soc_final
            else:
                s0 = self.opt.plant_conf["battery_target_state_of_charge"]
            traj = np.concatenate([[float(s0)], res["SOC_opt"].values]) * self.cap_wh / 1000.0
        else:
            traj = np.concatenate([[0.0], np.cumsum(plan_kw) * self.dt])
        self._good = Answer(plan_kw, private, traj, status="ok", detail=res)
        return self._good

    def respond(self, q: Query) -> Answer:
        """Answer a coordinator's query (interface.Participant). A query equal
        to the last one returns the same answer without solving again."""
        key = (q.kind,) + tuple(
            None if a is None else np.asarray(a, dtype=float).tobytes()
            for a in (q.price_draw, q.price_supply, q.residual_kw)
        )
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        a = self._respond(q)
        self._last = (key, a)
        return a

    def _respond(self, q: Query) -> Answer:
        """Map the query to an EMHASS solve: a price response meters the
        participant alone; a best response meters it under `q.residual_kw`,
        split into PV (exported part) and load (imported part). A proximal
        query raises NotImplementedError."""
        if q.kind in ("price_response", "best_response") and (
            q.price_draw is None or q.price_supply is None
        ):
            raise ValueError(f"a {q.kind} query needs price_draw and price_supply")
        if q.kind == "price_response":
            assert q.price_draw is not None and q.price_supply is not None
            zero = np.zeros(self.n)
            return self._run(zero, zero, q.price_draw, q.price_supply)
        if q.kind == "best_response":
            assert q.price_draw is not None and q.price_supply is not None
            r = np.asarray(q.residual_kw, dtype=float)
            return self._run(
                np.maximum(-r, 0.0) * 1000.0,
                np.maximum(r, 0.0) * 1000.0,
                q.price_draw,
                q.price_supply,
            )
        raise NotImplementedError(
            "a proximal step needs a quadratic term EMHASS's model does not have yet"
        )

    def blend(
        self, weights: Sequence[float] | np.ndarray, details: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """EMHASS's result for a mix of this participant's plans: every number
        mixed with the same weights, the rest taken from the heaviest plan."""
        import pandas as pd

        out = details[int(np.argmax(weights))].copy()
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c].dtype):  # pandas 3: text is StringDtype
                out[c] = sum(w * d[c].values for w, d in zip(weights, details))
        return out

    def baseline(self) -> Answer:
        """A feasible plan to start from: its plan at the import tariff, metered alone."""
        return self.respond(Query("price_response", self.buy, self.buy))


def emhass_participant(
    opt: EmhassOptimization,
    data_opt: pd.DataFrame,
    soc_init: float | None,
    soc_final: float | None,
    key: str = "emhass",
) -> EmhassParticipant:
    """All of an EMHASS's own devices as one participant: what an EMHASS
    serving the participant API answers for (remote.ParticipantService). Its
    battery (with set_use_battery) and every deferrable load, planned
    together by its own model, its deferrable_load_groups held there.

    `opt`: that EMHASS's Optimization. `data_opt`: its input DataFrame for
    the horizon asked about (its index, the tariff its baseline plans at,
    thermal forecasts). `soc_init`, `soc_final`: its battery's start and
    end-of-day state of charge (fractions).

    Its own grid limits (maximum_power_from_grid / maximum_power_to_grid) are
    kept: they are its connection's - a panel's rating, no backfeed - which
    its model holds in every plan it offers, however big its own devices are.
    The coordinator holds only what it shares with the rest of the house.
    """
    oc = opt.optim_conf
    loads = list(range(int(oc.get("number_of_deferrable_loads", 0) or 0)))
    runtime = {
        "def_total_hours": oc.get("operating_hours_of_each_deferrable_load"),
        "def_total_timestep": oc.get("operating_timesteps_of_each_deferrable_load"),
        "def_start_timestep": oc.get("start_timesteps_of_each_deferrable_load"),
        "def_end_timestep": oc.get("end_timesteps_of_each_deferrable_load"),
    }
    n = len(data_opt)
    buy = (
        np.asarray(data_opt[opt.var_load_cost].values, dtype=float)
        if opt.var_load_cost in data_opt
        else np.full(n, 0.2)
    )
    return EmhassParticipant(
        opt,
        key,
        bool(oc.get("set_use_battery")),
        loads,
        data_opt,
        soc_init,
        soc_final,
        runtime,
        buy,
        reach_w=None,
        load_groups=list(oc.get("deferrable_load_groups") or []),
    )


def emhass_description(p: EmhassParticipant) -> dict[str, Any]:
    """The participant API's Description of `p` (remote.ParticipantService):
    it answers price responses only, so no household load is sent to it."""
    devices = (["battery"] if p.battery else []) + [f"deferrable{k}" for k in p.loads]
    return {
        "key": p.key,
        "devices": devices,
        "max_power_kw": float(p.max_power_kw),
        "modulating": False,
        "onoff": bool(p.onoff),
        "kinds": ["price_response"],
        "solver": "emhass",
    }


# ------------------------------------------------------------------ the call
OLD_KEYS = ("participants", "electrical_topology", "group_limits")
SITE_DEVICE = _SITE_DEVICE  # one vocabulary with the site check (emhass.coordinator.site)
_kind = site_kind


class Layout(NamedTuple):
    """The `site` list as the coordinator takes it. `groups`: the participant
    groups, each a dict with "key" (its name in the coordinator: the group's
    id, or the device's), "devices", "solver" and "config". `submeters`,
    `set_limits`: types.SubMeter / SetLimit over participant keys.
    `own_groups`: participant key -> the deferrable_load_groups its own EMHASS
    model holds. `inverter`: the node EMHASS's inverter keys describe (its AC
    power is P_hybrid_inverter), or None. `grid_w`: the main meter's (import,
    export) limits in W from `site`, or None (EMHASS's own keys then).
    `notes`: things worth a warning that do not stop the plan."""

    groups: list[dict[str, Any]]
    submeters: tuple[SubMeter, ...]
    set_limits: tuple[SetLimit, ...]
    own_groups: dict[str, list[dict[str, Any]]]
    inverter: str | None
    grid_w: tuple[float | None, float | None] | None
    notes: tuple[str, ...]


def _site(
    optim_conf: dict[str, Any], plant_conf: dict[str, Any], devices: list[str]
) -> tuple[Layout, str | None]:
    """The house from EMHASS's `site` list, and a reason it cannot be planned
    (or None).

    Each element has an `id` and, but the main meter and limits, a `parent`
    ("grid", the main meter, or a node's id):

    - "grid": the main meter's `max_import` / `max_export` (W).
    - a node (any other id): `type` (hybrid_inverter, inverter, panel,
      breaker, meter), `max_import` / `max_export` (W) on its connection to
      its parent, `efficiency_import` / `efficiency_export`.
    - a limit (`type: "limit"`): `max_import` / `max_export` (W) on what the
      devices tagged with it draw together, wherever they are.
    - a device (`pv`, `battery0` or `battery`, `deferrableN`):
      `group` (devices with the same group are planned together by EMHASS's
      model) and `limits` (limit ids); `solver` may only be "emhass".

    A device `devices` lists that `site` does not is planned alone by EMHASS,
    on the main meter. With no node in `site`, EMHASS's own inverter keys
    still describe a hybrid inverter. Either way, EMHASS's
    deferrable_load_groups: a group whose loads are all in one EMHASS group
    stays in that group's own model (mutual exclusion included); otherwise
    its max_power is a set limit.

    The coordinator sees an EMHASS group only as its total, so a group's
    devices share one parent, and a limit holds all of them or none.

    `devices`: EMHASS's devices (battery, deferrableN).
    """
    empty = Layout([], (), (), {}, None, None, ())
    elements = [e for e in optim_conf.get("site") or [] if isinstance(e, dict)]
    kinds = {
        k: [e for e in elements if _kind(e) == k]
        for k in ("grid", "node", "limit", "remote", "device")
    }
    for e in kinds["node"] + kinds["device"] + kinds["remote"]:
        if "parent" not in e:
            # one tariff, one meter: every row but the meter and the limits
            # says where it is wired, so the list reads as the tree it is
            return (
                empty,
                f"site: {e.get('id')!r} needs a parent ('grid', the main meter, or a node's id)",
            )

    def kw(v: Any) -> float | None:
        return None if v is None else float(v) / 1000.0

    nodes = {str(e["id"]): e for e in kinds["node"]}
    limit_ids = [str(e["id"]) for e in kinds["limit"]]
    notes: list[str] = []
    parent_of: dict[str, str] = {}  # device -> its parent
    tags: dict[str, list[str]] = {}  # limit id -> the devices tagged with it
    groups: dict[str, dict[str, Any]] = {}  # participant key -> its group
    for e in kinds["device"]:
        dev = "battery" if e["id"] == "battery0" else str(e["id"])  # one battery: EMHASS's P_batt
        if dev in parent_of:
            return empty, f"site lists {dev!r} twice"
        parent = str(e.get("parent", MAIN))
        if parent != MAIN and parent not in nodes:
            return empty, f"site: the parent of {dev!r}, {parent!r}, is no node"
        parent_of[dev] = parent
        for lim in e.get("limits") or []:
            if lim not in limit_ids:
                return empty, f"site: {dev!r} is tagged with {lim!r}, which is no limit"
            tags.setdefault(lim, []).append(dev)
        if dev == "pv":
            if any(k in e for k in ("solver", "group", "config")):
                return empty, "site: the PV is a forecast, planned by no solver"
            continue
        solver = e.get("solver", "emhass")
        if solver != "emhass":
            return (
                empty,
                f"site: {dev!r} has the solver {solver!r} (EMHASS's own models plan every device)",
            )
        if dev not in devices:
            return empty, f"site lists {dev!r}, which EMHASS does not plan here"
        key = str(e.get("group") or dev)
        g = groups.get(key)
        if g is None:
            groups[key] = {"key": key, "devices": [dev], "solver": solver, "parent": parent}
        elif g["parent"] != parent:
            return (
                empty,
                f"site: the group {key!r} sits on {g['parent']!r} and {parent!r} (one parent per group)",
            )
        else:
            g["devices"].append(dev)
    for e in kinds["remote"]:
        # a solver on the network: one participant, its devices its own business
        rid = str(e["id"])
        parent = str(e.get("parent", MAIN))
        if parent != MAIN and parent not in nodes:
            return empty, f"site: the parent of {rid!r}, {parent!r}, is no node"
        if rid in groups or rid in parent_of:
            return empty, f"site: {rid!r} names two elements"
        solver = e["solver"]
        groups[rid] = {
            "key": rid,
            "devices": [rid],
            "solver": "remote",
            "parent": parent,
            "config": {
                "url": str(solver.get("url", "")),
                "token_secret": solver.get("token_secret"),
                "trusted_network": solver.get("trusted_network") is True,
            },
        }
        parent_of[rid] = parent
        for lim in e.get("limits") or []:
            if lim not in limit_ids:
                return empty, f"site: {rid!r} is tagged with {lim!r}, which is no limit"
            tags.setdefault(lim, []).append(rid)
    for key, g in groups.items():
        if key not in g["devices"] and (
            key in nodes or key in limit_ids or key == MAIN or SITE_DEVICE.match(key)
        ):
            return empty, f"site: the group {key!r} has the name of an element"
    for d in devices:  # a device site leaves out stays EMHASS's
        if d not in parent_of:
            groups[d] = {"key": d, "devices": [d], "solver": "emhass", "parent": MAIN}
            parent_of[d] = MAIN

    key_of = {d: g["key"] for g in groups.values() for d in g["devices"]}
    whole = {g["key"]: set(g["devices"]) for g in groups.values() if g["solver"] == "emhass"}

    def keys_of(devs: list[str], what: str) -> tuple[tuple[str, ...], str | None]:
        missing = [d for d in devs if d not in key_of]
        if missing:
            return (), f"{what} names {missing}, which nothing plans"
        keys = {key_of[d] for d in devs}
        split = sorted(k for k in keys if k in whole and not whole[k] <= set(devs))
        if split:
            return (
                (),
                f"{what} splits the group {split[0]!r} (it must hold all of its devices or none)",
            )
        return tuple(sorted(keys)), None

    own: dict[str, list[dict[str, Any]]] = {}
    sets: list[SetLimit] = []
    for e in kinds["limit"]:
        lid = str(e["id"])
        if not tags.get(lid):
            notes.append(f"the limit {lid!r} has no device tagged with it")
            continue
        keys, reason = keys_of(tags[lid], f"site: the limit {lid!r}")
        if reason:
            return empty, reason
        sets.append(
            SetLimit(
                lid,
                keys,
                max_import_kw=kw(e.get("max_import")),
                max_export_kw=kw(e.get("max_export")),
            )
        )
    for dg in optim_conf.get("deferrable_load_groups") or []:
        names = list(dg.get("names") or [])
        inside = [k for k, devs in whole.items() if names and set(names) <= devs]
        if inside:
            own.setdefault(inside[0], []).append(dg)
            continue
        if dg.get("mutual_exclusion"):
            return empty, "deferrable_load_groups: mutual exclusion across participants"
        if dg.get("max_power") is None:
            continue
        keys, reason = keys_of(names, f"deferrable_load_groups {names}")
        if reason:
            return empty, reason
        sets.append(SetLimit("+".join(names), keys, max_import_kw=float(dg["max_power"]) / 1000.0))
    if len({s.name for s in sets}) < len(sets):
        return empty, "two limits have the same name"

    grid_w = None
    if kinds["grid"]:
        g0 = kinds["grid"][0]
        grid_w = (
            None if g0.get("max_import") is None else float(g0["max_import"]),
            None if g0.get("max_export") is None else float(g0["max_export"]),
        )
    layout_groups = [{k: v for k, v in g.items() if k != "parent"} for g in groups.values()]
    if not nodes:
        subs: tuple[SubMeter, ...] = ()
        if plant_conf.get("inverter_is_hybrid"):
            # its DC bus: the PV and EMHASS's battery, when the coordinator holds it
            if "battery" in key_of and key_of["battery"] != "battery":
                return empty, "a hybrid inverter with the battery in an EMHASS group"
            out_w = plant_conf.get("inverter_ac_output_max")
            out_w = float(plant_conf["pv_inverter_model"] if out_w is None else out_w)
            in_w = plant_conf.get("inverter_ac_input_max")
            subs = (
                hybrid_inverter(
                    ("battery",) if "battery" in key_of else (),
                    out_w / 1000.0,
                    (out_w if in_w is None else float(in_w)) / 1000.0,
                    float(plant_conf.get("inverter_efficiency_dc_ac", 1.0)),
                    float(plant_conf.get("inverter_efficiency_ac_dc", 1.0)),
                ),
            )
        return Layout(
            layout_groups,
            subs,
            tuple(sets),
            own,
            "inverter" if subs else None,
            grid_w,
            tuple(notes),
        ), None

    on: dict[str, list[str]] = {n: [] for n in nodes}
    for g in groups.values():
        if g["parent"] != MAIN:
            on[g["parent"]].append(g["key"])
    if parent_of.get("pv", MAIN) != MAIN:
        on[parent_of["pv"]].append("pv")
    has_child = {str(e.get("parent", MAIN)) for e in kinds["node"]}
    subs_out = []
    inverter = None
    for nid, e in nodes.items():
        if not on[nid] and nid not in has_child:
            notes.append(f"the node {nid!r} holds nothing")
        if e.get("type") == "hybrid_inverter" and inverter is None:
            inverter = nid
        parent = str(e.get("parent", MAIN))
        subs_out.append(
            SubMeter(
                nid,
                tuple(sorted(on[nid])),
                max_export_kw=kw(e.get("max_export")),
                max_import_kw=kw(e.get("max_import")),
                eta_export=float(e.get("efficiency_export", 1.0)),
                eta_import=float(e.get("efficiency_import", 1.0)),
                parent=None if parent == MAIN else parent,
            )
        )
    return Layout(
        layout_groups, tuple(subs_out), tuple(sets), own, inverter, grid_w, tuple(notes)
    ), None


def _battery(plant_conf: dict[str, Any], soc_init: float, buy: np.ndarray) -> BatteryConfig:
    """EMHASS's battery as the coordinator holds it: its SoC window becomes the store, and
    energy left at the end is valued at the horizon's average import price (as
    the package's own planner does), not pinned to soc_final.

    `plant_conf`: EMHASS's plant configuration (W, Wh, fractions). `soc_init`:
    the starting state of charge (fraction of nominal capacity). `buy`: the
    import tariff per slot (currency/kWh). Returns a BatteryConfig in kW/kWh.

    Power limits are at the meter here. EMHASS bounds the meter-side power by
    each limit twice - directly, and through the efficiency (charging at most
    limit / charge efficiency, discharging at most limit * discharge
    efficiency) - so the tighter of the two is used: with efficiencies below 1,
    the charge limit as given and the discharge limit times its efficiency.
    """
    pc = plant_conf
    cap = float(pc["battery_nominal_energy_capacity"]) / 1000.0
    lo, hi = (
        float(pc["battery_minimum_state_of_charge"]),
        float(pc["battery_maximum_state_of_charge"]),
    )
    eta_c, eta_d = float(pc["battery_charge_efficiency"]), float(pc["battery_discharge_efficiency"])
    chg, dis = (
        float(pc["battery_charge_power_max"]) / 1000.0,
        float(pc["battery_discharge_power_max"]) / 1000.0,
    )
    return BatteryConfig(
        capacity_kwh=cap * hi,
        p_charge_max_kw=min(chg, chg / eta_c),
        p_discharge_max_kw=min(dis, dis * eta_d),
        eta_charge=eta_c,
        eta_discharge=eta_d,
        soc_initial_frac=min(max(soc_init / hi, 0.0), 1.0),
        soe_min_frac=lo / hi,
        terminal_price=float(np.mean(buy)),
    )


def _linear_battery(
    oc: dict[str, Any], pc: dict[str, Any], soc_init: float | None = None, buy: Any = None
) -> bool:
    """Whether EMHASS's battery is exactly its linear constraints - the state
    of charge stepped by both efficiencies, power limits, the SoC window and
    the end-of-day target - so the coordinator can hold it in its master LP.

    Otherwise EMHASS's own model plans it, as a participant: with a penalty
    or reward on its state of charge; when it starts outside its SoC window
    (EMHASS's model recovers from there); and when an import price is
    negative, where the LP without EMHASS's charge-or-discharge binary would
    charge and discharge in one slot to be paid for the loss."""

    def zero(v: Any) -> bool:
        return all(float(x or 0) == 0 for x in (v if isinstance(v, list) else [v]))

    # EMHASS reads the SoC penalties from optim_conf; plant_conf too, as before
    penalties = ("battery_soc_deficit_cost", "battery_soc_surplus_cost")
    return (
        not oc.get("set_battery_dynamic")
        and zero(oc.get("weight_battery_discharge", 0))
        and zero(oc.get("weight_battery_charge", 0))
        and zero(pc.get("battery_stress_cost", 0))
        and all(zero(conf.get(k, 0)) for conf in (oc, pc) for k in penalties)
        and not pc.get("battery_charge_power_derating")
        and zero(oc.get("battery_soc_final_reward_factor", 0))
        and (
            soc_init is None
            or float(pc["battery_minimum_state_of_charge"])
            <= float(soc_init)
            <= float(pc["battery_maximum_state_of_charge"])
        )
        and (buy is None or float(np.min(buy)) >= 0.0)
    )


class _Committed:
    """A remote that has taken its plan: planning the house again holds it
    to that plan (interface.Participant, answering it to every query)."""

    modulating = False
    onoff = True

    def __init__(self, rp: RemoteParticipant, plan: Column) -> None:
        self.key, self.max_power_kw = rp.key, rp.max_power_kw
        self.solves, self.solve_s = 0, 0.0
        self._answer = Answer(
            np.asarray(plan.power, dtype=float),
            float(plan.cost),
            np.asarray(plan.trajectory, dtype=float),
            detail=plan.detail,
        )

    def respond(self, query: Query) -> Answer:  # noqa: ARG002 - the plan it runs
        return self._answer

    def baseline(self) -> Answer:
        return self._answer


def optimize(
    opt: EmhassOptimization,
    data_opt: pd.DataFrame,
    p_pv: npt.ArrayLike,
    p_load: npt.ArrayLike,
    unit_load_cost: npt.ArrayLike,
    unit_prod_price: npt.ArrayLike,
    soc_init: float | None = None,
    soc_final: float | None = None,
    **runtime: Any,
) -> pd.DataFrame | None:
    """Plan with the coordinator `opt.optim_conf["optimization_backend"]` names.

    Called by EMHASS's Optimization.perform_optimization with its own
    arguments. `opt`: that Optimization (configuration, logger, time step).
    `data_opt`: its input DataFrame, one row per slot. `p_pv`, `p_load`: PV and
    house load forecasts (W per slot). `unit_load_cost`, `unit_prod_price`:
    import and export prices (currency/kWh per slot). `soc_init`, `soc_final`:
    the battery's start and end-of-day state of charge (fractions); EMHASS's
    battery_target_state_of_charge when None. `runtime`: perform_optimization's
    other keyword arguments.

    Returns a DataFrame with the columns perform_optimization returns (the same
    names and units; optim_status is "Optimal" only for a plan proven within
    GAP_TOL of the best, else "Optimal_Inaccurate") plus `fed_meter_price`,
    `fed_lower_bound`, `fed_gap`, `fed_stop_reason`, `fed_iterations` and
    one `fed_share_<player>` per player (solar, each device or participant):
    its share of the saving over the horizon, in currency, the same in every row.
    Returns None, after logging why, when it cannot plan this configuration or
    the coordinator fails; EMHASS then runs its default solver.
    """
    log = opt.logger
    oc, pc = opt.optim_conf, opt.plant_conf
    backend = oc.get("optimization_backend", "cvxpy")
    opt.fed_fallback_reason = None  # EMHASS may hand the same object back next run
    reason = unsupported(oc, pc, opt.costfun, runtime)
    if backend != "dantzig_wolfe":
        reason = reason or f"backend {backend!r} (this version runs dantzig_wolfe)"
    if reason:
        _decline(opt, f"not available for {reason}")
        return None

    t0 = time.perf_counter()
    n, dt = len(data_opt), float(opt.time_step)
    buy = np.asarray(unit_load_cost, dtype=float)
    sell_in = np.asarray(unit_prod_price, dtype=float)
    sell = sell_in if opt.costfun == "profit" else np.zeros(n)
    if np.any(sell > buy + 1e-12):
        _decline(opt, "an export price above the import price")
        return None
    pv_w, load_w = np.asarray(p_pv, dtype=float), np.asarray(p_load, dtype=float)
    if soc_init is None and oc.get("set_use_battery"):
        soc_init = pc["battery_target_state_of_charge"]
    if soc_final is None and oc.get("set_use_battery"):
        soc_final = pc["battery_target_state_of_charge"]

    devices = (["battery"] if oc.get("set_use_battery") else []) + [
        f"deferrable{k}" for k in range(int(oc.get("number_of_deferrable_loads", 0)))
    ]
    nominal = oc.get("nominal_power_of_deferrable_loads") or []
    loads_n = sum(1 for d in devices if d.startswith("deferrable"))
    device_w = sum((max(v) if isinstance(v, list) else float(v)) for v in nominal[:loads_n])
    if oc.get("set_use_battery"):
        device_w += max(
            float(pc["battery_charge_power_max"]), float(pc["battery_discharge_power_max"])
        )
    reach_w = (
        2.0 * (float(np.max(np.abs(load_w))) + float(np.max(np.abs(pv_w))) + device_w) + 1000.0
    )
    participants: list[EmhassParticipant | RemoteParticipant | _Committed] = []
    remotes: list[RemoteParticipant] = []
    remote_status: dict[str, str] = {}
    targets: dict[str, float] = {}
    battery_cfg: BatteryConfig | None = None
    layout, reason = _site(oc, pc, devices)
    if reason:
        _decline(opt, reason)
        return None
    for note in layout.notes:
        log.warning(f"site: {note}")
    groups = layout.groups
    own_groups = layout.own_groups
    for g in groups:
        if (
            g["solver"] == "emhass"
            and g["key"] == "battery"  # alone, not a group the master would not know
            and g["devices"] == ["battery"]
            and _linear_battery(oc, pc, soc_init, buy)
        ):
            # EMHASS's battery, as its linear model in the master: the same
            # constraints from the same parameters, held exactly
            # unsupported() declined a battery without set_use_battery, which sets both
            assert soc_init is not None and soc_final is not None
            b = replace(_battery(pc, float(soc_init), buy), terminal_price=0.0)
            cap = float(pc["battery_nominal_energy_capacity"]) / 1000.0
            s0 = b.capacity_kwh * b.soc_initial_frac
            reach_up = s0 + n * dt * b.eta_c * b.p_charge_max_kw
            reach_dn = s0 - n * dt * b.p_discharge_max_kw / b.eta_d
            target = float(soc_final) * cap
            battery_cfg = b
            targets["battery"] = min(
                max(target, b.soe_floor_kwh, reach_dn), b.capacity_kwh, reach_up
            )
        elif g["solver"] == "emhass":
            loads = sorted(
                int(d[len("deferrable") :]) for d in g["devices"] if d.startswith("deferrable")
            )
            unknown = [d for d in g["devices"] if d != "battery" and not d.startswith("deferrable")]
            if unknown:
                _decline(opt, f"EMHASS has no device {unknown}")
                return None
            key = g["key"]
            participants.append(
                EmhassParticipant(
                    opt,
                    key,
                    "battery" in g["devices"],
                    loads,
                    data_opt,
                    soc_init,
                    soc_final,
                    runtime,
                    buy,
                    reach_w,
                    load_groups=own_groups.get(key),
                )
            )
        elif g["solver"] == "remote":
            # a solver on the network: asked to describe itself and plan with no
            # price now, so one that cannot is left out of this plan, said so
            key = g["key"]
            tokens = (getattr(opt, "retrieve_hass_conf", None) or {}).get("remote_tokens") or {}
            secret = g["config"].get("token_secret")
            first = getattr(data_opt, "index", [None])[0]
            window = PlanWindow(
                dt * 60.0, n, first.isoformat() if hasattr(first, "isoformat") else None
            )
            try:
                rp = RemoteParticipant(
                    key,
                    g["config"]["url"],
                    window,
                    token=tokens.get(secret) if secret else None,
                    timeout=REMOTE_TIMEOUT_S,
                    commit_wait_s=REMOTE_COMMIT_WAIT_S,
                    trusted_network=bool(g["config"].get("trusted_network", False)),
                )
                rp.baseline()
            except (RemoteError, ValueError) as exc:
                log.warning(
                    f"site: the remote solver {key!r} did not answer ({exc}); planned without it"
                )
                status = getattr(exc, "status", 0)
                if isinstance(exc, PlainHttpRefused):
                    remote_status[key] = f"declined: {exc}"[:200]
                elif status in (401, 403):
                    remote_status[key] = "unauthorized"
                elif status == 409:  # it answered: it cannot plan this horizon
                    remote_status[key] = f"declined: {exc}"[:200]
                else:
                    remote_status[key] = "unreachable"
                continue
            participants.append(rp)
            remotes.append(rp)
        else:
            _decline(opt, f"unknown solver {g['solver']!r}")
            return None

    def plan_house(
        participants: Sequence[EmhassParticipant | RemoteParticipant | _Committed],
    ) -> tuple[pd.DataFrame, DWResult] | None:
        """The coordinator's plan for `participants` (None, declined, when it
        cannot make one), with the results EMHASS publishes."""
        gone = set(remote_status) - {p.key for p in participants}  # planned without
        submeters = tuple(
            replace(sm, members=tuple(m for m in sm.members if m not in gone))
            for sm in layout.submeters
        )
        set_limits = tuple(
            sl
            for sl in (
                replace(sl, members=tuple(m for m in sl.members if m not in gone))
                for sl in layout.set_limits
            )
            if sl.members
        )
        horizon = Horizon(dt=dt, hours=n * dt)

        def col(name: str, default: float) -> np.ndarray:
            return (
                np.asarray(data_opt[name].values, dtype=float)
                if name in data_opt
                else np.full(n, default)
            )

        fc = Forecasts(buy=buy, sell=sell, load=load_w / 1000.0, solar=pv_w / 1000.0)
        imp_w, exp_w = layout.grid_w or (
            None,
            None,
        )  # the main meter's, from site, else EMHASS's keys
        grid = GridLimits(
            max_import_kw=float(pc.get("maximum_power_from_grid", 9000) if imp_w is None else imp_w)
            / 1000.0,
            max_export_kw=float(pc.get("maximum_power_to_grid", 9000) if exp_w is None else exp_w)
            / 1000.0,
            allow_curtailment=bool(pc.get("compute_curtailment", False)),
        )
        if (
            layout.inverter == "inverter"
            and not any(_kind(e) == "node" for e in oc.get("site") or [])
            and any(isinstance(p, EmhassParticipant) and p.battery for p in participants)
        ):
            _decline(opt, "a hybrid inverter with the battery in an EMHASS participant group")
            return None
        site = SiteConfig(
            horizon=horizon,
            battery=battery_cfg,
            grid=grid,
            submeters=submeters,
            set_limits=set_limits,
        )
        try:
            site.validate()
        except ValueError as exc:  # a loop, a device twice, a bad rating
            _decline(opt, f"site: {exc}")
            return None
        ceiling = (
            np.maximum(pv_w - load_w, 0.0) / 1000.0 if oc.get("set_nodischarge_to_grid") else None
        )
        run_kw: RunOptions = (
            RUN_DEFAULTS if participants else {}
        )  # the battery alone: the coordinator's own defaults
        co = DWCoordinator(
            site, fc, participants=participants, export_ceiling=ceiling, soe_targets=targets
        )
        # EMHASS's own solver time limit, for each coordinator run (the plan, then
        # the one for the saving split): past it, the best plan so far is kept
        budget = float(oc.get("lp_solver_timeout", 45) or 45)
        run_kw = {**run_kw, "max_seconds": budget}
        try:
            r = co.run(**run_kw)
        except Exception as exc:  # a plan is always published
            _decline(opt, f"the coordinator failed ({type(exc).__name__}: {exc})")
            return None
        if not np.isfinite(float(r.upper)):
            # no plan it found can run: past the export ceiling with nothing to curtail
            _decline(opt, "the coordinator found no plan within the export ceiling")
            return None
        held = r.plan.get("battery") if battery_cfg is not None else None
        if held is not None:
            # The master's battery has a charge and a discharge per slot and no
            # binary: both at once (to burn surplus it cannot export or curtail)
            # is a plan one battery cannot run. Its net power must step its state.
            p = np.asarray(held.power, dtype=float)
            step = np.where(p > 0, p * battery_cfg.eta_c, p / battery_cfg.eta_d) * dt
            if np.max(np.abs(np.diff(held.trajectory) - step), initial=0.0) > 1e-3:
                _decline(opt, "the coordinator's battery plan charges and discharges in one slot")
                return None
        breach = float(co.parts(r.plan)["breach_kwh"])
        if breach > 1e-3:
            # A grid, node or limit rating is hard in EMHASS: the default solver
            # meets it (relaxing an on/off load if it must); the coordinator only
            # priced it, and its on/off plans could not fit
            _decline(opt, f"the coordinator's plan exceeds a limit by {breach:.3f} kWh")
            return None

        res = _results(
            opt,
            co,
            r,
            data_opt,
            pv_w,
            load_w,
            buy,
            sell_in,
            soc_init,
            devices,
            participants,
            site,
            layout.inverter,
        )
        try:
            # The saving split needs one more plan, the same devices with no PV
            # (the export ceiling, PV surplus, is then zero).
            dark = replace(fc, solar=np.zeros(n))
            co_dark = DWCoordinator(
                site,
                dark,
                participants=participants,
                export_ceiling=None if ceiling is None else np.zeros(n),
                soe_targets=targets,
            )
            for player, share in shares(co, r.plan, co_dark, co_dark.run(**run_kw).plan).items():
                res[f"fed_share_{player}"] = share
        except Exception as exc:  # the plan stands without its split
            log.warning(f"optimization_backend: could not split the saving ({exc})")
        opt.optim_status = str(res["optim_status"].iloc[0])
        solves = sum(p.solves for p in participants)
        log.info(
            f"optimization_backend=dantzig_wolfe: {r.iterations} iterations, gap {r.upper - r.lower:.4f}, "
            f"{solves} participant solves, {time.perf_counter() - t0:.2f} s"
        )
        return res, r

    planned = plan_house(participants)
    if planned is None:
        return None
    res, r = planned
    # Each remote runs its part of a live plan, last: once one has committed,
    # nothing here may fail and send EMHASS to its default solver, which would
    # plan the house again around a remote already running this plan.
    # A remote that refuses is planned without, as one that did not answer:
    # the house is planned again, the remotes that committed held to the plans
    # they took, and the remotes still to ask are asked for their part of
    # the new plan. A dry run asks nothing of anyone.
    pending = list(remotes)
    while pending:
        rp = pending.pop(0)
        if getattr(opt, "fed_dry_run", False):
            remote_status[rp.key] = "planned (dry run)"
            continue
        chosen = r.plan[rp.key]
        try:
            ok, why = rp.commit(chosen.detail, chosen.power)
        except Exception as exc:  # a commit that fails, however, is a refusal
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if ok or ok is None:
            # None: its outcome is not known (no answer in time, or published
            # part way) and it may be running - its part stays in the plan
            # rather than be planned around as absent
            remote_status[rp.key] = "committed" if ok else f"uncertain: {why}"[:200]
            if ok is None:
                log.warning(
                    f"site: the outcome of the remote solver {rp.key!r}'s commit is not known "
                    f"({why}); its part is kept in the plan"
                )
            participants[participants.index(rp)] = _Committed(rp, chosen)
            continue
        log.warning(f"site: the remote solver {rp.key!r} did not take its plan ({why})")
        remote_status[rp.key] = f"refused: {why}"[:200]
        participants.remove(rp)
        try:
            again = plan_house(participants)
        except Exception as exc:  # whatever fails here, a plan not made again
            _decline(opt, f"the coordinator failed ({type(exc).__name__}: {exc})")
            again = None
        if again is not None:
            res, r = again
            continue
        if not any(v == "committed" or v.startswith("uncertain") for v in remote_status.values()):
            return None  # nothing runs yet: EMHASS's default solver plans
        # a remote runs its part already: the plan stands, with the refusal
        opt.fed_fallback_reason = None
        log.error(
            f"site: the house could not be planned again without {rp.key!r}; "
            "the plan published still counts its part"
        )
    for key, status in remote_status.items():
        res[f"fed_remote_status_{key}"] = status
    for rp in remotes:
        # how its last commit ended, as it records it (a commit of an earlier
        # run, unless this run's has just been made)
        last = rp.active_commit
        if last is not None:
            res[f"fed_remote_last_commit_{rp.key}"] = (
                f"{last['state']}: {last.get('reason', '')}"
                if last.get("reason")
                else last["state"]
            )[:200]
    return res


def shares(
    co: DWCoordinator,
    plan: Mapping[str, Column],
    co_dark: DWCoordinator,
    plan_dark: Mapping[str, Column],
) -> dict[str, float]:
    """Each player's share of the saving over the horizon (currency): solar and
    every device or participant (dw.attribution: reimbursed its private-cost
    change, then an Owen value between solar and the devices, Aumann-Shapley
    among the devices). `plan` / `plan_dark`: the coordinator's plans with and
    without PV. The shares, with the reimbursements, add up to the saving over
    the baseline with no PV and no coordination."""
    rows = ledger(co, plan, co_dark, plan_dark)["rows"]
    return {
        row["player"]: float(row["net_gain"]) for row in rows if row["player"] != "household load"
    }


def _results(
    opt: Any,
    co: DWCoordinator,
    r: DWResult,
    data_opt: pd.DataFrame,
    pv_w: np.ndarray,
    load_w: np.ndarray,
    buy: np.ndarray,
    sell_in: np.ndarray,
    soc_init: float | None,
    devices: list[str],
    participants: Sequence[EmhassParticipant | RemoteParticipant | _Committed],
    site: SiteConfig,
    inverter: str | None = None,
) -> pd.DataFrame:
    """The coordinator's result `r` as EMHASS's opt_res: the same columns and
    units as perform_optimization returns, plus fed_* columns.

    `co`: the coordinator that produced `r`. The other arguments are those
    `optimize` worked with: EMHASS's inputs (W, currency/kWh), the device names,
    the EMHASS participants (whose own results fill their devices' columns) and
    the package's SiteConfig; `inverter`, the node whose AC power is
    P_hybrid_inverter. Returns a DataFrame indexed like `data_opt`.
    """
    import pandas as pd

    dt, n = float(opt.time_step), len(data_opt)
    plan = r.plan
    flows = site_meter(co.cfg, co.fc, {k: c.power for k, c in plan.items()})
    net, curtail = flows.net, flows.curtail
    if co.export_ceiling is not None:
        over = np.maximum(-net - co.export_ceiling, 0.0)
        net, curtail = net + over, curtail + over

    out = pd.DataFrame(index=data_opt.index)
    out["P_PV"] = pv_w
    out["P_Load"] = load_w
    if opt.plant_conf.get("compute_curtailment"):
        out["P_PV_curtailment"] = curtail * 1000.0
    if inverter is not None and inverter in flows.ac:
        # EMHASS's sign: + DC to AC (the inverter delivering to the house)
        out["P_hybrid_inverter"] = flows.ac[inverter] * 1000.0
    for node, f in flows.ac.items():
        # every node's power to its parent, W (+ = up the tree)
        out[f"fed_node_power_{node}"] = f * 1000.0
    out["P_grid_pos"] = np.maximum(net, 0.0) * 1000.0
    out["P_grid_neg"] = np.minimum(net, 0.0) * 1000.0
    out["P_grid"] = out["P_grid_pos"] + out["P_grid_neg"]

    by_load, batt_detail, extra = {}, None, {}
    for p in participants:
        if isinstance(p, (RemoteParticipant, _Committed)):
            # its devices are its own business: their total, W (+ = drawn)
            out[f"fed_remote_power_{p.key}"] = plan[p.key].power * 1000.0
            continue
        res = plan[p.key].detail
        for local, k in enumerate(p.loads):
            by_load[k] = res[f"P_deferrable{local}"].values
            for c in res.columns:  # thermal detail, renumbered
                if c.endswith(f"heater{local}"):
                    extra[c[: -len(str(local))] + str(k)] = res[c].values
        if p.battery:
            batt_detail = res
    for d in devices:
        if d.startswith("deferrable"):
            k = int(d[len("deferrable") :])
            out[f"P_deferrable{k}"] = by_load.get(k, np.zeros(n))
    if opt.optim_conf.get("set_use_battery"):
        if batt_detail is not None:
            out["P_batt"] = batt_detail["P_batt"].values
            out["SOC_opt"] = batt_detail["SOC_opt"].values
            for c in ("soc_deficit_cost", "soc_surplus_cost", "batt_stress_cost"):
                if c in batt_detail:
                    out[c] = batt_detail[c].values
        elif "battery" in plan:
            col = plan["battery"]
            out["P_batt"] = -col.power * 1000.0
            out["SOC_opt"] = col.trajectory[1:] / (
                float(opt.plant_conf["battery_nominal_energy_capacity"]) / 1000.0
            )
            # EMHASS writes these for every battery; held linearly, both costs are zero
            out["soc_deficit_cost"] = 0.0
            out["soc_surplus_cost"] = 0.0
    out["unit_load_cost"] = buy
    out["unit_prod_price"] = sell_in
    out["maximum_power_from_grid"] = opt._prepare_power_limit_array(
        opt.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
    )
    out["maximum_power_to_grid"] = opt._prepare_power_limit_array(
        opt.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
    )
    scale = -0.001 * dt
    cost_profit = scale * (buy * out["P_grid_pos"].values + sell_in * out["P_grid_neg"].values)
    out["cost_profit"] = cost_profit
    if opt.costfun == "profit":
        out["cost_fun_profit"] = cost_profit
    else:
        out["cost_fun_cost"] = scale * buy * out["P_grid_pos"].values
    out["optim_status"] = plan_status(float(r.upper), float(r.lower))
    for c, v in extra.items():
        out[c] = v
    out["fed_meter_price"] = r.prices
    out["fed_lower_bound"] = float(r.lower)
    out["fed_gap"] = float(r.upper - r.lower)
    for name, price in r.local_prices.items():
        # each node's own price (its balance row's dual): what the devices on
        # it - the hybrid inverter's battery, a panel's loads - are paid
        out[f"fed_local_price_{name}"] = price
    for name, price in r.limit_prices.items():
        # each constraint's premium (its row's dual), on top of its members' bus prices
        out[f"fed_limit_price_{name}"] = price
    out["fed_stop_reason"] = r.stop_reason or "converged"
    out["fed_iterations"] = int(r.iterations)
    return out
