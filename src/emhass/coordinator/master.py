"""The Dantzig-Wolfe coordinator.

Reference: G. B. Dantzig and P. Wolfe, "Decomposition Principle for Linear
Programs", Operations Research 8(1):101-111, 1960.

The meter is the only thing the devices share, so it becomes the (small,
linear) master programme; each participant is a sub-programme that only ever
answers one question:

    "at these per-slot prices, what would you do, and what would it cost you
     privately?"

Every answer becomes a column; the master mixes the columns to balance the
meter (and every node and limit of the site) at least cost, and emits new
prices, its duals. So:

* the bill, the grid limits (soft, priced at a breach multiplier) and
  curtailment are modelled exactly in the master LP;
* every iteration gives a Lagrangian lower bound, so the result carries a
  certificate: gap = upper - lower;
* on/off devices need no special treatment: pricing only asks the
  participant.

A battery whose model is linear is held in the master itself, as LP
variables. The fractional master solution is a convex mix of plans;
`recover()` turns it into one plan per participant (a MILP, HiGHS).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

import numpy as np

if TYPE_CHECKING:  # 3.11+; only ever an annotation here
    pass

from emhass.coordinator.costs import (
    battery_terminal_penalty,
    breach_energy,
    breach_price,
    grid_penalty,
    terminal_price,
    total_objective,
)
from emhass.coordinator.interface import Participant, Query
from emhass.coordinator.lpsolver import Triplets
from emhass.coordinator.lpsolver import linprog as np_linprog
from emhass.coordinator.meter import Bus, bus_cost
from emhass.coordinator.model import (
    BatteryConfig,
    Forecasts,
    SiteConfig,
)
from emhass.coordinator.submeter import (
    MAIN,
    beyond_main,
    device_bus,
    extra_bills,
    fixed_demand,
    pv_on,
    root_grid,
    site_meter,
    tariff,
    topology,
)


def _pinned_price(b: BatteryConfig) -> float:
    """The terminal price of a battery the coordinator holds. Each has it
    pinned to the tariff's when it is added (DWCoordinator.__init__)."""
    assert b.terminal_price is not None, "a coordinator battery without a pinned terminal price"
    return float(b.terminal_price)


@dataclass
class Column:
    power: np.ndarray  # kW per slot, + = consumption
    trajectory: np.ndarray  # state over time
    cost: float  # PRIVATE cost only: comfort / terminal energy
    source: str  # which oracle proposed it
    mix: list | None = None  # for a kept blend: [(weight, source, born), ...]
    born: int = 0  # iteration that proposed it (0 = seed, -1 = polish)
    runnable: bool = True  # False for an aggregate of an on/off device (a blend it cannot run)
    detail: Any = None  # a participant's own result for this plan (interface.Answer.detail)
    exact: bool = True  # False: a participant's stand-in, not the optimum of the query asked


@dataclass
class Device:
    key: str
    kind: str  # participant (interface.Participant)
    cfg: Participant
    columns: list[Column] = field(default_factory=list)
    stamp: int = 0  # iteration number given to columns added now
    on_add: Callable[[Device, Column], None] | None = None  # coordinator hook: keeps the
    # bound's per-price minima current

    @property
    def modulating(self) -> bool:
        """Whether a blend of this participant's plans can be run as it is
        (linear dynamics, a convex cost): it says so itself."""
        return bool(self.cfg.modulating)

    @property
    def participant(self) -> Participant:
        return self.cfg

    def add(self, col: Column, tol: float = 1e-9) -> bool:
        """Add unless an identical plan is already in the pool."""
        for c in self.columns:
            if np.max(np.abs(c.power - col.power)) < tol:
                return False
        col.born = self.stamp
        self.columns.append(col)
        if self.on_add is not None:
            self.on_add(self, col)
        return True


class RunOptions(TypedDict, total=False):
    """`DWCoordinator.run`'s keyword options, for passing them through
    `**kwargs` (`dw_plan`, `dw_coordinate`, `plan`). Each means what it means on
    `run`, and an absent one takes `run`'s default."""

    max_iter: int
    gap_tol: float
    smoothing: float | str
    heuristic_columns: bool
    integer: str
    polish: bool
    verbose: bool
    record: bool
    anytime: bool
    pool: str
    stall: int
    dive: int
    progress: Callable[..., None] | None


@dataclass
class DWResult:
    upper: float  # objective of the recovered (implementable) plan
    lower: float  # best Lagrangian bound
    relaxed: float  # master LP value (convexified problem)
    iterations: int
    plan: dict[str, Column]
    weights: dict[str, np.ndarray]
    prices: np.ndarray  # final meter price, currency/kWh
    history: list[dict]
    ms: float
    n_columns: int
    fractional_devices: int
    pricing_error: float = 0.0  # worst amount a pool column undercut the DP oracle
    relaxed_view: dict | None = None  # snapshot of the final master (record=True)
    stop_reason: str = ""  # converged | no new proposals | iteration cap
    plan_parts: dict | None = None  # objective breakdown of the returned plan
    incumbent_used: bool = False  # the returned plan came from an earlier iteration
    columns: dict | None = None  # device key -> list of Column (the pool)
    # node or further grid connection -> its bus's price, currency/kWh (the
    # dual of its balance row): what the devices on it were priced at. Empty
    # with no tree.
    local_prices: dict[str, np.ndarray] = field(default_factory=dict)
    # set limit -> its premium, currency/kWh (its row's dual): what its
    # members paid on top of their bus's price while it bound.
    limit_prices: dict[str, np.ndarray] = field(default_factory=dict)


class DWCoordinator:
    def __init__(
        self,
        cfg: SiteConfig,
        fc: Forecasts,
        solver: str = "numpy",
        participants: Sequence[Participant] = (),
        export_ceiling: np.ndarray | None = None,
        soe_targets: dict | None = None,
    ) -> None:
        """`participants`: devices answered through the interface (interface.
        Participant) rather than modelled here, priced in the master beside the
        site's own. `export_ceiling`: a hard per-slot cap on export, kW (for
        example "export no more than the PV surplus"). `soe_targets`: battery
        key -> the stored energy (kWh) a battery held in the master must end at."""
        cfg.validate()
        fc.validate(cfg.horizon)
        self.cfg, self.fc = cfg, fc
        if solver not in ("numpy", "highs"):
            raise ValueError("solver must be 'numpy' or 'highs'")
        self.solver = solver
        self.n, self.dt = cfg.horizon.steps, cfg.horizon.dt
        # The main meter's inflexible part. PV behind a node (a hybrid
        # inverter) is not in it: it reaches the meter through its bus.
        self.d = fixed_demand(cfg, fc).astype(float)
        self.topo = topology(cfg, fc)
        self.ref = float(np.mean(fc.buy))
        for root in self.topo.roots:
            buy, sell = tariff(cfg, fc, root)
            if np.any(sell > buy + 1e-12):
                # The meter cost is then non-convex (buy low, sell high in the same
                # slot) and the master LP would import and export at once.
                raise ValueError("DW master needs sell <= buy in every slot")

        self.devices: list[Device] = []
        # Batteries the master models directly as LP variables (the "hybrid
        # master"): a battery IS a linear programme, so convexifying it into
        # columns only makes the master rediscover its polytope one extreme
        # point at a time. Only the participants are DW blocks.
        self.lp_batts: list[tuple[str, BatteryConfig]] = []
        for i, b in enumerate(cfg.battery_list):
            if b.capacity_kwh > 0:
                # Pin the terminal price to the TARIFF's: by default the DP
                # derives it from min(buy), and buy will be the master's duals.
                pinned = replace(b, terminal_price=terminal_price(b, fc.buy))
                key = SiteConfig.battery_key(i)
                self.lp_batts.append((key, pinned))
        for p in participants:
            self.devices.append(Device(p.key, "participant", p))
        self.export_ceiling = (
            None if export_ceiling is None else np.asarray(export_ceiling, dtype=float)
        )
        self.soe_targets = dict(soe_targets or {})
        keys = {d.key for d in self.devices} | {k for k, _ in self.lp_batts}
        pv_keys = set(fc.pv_keys)
        for name, members in (
            [(sm.name, sm.members) for sm in cfg.submeters]
            + [(c.name, c.members) for c in cfg.connections]
            + [(lim.name, lim.members) for lim in cfg.set_limits]
        ):
            unknown = set(members) - keys - pv_keys
            if unknown:
                raise ValueError(f"{name!r}: no device {sorted(unknown)}")
        # The master's coupling rows, n each: every bus (the roots, then the
        # nodes), then every set limit. A device is in its bus's row and in
        # each of its set limits' rows; its price is the sum of their duals.
        self.rows = list(self.topo.buses) + [f"set:{lim.name}" for lim in cfg.set_limits]
        self._row = {name: i for i, name in enumerate(self.rows)}

        self._build_meter_block()
        # Running minimum of (cost + price . power) per device at every price
        # point priced so far, over EVERY plan ever proposed - including ones
        # the pool later forgets. That is what keeps the bound valid when the
        # pool is pruned or aggregated.
        self._priced: list[np.ndarray] = []
        self._seen_min: list[dict[str, float]] = []
        for dev in self.devices:
            dev.on_add = self._note

    # ----------------------------------------------------------- local prices
    def _rows_of(self, key: str) -> list[int]:
        """The coupling rows device `key` is in: its bus's, then each of its set limits'."""
        rows = [self._row[self.cfg.bus_of(key)]]
        rows += [self._row[f"set:{lim.name}"] for lim in self.cfg.set_limits if key in lim.members]
        return rows

    def price_of(self, y: np.ndarray, dev: Device | str) -> np.ndarray:
        """The price device `dev` sees in the dual vector `y` (n per coupling
        row, `self.rows`): its bus's, plus each of its set limits'. Per
        kW-slot, like `y`."""
        n = self.n
        rows = self._rows_of(dev if isinstance(dev, str) else dev.key)
        if len(rows) == 1:
            return y[rows[0] * n : (rows[0] + 1) * n]
        return sum((y[r * n : (r + 1) * n] for r in rows), np.zeros(n))

    def _shown(self, pi: np.ndarray) -> np.ndarray:
        """The duals to report for the master's last solve, `pi`: the ones
        nearest the pass-through prices, if they are of the same solve."""
        raw, centered = getattr(self, "_centered", (None, pi))
        return centered if raw is pi else pi

    def local_prices(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Each bus's price in `y` but the main meter's - a node's, a further
        grid connection's - currency/kWh."""
        n = self.n
        return {
            b: y[self._row[b] * n : (self._row[b] + 1) * n] / self.dt
            for b in self.topo.buses
            if b != MAIN
        }

    def limit_prices(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Each set limit's premium in `y` (its dual), currency/kWh: what its
        members pay on top of their bus's price while it binds."""
        n = self.n
        return {
            lim.name: y[self._row[f"set:{lim.name}"] * n : (self._row[f"set:{lim.name}"] + 1) * n]
            / self.dt
            for lim in self.cfg.set_limits
        }

    # ------------------------------------------------------------------ costs
    def battery_cost(self, key: str, traj: np.ndarray) -> float:
        """The non-bill cost of a battery the master holds, for its stored-energy
        trajectory `traj` (kWh): energy used up, at its terminal price."""
        b = dict(self.lp_batts)[key]
        return float(traj[0] - traj[-1]) * _pinned_price(b)

    # --------------------------------------------------------------- oracles

    def _note(self, dev: Device, col: Column) -> None:
        for p, mins in zip(self._priced, self._seen_min):
            v = col.cost + float(self.price_of(p, dev) @ col.power)
            if v < mins.get(dev.key, np.inf):
                mins[dev.key] = v

    def _ask(self, dev: Device, query: Query, source: str) -> Column:
        """Ask participant `dev` the interface query `query`; return its answer
        as a pool column tagged `source` (how it was proposed: "price",
        "load_aware"). The column is marked inexact when the participant's
        solve failed and it answered with a stand-in plan."""
        a = dev.participant.respond(query)
        return Column(
            np.asarray(a.plan_kw, dtype=float).copy(),
            np.asarray(a.trajectory, dtype=float).copy(),
            float(a.private_cost),
            source,
            detail=a.detail,
            exact=a.status == "ok",
        )

    def price_oracle(self, dev: Device, price_kwh: np.ndarray) -> Column:
        """The Dantzig-Wolfe pricing problem: min private cost + price . power,
        asked of the participant as a price response."""
        return self._ask(dev, Query("price_response", price_kwh, price_kwh), "price")

    def load_aware_oracle(
        self, dev: Device, others: Mapping[str, np.ndarray], price_kwh: np.ndarray
    ) -> Column:
        """A HEURISTIC column: today's load-aware DP against the others' mix
        (`others`: every other device's power, by key).

        Any feasible plan is a legal column, so the master can absorb the
        best response too. The kink makes this column good at
        self-consumption plans that a linear price reaches only by mixing
        many extreme points. Limits reach it the way mu does in coordinate():
        as a premium where the master price exceeds the tariff.

        With sub-meters the rest of the meter is what `site_meter` makes of
        the others (a battery's bus-side power reaches the meter through its
        inverter), and a device behind one meets its bus as well (meter.Bus).
        """
        view = device_bus(self.cfg, self.fc, dev.key, others)
        bus, residual = view.bus, view.dp_load
        buy = np.maximum(view.buy, price_kwh)  # its grid connection's tariff
        sell = np.minimum(view.sell, price_kwh)
        if (
            bus is not None
            or view.root != MAIN
            or not getattr(dev.participant, "takes_residual", True)
        ):
            # Told no household load, it is asked at its connection's
            # marginal prices given everyone else's plans: what one more kWh
            # drawn there costs, what one more supplied is worth - the
            # meter's tariff and every rating on its way in them. Behind a
            # panel that cannot export, supplying is then worth the breach
            # price wherever the panel is full, and it plans accordingly.
            draw, supply = self._marginal_prices(bus, residual, buy, sell)
            return self._ask(dev, Query("price_response", draw, supply), "load_aware")
        return self._ask(dev, Query("best_response", buy, sell, residual_kw=residual), "load_aware")

    def _capped(self, key: str) -> bool:
        """Whether device `key` sits behind a rated node or counts in a set limit."""
        chain = self.topo.chain(self.cfg.bus_of(key))
        return any(
            self.topo.sm[n].max_export_kw is not None or self.topo.sm[n].max_import_kw is not None
            for n in chain
        ) or any(key in lim.members for lim in self.cfg.set_limits)

    def _repair(self, plan: dict[str, Column], integer: str, rounds: int = 3) -> dict[str, Column]:
        """A recovered plan that still breaches a rating, repaired: each device
        behind a rated node is asked again against the plans the others will
        actually run (not a mix of them) - a participant at the secant prices
        of its whole power, so an EV of 7 kW sees that a slot with 3 kW of
        headroom costs it the breach - and the plan is recovered again from
        the larger pool, kept only if it scores better. Up to `rounds` times."""
        for _ in range(rounds):
            if self.parts(plan)["breach_kwh"] <= 1e-6:
                break
            added = 0
            for dev in self.devices:
                if not self._capped(dev.key):
                    continue
                others = {k: c.power for k, c in plan.items() if k != dev.key}
                added += dev.add(self.load_aware_oracle(dev, others, self.fc.buy))
                view = device_bus(self.cfg, self.fc, dev.key, others)
                q = max(float(dev.participant.max_power_kw), 1e-3)
                draw, supply = self._marginal_prices(
                    view.bus, view.dp_load, view.buy, view.sell, eps=q
                )
                added += dev.add(self._ask(dev, Query("price_response", draw, supply), "repair"))
            if not added:
                break
            _, lam, _, _, bp = self.solve_master()
            new = self.recover(integer, self.split_weights(lam), bp)
            if self.score(new) >= self.score(plan) - 1e-9:
                break
            plan = new
        return plan

    def _add_contained(self, dev: Device, price_kwh: np.ndarray) -> int:
        """For a participant behind a node that caps what it may send up: its
        plan at `price_kwh` when supplying is worth the breach price less than
        drawing - one that never sends power up. A plan whose discharge fits
        the others' current mix may not fit the plans recovery picks for them
        (two on/off devices behind a panel that cannot feed back); this one
        fits any. Returns how many columns were added (0 or 1)."""
        capped = any(
            self.topo.sm[node].max_export_kw is not None
            for node in self.topo.chain(self.cfg.bus_of(dev.key))
        )
        if not capped:
            return 0
        penalty = breach_price(self.cfg.grid, self.fc.buy, self.fc.sell)
        return dev.add(
            self._ask(dev, Query("price_response", price_kwh, price_kwh - penalty), "contained")
        )

    def _marginal_prices(
        self,
        bus: Bus | None,
        residual: np.ndarray,
        buy: np.ndarray,
        sell: np.ndarray,
        eps: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per slot, currency/kWh: what one more kWh drawn by a device on
        `bus` (None: at the meter) costs, and what one more kWh it supplies
        is worth, with `residual` the rest of its meter and `buy` / `sell`
        that meter's prices; breach of any rating on the way included."""
        draw, supply = np.empty(self.n), np.empty(self.n)
        for t in range(self.n):
            q = np.array([0.0, eps, -eps])
            if bus is None:
                at_meter, over = q, np.zeros(3)
            else:
                at_meter, over = bus_cost(q, t, bus)
            imp = at_meter + residual[t]
            cost = buy[t] * np.maximum(imp, 0.0) - sell[t] * np.maximum(-imp, 0.0) + over
            draw[t] = (cost[1] - cost[0]) / eps
            supply[t] = (cost[0] - cost[2]) / eps
        return draw, supply

    def seed(self) -> None:
        """Starting columns. Any feasible plan per device keeps the master
        feasible (the meter variables absorb any mismatch), so Phase I is
        free: no artificial columns, no Farkas pricing."""
        fc = self.fc
        for dev in self.devices:
            a = dev.participant.baseline()
            dev.add(
                Column(
                    np.asarray(a.plan_kw, dtype=float),
                    np.asarray(a.trajectory, dtype=float),
                    float(a.private_cost),
                    "baseline",
                    detail=a.detail,
                )
            )
        # The two corners of the dual box: "everything imports" and
        # "everything exports". Cheap and spans the extremes.
        for p in (fc.buy, np.maximum(fc.sell, 0.0)):
            for dev in self.devices:
                dev.add(self.price_oracle(dev, p))

    # ---------------------------------------------------------------- master
    def _build_meter_block(self) -> None:
        """The master's own variables, as families of n (one per slot), each
        with its coefficient in one or two coupling rows, its cost per slot
        and its upper bound: `self.blocks`, a list of ([(row, coef)], cost,
        upper). And the coupling rows' right-hand sides, `self.rhs_bal`.

        At each grid connection (a root): import within / over its limit,
        export within / over it, and curtailment of the PV on it. Breach is
        priced exactly as grid_penalty prices it, so the master minimises
        total_objective. At each node: its link to its parent - sending x
        (bus-side) puts eta_export * x in the parent's row, taking x draws
        x / eta_import from it - within its ratings and, beyond them, at the
        breach price; and the PV clipped on it. At each set limit: its
        members' total within the limit, and beyond it at the breach price.
        """
        g, fc, dt = self.cfg.grid, self.fc, self.dt
        cfg, topo, n = self.cfg, self.topo, self.n
        # No flow can exceed everything the house could draw or push at
        # once. Using that instead of +inf changes no solution, but it matters
        # when export pays exactly the import price: "import one more kWh and
        # export it again" then costs nothing, the optimal set is unbounded,
        # and an interior-point method drifts along it until it stalls.
        reach = (
            np.abs(fc.load)
            + np.abs(fc.total_solar)
            + sum(max(b.p_charge_max_kw, b.p_discharge_max_kw) for b in cfg.battery_list)
            + sum(float(d.participant.max_power_kw) for d in self.devices)
        )
        inf = 2.0 * reach + 1.0
        full = lambda v: np.broadcast_to(v, (n,)).astype(float)  # noqa: E731
        zero = np.zeros(n)
        blocks: list[tuple[list[tuple[int, float]], np.ndarray, np.ndarray]] = []
        rhs = np.zeros(len(self.rows) * n)

        for root in topo.roots:
            r = self._row[root]
            rg = root_grid(cfg, root)
            buy, sell = tariff(cfg, fc, root)
            # breach: a constant price per kWh beyond a limit, as grid_penalty
            B = breach_price(rg, buy, sell) if rg.active else 0.0
            imp_cap = np.minimum(rg.max_import_kw, inf) if rg.max_import_kw is not None else inf
            exp_cap = np.minimum(rg.max_export_kw, inf) if rg.max_export_kw is not None else inf
            if root == MAIN and self.export_ceiling is not None:
                # a hard cap: no over-cap block below lets the master buy past it
                exp_cap = np.minimum(exp_cap, self.export_ceiling)
            blocks.append(([(r, +1.0)], buy * dt, full(imp_cap)))  # import
            if rg.max_import_kw is not None:
                blocks.append(([(r, +1.0)], (buy + B) * dt, full(inf)))  # over import cap
            blocks.append(([(r, -1.0)], -sell * dt, full(exp_cap)))  # export
            if rg.max_export_kw is not None:
                blocks.append(([(r, -1.0)], (B - sell) * dt, full(inf)))  # over export cap
            pv = pv_on(fc, topo, root)
            if rg.allow_curtailment and np.any(pv > 0):
                blocks.append(([(r, -1.0)], zero, pv.copy()))  # curtail
            rhs[r * n : (r + 1) * n] = self.d if root == MAIN else -pv

        Bn = breach_price(g, fc.buy, fc.sell)  # a rating is a limit even with no grid limit
        # node -> the block indices of its link: (send within / past, take within / past)
        self.link_blocks: dict[str, tuple[list[int], list[int]]] = {}
        for node in topo.nodes:
            sm = topo.sm[node]
            r, up = self._row[node], self._row[topo.parent[node]]
            out_cap, in_cap = sm.export_cap_dc, sm.import_cap_dc
            sends, takes = [len(blocks)], []
            blocks.append(
                ([(up, sm.eta_export), (r, -1.0)], zero, np.minimum(full(out_cap), inf))
            )  # send
            if np.isfinite(out_cap):
                sends.append(len(blocks))
                blocks.append(
                    ([(up, sm.eta_export), (r, -1.0)], full(Bn * dt), full(inf))
                )  # past it
            takes.append(len(blocks))
            blocks.append(
                ([(up, -1.0 / sm.eta_import), (r, 1.0)], zero, np.minimum(full(in_cap), inf))
            )  # take
            if np.isfinite(in_cap):
                takes.append(len(blocks))
                blocks.append(
                    ([(up, -1.0 / sm.eta_import), (r, 1.0)], full(Bn * dt), full(inf))
                )  # past it
            self.link_blocks[node] = (sends, takes)
            pv = pv_on(fc, topo, node)
            if np.any(pv > 0):
                # PV clipped on the bus: free; only what the rating forces,
                # unless curtailment is allowed (then any of it)
                clip = pv.copy() if g.allow_curtailment else np.maximum(pv - out_cap, 0.0)
                blocks.append(([(r, -1.0)], zero, clip))
            rhs[r * n : (r + 1) * n] = -pv

        for lim in cfg.set_limits:
            # its row: imp - exp (+ over imp - over exp) = its members' total
            r = self._row[f"set:{lim.name}"]
            imp_cap = full(inf if lim.max_import_kw is None else np.minimum(lim.max_import_kw, inf))
            exp_cap = full(inf if lim.max_export_kw is None else np.minimum(lim.max_export_kw, inf))
            blocks.append(([(r, +1.0)], zero, imp_cap))
            if lim.max_import_kw is not None:
                blocks.append(([(r, +1.0)], full(Bn * dt), full(inf)))
            blocks.append(([(r, -1.0)], zero, exp_cap))
            if lim.max_export_kw is not None:
                blocks.append(([(r, -1.0)], full(Bn * dt), full(inf)))
        self.blocks = blocks
        self.rhs_bal = rhs
        # the main meter's own, as earlier releases exposed them: (coef, cost, upper)
        self.meter_blocks = [
            (rc[0][1], cost, up) for rc, cost, up in blocks if rc[0][0] == 0 and len(rc) == 1
        ]

    # ---------------------------------------------------------------- solvers
    def _lp(
        self,
        c: np.ndarray,
        A: Triplets,
        b: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        border: int = 0,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """min c.x s.t. A x = b, lb <= x <= ub -> (x, fun, equality duals)."""
        if self.solver == "highs":
            from scipy import sparse
            from scipy.optimize import linprog

            r, k, v = A.coo()
            M = sparse.csr_matrix((v, (r, k)), shape=A.shape)
            res = linprog(c, A_eq=M, b_eq=b, bounds=list(zip(lb, ub)), method="highs")
            if res.status != 0:
                raise RuntimeError(f"LP failed: {res.message}")
            return res.x, res.fun, res.eqlin.marginals
        res = np_linprog(c, A, b, lb, ub, border=border)
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        return res.x, res.fun, res.y

    def _choose(
        self,
        c: np.ndarray,
        A: Triplets,
        b: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        groups: list[np.ndarray],
        time_limit: float,
        node_limit: int = 150,
        border: int = 0,
    ) -> tuple[np.ndarray, float]:
        """The recovery MILP: one plan per group (HiGHS). -> (x, fun)."""
        from scipy import sparse
        from scipy.optimize import Bounds, LinearConstraint, milp

        r, k, v = A.coo()
        M = sparse.csr_matrix((v, (r, k)), shape=A.shape)
        integrality = np.zeros(len(c))
        for g in groups:
            integrality[g] = 1
        res = milp(
            c,
            constraints=LinearConstraint(M, b, b),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": time_limit, "mip_rel_gap": 1e-6},
        )
        if res.x is None:
            raise RuntimeError(f"integer master failed: {res.message}")
        return res.x, res.fun

    @overload
    def solve_master(
        self, integer: Literal[False] = ..., time_limit: float = ..., node_limit: int = ...
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Column]]: ...
    @overload
    def solve_master(
        self, integer: Literal[True], time_limit: float = ..., node_limit: int = ...
    ) -> tuple[float, np.ndarray, None, None, dict[str, Column]]: ...
    def solve_master(
        self, integer: bool = False, time_limit: float = 30.0, node_limit: int = 150
    ) -> tuple[float, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Column]]:
        """Returns (value, column weights, meter duals, convexity duals, battery plans)."""
        n, devs, dt = self.n, self.devices, self.dt
        # Rows: the coupling rows (every bus, then every set limit; n each),
        # the convexity rows, then the batteries' and the tank's own rows.
        # The first two are the border that couples the rest.
        n_bal = n * len(self.rows)
        n_meter = len(self.blocks) * n
        n_cols = sum(len(d.columns) for d in devs)
        nb = len(self.lp_batts)
        # per LP battery: charge c[n], discharge e[n], soe s[1..n]
        nv = n_meter + n_cols + 3 * n * nb
        n_rows = n_bal + len(devs) + n * nb

        c = np.zeros(nv)
        ub = np.full(nv, np.inf)
        lb = np.zeros(nv)
        # Coordinate triplets: a 10-battery, 48 h site is ~2k rows x ~7k
        # columns, which dense would put at ~100 MB.
        A = Triplets((n_rows, nv))
        rhs = np.zeros(n_rows)
        rhs[:n_bal] = self.rhs_bal
        rhs[n_bal : n_bal + len(devs)] = 1.0
        t_all = np.arange(n)
        j = 0
        for coefs, cost, upper in self.blocks:
            c[j : j + n] = cost
            ub[j : j + n] = upper
            for row, coef in coefs:
                A[row * n + t_all, j + t_all] = coef
            j += n
        for b, shut in getattr(self, "link_fix", {}).items():
            ub[b * n : (b + 1) * n][shut] = 0.0  # a link's direction, fixed for recovery
        for di, dev in enumerate(devs):
            rows = self._rows_of(dev.key)
            for col in dev.columns:
                c[j] = col.cost
                for row in rows:
                    A[row * n : (row + 1) * n, j] = -col.power.reshape(-1, 1)
                A[n_bal + di, j] = 1.0
                # An aggregate an on/off device cannot run is fine for the LP
                # (it is a point of the convex hull) but not for recovery.
                ub[j] = 1.0 if (not integer or col.runnable or dev.modulating) else 0.0
                j += 1
        base_row = n_bal + len(devs)
        batt_cols: list[tuple[str, BatteryConfig, int, int, int]] = []
        for bi, (key, b) in enumerate(self.lp_batts):
            C0, E0, S0 = j, j + n, j + 2 * n
            j += 3 * n
            batt_cols.append((key, b, C0, E0, S0))
            ub[C0 : C0 + n] = b.p_charge_max_kw
            ub[E0 : E0 + n] = b.p_discharge_max_kw
            ub[S0 : S0 + n] = b.capacity_kwh
            lb[S0 : S0 + n] = b.soe_floor_kwh
            if key in self.soe_targets:
                lb[S0 + n - 1] = ub[S0 + n - 1] = self.soe_targets[key]
            t = np.arange(n)
            for row in self._rows_of(key):  # its bus's row, and its set limits'
                A[row * n + t, C0 + t] = -1.0  # minus consumption
                A[row * n + t, E0 + t] = +1.0
            r = base_row + bi * n + t  # s[t+1] - s[t] - dt*eta_c*c + dt*e/eta_d = 0
            A[r, S0 + t] = 1.0
            A[r[1:], S0 + t[:-1]] = -1.0
            A[r, C0 + t] = -dt * b.eta_c
            A[r, E0 + t] = dt / b.eta_d
            s0 = b.capacity_kwh * b.soc_initial_frac
            rhs[r[0]] = s0
            # terminal: bill the net depletion at the terminal price
            c[S0 + n - 1] = -_pinned_price(b)

        const = sum(
            _pinned_price(b) * b.capacity_kwh * b.soc_initial_frac for _, b in self.lp_batts
        )
        lam_sl = slice(n_meter, n_meter + n_cols)

        def batt_plans(x: np.ndarray) -> dict[str, Column]:
            out = {}
            for key, b, C0, E0, S0 in batt_cols:
                s0 = b.capacity_kwh * b.soc_initial_frac
                traj = np.concatenate([[s0], x[S0 : S0 + n]])
                out[key] = Column(
                    x[C0 : C0 + n] - x[E0 : E0 + n], traj, (s0 - traj[-1]) * _pinned_price(b), "lp"
                )
            return out

        if integer:
            groups, j = [], n_meter
            for dev in devs:
                if not dev.modulating:
                    groups.append(np.arange(j, j + len(dev.columns)))
                j += len(dev.columns)
            # the meter + convexity rows are the border that couples the
            # batteries' and the tank's otherwise independent row blocks
            x, fun = self._choose(c, A, rhs, lb, ub, groups, time_limit, node_limit, base_row)
            self._note_links(x)
            return fun + const, x[lam_sl], None, None, batt_plans(x)

        x, fun, duals = self._lp(c, A, rhs, lb, ub, border=base_row)
        self._note_links(x)
        # d(cost)/d(d_t), currency per kW-slot: the meter's, then each bus's.
        # Every price this class hands around is this whole vector; price_of
        # picks a device's own.
        pi = duals[:n_bal]
        # the same optimum's duals nearest the pass-through prices: a second
        # point to price at, and the local prices shown (_shown)
        self._centered = (
            pi,
            self._center_duals(c, A, lb, ub, x, duals)[:n_bal]
            if self.topo.nodes or self.cfg.set_limits
            else pi,
        )
        sigma = duals[n_bal : n_bal + len(devs)]
        self.batt_lambda = {  # SoE-balance duals: the exact LP costate
            key: duals[base_row + bi * n : base_row + (bi + 1) * n]
            for bi, (key, *_rest) in enumerate(batt_cols)
        }
        return fun + const, x[lam_sl], pi, sigma, batt_plans(x)

    def _center_duals(
        self,
        c: np.ndarray,
        A: Triplets,
        lb: np.ndarray,
        ub: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Among the master's optimal duals, the one nearest the pass-through
        prices: each node's price within its connection's band of its
        parent's, [eta_export, 1/eta_import] times it, and each set limit's
        premium at 0 - as far as staying optimal allows.

        A node's price is not unique while its connection carries nothing:
        behind a panel that cannot export, with nothing there drawing, any
        price from the parent's minus the breach price up to the parent's is
        optimal. A solver picks one end, the devices there are then priced
        far below the meter, and the loop chases that price for dozens of
        iterations without closing the gap. The loop prices at this point as
        well as at the solver's (the solver's own, an interior point's near
        the centre of the optimal set, is often the better guide elsewhere),
        and reports it as the local prices.

        Parents before children, a slot at a time, a node moves with every
        node below it whose link is tight (a link variable at zero reduced
        cost - flowing, or about to), each scaled by its link's efficiency
        so those links stay tight; otherwise raising a panel would make its
        breaker's idle link worth using. The step keeps every reduced cost's
        sign (a variable strictly between its bounds pins it), so the vector
        stays dual-feasible and complementary to x.
        """
        n, topo = self.n, self.topo
        rows, cols, vals = A.coo()
        y = y.copy()
        red = c - np.bincount(cols, weights=vals * y[rows], minlength=len(c))
        order = np.argsort(rows, kind="stable")
        rows, cols, vals = rows[order], cols[order], vals[order]
        start = np.searchsorted(rows, np.arange(A.shape[0] + 1))
        tol_r = 1e-7
        tol_x = 1e-6 * (
            1.0
            + np.minimum(
                np.abs(np.where(np.isfinite(lb), lb, 0.0)),
                np.abs(np.where(np.isfinite(ub), ub, 0.0)),
            )
        )
        free = ub - lb > tol_x
        # +1: holds at its lower bound (keep red >= 0), -1: at its upper
        # (keep red <= 0), 0: strictly between (pins every row it is in)
        side = np.where(
            red > tol_r,
            1,
            np.where(
                red < -tol_r, -1, np.where(x - lb <= tol_x, 1, np.where(ub - x <= tol_x, -1, 0))
            ),
        )

        def tight(node: str, t: int) -> float:
            """The factor a tight link scales its node's step by, or 0."""
            sends, takes = self.link_blocks[node]
            sm = topo.sm[node]
            if any(abs(red[b * n + t]) <= tol_r for b in sends):
                return sm.eta_export
            if any(abs(red[b * n + t]) <= tol_r for b in takes):
                return 1.0 / sm.eta_import
            return 0.0

        def move(at: list[tuple[int, float]], target: float) -> None:
            """Shift y[i] by f * d for each (i, f): d toward target - y[at[0]]."""
            k = np.concatenate([np.arange(start[i], start[i + 1]) for i, _ in at])
            f = np.concatenate([np.full(start[i + 1] - start[i], fi) for i, fi in at])
            J, inv = np.unique(cols[k], return_inverse=True)
            a = np.bincount(inv, weights=vals[k] * f, minlength=len(J))
            on = free[J] & (np.abs(a) > 1e-12)
            J, a = J[on], a[on]
            s = side[J]
            if np.any(s == 0):
                return
            r = red[J]
            # red_j - a_j * d keeps its sign: a bound on d from each
            bound = np.where(s > 0, np.maximum(r, 0.0), np.minimum(r, 0.0)) / a
            upper = (s > 0) == (a > 0)
            hi = float(np.min(bound[upper], initial=np.inf))
            lo = float(np.max(bound[~upper], initial=-np.inf))
            d = min(max(target - y[at[0][0]], min(lo, 0.0)), max(hi, 0.0))
            if d != 0.0:
                for i, fi in at:
                    y[i] += fi * d
                red[J] -= a * d

        for node in reversed(topo.order):  # parents first
            sm = topo.sm[node]
            me, up = self._row[node] * n, self._row[topo.parent[node]] * n
            for t in range(n):
                at, stack = [(me + t, 1.0)], [(node, 1.0)]
                while stack:
                    b, fb = stack.pop()
                    for ch in topo.children[b]:
                        f_ch = tight(ch, t) * fb
                        if f_ch:
                            at.append((self._row[ch] * n + t, f_ch))
                            stack.append((ch, f_ch))
                p = y[up + t]
                band = sorted((sm.eta_export * p, p / sm.eta_import))
                move(at, min(max(y[me + t], band[0]), band[1]))
        for lim in self.cfg.set_limits:
            me = self._row[f"set:{lim.name}"] * n
            for t in range(n):
                move([(me + t, 1.0)], 0.0)
        return y

    def split_weights(self, lam: np.ndarray) -> dict[str, np.ndarray]:
        out, j = {}, 0
        for dev in self.devices:
            out[dev.key] = lam[j : j + len(dev.columns)]
            j += len(dev.columns)
        return out

    def blend_plan(self, dev: Device, w: np.ndarray) -> Column:
        """Execute a weighted mix of a modulating device's plans.

        The mixed power is re-run through the device's physics, so the
        trajectory and cost are what it actually produces - exact even where a
        mix would carry a tank past its cut-out or a battery past full.
        """
        power = sum(wi * c.power for wi, c in zip(w, dev.columns))
        mix = [(float(wi), c.source, c.born) for wi, c in zip(w, dev.columns) if wi > 1e-6]
        # A participant that says it modulates has linear dynamics and a
        # convex cost: the mix of its plans runs as it is (within each slot,
        # as a share of the slot per plan), at no more than the mixed cost.
        used = [(wi, c) for wi, c in zip(w, dev.columns) if wi > 1e-12]
        blend = getattr(dev.cfg, "blend", None)  # optional (interface.Participant)
        detail = (
            blend([wi for wi, _ in used], [c.detail for _, c in used])
            if blend is not None
            else None
        )
        return Column(
            power,
            sum(wi * c.trajectory for wi, c in used),
            float(sum(wi * c.cost for wi, c in used)),
            "blend",
            mix,
            detail=detail,
        )

    def mixed_power(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            dev.key: sum(w * col.power for w, col in zip(weights[dev.key], dev.columns))
            for dev in self.devices
        }

    # --------------------------------------------------------------- bounds
    def lagrangian(self, pi: np.ndarray, best: dict[str, Column]) -> float:
        """L(pi) = pi.d + sum_k min_x (f_k + pi.p_k) + sum_t min_meter(...).

        Valid at ANY pi, not only the master's, which is what lets the
        smoothed price produce a bound too. -inf outside the dual box.
        """
        n = self.n
        total = float(pi[: len(self.rows) * n] @ self.rhs_bal)
        for dev in self.devices:
            col = best[dev.key]
            total += col.cost + float(self.price_of(pi, dev) @ col.power)
        boxes = []
        for coefs, cost, upper in self.blocks:
            red = cost - sum((pi[r * n : (r + 1) * n] * coef for r, coef in coefs), np.zeros(n))
            boxes.append((red, upper))
        for red, upper in boxes:
            if np.any((red < -1e-12) & ~np.isfinite(upper)):
                return -np.inf
            total += float(np.sum(np.minimum(0.0, red * np.where(np.isfinite(upper), upper, 0.0))))
        return total

    def lagrangian_lp(self, pi: np.ndarray, best: dict[str, Column]) -> float:
        """L(pi) when batteries live in the master: each battery contributes
        min over its own polytope of (terminal cost + pi . p), a tiny LP."""
        total = self.lagrangian(pi, best)
        if not np.isfinite(total):
            return total
        n, dt = self.n, self.dt
        for key, b in self.lp_batts:
            # vars c[n], e[n], s[n]
            nv = 3 * n
            cc = np.zeros(nv)
            price = self.price_of(pi, key)
            cc[:n] = price
            cc[n : 2 * n] = -price
            cc[3 * n - 1] = -_pinned_price(b)
            A = Triplets((n, nv))
            t = np.arange(n)
            A[t, 2 * n + t] = 1.0
            A[t[1:], 2 * n + t[:-1]] = -1.0
            A[t, t] = -dt * b.eta_c
            A[t, n + t] = dt / b.eta_d
            s0 = b.capacity_kwh * b.soc_initial_frac
            rhs = np.zeros(n)
            rhs[0] = s0
            ub = np.concatenate(
                [
                    np.full(n, b.p_charge_max_kw),
                    np.full(n, b.p_discharge_max_kw),
                    np.full(n, b.capacity_kwh),
                ]
            )
            lbb = np.zeros(nv)
            lbb[2 * n :] = b.soe_floor_kwh
            if key in self.soe_targets:
                lbb[3 * n - 1] = ub[3 * n - 1] = self.soe_targets[key]
            _, fun, _ = self._lp(cc, A, rhs, lbb, ub)
            total += fun + _pinned_price(b) * s0
        return total

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, powers: dict[str, np.ndarray], trajs: dict[str, np.ndarray]) -> float:
        """Score a plan exactly as coordinate() scores its rounds."""
        flows = site_meter(self.cfg, self.fc, powers)
        net, curtail = flows.net, flows.curtail
        if self.export_ceiling is not None:
            # A hard cap: what is exported past it must be curtailed PV, as the
            # master would; a plan that cannot is not runnable.
            over = np.maximum(-net - self.export_ceiling, 0.0)
            if np.any(over > 1e-6):
                if not (
                    self.cfg.grid.allow_curtailment
                    and np.all(over <= self.fc.solar - curtail + 1e-6)
                ):
                    return np.inf
                net = net + over
        return extended_objective(self.cfg, self.fc, net, trajs) + beyond_main(
            self.cfg, self.fc, flows
        )

    def participants_cost(self, plan: dict[str, Column]) -> float:
        """The sum of the participants' private costs (currency) in `plan`
        (device key -> chosen column), as their own models priced them. The
        site's own devices are not included: `evaluate` prices those."""
        return float(sum(plan[d.key].cost for d in self.devices if d.key in plan))

    def score(self, plan: dict[str, Column]) -> float:
        """The objective (currency) of `plan` (device key -> chosen column):
        the bill and the site's own devices' costs (`evaluate`), plus the
        participants' private costs. Infinite if the plan breaks the export
        ceiling and cannot curtail its way under it."""
        return self.evaluate(
            {k: c.power for k, c in plan.items()}, {k: c.trajectory for k, c in plan.items()}
        ) + self.participants_cost(plan)

    def _note_links(self, x: np.ndarray) -> None:
        """Record each node's link flows in master solution `x`: node ->
        (sent, taken), bus-side kW per slot."""
        n = self.n
        self.link_flows = {
            node: (
                sum((x[b * n : (b + 1) * n] for b in sends), np.zeros(n)),
                sum((x[b * n : (b + 1) * n] for b in takes), np.zeros(n)),
            )
            for node, (sends, takes) in self.link_blocks.items()
        }

    def _fix_directions(self) -> bool:
        """Where the last master solve both sent and took on a lossy link in
        one slot - burning energy in conversion losses, which the LP allows
        (theory, "One relaxation") but no converter can do - fix that slot's
        direction to the net flow, for recovery. Returns whether any was fixed."""
        fixed = False
        fix: dict[int, np.ndarray] = dict(getattr(self, "link_fix", {}))
        for node, (sent, taken) in getattr(self, "link_flows", {}).items():
            loop = np.minimum(sent, taken) > 1e-6
            if not np.any(loop):
                continue
            sends, takes = self.link_blocks[node]
            shut_take, shut_send = loop & (sent >= taken), loop & (sent < taken)
            for b in takes:
                fix[b] = fix.get(b, np.zeros(self.n, dtype=bool)) | shut_take
            for b in sends:
                fix[b] = fix.get(b, np.zeros(self.n, dtype=bool)) | shut_send
            fixed = True
        self.link_fix = fix
        return fixed

    def recover(
        self,
        integer: str,
        weights: dict[str, np.ndarray] | None = None,
        bplans: dict[str, Column] | None = None,
        time_limit: float = 30.0,
        node_limit: int = 150,
    ) -> dict[str, Column]:
        """One implementable plan per device from the CURRENT pool.

        milp: choose jointly - binary weights for devices that cannot run a
        blend, continuous for modulating ones, batteries re-optimised in the
        same solve. maxweight: each device's heaviest column in `weights`
        (the relaxed master's), batteries as the master left them.
        """

        def pick(dev: Device, w: np.ndarray) -> Column:
            if dev.modulating and np.sum(w > 1e-6) > 1:
                return self.blend_plan(dev, w)
            runnable = np.array([wi if c.runnable else -1.0 for wi, c in zip(w, dev.columns)])
            k = int(np.argmax(runnable))
            if dev.modulating and dev.columns[k].source == "aggregate":
                # an aggregate carries an averaged trajectory: re-run its power
                # through the physics so the plan is scored as it would run
                one = np.zeros(len(dev.columns))
                one[k] = 1.0
                return self.blend_plan(dev, one)
            return dev.columns[k]

        if integer == "milp":
            try:
                _, lam_int, _, _, bplans_int = self.solve_master(
                    integer=True, time_limit=time_limit, node_limit=node_limit
                )
            except RuntimeError:
                # No integer solution within the time limit (large pools):
                # fall back to the heaviest runnable column per device.
                integer, lam_int = "maxweight", None
        if integer == "milp" and lam_int is not None:
            w_int = self.split_weights(lam_int)
            plan = {d.key: pick(d, w_int[d.key]) for d in self.devices}
            plan.update(bplans_int)
        else:
            if weights is None or bplans is None:
                _, lam, _, _, bplans = self.solve_master()
                weights = self.split_weights(lam)
            plan = {d.key: pick(d, weights[d.key]) for d in self.devices}
            plan.update(bplans)
        return plan

    def parts(self, plan: dict[str, Column]) -> dict:
        """The objective of a plan, term by term - what `evaluate` sums."""
        flows = site_meter(self.cfg, self.fc, {k: c.power for k, c in plan.items()})
        net = flows.net
        trajs = {k: c.trajectory for k, c in plan.items()}
        fc, dt = self.fc, self.dt
        bill = float(
            np.sum(np.maximum(net, 0) * fc.buy * dt) - np.sum(np.maximum(-net, 0) * fc.sell * dt)
        )
        bill += extra_bills(self.cfg, fc, flows) if self.cfg.connections else 0.0
        breach = grid_penalty(self.cfg, net, fc.buy, fc.sell)
        if self.cfg.has_tree:
            breach += beyond_main(self.cfg, fc, flows) - (
                extra_bills(self.cfg, fc, flows) if self.cfg.connections else 0.0
            )
        batt = battery_terminal_penalty(
            self.cfg, {k: v for k, v in trajs.items() if k.startswith("battery")} or None, fc.buy
        )
        part = self.participants_cost(plan)
        return {
            "bill": bill,
            "breach": breach,
            "battery_terminal": batt,
            "participants": part,
            "total": bill + breach + batt + part,
            "breach_kwh": float(np.sum(breach_energy(self.cfg, net)) + np.sum(flows.breach_kwh)),
            "peak_import": float(net.max()),
            "peak_export": float(-net.min()),
        }

    def polish(self, plan: dict[str, Column], sweeps: int = 3) -> tuple[dict[str, Column], float]:
        """Gauss-Seidel load-aware best response from the recovered plan.

        Accept a device's new plan only when the TRUE objective falls, so this
        is monotone - it cannot undo what the master found.
        """
        plan = dict(plan)
        cur = self.score(plan)
        for _ in range(sweeps):
            improved = False
            for dev in self.devices:
                others = {k: c.power for k, c in plan.items() if k != dev.key}
                cand = self.load_aware_oracle(dev, others, self.fc.buy)
                trial = dict(plan)
                trial[dev.key] = cand
                val = self.score(trial)
                if val < cur - 1e-9:
                    plan, cur, improved = trial, val, True
                    dev.add(cand)
            if not improved:
                break
        return plan, cur

    def _dive(self, iters: int, heuristic_columns: bool, history: list, t0: float) -> None:
        """Fix every on/off device to the plan the integer master picks, then
        price the others for up to `iters` more iterations, so their plans answer
        the on/off plans that will run rather than a fractional mix of them. A
        participant says whether it is on/off (`onoff`); a device of the site's
        own is when it cannot run a blend. The bound is unaffected: it is
        re-stated from every plan ever proposed (`_seen_min`).

        `iters`: the most pricing rounds after fixing. `heuristic_columns`: also
        ask each free device its best response to the rest (as `run` does).
        `history`: `run`'s per-iteration log, appended to ("dive 1", ...). `t0`:
        `run`'s start time (perf_counter), for the log's elapsed ms. Changes the
        device pools in place; returns nothing. Does nothing if the integer
        master fails, or if every device or none is on/off."""
        try:
            _, lam_int, _, _, _ = self.solve_master(integer=True)
        except RuntimeError:
            return
        w_int = self.split_weights(lam_int)
        onoff = lambda d: bool(getattr(d.cfg, "onoff", not d.modulating))  # noqa: E731
        fixed = {d.key for d in self.devices if onoff(d)}
        if not fixed or len(fixed) == len(self.devices):
            return
        for d in self.devices:
            if d.key in fixed:
                runnable = np.array(
                    [wi if c.runnable else -1.0 for wi, c in zip(w_int[d.key], d.columns)]
                )
                d.columns = [d.columns[int(np.argmax(runnable))]]
        dt = self.dt
        for k in range(iters):
            rmp, lam, pi, _, bplans = self.solve_master()
            weights = self.split_weights(lam)
            mix = self.mixed_power(weights)
            powers = {**mix, **{k: c.power for k, c in bplans.items()}}
            added = 0
            for d in self.devices:
                if d.key not in fixed:
                    added += d.add(self.price_oracle(d, self.price_of(pi, d) / dt))
                    if heuristic_columns:
                        others = {k: v for k, v in powers.items() if k != d.key}
                        added += d.add(self.load_aware_oracle(d, others, self.price_of(pi, d) / dt))
                        added += self._add_contained(d, self.price_of(pi, d) / dt)
            history.append(
                {
                    "iter": f"dive {k + 1}",
                    "rmp": rmp,
                    "lb": history[-1]["lb"] if history else -np.inf,
                    "gap": np.nan,
                    "columns": sum(len(d.columns) for d in self.devices),
                    "added": added,
                    "ms": (time.perf_counter() - t0) * 1000,
                }
            )
            if added == 0:
                break

    def extra_price_points(
        self, pi: np.ndarray, center: np.ndarray | None, it: int
    ) -> list[np.ndarray]:
        """Further meter prices to ask every device at, this iteration.

        None by default. Any price is a valid question - the answer is a
        feasible plan, and its Lagrangian value is a valid bound - so a
        subclass can add, e.g., momentum points.
        """
        return []

    # ------------------------------------------------------------------ run
    def run(
        self,
        max_iter: int = 40,
        gap_tol: float = 1e-3,
        smoothing: float | str = "auto",  # "auto": 0.8 if a plain battery bids plans, else 0.5
        heuristic_columns: bool = True,
        integer: str = "milp",  # milp | maxweight
        polish: bool = True,
        verbose: bool = False,
        record: bool = False,  # keep a per-iteration snapshot (for the GUI)
        anytime: bool = False,  # also recover a plan at every iteration
        pool: str = "auto",  # active: what the master uses + the newest | full: keep all | auto (below)
        stall: int = 0,  # stop once the master's value has not fallen for this many
        # iterations (0: never). For participants whose answers give a
        # weak bound, where the gap test alone runs to max_iter.
        dive: int = 0,  # then fix each on/off device to the plan the integer master
        # picks and re-price the modulating ones this many more
        # iterations (0: no dive). Their pools were built against a
        # fractional mix of the on/off plans, which cannot run.
        progress: Callable[..., None] | None = None,
        # progress(it, max_iter) as each iteration starts, then
        # progress(it, max_iter, blend, bound, best runnable) as it ends
        # (None where not known yet); (-1, max_iter) when finishing
    ) -> DWResult:
        t0 = time.perf_counter()
        # "auto": a plain battery that bids plans (batteries off the LP)
        # converges slowly - its optimum lies inside its polytope, which the
        # master rebuilds one plan at a time. Keeping every plan and pricing
        # nearer the best-bound prices closed the gap 0.12 -> 0.015 and
        # 0.25 -> 0.029 in 45 iterations. Otherwise the
        # bounded pool and the lighter smoothing are as good and leaner.
        bidding_battery = False  # batteries are held in the master
        if pool == "auto":
            pool = "full" if bidding_battery else "active"
        smooth = (0.8 if bidding_battery else 0.5) if smoothing == "auto" else float(smoothing)
        self.pool_used, self.smoothing_used = pool, smooth
        self.seed()
        dt = self.dt
        best_lb, center = -np.inf, None
        priced: list[np.ndarray] = []
        history: list[dict] = []
        it = 0
        stop_reason = "iteration cap"
        best_plan_val = np.inf
        best_rmp, flat = np.inf, 0
        incumbent = None  # best runnable plan seen at any iteration
        for it in range(1, max_iter + 1):
            if progress is not None:
                progress(it, max_iter)
            rmp, lam, pi, _sigma, bplans = self.solve_master()
            weights = self.split_weights(lam)
            mix = self.mixed_power(weights)
            if pool == "active":
                # Bounded memory: a column the master gives zero weight is not
                # needed to represent the current solution, only (maybe) a
                # later one. Keep the support (at most rows-many, by LP basis
                # theory) plus last iteration's proposals, drop the rest.
                for dev in self.devices:
                    w = weights[dev.key]
                    keep = [c for c, wi in zip(dev.columns, w) if wi > 1e-9 or c.born >= it - 1]
                    weights[dev.key] = np.array(
                        [wi for c, wi in zip(dev.columns, w) if wi > 1e-9 or c.born >= it - 1]
                    )
                    dev.columns = keep
                self.peak_pool = max(
                    getattr(self, "peak_pool", 0), sum(len(d.columns) for d in self.devices)
                )
            for dev in self.devices:
                dev.stamp = it

            # "If we stopped now": recover from the pool as it stands. Costs a
            # small MILP per iteration, so only when asked (the GUI asks).
            anytime_rec = {}
            if anytime:
                # quick: a few nodes only - the incumbent keeps the best anyway
                snap_plan = self.recover(integer, weights, bplans, time_limit=0.5, node_limit=6)
                pp = self.parts(snap_plan)
                if pp["total"] < best_plan_val:
                    best_plan_val, incumbent = pp["total"], dict(snap_plan)
                anytime_rec = {
                    "plan_value": pp["total"],
                    "plan_parts": pp,
                    "best_plan": best_plan_val,
                }

            # Price at the master's dual, and (Wentges) at a point pulled toward
            # the best-bound centre. Both give valid columns AND valid bounds.
            points = [pi]
            if np.max(np.abs(self._shown(pi) - pi), initial=0.0) > 1e-9:
                points.append(self._shown(pi))
            if center is not None and smooth > 0:
                points.append(smooth * center + (1 - smooth) * pi)
            points.extend(self.extra_price_points(pi, center, it))

            added = 0
            for p in points:
                best = {}
                exact = True
                for dev in self.devices:
                    q = self.price_of(p, dev)
                    col = self.price_oracle(dev, q / dt)
                    # The DP is an exact pricing oracle only up to its grid:
                    # a pool column can undercut it (measured: HVAC, 0.0026 at
                    # a binding import limit). Take the min over both, so the
                    # bound never exceeds the master; record the shortfall,
                    # because it is how far the certificate itself can be off.
                    pool_best = min(dev.columns, key=lambda c: c.cost + q @ c.power)
                    shortfall = (col.cost + q @ col.power) - (pool_best.cost + q @ pool_best.power)
                    self.pricing_error = max(getattr(self, "pricing_error", 0.0), float(shortfall))
                    best[dev.key] = col if shortfall <= 0 else pool_best
                    exact = exact and col.exact

                    added += dev.add(col)
                if not exact:
                    # a participant's solve failed here: its stand-in is a plan,
                    # not its optimum, so this price proves no bound
                    continue
                lb = self.lagrangian_lp(p, best) if self.lp_batts else self.lagrangian(p, best)
                priced.append(p.copy())
                self._priced.append(p.copy())
                self._seen_min.append(
                    {
                        d.key: min(c.cost + float(self.price_of(p, d) @ c.power) for c in d.columns)
                        for d in self.devices
                    }
                )
                if lb > best_lb:
                    best_lb, center = lb, p.copy()

            if heuristic_columns:
                powers = {**mix, **{k: c.power for k, c in bplans.items()}}
                for dev in self.devices:
                    others = {k: v for k, v in powers.items() if k != dev.key}
                    added += dev.add(
                        self.load_aware_oracle(dev, others, self.price_of(pi, dev) / dt)
                    )
                    added += self._add_contained(dev, self.price_of(pi, dev) / dt)

            gap = rmp - best_lb
            history.append(
                {
                    "iter": it,
                    "rmp": rmp,
                    "lb": best_lb,
                    "gap": gap,
                    "columns": sum(len(d.columns) for d in self.devices),
                    "added": added,
                    "ms": (time.perf_counter() - t0) * 1000,
                    **anytime_rec,
                }
            )
            if verbose:
                print(
                    f"  it {it:2d}  rmp {rmp:9.4f}  lb {best_lb:9.4f}  gap {gap:8.4f}  cols {history[-1]['columns']}"
                )
            if progress is not None:
                fin = lambda v: float(v) if np.isfinite(v) else None  # noqa: E731
                progress(it, max_iter, fin(rmp), fin(best_lb), fin(best_plan_val))
            if gap <= gap_tol * max(1.0, abs(rmp)):
                stop_reason = "converged"
                break
            if rmp < best_rmp - 1e-6 * max(1.0, abs(rmp)):
                best_rmp, flat = rmp, 0
            else:
                flat += 1
            if stall and flat >= stall:
                stop_reason = "stalled"
                break
            if added == 0:
                stop_reason = "no new proposals"
                break

        if dive:
            self._dive(dive, heuristic_columns, history, t0)
        if progress is not None:
            progress(-1, max_iter)
        relaxed, lam, pi, _, bplans = self.solve_master()
        shown = self._shown(pi)  # before recovery solves the master again

        # Re-state the bound against the FINAL pool. Columns added after a
        # point was priced (heuristic ones especially) can undercut the DP's
        # answer there; with the min taken over the whole pool, L(pi) <= the
        # final master value by weak duality, so the reported ordering holds.
        # What remains unguarded is a plan NO oracle proposed that beats the
        # DP - the DP's own discretisation error (see pricing_error).
        best_lb = -np.inf
        for i, p in enumerate(priced):
            best = {}
            for d in self.devices:
                # a stand-in column carrying the minimum value at this price
                v = min(
                    self._seen_min[i][d.key],
                    min(c.cost + float(self.price_of(p, d) @ c.power) for c in d.columns),
                )
                best[d.key] = Column(np.zeros(self.n), np.zeros(self.n + 1), v, "min")
            lb = self.lagrangian_lp(p, best) if self.lp_batts else self.lagrangian(p, best)
            best_lb = max(best_lb, lb)
        weights = self.split_weights(lam)
        frac = sum(int(np.sum(w > 1e-6) > 1) for w in weights.values())
        relaxed_view = None

        # ---- primal recovery -------------------------------------------------
        # A modulating device keeps its blend (continuous weights in the MILP
        # too); everything else is rounded to one plan.
        for dev in self.devices:
            dev.stamp = -1  # anything added from here on is polish's
        plan = self.recover(integer, weights, bplans)
        # A lossy link used both ways in a slot is no plan a converter can run:
        # fix those directions and recover again (the bound is unaffected).
        for _ in range(5):
            if not self._fix_directions():
                break
            _, lam_f, _, _, bplans_f = self.solve_master()
            plan = self.recover(integer, self.split_weights(lam_f), bplans_f)
        self.link_fix = {}
        plan = self._repair(plan, integer)
        upper = self.score(plan)
        if polish:
            plan, upper = self.polish(plan)
        # Keep the incumbent. Recovery from the final pool is a heuristic
        # (exactly so for "maxweight"), and can return a worse plan than one
        # already seen: measured 1.168 returned against 0.637 at iteration 2.
        incumbent_used = False
        if incumbent is not None and best_plan_val < upper - 1e-9:
            plan, upper, incumbent_used = incumbent, best_plan_val, True

        return DWResult(
            upper=upper,
            lower=best_lb,
            relaxed=relaxed,
            iterations=it,
            plan=plan,
            weights=weights,
            prices=pi[: self.n] / dt,
            history=history,
            ms=(time.perf_counter() - t0) * 1000,
            n_columns=sum(len(d.columns) for d in self.devices),
            fractional_devices=frac,
            pricing_error=getattr(self, "pricing_error", 0.0),
            relaxed_view=relaxed_view,
            stop_reason=stop_reason,
            plan_parts=self.parts(plan),
            incumbent_used=incumbent_used,
            columns={d.key: d.columns for d in self.devices},
            local_prices=self.local_prices(shown),
            limit_prices=self.limit_prices(shown),
        )


def extended_objective(
    cfg: SiteConfig, fc: Forecasts, net: np.ndarray, trajs: dict[str, np.ndarray]
) -> float:
    """total_objective for a plan's batteries (net must already be curtailed)."""
    soe = {k: v for k, v in trajs.items() if k.startswith("battery")} or None
    val = total_objective(cfg, net, fc, soe)
    return val
