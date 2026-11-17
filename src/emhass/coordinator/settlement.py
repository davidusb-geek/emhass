"""Who saves what: each player's share of the saving.

The house pays one bill, and it does not split by player: whether a kWh is
imported or exported depends on everything else on the meter. This module
splits it anyway, against a baseline with no PV and nothing coordinated
(each participant's own `baseline()` plan, batteries idle). Each device is
first reimbursed its private-cost change (stored energy, a missed target),
then the rest of the saving is shared between solar and the devices.

The split is an Owen value (Owen 1977) with two groups, solar and the devices:
the average of two orders of arrival.

* PV first, then the devices. Solar earns what PV saves on its own; the
  devices share the rest by Aumann-Shapley: every kWh a device shifts is
  priced at the average slope of the bill along the straight path from the
  baseline to the plan, all devices moving together (per slot, an exact
  secant).
* Devices first (a plan made with no PV), then PV. The devices share that
  plan's saving the same way; solar earns the rest.

It needs one extra plan, the devices' plan with no PV. The shares add up to
the total exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from emhass.coordinator.costs import apply_curtailment, breach_price
from emhass.coordinator.master import Column, Device, DWCoordinator
from emhass.coordinator.model import Forecasts, SiteConfig
from emhass.coordinator.submeter import (
    MAIN,
    ac_solar,
    fixed_demand,
    meter_from_draws,
    root_grid,
    tariff,
)

AC = "ac"  # the main meter's own bus, beside each node's and further connection's (by name)
SET = "set:"  # a set limit's members' total, beside the buses (types.SetLimit)

# A plan entry as callers hand it in: a coordinator Column, or a
# (power kW, trajectory[, private cost]) tuple.
PlanEntry = Column | tuple


def _tariff_cost(co: DWCoordinator, z: np.ndarray) -> np.ndarray:
    """Grid cost per slot at a meter flow z that is already final (no
    curtailment left to apply): the tariff plus any priced breach."""
    return _tariff_cost_at(co, co.fc, z)


def _tariff_cost_at(co: DWCoordinator, fc: Forecasts, z: np.ndarray) -> np.ndarray:
    """`_tariff_cost` with the tariff of `fc` (the same slots, or a subset);
    the breach price is the horizon's, from co.fc."""
    cfg, dt = co.cfg, co.dt
    c = (np.maximum(z, 0.0) * fc.buy - np.maximum(-z, 0.0) * fc.sell) * dt
    g = cfg.grid
    if g.active:
        b = breach_price(g, co.fc.buy, co.fc.sell)
        if g.max_import_kw is not None:
            c = c + np.maximum(z - g.max_import_kw, 0.0) * b * dt
        if g.max_export_kw is not None:
            c = c + np.maximum(-z - g.max_export_kw, 0.0) * b * dt
    return c


def _meter_cost(co: DWCoordinator, z: np.ndarray) -> np.ndarray:
    """Grid cost per slot at raw meter flow z, exactly as the objective scores
    it: the curtailment rule, then the tariff and any priced breach."""
    zc, _ = apply_curtailment(z, ac_solar(co.cfg, co.fc), co.fc.sell, co.cfg.grid)
    return _tariff_cost(co, _under_ceiling(co.fc, zc))


def _under_ceiling(fc: Forecasts, z: np.ndarray) -> np.ndarray:
    """The meter flow `z` as delivered under `fc.export_ceiling`: export past
    it is curtailed PV, as the plan's own score and results take it."""
    if fc.export_ceiling is None:
        return z
    return z + np.maximum(-z - np.asarray(fc.export_ceiling, dtype=float), 0.0)


def _path_prices(
    co: DWCoordinator,
    start: np.ndarray,
    end: np.ndarray,
    cost: Callable[[DWCoordinator, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Average slope of the grid cost `cost(co, z)` along the straight path start -> end,
    per slot (currency per kW-slot).

    Exact, with no integration: the grid cost is separable by slot and the path
    is a straight line in each slot's meter flow, so the average slope is the
    secant (cost at the end - cost at the start) / (flow at the end - flow at
    the start). Where the flow does not move, it is the slope there.
    """
    dz = end - start
    dm = cost(co, end) - cost(co, start)
    eps = 1e-6
    local = (cost(co, start + eps) - cost(co, start - eps)) / (2 * eps)
    moving = np.abs(dz) > 1e-9
    return np.where(moving, dm / np.where(moving, dz, 1.0), local)


def _bus_of(co: DWCoordinator, key: str) -> str:
    """Which bus device `key` is on: AC (the main meter's own), or a node's or
    a further grid connection's name."""
    b = co.cfg.bus_of(key)
    return AC if b == MAIN else b


def _totals_of(co: DWCoordinator, key: str) -> list[str]:
    """The totals device `key` counts in (`_draws`): its bus, and each set
    limit it is a member of. Its local price is the sum of theirs."""
    return [_bus_of(co, key)] + [SET + lim.name for lim in co.cfg.set_limits if key in lim.members]


def _price_of(co: DWCoordinator, q: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    """Device `key`'s Aumann-Shapley price: the sum of its totals' (`_bus_prices`)."""
    return sum((q[t] for t in _totals_of(co, key)), np.zeros(co.n))


def _draws(
    co: DWCoordinator, fc: Forecasts, powers: Mapping[str, np.ndarray], load: bool = True
) -> dict[str, np.ndarray]:
    """The totals the bill depends on (kW per slot), for device powers
    `powers`: each bus's draw - AC carries the load (if `load`) less the PV
    on it; the other buses their devices only, their PV coming from `fc` - and
    each set limit's members' total."""
    n = co.n
    x = {AC: (fc.load if load else np.zeros(n)) - ac_solar(co.cfg, fc)}
    x.update({b: np.zeros(n) for b in co.topo.buses if b != MAIN})
    x.update({SET + lim.name: np.zeros(n) for lim in co.cfg.set_limits})
    for k, p in powers.items():
        for t in _totals_of(co, k):
            x[t] = x[t] + p
    return x


def _split(
    x: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """`_draws` as meter_from_draws takes it: (AC draw, {bus: draw}, {set limit: total})."""
    buses = {k: v for k, v in x.items() if k != AC and not k.startswith(SET)}
    sets = {k[len(SET) :]: v for k, v in x.items() if k.startswith(SET)}
    return x[AC], buses, sets


def _slot_cost(co: DWCoordinator, fc: Forecasts, x: Mapping[str, np.ndarray]) -> np.ndarray:
    """The bill per slot (currency) at totals `x` (`_draws`), as the objective
    scores it: the tree's flows, curtailment, every connection's tariff, every
    breach."""
    flows = meter_from_draws(co.cfg, fc, *_split(x))
    c = _tariff_cost_at(co, fc, _under_ceiling(fc, flows.net))
    if np.any(flows.breach_kwh > 0):
        c = c + flows.breach_kwh * breach_price(co.cfg.grid, co.fc.buy, co.fc.sell)
    for conn in co.cfg.connections:
        buy, sell = tariff(co.cfg, fc, conn.name)
        z = flows.nets[conn.name]
        c = c + (np.maximum(z, 0.0) * buy - np.maximum(-z, 0.0) * sell) * co.dt
        g = root_grid(co.cfg, conn.name)
        if g.active:
            full_buy, full_sell = tariff(co.cfg, co.fc, conn.name)
            over = np.zeros_like(z)
            if g.max_import_kw is not None:
                over = over + np.maximum(z - g.max_import_kw, 0.0)
            if g.max_export_kw is not None:
                over = over + np.maximum(-z - g.max_export_kw, 0.0)
            c = c + over * breach_price(g, full_buy, full_sell) * co.dt
    return c


def _slots(fc: Forecasts, idx: np.ndarray) -> Forecasts:
    """`fc` restricted to slots `idx` (repeats allowed): one entry per index."""
    return replace(
        fc,
        buy=fc.buy[idx],
        sell=fc.sell[idx],
        load=fc.load[idx],
        solar=fc.solar[idx],
        pv_arrays={k: np.asarray(v)[idx] for k, v in fc.pv_arrays.items()},
        tariffs={k: (np.asarray(b)[idx], np.asarray(v)[idx]) for k, (b, v) in fc.tariffs.items()},
        export_ceiling=None if fc.export_ceiling is None else np.asarray(fc.export_ceiling)[idx],
    )


def dark(fc: Forecasts) -> Forecasts:
    """`fc` with no PV at all: every array at zero."""
    return replace(
        fc,
        solar=np.zeros_like(fc.solar),
        pv_arrays={k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in fc.pv_arrays.items()},
        # the ceiling is the PV surplus: none without PV
        export_ceiling=None if fc.export_ceiling is None else np.zeros_like(fc.export_ceiling),
    )


def _bus_prices(
    co: DWCoordinator,
    fc: Forecasts,
    x0: Mapping[str, np.ndarray],
    x1: Mapping[str, np.ndarray],
    cells: int = 16,
    depth: int = 40,
) -> dict[str, np.ndarray]:
    """Aumann-Shapley local prices: per bus and slot, the average along the
    straight path x0 -> x1 of the bill's slope in that bus's draw (currency
    per kW-slot). Sum over buses of price x (x1 - x0) is the bill's change.

    Per slot the bill is piecewise linear along the path, so its gradient is
    constant between kinks. Each slot's path is cut into `cells`; a cell
    whose gradient is the same just inside both ends is one linear piece and
    adds its gradient times its length. Any other cell is halved, up to
    `depth` times; a cell still not one piece is then a sliver around a kink,
    and adds its midpoint gradient. Every cell's contribution is corrected to
    its exact change in the bill, so the prices times the moves add up to the
    bill's change exactly. With one total, this is `_path_prices`' secant.
    """
    buses = list(x0)
    n = co.n
    d = {b: np.asarray(x1[b], dtype=float) - np.asarray(x0[b], dtype=float) for b in buses}
    if len(buses) == 1:
        b = buses[0]
        return {b: _path_prices(co, x0[b], x1[b], lambda c, z: _slot_cost(c, fc, {b: z}))}

    def point(
        idx: np.ndarray, s: np.ndarray, shift: str | None = None, h: float = 0.0
    ) -> dict[str, np.ndarray]:
        return {b: x0[b][idx] + s * d[b][idx] + (h if b == shift else 0.0) for b in buses}

    def cost(idx: np.ndarray, x: dict[str, np.ndarray]) -> np.ndarray:
        return _slot_cost(co, _slots(fc, idx), x)

    def grad(idx: np.ndarray, s: np.ndarray) -> np.ndarray:
        h = 1e-5  # kW: central differences are exact on a linear piece, up to rounding
        f = _slots(fc, idx)
        return np.array(
            [
                (_slot_cost(co, f, point(idx, s, b, h)) - _slot_cost(co, f, point(idx, s, b, -h)))
                / (2 * h)
                for b in buses
            ]
        )

    avg = np.zeros((len(buses), n))
    T = np.repeat(np.arange(n), cells)
    L = np.tile(np.arange(cells) / cells, n)
    R = L + 1.0 / cells
    for level in range(depth + 1):
        if not len(T):
            break
        w = R - L
        gl, gr = grad(T, L + 1e-3 * w), grad(T, R - 1e-3 * w)
        one = np.all(np.abs(gl - gr) <= 1e-7 * (1.0 + np.abs(gl) + np.abs(gr)), axis=0)
        if level == depth or len(T) > 64 * n:  # slivers (the cap only guards against a runaway)
            one = np.ones_like(one)
            gl = grad(T, 0.5 * (L + R))
        if np.any(one):
            # Each accepted cell's contribution is made exactly its change in
            # the bill (the gradients are central differences, exact on a
            # linear piece only up to rounding), so the totals add up exactly.
            Ti, wi, gi = T[one], w[one], gl[:, one]
            dc = cost(Ti, point(Ti, R[one])) - cost(Ti, point(Ti, L[one]))
            dd = np.array([d[b][Ti] for b in buses])
            rest = dc - np.sum(gi * dd, axis=0) * wi
            big = np.argmax(np.abs(dd), axis=0)
            at = np.arange(len(Ti))
            moved = np.abs(dd[big, at]) > 1e-12
            gi[big, at] += np.where(moved, rest / (wi * np.where(moved, dd[big, at], 1.0)), 0.0)
            for i in range(len(buses)):
                np.add.at(avg[i], Ti, gi[i] * wi)
        M = 0.5 * (L + R)
        keep = ~one
        T, L, R = (
            np.concatenate([T[keep], T[keep]]),
            np.concatenate([L[keep], M[keep]]),
            np.concatenate([M[keep], R[keep]]),
        )
    return {b: avg[i] for i, b in enumerate(buses)}


def _participant(co: DWCoordinator, key: str) -> Device | None:
    """The coordinator's participant called `key`, or None for a site device."""
    return next((d for d in co.devices if d.key == key), None)


def _entry(e: PlanEntry) -> tuple[np.ndarray, np.ndarray, float | None]:
    """A plan entry as (power kW, state trajectory, private cost or None).
    Accepts a coordinator Column or a (power, trajectory[, cost]) tuple."""
    if isinstance(e, tuple):
        cost = e[2] if len(e) > 2 else None
        return e[0], e[1], None if cost is None else float(cost)
    return e.power, e.trajectory, float(e.cost)


def _private_cost(co: DWCoordinator, key: str, traj: np.ndarray, cost: float | None) -> float:
    """A plan's private cost (currency): the cost a participant reported for
    it, or the coordinator's own pricing of a site device's trajectory."""
    if _participant(co, key) is not None:
        if cost is None:
            raise ValueError(f"participant {key!r}: its plan entry must carry its private cost")
        return cost
    return co.battery_cost(key, traj)


def _baseline(co: DWCoordinator, key: str) -> tuple[np.ndarray, np.ndarray, float | None]:
    """`baseline`, plus the private cost a participant reported for it (None
    for a site device). A participant is asked once per coordinator."""
    part = _participant(co, key)
    if part is not None:
        memo = co.__dict__.setdefault("_baseline_answers", {})
        if key not in memo:
            memo[key] = part.participant.baseline()
        a = memo[key]
        return (
            np.asarray(a.plan_kw, dtype=float),
            np.asarray(a.trajectory, dtype=float),
            float(a.private_cost),
        )
    h = co.cfg.horizon
    b = dict(co.lp_batts)[key]  # a battery the master holds: idle
    return np.zeros(h.steps), np.full(h.steps + 1, b.capacity_kwh * b.soc_initial_frac), None


def _device_shares(
    co: DWCoordinator, plan: Mapping[str, PlanEntry], keys: list[str]
) -> tuple[dict, dict]:
    """Aumann-Shapley shares of the devices' saving over their baseline, with
    the PV (whatever co.fc holds) fixed: each device's net gain, and its
    private-cost change."""
    fc = co.fc
    base = {k: _baseline(co, k) for k in keys}
    entries = {k: _entry(plan[k]) for k in keys}
    df = {
        k: _private_cost(co, k, entries[k][1], entries[k][2])
        - _private_cost(co, k, base[k][1], base[k][2])
        for k in keys
    }
    if not co.cfg.has_tree:
        z0 = fixed_demand(co.cfg, fc) + sum(base[k][0] for k in keys)
        z1 = fixed_demand(co.cfg, fc) + sum(entries[k][0] for k in keys)
        q = _path_prices(co, z0, z1, _meter_cost)
        return {k: -float(q @ (entries[k][0] - base[k][0])) - df[k] for k in keys}, df
    qb = _bus_prices(
        co,
        fc,
        _draws(co, fc, {k: base[k][0] for k in keys}),
        _draws(co, fc, {k: entries[k][0] for k in keys}),
    )
    return {
        k: -float(_price_of(co, qb, k) @ (entries[k][0] - base[k][0])) - df[k] for k in keys
    }, df


def device_keys(co: DWCoordinator) -> list[str]:
    """Every device that takes a share: the batteries the master holds, then
    the participants, in the coordinator's order."""
    keys = [
        SiteConfig.battery_key(i) for i, b in enumerate(co.cfg.battery_list) if b.capacity_kwh > 0
    ]
    return keys + [d.key for d in co.devices]


def ledger(
    co: DWCoordinator,
    plan: Mapping[str, PlanEntry],
    co_dark: DWCoordinator,
    plan_dark: Mapping[str, PlanEntry],
) -> dict:
    """Each player's bill before and after, the private-cost change and the gain.

    `plan` maps device key -> its plan, for every device on the site and every
    participant: a coordinator Column, or (power kW, state trajectory), plus
    the private cost as a third item for a participant. `co_dark` /
    `plan_dark` are the same site and method with the PV set to zero.

    Returns {"rows": [...], "totals": {...}}: one row per player (household
    load, solar, then each device) with its bill before and after, its
    private-cost change and its net gain (currency), and the house totals.
    """
    fc = co.fc
    keys = device_keys(co)
    base = {k: _baseline(co, k) for k in keys}
    entries = {k: _entry(plan[k]) for k in keys}

    # the bill: baseline with no PV, thermostats with PV, the plan
    nopv = dark(fc)
    p_base = {k: base[k][0] for k in keys}
    p_plan = {k: entries[k][0] for k in keys}
    bill_dark = float(_slot_cost(co, nopv, _draws(co, nopv, p_base)).sum())
    bill_thermo = float(_slot_cost(co, fc, _draws(co, fc, p_base)).sum())
    bill_plan = float(_slot_cost(co, fc, _draws(co, fc, p_plan)).sum())

    # the two orders of arrival
    first_pv, df = _device_shares(co, entries, keys)  # devices after PV
    first_dev, _ = _device_shares(co_dark, plan_dark, keys)  # devices before PV
    total = bill_dark - bill_plan - sum(df.values())
    v_pv, v_dev = bill_dark - bill_thermo, sum(first_dev.values())
    gain = {"household load": 0.0, "solar": 0.5 * (v_pv + (total - v_dev))}
    gain.update({k: 0.5 * (first_pv[k] + first_dev[k]) for k in keys})
    df = {"household load": 0.0, "solar": 0.0, **df}

    # bills before: the baseline bill, split along 0 -> baseline
    x_base = {"household load": fc.load, "solar": np.zeros_like(fc.solar)}
    x_base.update({k: base[k][0] for k in keys})
    if co.cfg.has_tree:
        zero = {b: np.zeros(co.n) for b in _draws(co, nopv, {})}
        qb0 = _bus_prices(co, nopv, zero, _draws(co, nopv, p_base))
        q_of = {
            "household load": qb0[AC],
            "solar": qb0[AC],
            **{k: _price_of(co, qb0, k) for k in keys},
        }
    else:
        z_dark = fc.load + sum(base[k][0] for k in keys)
        q0 = _path_prices(co, np.zeros_like(z_dark), z_dark, _tariff_cost)
        q_of = dict.fromkeys(x_base, q0)

    rows = []
    for j, x in x_base.items():
        before = float(q_of[j] @ x)
        after = before - (gain[j] + df[j])  # paid: its gain plus its reimbursement
        rows.append(
            {
                "player": j,
                "bill_before": before,
                "bill_after": after,
                "bill_change": after - before,
                "private_cost_change": df[j],
                "net_gain": gain[j],
            }
        )
    tot = {
        c: sum(r[c] for r in rows)
        for c in ("bill_before", "bill_after", "bill_change", "private_cost_change", "net_gain")
    }
    curtailed = meter_from_draws(co.cfg, fc, *_split(_draws(co, fc, p_plan))).curtail
    tot.update(
        house_bill_before=bill_dark,
        house_bill_after=bill_plan,
        house_bill_with_pv=bill_thermo,
        coordination_saving=bill_thermo - bill_plan - tot["private_cost_change"],
        pv_alone_saving=v_pv,
        devices_alone_saving=v_dev,
        curtailed_kwh=float(curtailed.sum() * co.dt),
    )
    return {"rows": rows, "totals": tot}
