"""The meter, given every device's power: the one place that knows the site's
electrical tree (model.SubMeter, GridConnection, SetLimit).

Every scorer (the coordinator's objective, the saving split, the results)
goes through `site_meter`, so they all agree with each other and with the
Dantzig-Wolfe master, which models the same tree as LP rows. With no tree it
is exactly what it replaces: the inflexible load, minus the PV, plus every
device, then `apply_curtailment`.

The tree. The main meter (MAIN) and any further grid connection are roots,
each with its own tariff and limits. A sub-meter (a hybrid inverter, a panel,
a breaker) is a node with one parent - a root or another node - and its
connection to that parent has ratings and a conversion efficiency each way.
Devices and PV arrays sit on one node or root each.

Each slot, bottom-up, at each node:

* what its devices draw, minus its PV, minus what its child nodes send up,
  is what it needs from its parent (positive) or can send to it (negative);
* sending is capped at the connection's rating. PV beyond it is clipped, at
  no cost: the node's own PV first, then - if the site allows curtailment -
  PV further down, which a child then does not send. Anything the node
  still cannot send - a battery discharging into a full connection - is sent
  anyway and charged as a breach, at the grid's breach price;
* taking beyond the rating is likewise planned, and charged;
* its parent sees what is sent times eta_export, or what is taken divided by
  eta_import.

Then, at each root, curtailment for the meter's own reasons (an export cap, a
negative export price), from the PV on the root and from what the nodes below
can still give up; each kWh a node does not send is 1/eta_export kWh of PV at
that node. A set limit's breach is charged the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from emhass.coordinator.costs import apply_curtailment, breach_price
from emhass.coordinator.meter import Bus, BusLevel, bus_flow
from emhass.coordinator.model import MAIN, Forecasts, GridLimits, SiteConfig, SubMeter


class Topology(NamedTuple):
    """The site's tree, as every consumer reads it."""

    roots: tuple[str, ...]  # MAIN, then further grid connections
    nodes: tuple[str, ...]  # sub-meters, as configured
    parent: dict[str, str]  # node -> its parent (a root or a node)
    children: dict[str, tuple[str, ...]]  # root or node -> its child nodes
    order: tuple[str, ...]  # nodes, every child before its parent
    sm: dict[str, SubMeter]  # node -> its SubMeter
    pv_at: dict[str, tuple[str, ...]]  # root or node -> the PV arrays on it
    root_of: dict[str, str]  # root or node -> its root

    @property
    def buses(self) -> tuple[str, ...]:
        """Roots then nodes: the balance rows of the master, in order."""
        return self.roots + self.nodes

    def chain(self, bus: str) -> tuple[str, ...]:
        """The nodes from `bus` up to, not including, its root."""
        out = []
        while bus in self.parent:
            out.append(bus)
            bus = self.parent[bus]
        return tuple(out)


def topology(cfg: SiteConfig, fc: Forecasts) -> Topology:
    """The tree of `cfg`, with the PV arrays of `fc` placed on it."""
    roots = (MAIN,) + tuple(c.name for c in cfg.connections)
    nodes = tuple(sm.name for sm in cfg.submeters)
    parent = {sm.name: sm.parent or MAIN for sm in cfg.submeters}
    children = {b: tuple(x for x in nodes if parent[x] == b) for b in roots + nodes}
    order: list[str] = []

    def visit(b: str) -> None:
        for c in children[b]:
            visit(c)
            order.append(c)

    for r in roots:
        visit(r)
    pv_at: dict[str, list[str]] = {b: [] for b in roots + nodes}
    for key in fc.pv_keys:
        pv_at[cfg.bus_of(key)].append(key)
    root_of: dict[str, str] = {r: r for r in roots}
    for x in nodes:
        b = x
        while b in parent:
            b = parent[b]
        root_of[x] = b
    return Topology(
        roots,
        nodes,
        parent,
        children,
        tuple(order),
        {sm.name: sm for sm in cfg.submeters},
        {b: tuple(v) for b, v in pv_at.items()},
        root_of,
    )


class MeterFlows(NamedTuple):
    net: np.ndarray  # kW at the main meter, after curtailment (+ = import)
    curtail: np.ndarray  # PV not used, kW, all arrays (clipped at a node and curtailed)
    breach_kwh: np.ndarray  # energy beyond a node's ratings or a set limit, per slot
    ac: dict[str, np.ndarray]  # node -> its power to its parent, kW (+ = to the parent)
    bus: dict[str, np.ndarray]  # node -> what its bus sends, bus-side kW (+ = to the parent)
    raw: np.ndarray  # the main meter before its own curtailment (an export cap,
    # a negative export price): what a device DP takes as dp_load
    ac_raw: dict[str, np.ndarray]  # `ac` before the roots' own curtailment
    nets: dict[str, np.ndarray]  # root -> its meter after curtailment (MAIN's is `net`)
    raws: dict[str, np.ndarray]  # root -> its meter before its own curtailment
    own: dict[str, np.ndarray]  # node -> its devices' draw minus its PV, kW


def tariff(cfg: SiteConfig, fc: Forecasts, root: str) -> tuple[np.ndarray, np.ndarray]:
    """(buy, sell) at grid connection `root`, currency/kWh."""
    if root == MAIN:
        return fc.buy, fc.sell
    buy, sell = fc.tariffs[root]
    return np.asarray(buy, dtype=float), np.asarray(sell, dtype=float)


def root_grid(cfg: SiteConfig, root: str) -> GridLimits:
    """The grid limits at connection `root`."""
    if root == MAIN:
        return cfg.grid
    return next(c.grid for c in cfg.connections if c.name == root)


def pv_on(fc: Forecasts, topo: Topology, bus: str) -> np.ndarray:
    """All PV on root or node `bus`, kW."""
    return sum((fc.pv(k) for k in topo.pv_at[bus]), np.zeros_like(fc.solar))


def ac_solar(cfg: SiteConfig, fc: Forecasts) -> np.ndarray:
    """The PV on the main meter's own bus, kW."""
    if not cfg.has_tree and not fc.pv_arrays:
        return fc.solar
    return pv_on(fc, topology(cfg, fc), MAIN)


def fixed_demand(cfg: SiteConfig, fc: Forecasts) -> np.ndarray:
    """The main meter's inflexible part: the load minus the PV on its own bus."""
    return fc.load - ac_solar(cfg, fc)


def clip_cap(
    cfg: SiteConfig, fc: Forecasts, sm: SubMeter, topo: Topology | None = None
) -> np.ndarray:
    """PV node `sm` may clip of its own, per slot (kW): free, but beyond what
    its rating forces only if the grid allows curtailment (as the master's
    clip variable)."""
    pv = pv_on(fc, topo or topology(cfg, fc), sm.name)
    return pv if cfg.grid.allow_curtailment else np.maximum(pv - sm.export_cap_dc, 0.0)


def site_meter(cfg: SiteConfig, fc: Forecasts, powers: Mapping[str, np.ndarray]) -> MeterFlows:
    """The meters, curtailment and node flows for device powers `powers`
    (device key -> kW, + = drawing). See the module docstring."""
    n = len(fc.load)
    if not cfg.has_tree:
        # the devices summed first, then added to the load: the order every
        # scorer used before the tree, so a site without one is bit-for-bit
        # what it was
        ac = fixed_demand(cfg, fc) + sum((p for p in powers.values()), np.zeros(n))
        return meter_from_draws(cfg, fc, ac, {})
    where = {k: cfg.bus_of(k) for k in powers}
    ac = fixed_demand(cfg, fc) + sum(
        (p for k, p in powers.items() if where[k] == MAIN), np.zeros(n)
    )
    draws = {
        b: sum((p for k, p in powers.items() if where[k] == b), np.zeros(n))
        for b in {w for w in where.values() if w != MAIN}
    }
    sets = {
        lim.name: sum((powers[k] for k in lim.members if k in powers), np.zeros(n))
        for lim in cfg.set_limits
    }
    return meter_from_draws(cfg, fc, ac, draws, sets)


def meter_from_draws(
    cfg: SiteConfig,
    fc: Forecasts,
    ac_draw: np.ndarray,
    bus_draw: Mapping[str, np.ndarray],
    set_draw: Mapping[str, np.ndarray] | None = None,
) -> MeterFlows:
    """`site_meter` from totals: `ac_draw`, everything on the main meter's own
    bus (the load, minus the PV there, plus the devices there); `bus_draw`,
    node or further connection -> its devices' total draw (kW; its PV comes
    from `fc`); and `set_draw`, set limit -> its members' total draw."""
    dt = cfg.horizon.dt
    n = len(fc.load)
    zero = np.zeros(n)
    topo = topology(cfg, fc)
    allow = cfg.grid.allow_curtailment
    flow: dict[str, np.ndarray] = {}
    send: dict[str, np.ndarray] = {}
    take: dict[str, np.ndarray] = {}
    own: dict[str, np.ndarray] = {}
    spare: dict[str, np.ndarray] = {}  # node -> AC-side kW it could still not send, by curtailing
    local: dict[str, np.ndarray] = {}  # node -> its own PV not yet clipped
    curtail = np.zeros(n)
    breach = np.zeros(n)

    def give_up(kids: tuple[str, ...], amount: np.ndarray) -> np.ndarray:
        """Have `kids` send `amount` (kW, in their parent's units) less, by
        curtailing PV under them; returns the PV curtailed, kW."""
        cut = np.zeros(n)
        for c in kids:
            part = np.minimum(amount, spare[c])
            if not np.any(part > 0):
                continue
            amount = amount - part
            sm = topo.sm[c]
            bus_amt = part / sm.eta_export
            mine = np.minimum(bus_amt, local[c])
            local[c] = local[c] - mine
            cut = cut + mine + give_up(topo.children[c], bus_amt - mine)
            flow[c] = flow[c] - part
            send[c] = send[c] - bus_amt
            spare[c] = spare[c] - part
        return cut

    for node in topo.order:
        sm = topo.sm[node]
        pv = pv_on(fc, topo, node)
        own[node] = np.asarray(bus_draw.get(node, zero), dtype=float) - pv
        kids = topo.children[node]
        draw = own[node] - sum((flow[c] for c in kids), zero)
        cap_local = pv if allow else np.maximum(pv - sm.export_cap_dc, 0.0)
        below = sum((spare[c] for c in kids), zero) if allow else zero
        f, clip, over, s = bus_flow(
            draw,
            cap_local + below,
            sm.eta_export,
            sm.eta_import,
            sm.export_cap_dc,
            sm.import_cap_dc,
        )
        mine = np.minimum(clip, cap_local)
        local[node] = pv - mine
        curtail = curtail + mine + give_up(kids, clip - mine)
        breach = breach + over * dt
        flow[node], send[node], take[node] = f, s, np.maximum(draw, 0.0)
        left = local[node] + sum((spare[c] for c in kids), zero)
        spare[node] = sm.eta_export * np.minimum(s, left) if allow else zero

    ac_raw = {k: v.copy() for k, v in flow.items()}
    nets: dict[str, np.ndarray] = {}
    raws: dict[str, np.ndarray] = {}
    for root in topo.roots:
        on_root = pv_on(fc, topo, root)
        if root == MAIN:
            base = np.array(ac_draw, dtype=float)
        else:
            base = np.asarray(bus_draw.get(root, zero), dtype=float) - on_root
        kids = topo.children[root]
        net = base - sum((flow[c] for c in kids), zero) if kids else base
        raws[root] = net
        curtailable = on_root + sum((spare[c] for c in kids), zero) if kids else on_root
        net, cut = apply_curtailment(
            net, curtailable, tariff(cfg, fc, root)[1], root_grid(cfg, root)
        )
        from_root = np.minimum(cut, on_root)
        curtail = curtail + from_root
        if kids and np.any(cut - from_root > 0):
            curtail = curtail + give_up(kids, cut - from_root)
        nets[root] = net
    for lim in cfg.set_limits:
        tot = (set_draw or {}).get(lim.name)
        if tot is None:
            continue
        over = np.zeros(n)
        if lim.max_import_kw is not None:
            over = over + np.maximum(tot - lim.max_import_kw, 0.0)
        if lim.max_export_kw is not None:
            over = over + np.maximum(-tot - lim.max_export_kw, 0.0)
        breach = breach + over * dt
    bus_out = {k: send[k] - take[k] for k in flow}
    return MeterFlows(
        nets[MAIN], curtail, breach, flow, bus_out, raws[MAIN], ac_raw, nets, raws, own
    )


def extra_bills(cfg: SiteConfig, fc: Forecasts, flows: MeterFlows) -> float:
    """The further grid connections' bills (currency): each meter at its own tariff."""
    total = 0.0
    for c in cfg.connections:
        buy, sell = tariff(cfg, fc, c.name)
        z = flows.nets[c.name]
        total += float(
            np.sum((np.maximum(z, 0.0) * buy - np.maximum(-z, 0.0) * sell) * cfg.horizon.dt)
        )
    return total


def extra_breach(cfg: SiteConfig, fc: Forecasts, flows: MeterFlows) -> float:
    """What the energy beyond the nodes' ratings, the set limits and the
    further connections' grid limits costs (currency): the main grid's breach
    price, and each further connection's own for its own limits."""
    total = 0.0
    if np.any(flows.breach_kwh > 0):
        total = float(np.sum(flows.breach_kwh)) * breach_price(cfg.grid, fc.buy, fc.sell)
    for c in cfg.connections:
        g = c.grid
        if not g.active:
            continue
        buy, sell = tariff(cfg, fc, c.name)
        z = flows.nets[c.name]
        over = np.zeros_like(z)
        if g.max_import_kw is not None:
            over = over + np.maximum(z - g.max_import_kw, 0.0)
        if g.max_export_kw is not None:
            over = over + np.maximum(-z - g.max_export_kw, 0.0)
        total += float(np.sum(over) * cfg.horizon.dt) * breach_price(g, buy, sell)
    return total


def beyond_main(cfg: SiteConfig, fc: Forecasts, flows: MeterFlows) -> float:
    """Everything the main meter's bill and grid limits do not price
    (currency): the further connections' bills, and every other breach."""
    if not cfg.has_tree:
        return 0.0
    return extra_bills(cfg, fc, flows) + extra_breach(cfg, fc, flows)


# what earlier releases called it
submeter_penalty = beyond_main


class DeviceView(NamedTuple):
    """What a device meets, the others held at their plans: its Bus (its
    chain of nodes up to its grid connection; None on a connection's own
    bus), the rest of that connection's meter (kW, the dp_load a device DP
    takes), and that connection's name, tariff and limits."""

    bus: Bus | None
    dp_load: np.ndarray
    root: str
    buy: np.ndarray
    sell: np.ndarray
    grid: GridLimits


def device_bus(
    cfg: SiteConfig, fc: Forecasts, key: str, powers: Mapping[str, np.ndarray]
) -> DeviceView:
    """What device `key` meets, holding every other device at `powers`."""
    rest = {k: p for k, p in powers.items() if k != key}
    flows = site_meter(cfg, fc, rest)
    if not cfg.has_tree:
        return DeviceView(None, flows.raw, MAIN, fc.buy, fc.sell, cfg.grid)
    topo = topology(cfg, fc)
    start = cfg.bus_of(key)
    root = topo.root_of[start]
    buy, sell = tariff(cfg, fc, root)
    chain = topo.chain(start)
    if not chain:
        return DeviceView(None, flows.raws[root], root, buy, sell, root_grid(cfg, root))
    n = len(fc.load)
    levels = []
    below: str | None = None
    for node in chain:
        sm = topo.sm[node]
        others = flows.own[node] - sum(
            (flows.ac_raw[c] for c in topo.children[node] if c != below), np.zeros(n)
        )
        levels.append(
            BusLevel(
                others,
                sm.eta_export,
                sm.eta_import,
                sm.export_cap_dc,
                sm.import_cap_dc,
                clip_cap(cfg, fc, sm, topo),
            )
        )
        below = node
    dp_load = flows.raws[root] + flows.ac_raw[chain[-1]]  # the meter without this chain at all
    return DeviceView(
        Bus(tuple(levels), breach_price(cfg.grid, fc.buy, fc.sell)),
        dp_load,
        root,
        buy,
        sell,
        root_grid(cfg, root),
    )
