"""The site and its forecasts, as the coordinator takes them: kW, kWh,
currency/kWh, one value per slot. Built by `backend` from EMHASS's
configuration (W, Wh).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# --------------------------------------------------------------------------
# Horizon
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Horizon:
    """Discretisation of the planning window.

    dt is in HOURS (0.25 = 15 min slots), matching the units every price and
    power quantity in this package uses: power in kW, energy in kWh, price in
    currency/kWh.
    """

    dt: float = 0.25
    hours: float = 48.0

    @property
    def steps(self) -> int:
        return int(round(self.hours / self.dt))

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.hours <= 0:
            raise ValueError(f"hours must be positive, got {self.hours}")
        if self.steps < 2:
            raise ValueError(f"horizon needs >= 2 steps, got {self.steps}")


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryConfig:
    """A battery held in the master by its linear model, kW and kWh.

    Charging power `a` stores `eta_charge * a`; delivering `d` to the meter
    drains `d / eta_discharge` from the store. Energy left at the end is worth
    `terminal_price` (currency/kWh); None means the cheapest import price.
    """

    capacity_kwh: float = 10.0
    p_charge_max_kw: float = 5.0
    p_discharge_max_kw: float = 5.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_initial_frac: float = 0.5
    # Reserve the plan never draws below, as a fraction of capacity. If the
    # battery starts below it, the floor is where it starts (soe_floor_kwh).
    soe_min_frac: float = 0.0
    terminal_price: float | None = None

    @property
    def eta_c(self) -> float:
        return self.eta_charge

    @property
    def eta_d(self) -> float:
        return self.eta_discharge

    @property
    def soe_floor_kwh(self) -> float:
        """The lowest stored energy a plan may reach, kWh."""
        return self.capacity_kwh * min(self.soe_min_frac, self.soc_initial_frac)

    def validate(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        for name, val in (("eta_charge", self.eta_c), ("eta_discharge", self.eta_d)):
            if not 0 < val <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {val}")
        if not 0 <= self.soc_initial_frac <= 1:
            raise ValueError("soc_initial_frac must be in [0, 1]")
        if not 0 <= self.soe_min_frac < 1:
            raise ValueError("soe_min_frac must be in [0, 1)")


@dataclass(frozen=True)
class GridLimits:
    """Hard limits on the grid connection (fuse, or an operator dimming order).

    These are a COUPLING constraint: they bind the sum of every device's power
    plus the inflexible load, so no per-device plan can enforce one on its own.
    The coordinators price them instead, at `breach_price` per kWh beyond the
    limit: Dantzig-Wolfe in its master LP, ADMM in the grid connection's step
    (and each device where it re-plans against the others).

    A plan that exceeds a limit is not cut back afterwards: the exceedance
    stays in the plan, is reported, and is charged at `breach_price`, so its
    impact can be seen. Clipping a setpoint to the measured headroom is
    opt-in (policy.clamp, /api/setpoint). `enforce` is not used.
    """

    max_import_kw: float | None = None
    max_export_kw: float | None = None
    # Throwing PV away is the only way to respect an export cap once storage is
    # full - without it a cap is unenforceable, because surplus has nowhere to
    # go and no price can make it disappear (docs/NOTES.md section 1).
    # Curtailment is also the right answer to a negative export price: being
    # paid to stop generating beats paying to generate.
    allow_curtailment: bool = True
    enforce: bool = True
    # What a kWh drawn (or pushed) beyond the limit costs, in currency per kWh,
    # on top of the energy's own price. ONE constant for the whole horizon: a
    # fuse or a regulatory cap is not less binding when energy is cheap. (It
    # used to be a multiple of each slot's import tariff, which made a breach
    # free at a zero price and paid for at a negative one - exactly when a
    # battery wants to import hardest.)
    #
    # `breach_price` sets it directly. Left as None, it is
    # `breach_price_multiplier` times the largest |price| on the horizon, so it
    # outweighs anything a breach could save in any currency. It is still a
    # finite price rather than a hard constraint, so an unavoidable breach is
    # planned, reported and charged instead of the plan failing. Nothing clips
    # the plan to the limit afterwards. 0 disables the term.
    breach_price: float | None = None
    breach_price_multiplier: float = 10.0

    def validate(self) -> None:
        for name in ("max_import_kw", "max_export_kw"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive or None, got {v}")
        if self.breach_price_multiplier < 0:
            raise ValueError("breach_price_multiplier must be >= 0")
        if self.breach_price is not None and self.breach_price < 0:
            raise ValueError("breach_price must be >= 0")

    @property
    def active(self) -> bool:
        return self.max_import_kw is not None or self.max_export_kw is not None


@dataclass(frozen=True)
class SubMeter:
    """A node of the site's electrical tree: devices (and PV arrays) behind
    one connection to their parent, with that connection's own limits - a
    hybrid inverter, a sub-panel or a breaker. The parent is another
    sub-meter, a grid connection (GridConnection), or, by default, the main
    meter. Nodes nest to any depth.

    Like GridLimits these are COUPLING constraints, but on a subset of the
    devices. Dantzig-Wolfe models each sub-meter in its master LP: a balance
    row at the sub-meter (its "bus") and the connection's flows to the house,
    so every member device is priced at the sub-meter's own price (that row's
    dual) instead of the meter's. Where the connection is not at a limit the
    two differ only by the conversion efficiency; where it is, they separate
    (docs/theory.tex, "Sub-meters and local prices").

    Signs follow the devices: a member's power is positive when it draws. The
    connection's limits are AC-side, in kW: `max_export_kw` from the bus to
    the house (an inverter's AC output rating), `max_import_kw` from the house
    to the bus (its AC input rating). None means no limit. `eta_export` and
    `eta_import` are the conversion efficiencies each way (1.0 for a
    sub-panel). Power beyond a limit is planned only where nothing else fits,
    and charged at the grid's breach price, as a grid-limit breach is.

    `members` are device keys and PV array keys ("pv" for Forecasts.solar,
    or a name in Forecasts.pv_arrays). A PV array on a node reaches the house
    through the node's connection, and is clipped there - at no cost - when
    the connection cannot pass it. `pv=True` is shorthand for "pv" in
    `members`. "Export" is towards the parent, "import" from it.
    """

    name: str
    members: tuple[str, ...] = ()
    pv: bool = False
    max_export_kw: float | None = None
    max_import_kw: float | None = None
    eta_export: float = 1.0
    eta_import: float = 1.0
    parent: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("a sub-meter needs a name")
        for name in ("max_export_kw", "max_import_kw"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{self.name}: {name} must be >= 0 or None, got {v}")
        for name in ("eta_export", "eta_import"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{self.name}: {name} must be in (0, 1], got {v}")
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"{self.name}: a member is listed twice")
        if self.parent == self.name:
            raise ValueError(f"{self.name}: a sub-meter cannot be its own parent")

    @property
    def keys(self) -> tuple[str, ...]:
        """Its members, with "pv" when `pv` is set."""
        return self.members + (("pv",) if self.pv and "pv" not in self.members else ())

    @property
    def export_cap_dc(self) -> float:
        """The most the bus can send to the house, bus-side (kW); inf if unlimited."""
        return np.inf if self.max_export_kw is None else self.max_export_kw / self.eta_export

    @property
    def import_cap_dc(self) -> float:
        """The most the bus can take from the house, bus-side (kW); inf if unlimited."""
        return np.inf if self.max_import_kw is None else self.max_import_kw * self.eta_import


def hybrid_inverter(
    batteries: tuple[str, ...] = ("battery",),
    max_output_kw: float | None = None,
    max_input_kw: float | None = None,
    eta_dc_ac: float = 1.0,
    eta_ac_dc: float = 1.0,
    name: str = "inverter",
    pv: tuple[str, ...] = ("pv",),
    parent: str | None = None,
) -> SubMeter:
    """A hybrid inverter: the PV arrays `pv` and `batteries` on its DC bus,
    its parent (by default the main meter) on its AC side. `max_output_kw` is
    its AC output rating (None: no limit), `max_input_kw` its AC input rating
    (None: the same as the output). EMHASS's inverter_ac_output_max,
    inverter_ac_input_max, inverter_efficiency_dc_ac and
    inverter_efficiency_ac_dc."""
    return SubMeter(
        name,
        tuple(batteries) + tuple(pv),
        max_export_kw=max_output_kw,
        max_import_kw=max_output_kw if max_input_kw is None else max_input_kw,
        eta_export=eta_dc_ac,
        eta_import=eta_ac_dc,
        parent=parent,
    )


MAIN = "grid"  # the main meter's name in the site's tree


@dataclass(frozen=True)
class GridConnection:
    """A further connection to the grid, with its own meter, tariff and
    limits - a heat-pump tariff meter, a second incomer. Its tariff is
    Forecasts.tariffs[name]. `members`: devices and PV arrays directly on it;
    sub-meters reach it by `parent=name`. The main meter is MAIN, with
    SiteConfig.grid and the forecasts' own tariff."""

    name: str
    grid: GridLimits = field(default_factory=lambda: GridLimits())
    members: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name or self.name == MAIN:
            raise ValueError(f"a grid connection needs a name other than {MAIN!r}")
        self.grid.validate()


@dataclass(frozen=True)
class SetLimit:
    """A limit on what a set of devices draw together, wherever they are in
    the tree (sets may overlap each other and the tree): a phase, a shared
    supply cable, a contract. kW, on the devices' own power: at most
    `max_import_kw`, at most `max_export_kw` pushed back. None: no limit that
    way. Dantzig-Wolfe holds it as a row of its own; each member is then
    priced at its bus's price plus the limit's (its dual)."""

    name: str
    members: tuple[str, ...]
    max_import_kw: float | None = None
    max_export_kw: float | None = None

    def validate(self) -> None:
        if not self.name or not self.members:
            raise ValueError("a set limit needs a name and members")
        for v in (self.max_import_kw, self.max_export_kw):
            if v is not None and v < 0:
                raise ValueError(f"{self.name}: limits must be >= 0 or None")


@dataclass(frozen=True)
class SiteConfig:
    """Everything the coordinator needs. Devices are optional (None = absent)."""

    horizon: Horizon = field(default_factory=Horizon)
    battery: BatteryConfig | None = field(default_factory=BatteryConfig)
    # Additional batteries beyond `battery`. evcc sends one BatteryConfig per
    # stationary battery AND one per loadpoint (core/site_optimizer.go:532,564),
    # so anyone with a charger has at least two. `battery` stays the first one
    # rather than being folded into a list, so single-battery call sites and
    # `result.devices["battery"]` keep working unchanged.
    batteries: tuple[BatteryConfig, ...] = ()
    grid: GridLimits = field(default_factory=GridLimits)
    # The site's electrical tree (docs/theory.tex, "Sub-meters and local
    # prices"): nodes behind their own connection (a hybrid inverter, a panel,
    # a breaker), nesting to any depth; further grid connections with their
    # own tariffs; and limits on sets of devices. A device or PV array sits in
    # at most one node or connection; the rest are on the main meter.
    submeters: tuple[SubMeter, ...] = ()
    connections: tuple[GridConnection, ...] = ()
    set_limits: tuple[SetLimit, ...] = ()

    @property
    def battery_list(self) -> tuple[BatteryConfig, ...]:
        """Every battery, `battery` first. The canonical ordering."""
        head = (self.battery,) if self.battery is not None else ()
        return head + self.batteries

    @staticmethod
    def battery_key(index: int) -> str:
        """Device key for battery `index`. Index 0 is plain "battery" so that
        every existing single-battery call site and test keeps working."""
        return "battery" if index == 0 else f"battery{index}"

    def validate(self) -> None:
        self.horizon.validate()
        self.grid.validate()
        if self.batteries and self.battery is None:
            raise ValueError("cannot set `batteries` without `battery`; battery is index 0")
        for dev in self.battery_list:
            dev.validate()
        seen: set[str] = set()
        names: set[str] = {MAIN}
        for c in self.connections:
            c.validate()
            if c.name in names:
                raise ValueError(f"two nodes or connections are named {c.name!r}")
            names.add(c.name)
            if seen & set(c.members):
                raise ValueError(f"{c.name}: a device is in two places")
            seen |= set(c.members)
        for sm in self.submeters:
            sm.validate()
            if sm.name in names:
                raise ValueError(f"two nodes or connections are named {sm.name!r}")
            names.add(sm.name)
            if seen & set(sm.keys):
                raise ValueError(f"{sm.name}: a device is behind two sub-meters")
            seen |= set(sm.keys)
        for sm in self.submeters:
            if sm.parent is not None and sm.parent not in names:
                raise ValueError(f"{sm.name}: no parent {sm.parent!r}")
        parent = {sm.name: sm.parent or MAIN for sm in self.submeters}
        for sm in self.submeters:  # every chain must end at a connection
            node, steps = sm.name, 0
            while node in parent:
                node, steps = parent[node], steps + 1
                if steps > len(parent):
                    raise ValueError(f"{sm.name}: its parents form a loop")
        set_names: set[str] = set()
        for lim in self.set_limits:
            lim.validate()
            if lim.name in set_names:
                raise ValueError(f"two set limits are named {lim.name!r}")
            set_names.add(lim.name)

    def bus_of(self, key: str) -> str:
        """The node or grid connection device (or PV array) `key` is on: a
        sub-meter's name, a connection's, or MAIN."""
        for sm in self.submeters:
            if key in sm.keys:
                return sm.name
        for c in self.connections:
            if key in c.members:
                return c.name
        return MAIN

    @property
    def has_tree(self) -> bool:
        """Whether anything is off the main meter's own bus or set-limited."""
        return bool(self.submeters or self.connections or self.set_limits)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecasts:
    """Exogenous time series, all length `horizon.steps`.

    buy/sell in currency per kWh; every power series in kW.
    `load` is inflexible household demand; `solar` is PV production.
    """

    buy: np.ndarray
    sell: np.ndarray
    load: np.ndarray
    solar: np.ndarray
    # Further PV arrays, kW, by name (`solar` is the array named "pv"); each
    # sits where a node or connection lists it, else on the main meter.
    pv_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    # Tariffs of further grid connections (SiteConfig.connections), by name:
    # (buy, sell), currency/kWh.
    tariffs: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def validate(self, horizon: Horizon) -> None:
        n = horizon.steps
        series: list[tuple[str, np.ndarray]] = [
            (name, getattr(self, name))
            for name in (
                "buy",
                "sell",
                "load",
                "solar",
            )
        ]
        series += [(f"pv_arrays[{k!r}]", v) for k, v in self.pv_arrays.items()]
        for k, (b, s_) in self.tariffs.items():
            series += [(f"tariffs[{k!r}] buy", b), (f"tariffs[{k!r}] sell", s_)]
        for name, arr in series:
            arr = np.asarray(arr)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if "pv" in self.pv_arrays:
            raise ValueError('"pv" is `solar`; name further PV arrays otherwise')

    def pv(self, key: str) -> np.ndarray:
        """PV array `key`'s forecast, kW: "pv" is `solar`."""
        return self.solar if key == "pv" else np.asarray(self.pv_arrays[key], dtype=float)

    @property
    def pv_keys(self) -> tuple[str, ...]:
        """Every PV array: "pv", then the further ones."""
        return ("pv",) + tuple(self.pv_arrays)

    @property
    def total_solar(self) -> np.ndarray:
        """All PV arrays together, kW."""
        return self.solar + sum(
            (np.asarray(v, dtype=float) for v in self.pv_arrays.values()), np.zeros_like(self.solar)
        )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
