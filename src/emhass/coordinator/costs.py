"""The terms of the coordinator's objective, all in currency: the bill, the
grid-limit breach, the battery's end-of-horizon value, and the
curtailment rule.
"""

from __future__ import annotations

import numpy as np

from emhass.coordinator.model import (
    BatteryConfig,
    Forecasts,
    GridLimits,
    SiteConfig,
)


def apply_curtailment(
    net_raw: np.ndarray, solar: np.ndarray, sell: np.ndarray, grid: GridLimits | None
) -> tuple[np.ndarray, np.ndarray]:
    """Throw away the PV that cannot usefully leave the site.

    Returns (net_grid_after, curtailment). Curtailment reduces export 1:1 and
    is bounded by the PV actually available, so this is exact rather than an
    approximation - there is no interaction with the devices to iterate over.

    Two reasons to curtail, and only two:

    * **an export cap** that storage cannot absorb. Once the battery is full
      the surplus has nowhere to go, and no price can make it vanish - this is
      the only lever left.
    * **a negative export price**, where being paid to stop generating beats
      paying someone to take the energy.

    Deliberately applied AFTER the device solves rather than inside them: it
    only ever discards energy that was already going to be exported, so it
    cannot change what a device should have done. That keeps it out of the
    coordination loop entirely.
    """
    curtail = np.zeros_like(net_raw)
    if grid is None or not grid.allow_curtailment:
        return net_raw, curtail

    export = np.maximum(-net_raw, 0.0)
    if grid.max_export_kw is not None:
        curtail = np.minimum(solar, np.maximum(export - grid.max_export_kw, 0.0))
    # Never pay to export.
    negative_price = sell < 0
    if negative_price.any():
        curtail = np.where(negative_price, np.minimum(solar, export), curtail)
    return net_raw + curtail, curtail


def net_cost(net_grid: np.ndarray, buy: np.ndarray, sell: np.ndarray, dt: float) -> float:
    """Bill for a net grid exchange series. Positive net_grid = import."""
    imp = float(np.sum(np.maximum(net_grid, 0.0) * buy * dt))
    exp = float(np.sum(np.maximum(-net_grid, 0.0) * sell * dt))
    return imp - exp


def breach_energy(cfg: SiteConfig, net_grid: np.ndarray) -> np.ndarray:
    """Energy drawn or pushed beyond the grid limits, per slot, in kWh.

    Zero everywhere when no limit is configured, which is why every benchmark
    objective is untouched by the penalty built on it.
    """
    over = np.zeros_like(net_grid)
    if cfg.grid.max_import_kw is not None:
        over = over + np.maximum(net_grid - cfg.grid.max_import_kw, 0.0)
    if cfg.grid.max_export_kw is not None:
        over = over + np.maximum(-net_grid - cfg.grid.max_export_kw, 0.0)
    return over * cfg.horizon.dt


def breach_price(grid: GridLimits, buy: np.ndarray, sell: np.ndarray | None = None) -> float:
    """Currency per kWh beyond a grid limit: one constant for the horizon.

    `grid.breach_price` if set, else `breach_price_multiplier` times the
    largest |price| on the horizon (1.0 if every price is zero).
    """
    if grid.breach_price is not None:
        return float(grid.breach_price)
    prices = np.abs(np.asarray(buy, dtype=float))
    if sell is not None:
        prices = np.concatenate([prices, np.abs(np.asarray(sell, dtype=float))])
    scale = float(prices.max()) if prices.size and prices.max() > 0 else 1.0
    return grid.breach_price_multiplier * scale


def grid_penalty(
    cfg: SiteConfig, net_grid: np.ndarray, buy: np.ndarray, sell: np.ndarray | None = None
) -> float:
    """Cost of breaching the grid limits, in currency.

    Over-limit energy is priced at `breach_price` - a constant, not the slot's
    tariff. A grid limit is usually a fuse or a regulatory cap; it does not
    matter less at a cheap hour, and a tariff-scaled penalty vanishes at a zero
    import price and turns into a reward at a negative one.

    Putting the breach in the objective is what lets round selection be a
    single comparison on money. The alternative - rank on violation first and
    cost second - never lets the two trade at all, which sounds safer but in
    practice decides on the eighth decimal place of a breach that is
    infeasible either way.

    This does NOT enforce anything: the master prices the limit the same way,
    and the backend declines a plan that still breaches one, so the default
    solver plans. This term scores a finished plan; it does not produce one.
    """
    if not cfg.grid.active:
        return 0.0
    price = breach_price(cfg.grid, buy, sell)
    if price <= 0:
        return 0.0
    return float(np.sum(breach_energy(cfg, net_grid)) * price)


def battery_terminal_penalty(
    cfg: SiteConfig,
    soe: np.ndarray | dict[str, np.ndarray] | None,
    buy: np.ndarray | None = None,
) -> float:
    """Cost of ending the horizon with less energy than you started with.

    It bills the net depletion at the terminal price, which is what stops a
    plan from looking cheap simply by emptying the battery.

    Accepts either a single trajectory (battery 0, the common case) or a
    key -> trajectory mapping, in which case every battery is charged.
    """
    if soe is None or not cfg.battery_list:
        return 0.0

    if not isinstance(soe, dict):
        soe = {SiteConfig.battery_key(0): soe}

    total = 0.0
    for i, b in enumerate(cfg.battery_list):
        traj = soe.get(SiteConfig.battery_key(i))
        if traj is None:
            continue
        if buy is not None:
            total += float(traj[0] - traj[-1]) * terminal_price(b, buy)
    return total


def total_objective(
    cfg: SiteConfig,
    net_grid: np.ndarray,
    fc: Forecasts,
    soe: np.ndarray | dict[str, np.ndarray] | None,
) -> float:
    """Bill + comfort + grid-breach penalties + battery terminal penalty.

    Lower is better, and every term is in currency, so the sum is a money
    number that can be compared against another solver's objective. Used for
    round selection, which is therefore a single comparison: the coordinator
    cannot trade a comfort violation or a grid breach for a smaller bill,
    because both already cost money here.

    The grid term is identically zero with no limit configured, so the
    benchmark objectives are exactly the bill-plus-comfort they always were.
    """
    return (
        net_cost(net_grid, fc.buy, fc.sell, cfg.horizon.dt)
        + grid_penalty(cfg, net_grid, fc.buy, fc.sell)
        + battery_terminal_penalty(cfg, soe, fc.buy)
    )


def terminal_price(cfg: BatteryConfig, buy: np.ndarray) -> float:
    """Value of a kWh left in the battery at the horizon edge.

    Defaults to the cheapest import price on the horizon: a lower bound on
    what that energy can save, so it never encourages hoarding beyond what
    the prices justify.
    """
    if cfg.terminal_price is not None:
        return float(cfg.terminal_price)
    return float(np.min(buy))
