"""The participant interface: one query in, a plan and its private cost back.

A participant is one device, or a group of devices planned together by one
model. Every participant answers the same query, so the coordinator never
sees a model: it asks, at given prices, what the participant would do and
what that would cost it privately (everything but the bill).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

SCHEMA_VERSION = 1
KINDS = ("price_response", "best_response", "proximal")
STATUSES = ("ok", "infeasible", "fallback")


@dataclass(frozen=True)
class Query:
    """What a coordinator asks a participant.

    price_response  its cheapest plan when its energy costs `price_draw` drawn
                    and `price_supply` supplied (equal: Dantzig-Wolfe's query).
    best_response   the same at the real tariff, metered under `residual_kw`,
                    the rest of the house (a polish step, or an extra offer).
    proximal        no price, a pull of weight `rho` toward `target_kw` (reserved).

    Prices are currency/kWh per slot; powers kW, + = drawn from the meter.
    """

    kind: str
    price_draw: np.ndarray | None = None
    price_supply: np.ndarray | None = None
    residual_kw: np.ndarray | None = None
    target_kw: np.ndarray | None = None
    rho: float = 0.0


@dataclass
class Answer:
    """A plan (kW per slot, + = drawn), its private cost (everything but the
    bill, in currency) and its state trajectory (one point more than the plan).
    `detail` is the solver's own result, kept in-process and never serialised."""

    plan_kw: np.ndarray
    private_cost: float
    trajectory: np.ndarray
    status: str = "ok"
    detail: Any = None


@runtime_checkable
class Participant(Protocol):
    """What a coordinator needs from a participant.

    Two optional attributes refine how the coordinator treats it:
    `onoff` (bool) - True if its plans are on/off schedules, so the coordinator
    fixes it to one plan before re-pricing the others (DWCoordinator.run's
    dive); taken as `not modulating` when absent. `blend(weights, details)` -
    its own result for a weighted mix of its plans, given each plan's
    `Answer.detail`; called only when `modulating` is True.
    """

    key: str  # unique name in the coordinator (e.g. "battery", "deferrable0+deferrable1")
    max_power_kw: float  # the most it can draw or supply in one slot (bounds the meter)
    modulating: bool  # can a weighted blend of its plans run as it is?

    def respond(self, query: Query) -> Answer:
        """Its answer to `query`: the plan it would run and that plan's private
        cost. If its solve fails it still answers with a plan it can run (for
        example its last good one), with status "fallback" or "infeasible"; the
        coordinator then uses that plan but proves no bound from it."""
        ...

    def baseline(self) -> Answer:
        """A feasible plan to start from, with no price given (for example its
        thermostat or its plan at the import tariff)."""
        ...


# --------------------------------------------------------------------- JSON
