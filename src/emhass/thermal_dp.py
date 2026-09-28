"""Dynamic-programming solver for a heat-pump-charged thermal store.

A heat pump's COP depends on the temperature it charges the store to
(``COP = carnot * T_supply_K / (T_supply - T_outdoor)``), and the delivered heat is
``P_elec * COP(T_store)`` - a bilinear, non-convex coupling that a single MILP cannot
solve exactly. Dynamic programming sidesteps it: the state is the store
temperature(s); backward induction evaluates the true COP at every state and returns
the globally-optimal dispatch in a single pass, with no convexity assumption and no
linearisation.

The store may optionally be coupled to a second store it feeds through a temperature
gradient (e.g. a tank that supplies a larger sink), which adds a second state and is
still tractable on a day-ahead horizon. Costs are evaluated against a per-timestep
electricity price for the heat pump and a commodity price for an optional backup
source (gas/oil/electric element).

The model is intentionally generic - one heat-pump store, an optional coupled store,
an optional backup source - so it applies to any such configuration. Parameters carry
neutral defaults; callers supply the values for the store they are solving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Upper bound on the number of discretized temperature states for the coupled
# store. Its grid step defaults to a fine 0.1 C, so a wide configured band (e.g.
# a mis-set 0-100 C pool) would otherwise create ~1000 states and blow up the
# backward induction. When the band would exceed this, the step is coarsened so
# the state count stays bounded; the band itself is preserved.
MAX_COUPLED_STATES = 64

# Upper bound on the number of discretized temperature states for the heat-pump
# store itself. grid_step is user-configurable, so a fine step on a wide band (e.g.
# 0.1 C over 25-65 C -> 401 states) would make the backward induction slow and, past
# ~32767 states, overflow the int16 policy array into a silently wrong plan. When the
# band would exceed this, the step is coarsened so the count stays bounded; the band
# is preserved. The default 0.5 C step stays well under the cap (unchanged behaviour).
MAX_TANK_STATES = 200


@dataclass
class ThermalDPParams:
    """Physical parameters for the heat-pump store (and an optional coupled store)."""

    # Heat-pump store: the charged store whose temperature sets the COP.
    heat_capacity: float = 1.0  # kWh/K
    loss_coeff: float = 0.0  # kW/K to ambient
    min_temp: float = 25.0
    max_temp: float = 65.0
    grid_step: float = 0.5
    # Reference BOTH stores' standing loss is taken against. Scalar, or a per-step
    # array so a diurnal swing (cold night / mild day) prices loss per step exactly
    # like the LP's own tank dynamics do - a horizon mean disagrees with the LP on
    # precisely the swing days the DP exists to refine.
    ambient_temperature: float | np.ndarray = 20.0

    # Heat pump charging the store, plus an optional flat-efficiency backup source.
    # mode "heat": the unit ADDS heat, condenser runs at store_temp + approach, charging
    # raises temperature. mode "cool": the unit REMOVES heat (chiller), evaporator runs at
    # store_temp - approach, charging LOWERS temperature; the COP, standing-loss sign and
    # charging direction all invert. Cool mode does not support a coupled store yet.
    mode: str = "heat"  # "heat" | "cool"
    carnot_efficiency: float = 0.45
    hx_approach: float = 5.0  # condenser runs at store_temp + approach
    cop_bounds: tuple[float, float] = (1.0, 8.0)
    hp_max_power: float = 3.0  # kW electric
    # Optional rated thermal-output ceiling (kW thermal). hp_max_power * COP is
    # electrical headroom, not what the compressor can physically move: on a
    # mild day a high COP would otherwise let the DP plan deliveries far above
    # the unit's rating (mirrors the LP's per-source max_thermal_power cap).
    max_thermal_power: float | None = None
    backup_price: float = 0.10  # commodity price of the backup per kWh input
    backup_efficiency: float = 0.95
    backup_max_power: float = 0.0  # kW input; 0 disables the backup

    # Optional second store fed by a temperature-gradient transfer from the store.
    coupled_heat_capacity: float | None = None  # None keeps the model single-state
    coupled_loss_coeff: float = 0.0  # kW/K
    coupled_min_temp: float = 0.0
    coupled_max_temp: float = 100.0
    coupled_grid_step: float = 0.1
    coupling_coeff: float = 1.0  # kW/K transfer conductance (store -> coupled)
    coupling_max_power: float = 20.0  # kW
    coupling_levels: int = 11

    # Optional soft comfort target on the store: comfort_target[t] is the target
    # for the END of step t (NaN: none), and comfort_penalty prices each degree of
    # shortfall below it (above it in cool mode) - the LP's desired_temperature /
    # penalty_factor term. Without it the DP sees only the hard floor and plans the
    # store cold while the LP heats it toward the target. Not included in
    # ThermalDPResult.total_cost, which stays the energy cost.
    comfort_target: float | np.ndarray | None = None
    comfort_penalty: float = 0.0
    # Optional ceiling (scalar or per step) on the temperature the heat pump can
    # charge the store to - its max_supply_temperature. Above it only the backup
    # adds heat, as in the LP (heat mode).
    hp_max_temp: float | np.ndarray | None = None

    # External demand drawn directly from the heat-pump store. May be negative
    # (a net solar/window gain): free heat that warms the store; any surplus the
    # discretized state cannot absorb is shed at zero cost, never purchased away.
    demand_kw: float | np.ndarray = 0.0


@dataclass
class ThermalDPResult:
    tank_trajectory: np.ndarray
    coupled_trajectory: np.ndarray | None
    hp_electric_per_step: np.ndarray  # kW electric drawn by the heat pump each step
    backup_input_per_step: np.ndarray  # kW input to the backup source each step
    total_cost: float
    solve_seconds: float
    meta: dict = field(default_factory=dict)


def _per_step(value, n, fill):
    """A scalar or a per-step series as a float array of length n (a short
    series is padded with `fill`)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if len(arr) < n:
        arr = np.concatenate([arr, np.full(n - len(arr), fill)])
    return arr[:n]


def solve_thermal_dp(
    price: np.ndarray,
    outdoor_temperature: float | np.ndarray,
    params: ThermalDPParams,
    time_step: float = 0.5,
    tank_start: float = 35.0,
    coupled_start: float = 26.5,
) -> ThermalDPResult:
    """Exact DP for a heat-pump store (and optional coupled store) against a price.

    ``price[t]`` is the marginal cost of heat-pump electricity at step t. The DP
    returns the globally optimal store temperature trajectory and the heat-pump /
    backup dispatch.

    Outdoor temperature may be a scalar or a per-step array; the COP is evaluated per
    timestep against that step's outdoor temperature, so intraday swings (a cold dawn,
    a mild PV-rich midday) are reflected - not averaged away. For a charging step the COP
    is taken at the TARGET (end-of-step) temperature ``T[t+1] + approach``: over a control
    step the heat pump runs at the supply temperature needed to reach the step's setpoint,
    so the COP is set by the temperature it heats to, not the cooler one it starts from.
    """
    import time

    t0 = time.time()
    N = len(price)
    dt = time_step
    p = params
    # Sense of the unit: +1 heat (adds heat; charging raises temperature), -1 cool
    # (removes heat; charging lowers temperature). Flips the energy-balance / loss sign.
    sc = -1.0 if p.mode == "cool" else 1.0
    outdoor_arr = (
        np.full(N, float(outdoor_temperature))
        if np.ndim(outdoor_temperature) == 0
        else np.asarray(outdoor_temperature, dtype=float)
    )
    if len(outdoor_arr) < N:  # forward-fill a short outdoor series to the horizon
        outdoor_arr = np.concatenate([outdoor_arr, np.full(N - len(outdoor_arr), outdoor_arr[-1])])
    outdoor_arr = outdoor_arr[:N]
    # Ambient reference for standing loss: scalar or per-step (forward-filled like
    # the outdoor series), so loss is priced per step exactly as the LP prices it.
    amb = (
        np.full(N, float(p.ambient_temperature))
        if np.ndim(p.ambient_temperature) == 0
        else np.asarray(p.ambient_temperature, dtype=float)
    )
    if len(amb) < N:
        amb = np.concatenate([amb, np.full(N - len(amb), amb[-1])])
    amb = amb[:N]
    demand = p.demand_kw if np.ndim(p.demand_kw) else np.full(N, float(p.demand_kw))
    demand = np.broadcast_to(demand, (N,))

    span_t = p.max_temp - p.min_temp
    if span_t > 0:
        # Resolution from grid_step, hard-capped at MAX_TANK_STATES (bounds runtime and
        # keeps nt < int16 so the policy array cannot overflow). linspace for an exact
        # count; the default 0.5 C step lands below the cap, leaving the grid unchanged.
        nt = min(MAX_TANK_STATES, max(2, int(round(span_t / p.grid_step)) + 1))
        grid = np.linspace(p.min_temp, p.max_temp, nt)
    else:
        grid = np.array([p.min_temp])
    nt = len(grid)
    # COP for a charging step i -> j is taken at the TARGET temperature grid[j]: over a
    # 30-minute control step the heat pump receives one setpoint and runs its condenser
    # at the supply temperature needed to REACH it (grid[j] + approach), so the COP is set
    # by the temperature it heats TO, not the cooler temperature it starts from. This is
    # both the physically faithful model for a setpoint-driven HP and the conservative one
    # (it prices a large heating step at its true cost). cop2d[t, j] evaluates the target
    # supply against step t's outdoor temperature, so an intraday swing is still captured.
    # Supply temperature the unit runs at to reach target grid[j]. Heat: the condenser runs
    # ABOVE the store (grid + approach). Cool: the evaporator runs BELOW the store
    # (grid - approach) to extract heat. The Carnot COP pays for the lift between the hot and
    # cold sides: condenser - outdoor (heat) or outdoor - evaporator (cool). A vanishing or
    # negative lift (heat: outdoor >= supply, a low store on a hot day; cool: outdoor <=
    # supply, a chilled store on a cold day) clamps to the resistive FLOOR
    # (cop_bounds[0]) to match the canonical LP helper (utils.calculate_cop_heatpump),
    # which prices non-physical lift at the neutral 1.0. The physically tempting
    # "free heat" upper clamp was tried and reverted: the DP then routed trajectories
    # through states the LP re-solve re-priced ~8x more expensive - a self-
    # inconsistent refinement. Consistency with the executor wins.
    if p.mode == "cool":
        supply = grid - p.hx_approach  # (nt,) cold evaporator side
        lift = outdoor_arr[:, None] - supply[None, :]  # (N, nt) outdoor - evaporator
    else:
        supply = grid + p.hx_approach  # (nt,) condenser supply needed to reach each target
        lift = supply[None, :] - outdoor_arr[:, None]  # (N, nt) condenser - source
    carnot = p.carnot_efficiency * (supply[None, :] + 273.15) / np.where(lift > 1e-6, lift, 1e-6)
    cop2d = np.clip(
        np.where(lift > 1e-6, carnot, p.cop_bounds[0]), *p.cop_bounds
    )  # (N, nt): cop2d[t, j] = COP to charge to grid[j] at step t
    hp_cap_th = p.hp_max_power * cop2d * dt  # (N, nt) thermal kWh the HP delivers reaching j
    if p.max_thermal_power is not None:
        # The compressor's rated thermal output binds before electrical * COP does
        # whenever the COP is high - the same physical ceiling the LP enforces.
        hp_cap_th = np.minimum(hp_cap_th, p.max_thermal_power * dt)
    if p.hp_max_temp is not None and p.mode != "cool":
        hp_ceiling = _per_step(p.hp_max_temp, N, np.inf)
        hp_cap_th = np.where(grid[None, :] <= hp_ceiling[:, None] + 1e-9, hp_cap_th, 0.0)
    qin_max = hp_cap_th + p.backup_max_power * p.backup_efficiency * dt  # (N, nt)
    # Standing loss per step and per state: (N, nt). Negative entries (store colder
    # than ambient) are a passive gain - handled by the surplus slack below.
    loss = p.loss_coeff * (grid[None, :] - amb[:, None]) * dt
    cp_backup = p.backup_price / p.backup_efficiency
    # Comfort shortfall penalty for reaching target grid[j] at the end of step t.
    comfort = np.zeros((N, nt))
    if p.comfort_target is not None and p.comfort_penalty > 0:
        tgt = np.asarray(
            np.broadcast_to(p.comfort_target, (N,))
            if np.ndim(p.comfort_target) == 0
            else p.comfort_target,
            dtype=float,
        )
        if len(tgt) < N:
            tgt = np.concatenate([tgt, np.full(N - len(tgt), np.nan)])
        short = sc * (tgt[:N, None] - grid[None, :])  # (N, nt), NaN where no target
        comfort = p.comfort_penalty * np.where(np.isnan(short), 0.0, np.maximum(short, 0.0))

    use_coupled = p.coupled_heat_capacity is not None
    if use_coupled and p.mode == "cool":
        raise NotImplementedError(
            "cool-mode thermal DP does not support a coupled store yet; "
            "the optimizer keeps the static COP for cool tanks with coupling."
        )
    if use_coupled:
        span = p.coupled_max_temp - p.coupled_min_temp
        if span > 0:
            # Resolution from the configured step, but hard-capped at MAX_COUPLED_STATES.
            # linspace (not arange) so the count is exact - arange's float endpoint fudge
            # could otherwise blow the count up on a tiny span.
            npts = min(MAX_COUPLED_STATES, max(2, int(round(span / p.coupled_grid_step)) + 1))
            cgrid = np.linspace(p.coupled_min_temp, p.coupled_max_temp, npts)
        else:
            cgrid = np.array([p.coupled_min_temp])
        ncl = len(cgrid)
        # Coupled-store standing loss, per step and per state: (N, ncl) - against the
        # same per-step ambient as the primary store.
        closs = p.coupled_loss_coeff * (cgrid[None, :] - amb[:, None]) * dt
        qxf_levels = np.linspace(0.0, p.coupling_max_power * dt, p.coupling_levels)
    else:
        cgrid = np.array([0.0])
        ncl = 1
        qxf_levels = np.array([0.0])

    INF = 1e15

    # Backward induction, vectorized over (current state i, target j) with a short
    # Python loop over the transfer levels qi only: N * nq iterations instead of
    # N * nt * nq (measured ~36x wall-clock on a coupled buffer+pool day-ahead
    # horizon, where the per-(t, j, qi) Python/numpy call overhead dominated).
    #
    # Precomputed invariants:
    # - delta_e[i, j]: energy to move the store from state i to target j.
    # - xf_bound[i, c]: conductance transfer bound. Clamped at 0: when the feeder is
    #   COLDER than the receiver heat cannot flow uphill, yet transferring NOTHING
    #   (qxf=0) must always stay legal - without the clamp the bound goes negative
    #   and every state with a warmer coupled store turns spuriously infeasible.
    #
    # Tie-breaking: ties are resolved lexicographically by (qi, j) - the previous
    # per-(j, qi) loop resolved by (j, qi). Both pick an exactly-optimal policy;
    # only the choice AMONG equal-cost optima can differ.
    delta_e = sc * (grid[None, :] - grid[:, None]) * p.heat_capacity  # (i, j)
    bk_cap = p.backup_max_power * p.backup_efficiency * dt
    if use_coupled:
        xf_bound = np.maximum(0.0, p.coupling_coeff * (grid[:, None] - cgrid[None, :])) * dt

    V = np.zeros((nt, ncl))
    POL = np.zeros((N, nt, ncl, 2), dtype=np.int16)
    for t in range(N - 1, -1, -1):
        Vn = np.full((nt, ncl), INF)
        best_j = np.zeros((nt, ncl), np.int16)
        best_q = np.zeros((nt, ncl), np.int16)
        # Net external term per current state i. Heat (sc=+1): Qin to ADD =
        # delta_e + demand + loss. Cool (sc=-1): Qin to REMOVE = -delta + demand
        # - loss, i.e. sc flips the temperature delta and the passive-loss term
        # while the demand stays a positive load.
        ext = demand[t] * dt + sc * loss[t]  # (i,)
        # Surplus free inflow per current state (nt,): when the net external term is
        # negative (solar/window gain beyond losses, or a store colder than ambient),
        # the discretized state may be unable to absorb it - e.g. a sub-grid-cell
        # gain makes even coasting need Qin < 0. Allow shedding EXACTLY that surplus
        # at zero cost (the cost below prices max(Qin, 0)): the physical store just
        # warms within its cell / saturates at its ceiling. Bounding the slack by
        # the actual free inflow is what keeps energy conserved - a universal slack
        # would let coupled-store transfers be fed with phantom heat.
        surplus = np.maximum(0.0, -ext)
        hp_cheaper = (price[t] / cop2d[t]) <= cp_backup  # (j,)
        for qi, qxf in enumerate(qxf_levels):
            Qin = delta_e + ext[:, None] + qxf  # (i, j)
            Qpos = np.maximum(Qin, 0.0)
            # Cheapest split between the HP (at target j's COP) and the backup.
            qhp = np.where(
                hp_cheaper[None, :],
                np.minimum(Qpos, hp_cap_th[t][None, :]),
                np.maximum(Qpos - bk_cap, 0.0),
            )
            qbk = Qpos - qhp
            cost = (qhp / cop2d[t][None, :]) * price[t] + qbk * cp_backup + comfort[t][None, :]
            feas_t = (Qin >= -surplus[:, None] - 1e-9) & (Qin <= qin_max[t][None, :] + 1e-9)
            if use_coupled and ncl > 1:
                Tc2 = cgrid + (qxf - closs[t]) / p.coupled_heat_capacity  # (c,)
                feas_c = (Tc2 >= cgrid[0] - 1e-9) & (Tc2 <= cgrid[-1] + 1e-9)
                # Linear interpolation of V[j, :] at Tc2 for every j at once
                # (replaces one np.interp call per (j, qi)): index/weight form.
                xc = np.clip(Tc2, cgrid[0], cgrid[-1])
                hi = np.clip(np.searchsorted(cgrid, xc, side="left"), 1, ncl - 1)
                lo = hi - 1
                w = (xc - cgrid[lo]) / (cgrid[hi] - cgrid[lo])
                Vnext = V[:, lo] * (1.0 - w)[None, :] + V[:, hi] * w[None, :]  # (j, c)
                mask = (
                    feas_t[:, :, None]
                    & feas_c[None, None, :]
                    & (qxf <= xf_bound + 1e-9)[:, None, :]
                )
                val = np.where(mask, cost[:, :, None] + Vnext[None, :, :], INF)  # (i, j, c)
            elif use_coupled:
                # Degenerate single-state coupled store (min == max): the transfer
                # must exactly offset its standing loss to stay on the only state.
                Tc2 = cgrid[0] + (qxf - closs[t, 0]) / p.coupled_heat_capacity
                feas_c = bool(cgrid[0] - 1e-9 <= Tc2 <= cgrid[-1] + 1e-9)
                mask = feas_t & feas_c & (qxf <= xf_bound[:, 0] + 1e-9)[:, None]
                val = np.where(mask, cost + V[:, 0][None, :], INF)[:, :, None]  # (i, j, 1)
            else:
                val = np.where(feas_t, cost + V[:, 0][None, :], INF)[:, :, None]  # (i, j, 1)
            # Best target j for this transfer level, then fold into the running best.
            idx_j = np.argmin(val, axis=1)  # (i, c)
            vmin = np.take_along_axis(val, idx_j[:, None, :], axis=1)[:, 0, :]  # (i, c)
            upd = vmin < Vn
            Vn = np.where(upd, vmin, Vn)
            best_j = np.where(upd, idx_j.astype(np.int16), best_j)
            best_q = np.where(upd, np.int16(qi), best_q)
        V = Vn
        POL[t, :, :, 0] = best_j
        POL[t, :, :, 1] = best_q

    # Forward rollout from the start state.
    it = int(np.argmin(np.abs(grid - tank_start)))
    ic = int(np.argmin(np.abs(cgrid - coupled_start))) if use_coupled else 0
    # V now holds the optimal cost-to-go at t=0. If the start state is INF, no feasible
    # policy exists (e.g. demand exceeds what the heat pump and backup can deliver every
    # step): flag it rather than returning a plausible-looking but energy-violating plan.
    start_infeasible = bool(V[it, ic] >= INF * 0.5)
    traj = [grid[it]]
    # The coupled store's state is carried continuously: snapping it to the grid
    # every step would round small per-step changes (a large store, a fine grid)
    # back to the same point forever. The policy is read at the nearest grid point.
    tc = float(np.clip(coupled_start, cgrid[0], cgrid[-1])) if use_coupled else 0.0
    ctraj = [tc] if use_coupled else None
    hp_draw = np.zeros(N)
    bk_draw = np.zeros(N)
    cost = 0.0
    for t in range(N):
        j = int(POL[t, it, ic, 0])
        qxf = qxf_levels[int(POL[t, it, ic, 1])]
        Qin = max(
            0.0,
            sc * (grid[j] - grid[it]) * p.heat_capacity + demand[t] * dt + sc * loss[t, it] + qxf,
        )
        cop_ij = cop2d[t, j]  # COP to reach the target temperature grid[j]
        if price[t] / cop_ij <= cp_backup:
            qhp = min(Qin, hp_cap_th[t, j])
            qbk = Qin - qhp
        else:
            qbk = min(Qin, p.backup_max_power * p.backup_efficiency * dt)
            qhp = Qin - qbk
        cost += (qhp / cop_ij) * price[t] + (qbk / p.backup_efficiency) * p.backup_price
        hp_draw[t] = qhp / cop_ij / dt
        bk_draw[t] = qbk / p.backup_efficiency / dt
        if use_coupled:
            tc = float(
                np.clip(tc + (qxf - closs[t, ic]) / p.coupled_heat_capacity, cgrid[0], cgrid[-1])
            )
            ic = int(np.argmin(np.abs(cgrid - tc)))
            ctraj.append(tc)
        it = j
        traj.append(grid[it])

    return ThermalDPResult(
        tank_trajectory=np.array(traj),
        coupled_trajectory=np.array(ctraj) if use_coupled else None,
        hp_electric_per_step=hp_draw,
        backup_input_per_step=bk_draw,
        total_cost=float("inf") if start_infeasible else float(cost),
        solve_seconds=time.time() - t0,
        meta={
            "tank_states": nt,
            "coupled_states": ncl,
            "horizon": N,
            "infeasible": start_infeasible,
        },
    )
