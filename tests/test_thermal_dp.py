"""Tests for the DP thermal solver (src/emhass/thermal_dp.py)."""

import numpy as np

from emhass.thermal_dp import (
    MAX_COUPLED_STATES,
    MAX_TANK_STATES,
    ThermalDPParams,
    solve_thermal_dp,
)


def _spread_price(n=48, dt=0.5, cheap=0.05, dear=0.50, start_h=8):
    hrs = [(start_h + i * dt) % 24 for i in range(n)]
    return np.array([cheap if 8 <= h < 16 else dear for h in hrs])


def test_dp_super_heats_only_when_spread_pays():
    """No spread -> no super-heating; a big spread -> the store banks heat hot."""
    params = ThermalDPParams(demand_kw=4.0)
    flat = solve_thermal_dp(np.full(48, 0.20), outdoor_temperature=8.0, params=params)
    spread = solve_thermal_dp(_spread_price(), outdoor_temperature=8.0, params=params)
    assert flat.tank_trajectory.max() < spread.tank_trajectory.max()
    assert spread.tank_trajectory.max() > 45.0  # genuinely super-heats on a big spread


def test_dp_prefers_heat_pump_when_genuinely_cheaper():
    """When the heat pump is genuinely cheaper than the backup at every step (mild
    prices, cool store -> COP ~5-6 -> price/COP well below the backup commodity cost),
    the global optimum uses almost no backup. (On a big night price the HP/backup
    costs tie and the backup can be optimal for forced maintenance - the DP gets that
    right too; this test isolates the clear case.)"""
    params = ThermalDPParams(
        coupled_heat_capacity=163.0,
        coupled_min_temp=26.0,
        coupled_max_temp=30.0,
        demand_kw=2.0,
        heat_capacity=1.163,
    )
    mild = _spread_price(cheap=0.08, dear=0.22)  # 0.22/COP(~4.7) = 0.047 << gas 0.105
    res = solve_thermal_dp(mild, outdoor_temperature=8.0, params=params)
    backup_kwh = float(np.sum(res.backup_input_per_step) * 0.5)
    hp_kwh = float(np.sum(res.hp_electric_per_step) * 0.5)
    assert backup_kwh < 0.10 * hp_kwh  # backup is negligible when the HP clearly wins


def test_dp_cop_matches_canonical_floor_when_outdoor_above_supply():
    """Consistency regression: when it is warmer outside than the condenser supply
    (lift <= 0), the LP's canonical COP helper (utils.calculate_cop_heatpump) prices
    the state at the neutral floor 1.0 - and the DP MUST price it the same way.
    An earlier fix clamped this regime to the UPPER bound (the 'free heat' physical
    reading), but that let the DP select trajectories through COP-8 states that the
    LP re-solve then re-priced at COP 1.0, a self-inconsistent refinement (the DP's
    whole purpose is COP/trajectory consistency with the re-solve). Consistency with
    the executor wins over isolated physics. Effective COP is recovered from the
    energy balance (heat delivered / electricity drawn)."""
    params = ThermalDPParams(
        heat_capacity=2.0,
        loss_coeff=0.0,
        min_temp=25.0,
        max_temp=30.0,
        hx_approach=5.0,  # supply = tank + 5 = 30..35 C
        hp_max_power=6.0,  # enough electrical headroom at COP 1
        backup_max_power=0.0,  # only the HP can meet the demand
        demand_kw=2.0,
        cop_bounds=(1.0, 8.0),
    )
    n = 12
    # Outdoor (40 C) is hotter than every supply temperature -> lift <= 0 everywhere.
    res = solve_thermal_dp(np.full(n, 0.20), outdoor_temperature=40.0, params=params)
    traj = res.tank_trajectory
    heat_delivered = params.demand_kw * 0.5 * n + params.heat_capacity * (traj[-1] - traj[0])
    hp_elec = float(np.sum(res.hp_electric_per_step) * 0.5)
    assert hp_elec > 0, "the HP must run to meet the demand"
    eff_cop = heat_delivered / hp_elec
    assert eff_cop < 1.5, (
        f"effective COP {eff_cop:.2f} - non-physical lift must price at the canonical "
        "floor (1.0) to stay consistent with the LP re-solve, not the upper bound"
    )


def test_dp_per_step_ambient_prices_loss_per_step():
    """Standing loss must be priced against the per-step ambient, not a horizon
    mean: on a cold-night/mild-day horizon the store coasts down faster at night.
    (The LP's own tank dynamics use the per-step outdoor array; a mean-ambient DP
    disagrees with the LP on exactly the diurnal-swing days it should refine.)"""
    base = {
        "heat_capacity": 1.0,
        "loss_coeff": 0.2,  # strong loss so the effect is unambiguous
        "min_temp": 20.0,
        "max_temp": 60.0,
        "demand_kw": 0.0,
        "backup_max_power": 0.0,
    }
    n = 8
    cold_then_warm = np.array([0.0] * 4 + [40.0] * 4)
    # Expensive flat price + no demand -> optimal policy is (near-)pure coasting.
    res = solve_thermal_dp(
        np.full(n, 5.0),
        outdoor_temperature=40.0,
        params=ThermalDPParams(ambient_temperature=cold_then_warm, **base),
        tank_start=40.0,
    )
    # Known quantization wart (predates per-step ambient): a coast landing between
    # grid cells is rounded UP with paid heat, bounded by one cell (grid_step * hc)
    # thermal per step. With COP >= 1, electric energy <= that thermal pad. Allow
    # the pad, nothing more - real dispatch would be far larger.
    hp_elec_kwh = float(np.sum(res.hp_electric_per_step)) * 0.5
    assert hp_elec_kwh <= 4 * 0.5 * base["heat_capacity"] + 1e-6, (
        "HP energy exceeds the one-cell-per-step rounding pad: not coasting"
    )
    traj = res.tank_trajectory
    drop_cold = traj[0] - traj[4]  # loss over the 4 cold steps
    drop_warm = traj[4] - traj[8]  # loss over the 4 warm steps (ambient 40 >= tank)
    assert drop_cold > drop_warm + 1.0, (
        f"cold-step loss ({drop_cold:.2f} C) must exceed warm-step loss "
        f"({drop_warm:.2f} C): per-step ambient is not being applied"
    )


def test_dp_negative_demand_warms_store_for_free():
    """A negative demand step (net solar/window gain) is free heat the store can
    BANK against later demand - not zero demand (the old clamp). The gain steps
    provide 4 kWh; the final demand needs 4 kWh but only 2 kWh is drainable from
    storage (min_temp), so without the gain the HP must buy the difference."""

    def run(with_gain: bool):
        gain = -2.0 if with_gain else 0.0
        params = ThermalDPParams(
            heat_capacity=1.0,
            loss_coeff=0.0,
            min_temp=28.0,
            max_temp=60.0,
            demand_kw=np.array([0.0, 0.0] + [gain] * 4 + [4.0, 4.0]),
            backup_max_power=0.0,
        )
        # Price so high the HP only runs when physically forced.
        return solve_thermal_dp(
            np.full(8, 5.0), outdoor_temperature=10.0, params=params, tank_start=30.0
        )

    banked = run(with_gain=True)
    assert not banked.meta["infeasible"]
    assert float(np.sum(banked.hp_electric_per_step)) < 1e-9, (
        "banked free gain plus storage covers the demand - the HP must stay off"
    )
    # Load-bearing control: without the gain the same demand forces the HP on.
    control = run(with_gain=False)
    assert not control.meta["infeasible"]
    assert float(np.sum(control.hp_electric_per_step)) > 0.1, (
        "control must need the HP, else the gain assertion proves nothing"
    )

    # Sub-grid-cell gain (0.2 kWh/step < the 0.5 kWh cell): representable only by
    # shedding the surplus while coasting. Without the surplus slack the DP's only
    # feasible moves are a PAID jump to the next grid cell or infeasibility - the
    # optimizer must never buy heat to 'absorb' free gain.
    params_sub = ThermalDPParams(
        heat_capacity=1.0,
        loss_coeff=0.0,
        min_temp=20.0,
        max_temp=60.0,
        demand_kw=np.full(8, -0.4),
        backup_max_power=0.0,
    )
    res_sub = solve_thermal_dp(
        np.full(8, 5.0), outdoor_temperature=10.0, params=params_sub, tank_start=30.0
    )
    assert not res_sub.meta["infeasible"]
    assert float(np.sum(res_sub.hp_electric_per_step)) < 1e-9, (
        "sub-cell gain must never force the HP to buy heat"
    )


def test_dp_negative_demand_at_max_temp_stays_feasible():
    """Surplus gain with the store already at its ceiling must shed the excess
    (the physical tank saturates), not declare the whole horizon infeasible."""
    params = ThermalDPParams(
        heat_capacity=1.0,
        loss_coeff=0.0,
        min_temp=20.0,
        max_temp=40.0,
        demand_kw=np.full(6, -3.0),  # relentless gain
        backup_max_power=0.0,
    )
    res = solve_thermal_dp(
        np.full(6, 5.0), outdoor_temperature=10.0, params=params, tank_start=40.0
    )
    assert not res.meta["infeasible"]
    assert float(np.sum(res.hp_electric_per_step)) < 1e-9


def test_dp_gain_slack_cannot_mine_phantom_transfer_heat():
    """The surplus-dump slack must be bounded by the actual free inflow: a coupled
    store kept alive by transfers must still be paid for with real heat-pump energy
    (energy conservation), not phantom heat dumped through the slack."""
    params = ThermalDPParams(
        heat_capacity=1.0,
        loss_coeff=0.0,
        min_temp=20.0,
        max_temp=60.0,
        demand_kw=0.0,  # NO free inflow anywhere
        backup_max_power=0.0,
        coupled_heat_capacity=10.0,
        coupled_loss_coeff=0.5,  # the coupled store bleeds and must be fed
        coupled_min_temp=25.0,
        coupled_max_temp=30.0,
        coupling_coeff=5.0,
        coupling_max_power=20.0,
    )
    n = 8
    res = solve_thermal_dp(
        np.full(n, 0.20),
        outdoor_temperature=5.0,
        params=params,
        tank_start=40.0,
        coupled_start=27.0,
    )
    assert not res.meta["infeasible"]
    # Energy conservation: everything the tank sent into the coupled store
    # (its temperature change plus its standing losses, reconstructed from the
    # realized states) must be covered by real sources - HP heat (<= elec * COP
    # upper bound) plus heat drained from the feeder tank. Note the coupled
    # store may also legally coast down on its OWN bank (that shows up as a
    # negative temperature-change term), so only the net transfer is bounded.
    ctraj = res.coupled_trajectory
    closs_paid = float(
        sum(params.coupled_loss_coeff * (c - 20.0) * 0.5 for c in ctraj[:-1])
    )  # loss_coeff * (T - default ambient 20) * dt, at each realized state
    qxf_total = params.coupled_heat_capacity * (ctraj[-1] - ctraj[0]) + closs_paid
    hp_thermal_ub = float(np.sum(res.hp_electric_per_step) * 0.5) * 8.0  # COP <= 8
    tank_drain = params.heat_capacity * (res.tank_trajectory[0] - res.tank_trajectory[-1])
    assert qxf_total <= hp_thermal_ub + max(tank_drain, 0.0) + 1.0, (
        f"transfers into the coupled store ({qxf_total:.2f} kWh) exceed the real "
        f"energy available (HP <= {hp_thermal_ub:.2f} + tank drain "
        f"{max(tank_drain, 0.0):.2f} kWh): phantom heat"
    )


def test_dp_coupled_store_moves_in_small_steps():
    """A large coupled store changes by much less than one grid cell per step. The
    rollout carries its temperature continuously, so it drifts as its losses
    dictate instead of being snapped back to the same grid point every step."""
    params = ThermalDPParams(
        heat_capacity=1.0,
        min_temp=30.0,
        max_temp=55.0,
        demand_kw=0.0,
        backup_max_power=0.0,
        coupled_heat_capacity=163.0,
        coupled_loss_coeff=0.5,  # about 0.015 K per step at 30 C and 20 C ambient
        coupled_min_temp=26.0,
        coupled_max_temp=40.0,
        coupling_coeff=0.0,
        coupling_max_power=0.0,
    )
    res = solve_thermal_dp(
        np.full(48, 0.20),
        outdoor_temperature=5.0,
        params=params,
        tank_start=40.0,
        coupled_start=30.0,
    )
    ctraj = np.asarray(res.coupled_trajectory)
    assert ctraj[-1] < ctraj[0] - 0.3, ctraj


def test_dp_solves_quickly():
    """The 2-state DP must be fast enough to live inside the optimisation loop."""
    params = ThermalDPParams(
        coupled_heat_capacity=163.0, coupled_min_temp=26.0, coupled_max_temp=30.0, demand_kw=2.0
    )
    res = solve_thermal_dp(_spread_price(), outdoor_temperature=8.0, params=params)
    assert res.solve_seconds < 10.0
    assert res.coupled_trajectory is not None
    assert len(res.tank_trajectory) == 49  # N + 1


def test_dp_uses_per_step_outdoor_not_the_mean():
    """The COP follows the per-step outdoor temperature, not its average. With a warm
    first half (high COP) and a cold second half (low COP), the heat pump should bank
    heat early even though the price is marginally cheaper late - a schedule the
    mean-outdoor approximation (uniform COP) would not produce."""
    N = 6
    # Price marginally cheaper in the (cold) second half: a uniform-COP solver would
    # lean late; the true per-step COP makes the warm first half clearly cheaper.
    price = np.array([0.21, 0.21, 0.21, 0.20, 0.20, 0.20])
    warm_then_cold = np.array([15.0, 15.0, 15.0, -5.0, -5.0, -5.0])
    params = ThermalDPParams(
        heat_capacity=2.0,
        loss_coeff=0.0,
        min_temp=44.0,
        max_temp=60.0,
        hp_max_power=3.0,
        backup_max_power=0.0,
        demand_kw=2.0,
    )
    per_step = solve_thermal_dp(price, warm_then_cold, params, tank_start=45.0)
    mean_run = solve_thermal_dp(price, float(np.mean(warm_then_cold)), params, tank_start=45.0)
    assert len(per_step.tank_trajectory) == N + 1

    def early_share(res):
        total = res.hp_electric_per_step.sum()
        return res.hp_electric_per_step[:3].sum() / total if total > 0 else 0.0

    # The per-step run concentrates HP electricity in the warm, high-COP first half;
    # the mean-outdoor run (uniform COP, marginally cheaper late) does not.
    assert early_share(per_step) > early_share(mean_run) + 0.2


def test_dp_caps_coupled_state_count():
    """A pathologically wide coupled band must not blow up the state space: the grid
    step is coarsened so the coupled state count stays bounded (and the DP stays fast),
    while the band itself is preserved."""
    params = ThermalDPParams(
        coupled_heat_capacity=163.0,
        coupled_min_temp=0.0,
        coupled_max_temp=100.0,  # 1001 states at the raw 0.1 step
        coupled_grid_step=0.1,
        demand_kw=2.0,
    )
    res = solve_thermal_dp(_spread_price(), outdoor_temperature=8.0, params=params)
    assert res.meta["coupled_states"] <= MAX_COUPLED_STATES
    assert res.solve_seconds < 10.0
    assert res.coupled_trajectory is not None

    # Micro-span edge case: a near-degenerate band with a tiny configured step must
    # not blow up the count (an arange-with-epsilon would have produced thousands).
    micro = ThermalDPParams(
        coupled_heat_capacity=100.0,
        coupled_min_temp=26.0,
        coupled_max_temp=26.00000001,
        coupled_grid_step=1e-9,
        demand_kw=1.0,
    )
    res_micro = solve_thermal_dp(np.full(6, 0.2), outdoor_temperature=8.0, params=micro)
    assert res_micro.meta["coupled_states"] <= MAX_COUPLED_STATES


def test_dp_caps_tank_state_count():
    """A fine grid_step on a wide band must not blow up the heat-pump-store grid: it is
    coarsened to stay within MAX_TANK_STATES (bounding runtime and keeping the int16
    policy array from overflowing). The default 0.5 C step stays under the cap."""
    fine = ThermalDPParams(min_temp=25.0, max_temp=65.0, grid_step=0.1, demand_kw=2.0)
    res = solve_thermal_dp(_spread_price(), outdoor_temperature=8.0, params=fine)
    assert res.meta["tank_states"] <= MAX_TANK_STATES  # 401 raw -> capped
    # The default 0.5 C step is unchanged (well under the cap).
    default = ThermalDPParams(min_temp=25.0, max_temp=65.0, grid_step=0.5, demand_kw=2.0)
    res_def = solve_thermal_dp(_spread_price(), outdoor_temperature=8.0, params=default)
    assert res_def.meta["tank_states"] == 81


def test_dp_flags_infeasible_when_demand_exceeds_deliverable_heat():
    """A store starting at its floor (no stored heat to drain) with demand that exceeds
    what the heat pump can deliver every step and no backup has no feasible trajectory -
    the DP must flag it (meta['infeasible'], total_cost=inf) rather than return a
    plausible-looking plan that silently violates the energy balance."""
    params = ThermalDPParams(
        hp_max_power=1.0, backup_max_power=0.0, demand_kw=7.0, min_temp=30.0, max_temp=60.0
    )
    # start at the floor: cannot drain, and 3.5 kWh/step demand >> HP's ~2 kWh/step max.
    res = solve_thermal_dp(
        np.full(4, 0.2), outdoor_temperature=10.0, params=params, tank_start=30.0
    )
    assert res.meta["infeasible"] is True
    assert res.total_cost == float("inf")

    # The same store with a capable backup IS feasible (sanity: the flag is not stuck on).
    ok = ThermalDPParams(
        hp_max_power=1.0, backup_max_power=10.0, demand_kw=7.0, min_temp=30.0, max_temp=60.0
    )
    res_ok = solve_thermal_dp(np.full(4, 0.2), outdoor_temperature=10.0, params=ok, tank_start=30.0)
    assert res_ok.meta["infeasible"] is False
    assert np.isfinite(res_ok.total_cost)


def test_dp_feasible_when_feeder_colder_than_coupled_store():
    """A coupled store warmer than its feeder must NOT make the DP infeasible. Heat
    can't flow uphill into it, but transferring nothing (qxf=0) is always legal -
    the store just coasts. The conductance bound must be clamped at 0; without the
    clamp it goes negative and qxf in [0, negative] is empty, spuriously flagging
    every such state infeasible (the buffer/pool case where the buffer starts cool)."""
    params = ThermalDPParams(
        min_temp=25.0,
        max_temp=65.0,
        hp_max_power=4.0,
        backup_max_power=20.0,
        demand_kw=0.5,
        coupled_heat_capacity=100.0,
        coupled_loss_coeff=0.5,
        coupled_min_temp=15.0,
        coupled_max_temp=30.0,
        coupling_coeff=2.0,
        coupling_max_power=50.0,
    )
    # The feeder starts COLDER (25) than the coupled store (28): no uphill flow, but
    # the DP must still find the do-nothing policy rather than declaring infeasible.
    res = solve_thermal_dp(
        np.full(8, 0.2),
        outdoor_temperature=10.0,
        params=params,
        tank_start=25.0,
        coupled_start=28.0,
    )
    assert res.meta["infeasible"] is False
    assert np.isfinite(res.total_cost)


def _cool_params(**over):
    base = {
        "mode": "cool",
        "heat_capacity": 2.0,
        "loss_coeff": 0.0,
        "min_temp": 5.0,
        "max_temp": 15.0,
        "grid_step": 0.5,
        "carnot_efficiency": 0.45,
        "hx_approach": 5.0,
        "cop_bounds": (1.0, 8.0),
        "hp_max_power": 3.0,
        "ambient_temperature": 30.0,
        "demand_kw": 1.0,
    }
    base.update(over)
    return ThermalDPParams(**base)


def test_cool_mode_cools_against_demand():
    """Cool mode: a chilled store with a steady cooling load is held in band by the
    chiller removing heat (charging = lowering temperature), priced at the cooling
    Carnot COP - not the heating COP. With no cooling it would warm past its max."""
    res = solve_thermal_dp(np.full(24, 0.20), 30.0, _cool_params(), time_step=0.5, tank_start=10.0)
    assert res.meta["infeasible"] is False
    # The chiller must run to remove the steady load and hold the band.
    assert res.hp_electric_per_step.sum() > 0
    traj = res.tank_trajectory
    assert traj.min() >= 5.0 - 0.6
    assert traj.max() <= 15.0 + 0.6


def test_cool_mode_super_cools_only_when_spread_pays():
    """The cooling analog of heat banking: with a flat price the chiller cools lazily
    and the store never dips below its start; with a big cheap/dear spread it banks COLD
    during the cheap window (a markedly lower minimum) to coast the dear window."""
    # Fine grid (0.1 C) so the steady demand can coast the store up between cheap
    # windows; a coarse grid would pin warming below one grid step and forbid banking.
    params = _cool_params(demand_kw=2.0, heat_capacity=5.0, grid_step=0.1)
    flat = solve_thermal_dp(np.full(48, 0.20), 30.0, params, tank_start=10.0)
    spread = solve_thermal_dp(_spread_price(), 30.0, params, tank_start=10.0)
    assert spread.tank_trajectory.min() < flat.tank_trajectory.min() - 0.5


def test_cool_mode_rejects_coupled_store():
    """Cool mode does not yet support a coupled store; it must raise so the optimizer
    falls back to the static COP instead of silently using heating-physics coupling."""
    import pytest

    params = _cool_params(coupled_heat_capacity=50.0)
    with pytest.raises(NotImplementedError):
        solve_thermal_dp(np.full(24, 0.2), 30.0, params, time_step=0.5, tank_start=10.0)


def test_dp_prices_a_soft_comfort_target():
    """Without a comfort term the DP only sees the hard floor and keeps the store
    cold; with the LP's desired_temperature / penalty_factor term it heats toward
    the target (and not beyond it on a flat price). A NaN step carries no target."""
    params = ThermalDPParams(min_temp=20.0, max_temp=65.0, demand_kw=0.0)
    cold = solve_thermal_dp(np.full(24, 0.10), -5.0, params, tank_start=25.0)
    assert cold.tank_trajectory.max() <= 25.0
    target = np.full(24, 40.0)
    target[-1] = np.nan
    params.comfort_target = target
    params.comfort_penalty = 50.0
    warm = solve_thermal_dp(np.full(24, 0.10), -5.0, params, tank_start=25.0)
    assert warm.tank_trajectory[-2] >= 39.5
    assert warm.tank_trajectory.max() <= 40.0
    assert warm.total_cost > 0.0  # energy cost only; the penalty is not added


def test_dp_heat_pump_stops_at_its_max_supply_temperature():
    """Above the heat pump's max_supply_temperature only the backup can add heat,
    as in the LP. Without a backup the store stays at or below that ceiling even
    when a comfort target above it is priced heavily."""
    params = ThermalDPParams(
        min_temp=20.0,
        max_temp=65.0,
        comfort_target=np.full(24, 60.0),
        comfort_penalty=50.0,
    )
    free = solve_thermal_dp(np.full(24, 0.10), 5.0, params, tank_start=40.0)
    assert free.tank_trajectory.max() >= 59.5
    params.hp_max_temp = 55.0
    capped = solve_thermal_dp(np.full(24, 0.10), 5.0, params, tank_start=40.0)
    assert capped.tank_trajectory.max() <= 55.0
    params.backup_max_power = 10.0
    params.backup_price = 0.05
    backed = solve_thermal_dp(np.full(24, 0.10), 5.0, params, tank_start=40.0)
    assert backed.tank_trajectory.max() >= 59.5
    assert backed.backup_input_per_step.sum() > 0.0
