"""P9: perfect-optim solves every historical window completely across a DST change (#1263).

A perfect-optim window is 24 elapsed hours from a UTC midnight of ``days_list`` (not a local
calendar day): a window with a DST change has 23 or 25 local hours and as many steps as any other.
These tests run the real solver on offline history (``test_df_final.pkl``, no Home Assistant).

P9  the optimizer output holds, for every window, exactly the UTC instants an independent oracle
    gives (cases.perfect_window_instants): in order, once each, none dropped or repeated, with
    both passes of a repeated hour and the steps after a skipped one. Recorded PV, load and
    prices sit on the instant they belong to, and every step is solved.
"""

from __future__ import annotations

import functools
import pickle

import pandas as pd
import pytest
from cases import NORMAL_DAY_LOCAL, STEPS, TRANSITIONS, perfect_window_instants
from harness import LOG, Env, marker, run_async
from pandas.testing import assert_frame_equal

from emhass import command_line as cl

WINDOWS = 2
NOW = pd.Timestamp("2026-10-28T12:00Z")  # after every history used here
OPTIMAL = {"Optimal", "Optimal (Relaxed)"}


def _cases():
    """The window before a transition and the one containing it. Kathmandu (+05:45, no DST)
    stands for a fractional offset on an ordinary day. ``repeated`` is the number of steps whose
    local wall time occurs twice (both passes of a fall-back), counted on the oracle instants."""
    for zone in ("Europe/Berlin", "Australia/Sydney", "Australia/Lord_Howe", "Asia/Kathmandu"):
        transitions = TRANSITIONS[zone] or [("normal", f"{NORMAL_DAY_LOCAL[:10]}T00:00Z", 0)]
        for kind, t_iso, _ in transitions:
            first_day = pd.Timestamp(t_iso).normalize() - pd.Timedelta(days=1)
            for step in STEPS:
                local = perfect_window_instants(first_day, WINDOWS, step).tz_convert(zone)
                repeated = int(local.tz_localize(None).duplicated(keep=False).sum())
                yield pytest.param(zone, first_day, step, repeated, id=f"{zone}-{kind}-{step}min")


# Recorded values as functions of the instant: distinct periodic series, so a value on the wrong
# step (even by one) is caught.
def _k(utc, step):
    return int(round(marker(utc) / step))


def pv(utc, step):
    return 50.0 + 3.0 * (_k(utc, step) % 211)


def load(utc, step):
    return 300.0 + (_k(utc, step) % 307)


def load_cost(utc, step):
    return 0.10 + 0.001 * (_k(utc, step) % 97)


def prod_price(utc, step):
    return 0.05 + 0.0005 * (_k(utc, step) % 89)


@functools.cache
def _env(zone, step):
    return Env(zone, step, "nearest")


def _perfect_input(zone, step, first_day, drop=(), shift=pd.Timedelta(0)):
    """set_input_data_dict for perfect-optim on offline history: WINDOWS days to solve plus the
    current day, which perfect-optim never solves. ``drop`` removes history instants and ``shift``
    moves the whole history grid off the UTC midnights."""
    env = _env(zone, step)
    rc = env.params["retrieve_hass_conf"]
    names = [
        rc["sensor_power_load_no_var_loads"],
        rc["sensor_power_photovoltaics"],
        rc["sensor_power_photovoltaics_forecast"],
    ]
    idx = (perfect_window_instants(first_day, WINDOWS + 1, step) + shift).drop(list(drop))
    history = pd.DataFrame(
        {
            names[0]: [load(t, step) for t in idx],
            names[1]: [pv(t, step) for t in idx],
            names[2]: 1.0,
        },
        index=idx,
    )
    days = pd.date_range(first_day, periods=WINDOWS + 1, freq="D", tz="UTC")
    with open(env.data_path / "test_df_final.pkl", "wb") as f:
        pickle.dump((history, days, names, {}), f)
    idd, err = env.set_input(NOW, {}, "perfect-optim", get_data_from_file=True)
    assert err is None, err
    return idd


def _with_prices(idd, step):
    """The input frame perfect-optim solves, with prices that vary with the instant."""
    df = idd["df_input_data"].copy()
    utc = df.index.tz_convert("UTC")
    df["unit_load_cost"] = [load_cost(t, step) for t in utc]
    df["unit_prod_price"] = [prod_price(t, step) for t in utc]
    return df


@pytest.mark.parametrize("zone, first_day, step, repeated", list(_cases()))
def test_every_window_complete_and_bound_to_its_instants(zone, first_day, step, repeated):
    idd = _perfect_input(zone, step, first_day)
    res = idd["opt"].perform_perfect_forecast_optim(_with_prices(idd, step), idd["days_list"])

    instants = perfect_window_instants(first_day, WINDOWS, step)
    expected = pd.DataFrame(
        {
            "P_PV": [pv(t, step) for t in instants],
            "P_Load": [load(t, step) for t in instants],
            "unit_load_cost": [load_cost(t, step) for t in instants],
            "unit_prod_price": [prod_price(t, step) for t in instants],
        },
        index=instants.as_unit("ns"),
    )
    # in order, once each, none dropped or repeated, every value on its own instant
    actual = res[list(expected.columns)].set_axis(res.index.tz_convert("UTC").as_unit("ns"))
    assert_frame_equal(actual, expected, check_freq=False)
    assert set(res["optim_status"]) <= OPTIMAL
    # both passes of a repeated local hour are there: each such wall time twice
    assert res.index.tz_localize(None).duplicated(keep=False).sum() == repeated


def test_perfect_forecast_optim_action_across_fall_back():
    """The action itself (price preparation, solve, result), Berlin 2026-10-25."""
    first_day = pd.Timestamp("2026-10-24", tz="UTC")
    idd = _perfect_input("Europe/Berlin", 30, first_day)
    res = run_async(cl.perfect_forecast_optim(idd, LOG, debug=True))

    expected = perfect_window_instants(first_day, WINDOWS, 30)
    assert list(res.index.tz_convert("UTC")) == list(expected)
    assert list(res["P_PV"]) == pytest.approx([pv(t, 30) for t in expected])
    assert list(res["P_Load"]) == pytest.approx([load(t, 30) for t in expected])
    assert set(res["optim_status"]) <= OPTIMAL
    assert not res.isna().any().any()


def test_missing_history_skips_only_its_own_window():
    """A window with a missing step is still skipped (not filled in); the DST change is no reason
    to lose the windows around it."""
    first_day = pd.Timestamp("2026-10-24", tz="UTC")
    missing = pd.Timestamp("2026-10-25T00:30Z")  # inside the second window (the fall-back day)
    idd = _perfect_input("Europe/Berlin", 30, first_day, drop=[missing])
    res = idd["opt"].perform_perfect_forecast_optim(_with_prices(idd, 30), idd["days_list"])
    assert list(res.index.tz_convert("UTC")) == list(perfect_window_instants(first_day, 1, 30))


def test_history_on_a_local_midnight_grid_is_still_solved():
    """Websocket history is resampled from local midnight, so in a +05:45 zone its 30-minute steps
    sit at :15/:45 UTC, not on the UTC midnights. Every window still has all its steps."""
    first_day = pd.Timestamp("2026-10-13", tz="UTC")
    shift = pd.Timedelta(minutes=15)
    idd = _perfect_input("Asia/Kathmandu", 30, first_day, shift=shift)
    res = idd["opt"].perform_perfect_forecast_optim(_with_prices(idd, 30), idd["days_list"])

    expected = perfect_window_instants(first_day, WINDOWS, 30) + shift
    assert list(res.index.tz_convert("UTC")) == list(expected)
    assert list(res["P_PV"]) == pytest.approx([pv(t, 30) for t in expected])
    assert set(res["optim_status"]) <= OPTIMAL


def test_normal_day_equals_a_direct_solve_of_its_window():
    """Ordinary day: the window result is what perform_optimization gives for the same slice."""
    first_day = pd.Timestamp("2026-10-13", tz="UTC")
    idd = _perfect_input("Europe/Berlin", 30, first_day)
    df = _with_prices(idd, 30)
    res = idd["opt"].perform_perfect_forecast_optim(df, idd["days_list"])

    window = df.loc[perfect_window_instants(first_day, 1, 30)]
    opt = _perfect_input("Europe/Berlin", 30, first_day)["opt"]  # fresh problem, same config
    direct = opt.perform_optimization(
        window,
        window[opt.var_pv].values,
        window[opt.var_load_new].values,
        window[opt.var_load_cost].values,
        window[opt.var_prod_price].values,
    )
    assert_frame_equal(res.iloc[: len(direct)], direct)
