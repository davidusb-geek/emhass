# EMHASS time-grid / DST test suite (davidusb-geek/emhass#1189)

Offline pytest suite for #1189: one time grid per run, every value on its UTC instant, known failures as strict xfails.
Run from the repository root: `uv run --extra test --with time-machine python -m pytest tests/dst_1189 -q`
Expected on master `f571c59f` (14 Oct 2026): **3979 passed, 2485 xfailed**, no failures; 7 min 47 s on one core.

Without uv: `pip install -e '.[test]' time-machine && python -m pytest tests/dst_1189 -q`. A fix shows up as
`XPASS(strict)`, which fails the run: drop the matching predicate in `known.py` (or the test module)
and the cases run as normal tests. `-rxX` lists every expected failure with its class; `-k Berlin`
gives a quick look. `time-machine` is the only extra dependency; without it `conftest.py` skips
the folder, so a plain `pytest` (and upstream CI) is unaffected.

Free to use, adapt or trim, in whole or in part, without asking and without attribution.
Written with Claude Code (Anthropic), following `AGENTS.md` and `docs/develop_ai_coders.md`; the
commit carries a `Co-Authored-By` trailer. The findings behind the xfails were reproduced against
the real EMHASS package.

## What it checks

Black-box pytest suite that checks one contract: **a run has exactly one time grid, and
every value sits on the UTC instant it belongs to** - on normal days and across DST
transitions. It drives EMHASS only through public entry points (`set_input_data_dict`,
`prepare_forecast_and_weather_data`, `naive_mpc_optim`, `publish_data`,
`utils.get_forecast_dates`, `Forecast.get_forecast_out_from_csv_or_list`) and does **not**
patch EMHASS code. Home Assistant and the network are not needed: no token (so
`get_ha_config()` returns early), runtime lists / timestamp dicts / a CSV / a pre-seeded
Open-Meteo cache file, `publish_data(..., dont_post=True)`.

The clock is frozen with `time_machine`, which freezes `datetime.now()`,
`pd.Timestamp.now()` and `time.time()` together - i.e. every clock entry point EMHASS reads
(`utils._get_now`, `Forecast.__init__`, `_get_closest_index`, cache ages).

## Properties

| | property | where |
|---|---|---|
| P1 | runtime grid (`get_forecast_dates`) == `Forecast.forecast_dates` == optimizer input index | `test_grid_contract.py` |
| P2 | optimizer index unique, strictly increasing, constant real step | `test_grid_contract.py` |
| P3 | requested `prediction_horizon` delivered in full or an explicit error, never silently shorter | `test_grid_contract.py` |
| P4 | marked values (value = minutes since 2026-01-01Z of their instant) land on their UTC step: timestamp dicts, CSV with timestamps (UTC and local ISO), Open-Meteo cache, list path at method level | `test_alignment.py`, `test_list_path.py` |
| P5 | `publish-data` publishes the plan row chosen by `method_ts_round` on the real timeline, with the plan's own timestamp | `test_publish.py` |
| P6 | first step obeys `method_ts_round` (first may contain the running step, nearest <= half a step, last not in the past) | `test_grid_contract.py` |

Matrix: Europe/Berlin, Australia/Sydney, Europe/Dublin (negative DST), Africa/Casablanca
(Ramadan 2026, negative DST in pytz), Australia/Lord_Howe (30-min DST), Asia/Kathmandu
(+05:45, no DST); steps 5/15/30/60 min (CSV/Open-Meteo 15/30/60); `method_ts_round`
first/nearest/last; run instants before, at the start and in the middle of both passes of
the repeated interval, after it, around the spring-forward gap, the day before (window end
in the fold or gap, #1163/#1164), and on a normal day
(on a step boundary, early and late in the step).

## Expected failures

Known failure classes are listed in `known.py` (K1-K9, K9 in `test_list_path.py`) and marked `xfail(strict=True)` with
the reason. The predicates only describe *when* a failure is expected; they never feed into
the assertions. A fixed class therefore shows up as `XPASS(strict)` = FAIL, which is the
signal to update `known.py`. The `K3` items are design questions (wall-clock alignment vs.
real-time steps with a 30-minute DST and a 60-minute step), not necessarily bugs.

## Layout

Helper modules `cases.py`, `harness.py`, `known.py` sit next to the tests; `conftest.py` puts this
folder on `sys.path` and prints the EMHASS/pandas/pytz versions in the pytest header.

## Results

| EMHASS | passed | xfailed | failed |
|---|---|---|---|
| v0.18.5 = master `2f3dc75d` | 3979 | 2485 | 0 |
| master `f571c59f` (2026-10-14) | 3979 | 2485 | 0 |
| v0.18.4 `ddad1a8d` | 3345 | 2475 | 644 (the window-end and short-day cases fixed by #1164/#1166) |

