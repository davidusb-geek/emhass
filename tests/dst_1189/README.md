# EMHASS time-grid / DST test suite (davidusb-geek/emhass#1189)

Offline pytest suite for #1189: one time grid per run, every value on its UTC instant; the remaining
known failures are strict xfails.
Runs with the normal `uv run pytest` (`time-machine` is in the `test` extra). On its own, from the repository root: `uv run --extra test python -m pytest tests/dst_1189 -q`
Expected with the #1189 fix: **7501 passed, 42 skipped, 70 xfailed**, no failures (about 1 minute
with `-n 16` from `pytest-xdist`, about 4-5 minutes on one core).

Without uv: `pip install -e '.[test]' && python -m pytest tests/dst_1189 -q`. A case that
starts to pass shows up as `XPASS(strict)`, which fails the run: drop the matching predicate in
`known.py` (or the test module) and the case runs as a normal test. `-rxX` lists every expected
failure with its class; `-k Berlin` gives a quick look. If `time-machine` is missing from an
environment, `conftest.py` skips the folder instead of failing at collection.

Provenance: this suite was written and contributed by Thorsten Hornung (@thorstenhornung1),
https://github.com/thorstenhornung1/emhass/tree/tests/1189-dst-grid-regression (commit
`50400dea5627d6f0d909f40dec05b2882ac8c97a`, shared in
https://github.com/davidusb-geek/emhass/issues/1189#issuecomment-6178133707). The original commit is
kept in the history with its authorship; later commits adapt it to the #1189 fix (fixed failure
classes now run as ordinary tests, oracle helpers are independent of EMHASS (`cases.expected_start`), one event loop
per process in the harness). The original text follows.

Free to use, adapt or trim, in whole or in part, without asking and without attribution.
Written with Claude Code (Anthropic), following `AGENTS.md` and `docs/develop_ai_coders.md`; the
commit carries a `Co-Authored-By` trailer. The findings behind the xfails were reproduced against
the real EMHASS package.

## What it checks

Black-box pytest suite that checks one contract: **a run has exactly one time grid, and
every value sits on the UTC instant it belongs to** - on normal days and across DST
transitions. It drives EMHASS only through public entry points (`set_input_data_dict`,
`prepare_forecast_and_weather_data`, `naive_mpc_optim`, `perfect_forecast_optim`, `publish_data`,
`utils.get_forecast_dates`, `Forecast.get_forecast_out_from_csv_or_list`,
`Optimization.perform_perfect_forecast_optim`) and does **not**
patch EMHASS code. Home Assistant and the network are not needed: no token (so
`get_ha_config()` returns early), runtime lists / timestamp dicts / a CSV / a pre-seeded
Open-Meteo cache file / a pickled history (`test_df_final.pkl`, for `perfect-optim`),
`publish_data(..., dont_post=True)`.

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
| P7 | first step equals an independent oracle (`cases.expected_start`: wall-clock aligned neighbour found by scanning real time; `nearest` rounded on the UTC clock) | `test_grid_contract.py` |
| P8 | a request that crosses a step boundary between its two clock reads keeps one grid (one clock read per request) | `test_request_clock.py` |
| P9 | `perfect-optim` solves every 24-elapsed-hour window of the offline history across a DST change (#1263): exactly the UTC instants of an independent oracle (`cases.perfect_window_instants`), once each, PV/load/prices on their own instants, every step solved, also on history gridded from local midnight (websocket); a missing step skips only its own window; an ordinary day equals a direct solve of its window | `test_perfect_optim.py` |

Matrix: Europe/Berlin, Australia/Sydney, Europe/Dublin (negative DST), Africa/Casablanca
(Ramadan 2026, negative DST in pytz), Australia/Lord_Howe (30-min DST), Asia/Kathmandu
(+05:45, no DST); steps 5/15/30/60 min (CSV/Open-Meteo 15/30/60); `method_ts_round`
first/nearest/last; run instants before, at the start and in the middle of both passes of
the repeated interval, after it, around the spring-forward gap, the day before (window end
in the fold or gap, #1163/#1164), and on a normal day
(on a step boundary, early and late in the step).
P9 runs the real solver on Berlin, Sydney, Lord_Howe and Kathmandu (an ordinary day, for the
fractional offset) at 5/15/30/60 min: the window before each transition and the one containing
it, plus the `perfect_forecast_optim` action once.

## Expected failures

On master `f571c59f` the suite classified every failure into K1-K9 and marked about 2,500 cases as
strict xfail. The #1189 fix resolves K1, K2, K4, K6, K7, K8 and K9, so those cases now run as
ordinary tests. K5 (a CSV that does not cover the horizon) is the established hold-nearest
behaviour and is tested as such, including its warning. `known.py` lists what is left, K3: timestamped
CSV rows on a wall-clock aligned timeline in zones whose wall clock is not a whole number of steps from
UTC. Australia/Lord_Howe with a 60-minute step is an unsupported combination (its 30-minute DST change
moves the hourly rows off the grid, the prices become NaN and the solve fails explicitly), and
Asia/Kathmandu with `nearest` at 30/60 minutes has a grid on UTC steps (:15/:45 local) against
wall-clock CSV rows. Both are limitations, not defects of the fix, and are documented in
`docs/config.md` (`method_ts_round`). The predicates only describe *when* a failure is expected; they
never feed into the assertions. The independent oracles (`cases.expected_start`, source markers that
encode the UTC instant of each value) do not call EMHASS code.

## Layout

Helper modules `cases.py`, `harness.py`, `known.py` sit next to the tests; `conftest.py` puts this
folder on `sys.path` and prints the EMHASS/pandas/pytz versions in the pytest header.

## Results

| EMHASS | passed | xfailed | failed |
|---|---|---|---|
| v0.18.5 = master `2f3dc75d` (as contributed) | 3979 | 2485 | 0 |
| master `f571c59f` (2026-10-14, as contributed) | 3979 | 2485 | 0 |
| v0.18.4 `ddad1a8d` (as contributed) | 3345 | 2475 | 644 (the window-end and short-day cases fixed by #1164/#1166) |
| master `f571c59f` + #1189 fix (this suite, incl. P7/P8 and the K5 tests) | 7469 | 70 | 0 (42 skipped) |

The fix suite has more cases (7,581) than the original (6,464): fixed xfail classes now run and pass, and
P7/P8 and the K5 behaviour tests were added.

