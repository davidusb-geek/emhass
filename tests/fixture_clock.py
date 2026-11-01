"""Reference clock for tests that run on the fixed-length fixture data.

``Forecast`` builds its horizon from the real clock as ``delta_forecast_daily``
local calendar days (utils.add_local_calendar_days), so the grid has 48 half-hour
steps on an ordinary day but 46 on a spring-forward day and 50 on a fall-back day.
That is the intended production behaviour (see tests/dst_1189). Tests that feed
hand-written 48-step arrays, or the 48-row data/data_weather_forecast.csv, must
therefore not read the real clock, or they fail on the days around a DST change.

FIXTURE_NOW is the day after the shipped history fixture (data/test_df_final.pkl
covers 2025-03-12 to 2025-03-21) ends, and no DST change in any time zone the tests
use falls within a day of it. It was chosen for that reason only, not by running the
tests on candidate dates. Tests that need a variable-length grid belong in
tests/dst_1189 and TestDstForecastDates, which set their own instants.

Use it narrowly: tests that compare against real file modification times (cache and
model age) must keep the real clock.
"""

from datetime import UTC, datetime

import time_machine

FIXTURE_NOW = datetime(2025, 3, 22, 12, 0, tzinfo=UTC)


def fixture_clock():
    """A new, unstarted time_machine traveller frozen at FIXTURE_NOW.

    Works as a decorator on a test, a context manager, or via start()/stop().
    """
    return time_machine.travel(FIXTURE_NOW, tick=False)
