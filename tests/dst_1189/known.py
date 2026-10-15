"""Remaining expected failures of the #1189 suite, used to mark strict xfails.

The suite as contributed (davidusb-geek/emhass#1189, @thorstenhornung1, commit 50400dea)
classified every failure of master f571c59f into K1-K9 and marked ~2,500 cases as strict
xfail. The #1189 time-grid fix resolves K1, K2, K4, K6, K7, K8 and K9, so those cases now run as
ordinary tests. K5 (a CSV that does not cover the horizon) is the established hold-nearest
behaviour and is now tested as such, including its warning. What remains are limitations of
timestamped CSV rows on a wall-clock aligned timeline, not defects of the fix:

K3   timestamped CSV forecasts (price rows on wall-clock times) on a grid that is regular in
     elapsed time, in zones whose wall clock is not a whole number of steps from UTC:

     * Australia/Lord_Howe with a 60-minute step: its 30-minute DST change moves the hourly
       rows off the grid; the CSV prices become NaN and the solve fails explicitly
       (unsupported combination; a step of 30 minutes or less works).
     * Asia/Kathmandu (+05:45) with method_ts_round="nearest" at 30/60 minutes: nearest rounds
       on the UTC clock (the established contract), so the grid sits at :15/:45 local and every
       step takes the closest CSV row, up to half a step away (documented limitation;
       first/last give a grid on the local clock).

The predicates only describe *when* a failure is expected; they never feed into the
assertions. All markers are strict: a case that starts to pass turns into XPASS(strict) =
FAIL, the signal to update this table.
"""

from __future__ import annotations

import pytest

REF_1189 = "davidusb-geek/emhass#1189"


def xfail(request, *reasons):
    reasons = [r for r in reasons if r]
    if reasons:
        request.applymarker(pytest.mark.xfail(reason=" | ".join(reasons), strict=True))
