"""Purged, embargoed, expanding-window walk-forward fold structure - Phase 3 design
(see docs/phase3_working_notes.md, "embargo re-derivation" entry).

PURGE_DAYS=1 matches the label horizon (HORIZON_DAYS=1 in target.py) exactly - the
only training sample whose label could reach into a test period is the single day
immediately before it.

EMBARGO_DAYS=240 is NOT the WARM_UP_EDGE_DAYS=90 figure borrowed from Phase 2's
regime-edge-instability finding - that constant measures a different mechanism (how
long regime classification takes to stabilize near a fitting cutoff). 240 was
re-derived from the actual mechanism at stake here: real autocorrelation of the
halflife-60 features (the longest-memory family in the Phase 2 feature set) measured
directly on the data. DD_log_60 - the slowest to decay of the three - is still 0.56
autocorrelated at lag 90, and doesn't drop below the |ACF|<0.10 "negligible" bar until
lag 240. ret_60/sortino_60 clear that bar earlier (lag 180), so 240 is the correctly
binding number across the whole halflife-60 family.

Placement follows the mechanism precisely: the risk is training on data *shortly
after* a period whose features still echo it, so each fold's training set excludes
its own most recent EMBARGO_DAYS trading days (not an idle gap between test folds).
Test folds stay full, consecutive, annual blocks - no loss of evaluation coverage,
only a training-eligibility lag. This is a real cost, stated plainly: each fold's
usable training set trails about one fold-cycle behind what a naive expanding-window
walk-forward would use.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PURGE_DAYS = 1
EMBARGO_DAYS = 240
FIRST_TEST_YEAR = 2015


@dataclass(frozen=True)
class Fold:
    year: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def generate_folds(
    dates: pd.DatetimeIndex,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
    first_test_year: int = FIRST_TEST_YEAR,
) -> list[Fold]:
    """Consecutive annual test folds (first_test_year through the last year present
    in `dates`, partial for the final calendar year). Each fold's training set is
    the expanding window up to (test_start_position - purge_days - embargo_days),
    i.e. strictly excludes the purge+embargo gap immediately preceding its own test
    start - not merely a buffer after the previous fold."""
    dates = pd.DatetimeIndex(sorted(dates))
    last_year = dates.max().year

    folds = []
    for year in range(first_test_year, last_year + 1):
        test_dates = dates[dates.year == year]
        if len(test_dates) == 0:
            continue

        test_start_pos = dates.get_loc(test_dates[0])
        train_end_pos = test_start_pos - purge_days - embargo_days
        if train_end_pos <= 0:
            continue  # not enough history yet to satisfy purge+embargo for this fold

        train_dates = dates[:train_end_pos]
        folds.append(Fold(year=year, train_dates=train_dates, test_dates=test_dates))

    return folds
