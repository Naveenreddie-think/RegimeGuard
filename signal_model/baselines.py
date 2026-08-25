"""Simple rule-based baseline - Phase 3 design (approved, item #4b): sign of the
already-built `ret_5` feature (5-day EWM return) against the same +-20bps flat band
as the real target - "predict continuation of recent short-term momentum." Standard,
minimal, and reuses an existing Phase 2 feature exactly, giving a clean apples-to-
apples comparison since it's evaluated under the identical fold structure and regime
stratification as the LightGBM model.
"""

from __future__ import annotations

import pandas as pd

from signal_model.target import FLAT_THRESHOLD


def momentum_baseline(ret_5: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=ret_5.index, dtype=float)
    labels[ret_5 > FLAT_THRESHOLD] = 1
    labels[ret_5 < -FLAT_THRESHOLD] = -1
    return labels
