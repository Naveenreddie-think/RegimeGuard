"""Prediction target: 1-trading-day-forward direction on Nifty 50 (up/flat/down),
+-20bps flat band - both locked per the Phase 3 design review (see
docs/phase3_working_notes.md).

Grounded in real data, not picked abstractly:
- H=1 was chosen specifically to avoid the overlapping-label serial correlation a
  multi-day forward return would introduce (a 5-day forward return showed
  autocorr(1)=0.81 in the design pass - purely mechanical, from sharing 4/5 days
  between adjacent labels). H=1 sidesteps this: each day's label depends on exactly
  one price transition, so adjacent labels don't overlap.
- FLAT_THRESHOLD=20bps was chosen from the real class-balance check across the whole
  series and per already-detected regime (no regime showed a degenerate flat class
  at this threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLAT_THRESHOLD = 0.0020  # +-20bps on 1-day log return
HORIZON_DAYS = 1


def compute_forward_return(df: pd.DataFrame, nifty_close: pd.Series) -> pd.Series:
    """1-trading-day-forward log return, aligned to df's index. Forward-looking by
    construction (it's the label) - callers must never feed this back in as a
    feature. walk_forward.py's purge/embargo exists specifically to keep this
    label's forward-looking window from crossing a train/test boundary."""
    fwd_close = nifty_close.shift(-HORIZON_DAYS)
    fwd_ret = np.log(fwd_close / nifty_close)
    return fwd_ret.reindex(df.index)


def compute_direction_labels(fwd_ret: pd.Series) -> pd.Series:
    """+1 = up, 0 = flat, -1 = down, per the +-20bps band. NaN where no forward
    price exists (the last available trading day) - callers must drop these before
    fitting or evaluating."""
    labels = pd.Series(np.nan, index=fwd_ret.index)
    labels[fwd_ret > FLAT_THRESHOLD] = 1
    labels[fwd_ret < -FLAT_THRESHOLD] = -1
    labels[(fwd_ret >= -FLAT_THRESHOLD) & (fwd_ret <= FLAT_THRESHOLD)] = 0
    return labels
