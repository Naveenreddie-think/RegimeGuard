"""Point-in-time feature engineering for regime detection.

Per the approved Phase 2 plan (see the compiled plan doc, sections 1.1-1.6):
11 features, computed on Nifty 50 daily log returns and India VIX daily close,
directly adapted from the jumpmodels package authors' own reference implementation
(EWM return / log downside-deviation / Sortino ratio at halflives 5/20/60 trading
days) plus two VIX-derived additions (vix_log, vix_chg_5).

Design notes:
- Nifty-return-derived features (9 of the 11) are computed on the *full* Nifty 50
  close series - they don't depend on VIX and shouldn't be interrupted by VIX's own
  gaps (see calendar_days.KNOWN_SPECIAL_SESSIONS / load_india_vix_manual.py's
  KNOWN_CONFIRMED_UNAVAILABLE: 2021-02-12, 2021-03-30, 2024-03-02 have no VIX value).
- VIX-derived features are computed on VIX's own close series, independently.
- The two feature sets are inner-joined on trade_date at the end - this is what
  naturally drops the 3 VIX-gap dates from the final model-ready matrix, without
  ever forward-filling or interpolating a VIX value that doesn't exist.
- pandas' `.ewm()` is inherently causal (row t only uses rows <= t) - no lookahead
  in any of these statistics. Standardization (a real leakage risk if done globally)
  is deliberately NOT done here - that belongs to the modeling code, fit per split,
  never on the full history at once (see plan §1.6).
- `warm_up` flags the first ~90 trading days of the Nifty series (counted on the
  base return series, before the VIX join), where the longest EWM halflife (60
  days) hasn't accumulated enough history to be reliable yet. Rows are kept, not
  dropped, so the caller decides what to do with them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_agent.db import get_connection

HALFLIVES = (5, 20, 60)
WARM_UP_DAYS = 90


def _load_close_series(conn, symbol: str) -> pd.Series:
    rows = conn.execute(
        """
        SELECT b.trade_date, b.close FROM daily_bars b
        JOIN instruments i ON i.id = b.instrument_id
        WHERE i.symbol = ? AND b.superseded_by IS NULL
        ORDER BY b.trade_date
        """,
        (symbol,),
    ).fetchall()
    dates = pd.to_datetime([r[0] for r in rows])
    closes = pd.Series([r[1] for r in rows], index=dates, name=symbol, dtype=float)
    return closes


def _ewm_downside_deviation(returns: pd.Series, halflife: int) -> pd.Series:
    negative_returns = returns.clip(upper=0.0)
    mean_sq = negative_returns.pow(2).ewm(halflife=halflife).mean()
    return np.sqrt(mean_sq)


def compute_nifty_features(nifty_close: pd.Series) -> pd.DataFrame:
    """The 9 Nifty-return-derived features: EWM return, log downside deviation,
    and Sortino ratio at halflives 5/20/60 (see module docstring)."""
    log_returns = np.log(nifty_close / nifty_close.shift(1))

    feat = pd.DataFrame(index=nifty_close.index)
    feat["nifty_close"] = nifty_close
    for hl in HALFLIVES:
        ret_hl = log_returns.ewm(halflife=hl).mean()
        dd_hl = _ewm_downside_deviation(log_returns, hl)
        feat[f"ret_{hl}"] = ret_hl
        feat[f"DD_log_{hl}"] = np.log(dd_hl)
        feat[f"sortino_{hl}"] = ret_hl / dd_hl

    feat["days_since_start"] = np.arange(len(feat))
    feat["warm_up"] = feat["days_since_start"] < WARM_UP_DAYS
    return feat


def compute_vix_features(vix_close: pd.Series) -> pd.DataFrame:
    """The 2 VIX-derived features: log level and short-halflife EWM change."""
    vix_log_change = np.log(vix_close / vix_close.shift(1))

    feat = pd.DataFrame(index=vix_close.index)
    feat["vix_close"] = vix_close
    feat["vix_log"] = np.log(vix_close)
    feat["vix_chg_5"] = vix_log_change.ewm(halflife=5).mean()
    return feat


def build_feature_matrix(conn) -> pd.DataFrame:
    """Load Nifty 50 and India VIX close series, compute both feature families
    independently, and inner-join on trade_date. The join is what drops the known
    VIX-gap dates from the model-ready matrix - deliberately, not accidentally."""
    nifty_close = _load_close_series(conn, "NIFTY50")
    vix_close = _load_close_series(conn, "INDIAVIX")

    nifty_feat = compute_nifty_features(nifty_close)
    vix_feat = compute_vix_features(vix_close)

    merged = nifty_feat.join(vix_feat, how="inner")
    merged.index.name = "trade_date"
    return merged


FEATURE_COLUMNS = [
    "ret_5", "DD_log_5", "sortino_5",
    "ret_20", "DD_log_20", "sortino_20",
    "ret_60", "DD_log_60", "sortino_60",
    "vix_log", "vix_chg_5",
]


def main() -> None:
    conn = get_connection()
    try:
        nifty_close = _load_close_series(conn, "NIFTY50")
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    print(f"Feature matrix: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Warm-up rows (excluded downstream, not dropped here): {df['warm_up'].sum()}")
    print(f"Rows dropped by the VIX inner-join (Nifty had a row, VIX didn't): "
          f"{len(nifty_close) - len(df)}")
    print()
    print("Feature columns:", FEATURE_COLUMNS)
    print()
    print(df[FEATURE_COLUMNS].describe())


if __name__ == "__main__":
    main()
