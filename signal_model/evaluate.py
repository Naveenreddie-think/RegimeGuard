"""Regime-stratified evaluation metrics - Phase 3 design item #3 (approved), the core
validation contribution the project exists to demonstrate.

Uses FINAL/best-available regime labels (the full-sample JM k=3 fit registered as
model_version 1), an approved judgment call for THIS research question - "does the
model's edge genuinely vary by regime at all" deserves the cleanest available regime
classification. Logged explicitly as an open item requiring a point-in-time-honest
re-run (using the versioned model_versions/regime_labels infrastructure) before any
deployment claim - see docs/phase3_working_notes.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TRADING_DAYS_PER_YEAR = 252


def strategy_pnl(pred_labels: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    """Long on an 'up' prediction, short on 'down', flat (no position) on 'flat' -
    P&L in log-return units using the actual realized forward return for that date."""
    position = pred_labels.reindex(fwd_ret.index).fillna(0)
    return position * fwd_ret


def sharpe_like(pnl: pd.Series) -> float:
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    running_max = equity.cummax()
    return float((equity - running_max).min())


def compute_metrics(true_labels: pd.Series, pred_labels: pd.Series, fwd_ret: pd.Series) -> dict:
    y_true = true_labels.to_numpy()
    y_pred = pred_labels.reindex(true_labels.index).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1, 0, 1], average="macro", zero_division=0
    )
    pnl = strategy_pnl(pred_labels.reindex(true_labels.index), fwd_ret.reindex(true_labels.index))

    return {
        "n": int(len(true_labels)),
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "sharpe_like": sharpe_like(pnl),
        "max_drawdown_bps": max_drawdown(pnl) * 10000,
        "mean_pnl_bps_per_day": float(pnl.mean() * 10000),
    }


def regime_stratified_metrics(
    true_labels: pd.Series, pred_labels: pd.Series, fwd_ret: pd.Series, regime: pd.Series
) -> pd.DataFrame:
    rows = [{"regime": "aggregate", **compute_metrics(true_labels, pred_labels, fwd_ret)}]
    for reg in sorted(regime.dropna().unique()):
        mask = regime.reindex(true_labels.index) == reg
        if mask.sum() < 2:
            continue
        rows.append({
            "regime": f"regime_{int(reg)}",
            **compute_metrics(true_labels[mask], pred_labels[mask], fwd_ret[mask]),
        })
    return pd.DataFrame(rows)
