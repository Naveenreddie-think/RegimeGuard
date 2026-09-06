"""Fit one LightGBM signal model as-of a given date and register it as a
`signal_model_version` with its stored out-of-sample predictions - Phase 4 design §5
step 2, the signal-model analogue of `regime_detection/fit_and_register.py`.

CLI, so it doubles as the demo/validation tool and the real utility a scheduled or
triggered signal-model refit would eventually call (that cadence is the later Model
Training Agent's job, not decided here).

Train/predict separation mirrors `signal_model/walk_forward.py` exactly:
    train_end = as_of - PURGE_DAYS - EMBARGO_DAYS   (positional, on the feature-date index)
    predictions start at as_of and run to as_of + SIGNAL_COVERAGE_TD (or last data).
So a registered version's predictions for dates near its as_of are produced under the
same purge+embargo discipline the walk-forward evaluation used. The model blob is not
stored - re-fitting is deterministic (fixed seeds, established in Phase 3), and the
version row records everything needed to reproduce it.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from signal_model.lgbm_model import CLASS_TO_LABEL, LABEL_TO_CLASS, LGBM_PARAMS, fit_fold_model
from signal_model.registry import (
    SIGNAL_COVERAGE_TD,
    ensure_schema,
    save_signal_model_version,
    save_signal_predictions,
)
from signal_model.run_signal_model import load_nifty_close
from signal_model.target import FLAT_THRESHOLD, HORIZON_DAYS, compute_direction_labels, compute_forward_return
from signal_model.walk_forward import EMBARGO_DAYS, PURGE_DAYS

MODEL_KIND = "lgbm"
MIN_TRAIN_ROWS = 200  # below this, an as-of date is too early to register a version for


def fit_and_register_signal(
    conn, as_of: date | None, coverage_td: int = SIGNAL_COVERAGE_TD, notes: str | None = None
) -> tuple[int, int]:
    df = build_feature_matrix(conn)
    nifty_close = load_nifty_close(conn)
    df = df.loc[~df["warm_up"]].copy()

    fwd_ret = compute_forward_return(df, nifty_close)
    labels = compute_direction_labels(fwd_ret)
    dates = df.index  # sorted DatetimeIndex, post-warm-up

    as_of_ts = dates[-1] if as_of is None else pd.Timestamp(as_of)
    eligible = dates[dates <= as_of_ts]
    if len(eligible) == 0:
        raise ValueError(f"as_of {as_of_ts.date()} precedes the first feature row ({dates[0].date()})")
    as_of_ts = eligible[-1]  # snap back to the last trading day on/before as_of
    as_of_pos = dates.get_loc(as_of_ts)

    train_end_pos = as_of_pos - PURGE_DAYS - EMBARGO_DAYS
    if train_end_pos <= 0:
        raise ValueError(
            f"as_of {as_of_ts.date()} is too early: only {as_of_pos} trading days of history, "
            f"need > {PURGE_DAYS + EMBARGO_DAYS} for purge+embargo"
        )
    train_dates = dates[:train_end_pos]

    X_train = df.loc[train_dates, FEATURE_COLUMNS]
    y_train = labels.loc[train_dates]
    keep = y_train.notna()
    X_train, y_train = X_train[keep], y_train[keep]
    if len(X_train) < MIN_TRAIN_ROWS:
        raise ValueError(f"only {len(X_train)} labelled training rows for as_of {as_of_ts.date()} "
                         f"(min {MIN_TRAIN_ROWS})")

    model = fit_fold_model(X_train, y_train)

    coverage_end_pos = min(as_of_pos + coverage_td, len(dates) - 1)
    pred_dates = dates[as_of_pos:coverage_end_pos + 1]
    X_pred = df.loc[pred_dates, FEATURE_COLUMNS]

    proba = model.predict_proba(X_pred)
    col = {c: i for i, c in enumerate(model.classes_)}
    preds = pd.DataFrame(index=pred_dates)
    preds["prob_down"] = proba[:, col[LABEL_TO_CLASS[-1]]]
    preds["prob_flat"] = proba[:, col[LABEL_TO_CLASS[0]]]
    preds["prob_up"] = proba[:, col[LABEL_TO_CLASS[1]]]
    preds["pred_direction"] = (
        pd.Series(model.predict(X_pred), index=pred_dates).map(CLASS_TO_LABEL).astype(int)
    )

    version_id = save_signal_model_version(
        conn,
        model_kind=MODEL_KIND,
        target_horizon_days=HORIZON_DAYS,
        target_flat_bps=FLAT_THRESHOLD * 10000,
        fit_start_date=train_dates[0].date(),
        fit_end_date=train_dates[-1].date(),
        as_of_date=as_of_ts.date(),
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS,
        coverage_end_date=pred_dates[-1].date(),
        feature_columns=FEATURE_COLUMNS,
        lgbm_params=LGBM_PARAMS,
        best_iteration=int(model.best_iteration_) if model.best_iteration_ else None,
        n_train_rows=len(X_train),
        notes=notes,
    )
    n_preds = save_signal_predictions(
        conn, version_id, preds[["pred_direction", "prob_down", "prob_flat", "prob_up"]]
    )
    return version_id, n_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and register a LightGBM signal-model version")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="Model knows data through this date (default: latest available feature row). "
                             "Training ends PURGE+EMBARGO trading days before it; predictions run from it "
                             "for one coverage window.")
    parser.add_argument("--coverage-td", type=int, default=SIGNAL_COVERAGE_TD,
                        help=f"Trading days of forward prediction coverage (default {SIGNAL_COVERAGE_TD}).")
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    try:
        ensure_schema(conn)
        version_id, n_preds = fit_and_register_signal(conn, args.as_of, args.coverage_td, args.notes)
        row = conn.execute(
            "SELECT as_of_date, fit_start_date, fit_end_date, coverage_end_date, n_train_rows, best_iteration "
            "FROM signal_model_versions WHERE id = ?", (version_id,)
        ).fetchone()
    finally:
        conn.close()

    print(f"Registered signal_model_version {version_id} ({MODEL_KIND}, H={HORIZON_DAYS}, "
          f"flat={FLAT_THRESHOLD*10000:.0f}bps)")
    print(f"  as_of={row[0]}  train={row[1]}..{row[2]}  ({row[4]} rows, best_iter={row[5]})")
    print(f"  predictions: {n_preds} rows, {row[0]}..{row[3]}")


if __name__ == "__main__":
    main()
