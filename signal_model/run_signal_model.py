"""Phase 3 end-to-end run: purged/embargoed walk-forward LightGBM signal model,
regime-stratified evaluation (final/best-available regime labels - approved judgment
call, logged as requiring a point-in-time-honest re-run before any deployment claim,
see docs/phase3_working_notes.md), baselines, and significance tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from signal_model import significance
from signal_model.baselines import momentum_baseline
from signal_model.evaluate import regime_stratified_metrics
from signal_model.lgbm_model import fit_fold_model, predict_labels
from signal_model.target import compute_direction_labels, compute_forward_return
from signal_model.walk_forward import EMBARGO_DAYS, PURGE_DAYS, generate_folds

RESULTS_DIR = Path(__file__).parent / "results"
LGBM_N_ESTIMATORS = 500  # fallback label for fold_summary if best_iteration_ is None (no early stop triggered)


def load_nifty_close(conn) -> pd.Series:
    return pd.read_sql_query(
        """
        SELECT b.trade_date, b.close FROM daily_bars b
        JOIN instruments i ON i.id = b.instrument_id
        WHERE i.symbol = 'NIFTY50' AND b.superseded_by IS NULL
        ORDER BY b.trade_date
        """,
        conn, parse_dates=["trade_date"],
    ).set_index("trade_date")["close"]


HINDSIGHT_MODEL_VERSION_ID = 1  # the standalone full-sample fit registered for this
# comparison specifically. NOT current_regime_labels: that view reflects whatever
# model_version is currently non-superseded for each date, which drifts once other
# model_versions sharing the same (model_kind, k, jump_penalty) are registered later
# (e.g. the point-in-time quarterly walk) - confirmed this happened here (regime
# counts shifted by 1 row after the quarterly walk ran). Querying model_version_id=1
# directly pins the hindsight reference to the exact fit originally reported,
# regardless of what else gets registered in the DB afterward.


def load_hindsight_regime_labels(conn) -> pd.Series:
    return pd.read_sql_query(
        "SELECT trade_date, regime FROM regime_labels WHERE model_version_id = ?",
        conn, params=(HINDSIGHT_MODEL_VERSION_ID,), parse_dates=["trade_date"],
    ).set_index("trade_date")["regime"]


def compute_oof_predictions():
    """The purged/embargoed walk-forward LightGBM fit + rule-based baseline, run
    once and reused by every downstream evaluation (hindsight-label and
    point-in-time-label alike) - isolates the regime-label source as the only
    variable that changes between those two evaluations, rather than re-deriving
    predictions that happen to be deterministic anyway (fixed seeds throughout)."""
    conn = get_connection()
    df = build_feature_matrix(conn)
    nifty_close = load_nifty_close(conn)
    conn.close()

    df = df.loc[~df["warm_up"]].copy()
    fwd_ret = compute_forward_return(df, nifty_close)
    labels = compute_direction_labels(fwd_ret)

    valid = labels.notna()
    df, fwd_ret, labels = df.loc[valid], fwd_ret.loc[valid], labels.loc[valid]
    print(f"Usable rows after warm-up exclusion and dropping the undefined final-day label: {len(df)}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    folds = generate_folds(df.index)
    print(f"\nGenerated {len(folds)} folds (purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d)")

    oof_pred = pd.Series(index=df.index, dtype=float)
    oof_baseline = pd.Series(index=df.index, dtype=float)
    fold_summary = []

    for fold in folds:
        X_train = df.loc[fold.train_dates, FEATURE_COLUMNS]
        y_train = labels.loc[fold.train_dates]
        X_test = df.loc[fold.test_dates, FEATURE_COLUMNS]

        model = fit_fold_model(X_train, y_train)
        pred = predict_labels(model, X_test)
        oof_pred.loc[fold.test_dates] = pred

        base = momentum_baseline(df.loc[fold.test_dates, "ret_5"])
        oof_baseline.loc[fold.test_dates] = base

        fold_summary.append({
            "year": fold.year,
            "n_train": len(fold.train_dates),
            "n_test": len(fold.test_dates),
            "train_start": str(fold.train_dates[0].date()),
            "train_end": str(fold.train_dates[-1].date()),
            "test_start": str(fold.test_dates[0].date()),
            "test_end": str(fold.test_dates[-1].date()),
            "best_iteration": int(model.best_iteration_) if model.best_iteration_ else LGBM_N_ESTIMATORS,
        })
        print(f"  fold {fold.year}: train n={len(fold.train_dates)} "
              f"({fold.train_dates[0].date()} to {fold.train_dates[-1].date()}), "
              f"test n={len(fold.test_dates)}, best_iter={fold_summary[-1]['best_iteration']}")

    covered = oof_pred.notna()
    oof_pred, oof_baseline = oof_pred[covered], oof_baseline[covered]
    y_eval = labels.loc[oof_pred.index]
    fwd_ret_eval = fwd_ret.loc[oof_pred.index]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_summary).to_csv(RESULTS_DIR / "fold_summary.csv", index=False)
    oof = pd.DataFrame({
        "y_true": y_eval, "oof_pred": oof_pred, "oof_baseline": oof_baseline, "fwd_ret": fwd_ret_eval,
    })
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv")

    return oof_pred, oof_baseline, y_eval, fwd_ret_eval


def main() -> None:
    oof_pred, oof_baseline, y_eval, fwd_ret_eval = compute_oof_predictions()

    conn = get_connection()
    regime = load_hindsight_regime_labels(conn)
    conn.close()
    regime_eval = regime.reindex(oof_pred.index)

    print(f"\nTotal out-of-fold predictions: {len(oof_pred)} "
          f"({regime_eval.notna().sum()} with a hindsight regime label, model_version_id={HINDSIGHT_MODEL_VERSION_ID})")

    print("\n=== #3: LightGBM model - regime-stratified metrics ===")
    model_metrics = regime_stratified_metrics(y_eval, oof_pred, fwd_ret_eval, regime_eval)
    print(model_metrics.to_string(index=False))

    print("\n=== #4b: Rule-based baseline (momentum-1) - regime-stratified metrics ===")
    baseline_metrics = regime_stratified_metrics(y_eval, oof_baseline, fwd_ret_eval, regime_eval)
    print(baseline_metrics.to_string(index=False))
    print("\n(#4a - aggregate vs. regime-stratified view of the SAME model predictions - "
          "is the 'aggregate' row vs. the per-regime rows in the table above.)")

    print("\n=== #5: Significance tests ===")
    correct_model = (oof_pred == y_eval).astype(float)
    correct_baseline = (oof_baseline == y_eval).astype(float)
    diff = correct_model - correct_baseline
    pnl_model = oof_pred * fwd_ret_eval

    paired_result = significance.paired_block_bootstrap(diff)
    print(f"\nModel vs. rule-based baseline (paired block bootstrap on per-day correctness diff):")
    print(f"  {paired_result}")

    mean_pnl_test = significance.block_bootstrap_metric(pnl_model, np.mean)
    print(f"\nOverall model mean daily P&L (block bootstrap CI/p-value vs. 0):")
    print(f"  {mean_pnl_test}")

    perm_result = significance.regime_permutation_test(regime_eval, correct_model)
    print(f"\nRegime-permutation test (is regime-to-regime accuracy spread distinguishable from chance?):")
    print(f"  {perm_result}")

    p_values = {
        "overall_mean_pnl": mean_pnl_test["p_value"],
        "model_vs_baseline": paired_result["p_value"],
        "regime_spread": perm_result["p_value"],
    }
    for reg in sorted(regime_eval.dropna().unique()):
        mask = regime_eval == reg
        if mask.sum() < 30:
            continue
        p_values[f"regime_{int(reg)}_mean_pnl"] = significance.block_bootstrap_metric(pnl_model[mask], np.mean)["p_value"]

    bh = significance.benjamini_hochberg(p_values)
    print("\n=== Benjamini-Hochberg FDR correction across the test family (alpha=0.05) ===")
    print(bh.to_string(index=False))

    model_metrics.to_csv(RESULTS_DIR / "model_regime_stratified.csv", index=False)
    baseline_metrics.to_csv(RESULTS_DIR / "baseline_regime_stratified.csv", index=False)
    bh.to_csv(RESULTS_DIR / "significance_bh.csv", index=False)
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
