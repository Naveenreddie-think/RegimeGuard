"""Required Phase 3 follow-up (see docs/phase3_working_notes.md): re-run the
regime-stratified evaluation using point-in-time regime labels - what a live
model_version, active at each historical date, would actually have classified -
instead of the final/best-available (hindsight) labels used in the first pass.

Reuses the EXACT SAME out-of-fold predictions already computed and saved by
run_signal_model.py (signal_model/results/oof_predictions.csv) - only the regime-
label source changes, isolating that as the one variable under test, same as every
other check this project has run.

Point-in-time labels come from regime_detection.quarterly_walk - a full-history JM
refit at each quarterly cutoff (Phase 2's validated quarterly floor), predicting
forward into the gap before the next cutoff using that same fitted model's FIXED
parameters and its own scaler's .transform() only (never refit). See that module's
docstring for the full, reviewed design and its two explicitly-confirmed leakage
guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from data_agent.db import get_connection
from regime_detection.quarterly_walk import load_point_in_time_labels
from signal_model import significance
from signal_model.evaluate import regime_stratified_metrics
from signal_model.run_signal_model import HINDSIGHT_MODEL_VERSION_ID, RESULTS_DIR, load_hindsight_regime_labels

QUARTERLY_VERSION_ID_RANGE = range(2, 65)  # the 63 model_versions created by the quarterly walk


def main() -> None:
    oof = pd.read_csv(RESULTS_DIR / "oof_predictions.csv", index_col=0, parse_dates=True)
    y_eval, oof_pred, oof_baseline, fwd_ret_eval = (
        oof["y_true"], oof["oof_pred"], oof["oof_baseline"], oof["fwd_ret"]
    )

    conn = get_connection()
    pit_regime = load_point_in_time_labels(conn, list(QUARTERLY_VERSION_ID_RANGE))
    hindsight_regime = load_hindsight_regime_labels(conn)
    conn.close()

    pit_eval = pit_regime.reindex(oof_pred.index)
    hindsight_eval = hindsight_regime.reindex(oof_pred.index)
    common = pit_eval.dropna().index.intersection(hindsight_eval.dropna().index)
    ari = adjusted_rand_score(pit_eval.loc[common], hindsight_eval.loc[common])
    print(f"Point-in-time vs. hindsight regime labels on the {len(common)} evaluation dates: ARI={ari:.3f}")
    print("(permutation-invariant - raw label-id agreement isn't meaningful across independently-fit models)\n")

    print("=== #3 (point-in-time labels): LightGBM model - regime-stratified metrics ===")
    model_metrics_pit = regime_stratified_metrics(y_eval, oof_pred, fwd_ret_eval, pit_eval)
    print(model_metrics_pit.to_string(index=False))

    print("\n=== #3 (hindsight labels, for direct comparison): LightGBM model - regime-stratified metrics ===")
    model_metrics_hindsight = regime_stratified_metrics(y_eval, oof_pred, fwd_ret_eval, hindsight_eval)
    print(model_metrics_hindsight.to_string(index=False))

    print("\n=== Rule-based baseline - regime-stratified metrics (point-in-time labels) ===")
    baseline_metrics_pit = regime_stratified_metrics(y_eval, oof_baseline, fwd_ret_eval, pit_eval)
    print(baseline_metrics_pit.to_string(index=False))

    print("\n=== #5 (point-in-time labels): significance tests ===")
    correct_model = (oof_pred == y_eval).astype(float)
    pnl_model = oof_pred * fwd_ret_eval

    perm_result_pit = significance.regime_permutation_test(pit_eval, correct_model)
    print(f"\nRegime-permutation test (point-in-time labels): {perm_result_pit}")

    p_values = {"regime_spread_pit": perm_result_pit["p_value"]}
    for reg in sorted(pit_eval.dropna().unique()):
        mask = pit_eval == reg
        if mask.sum() < 30:
            continue
        p_values[f"regime_{int(reg)}_mean_pnl_pit"] = significance.block_bootstrap_metric(pnl_model[mask], np.mean)["p_value"]

    bh_pit = significance.benjamini_hochberg(p_values)
    print("\n=== Benjamini-Hochberg FDR correction (point-in-time regime tests, alpha=0.05) ===")
    print(bh_pit.to_string(index=False))

    model_metrics_pit.to_csv(RESULTS_DIR / "model_regime_stratified_pit.csv", index=False)
    bh_pit.to_csv(RESULTS_DIR / "significance_bh_pit.csv", index=False)
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
