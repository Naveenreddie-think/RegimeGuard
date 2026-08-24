"""Diagnostic: does per-window standardization drift explain the rolling-window
interior-ARI failures, independent of clustering?

For the interior dates (>90 trading days before the cutoff - same threshold and
split logic as rolling_window_stability.py, reused directly) of a truncated window,
this computes the standardized feature values TWO ways: under the truncated fit's
own clip+scale pipeline (what the truncated model actually sees), and under the
full-sample fit's clip+scale pipeline (what the full-sample model sees for those same
raw dates). Comparing the two isolates the preprocessing question from the clustering
question - the clustering algorithm never runs here.

If the two standardizations differ substantially, that's a real, mechanistic
explanation for at least part of the interior-ARI failures (a methodology artifact:
each window's scaler answers "extreme relative to what", and later data widening the
full-sample distribution changes that answer for earlier dates without any real
change in what happened on those dates). If they're nearly identical and the
clustering still produced very different labels for the same near-identical numbers,
that rules this out as the explanation and points at something in the clustering
step itself, or a genuinely unstable underlying regime structure.

Not fixing anything here - isolating the cause only, per direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.jump_model_fit import RESULTS_DIR, prepare_fit_data
from regime_detection.rolling_window_stability import WARM_UP_EDGE_DAYS

CUTOFFS_TO_CHECK = ["2016-12-31", "2020-06-30"]


def fit_pipeline(X_raw: pd.DataFrame) -> tuple[StandardScalerPD, DataClipperStd]:
    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    scaler.fit_transform(clipper.fit_transform(X_raw))
    return scaler, clipper


def transform_with(X_raw: pd.DataFrame, scaler: StandardScalerPD, clipper: DataClipperStd) -> pd.DataFrame:
    return scaler.transform(clipper.transform(X_raw))


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw_full, _ = prepare_fit_data(df)
    full_scaler, full_clipper = fit_pipeline(X_raw_full)

    print("=== Full-sample scaler stats (mean_ / scale_ per feature) ===")
    stats_df = pd.DataFrame(
        {"mean_": full_scaler.scaler.mean_, "scale_": full_scaler.scaler.scale_},
        index=FEATURE_COLUMNS,
    )

    for cutoff in CUTOFFS_TO_CHECK:
        print(f"\n{'='*70}\nCutoff: {cutoff}\n{'='*70}")
        trunc_X_raw = X_raw_full.loc[:cutoff]
        trunc_scaler, trunc_clipper = fit_pipeline(trunc_X_raw)

        # Same interior/edge split as rolling_window_stability.py
        overlap_dates = trunc_X_raw.index
        edge_start = (
            overlap_dates[-WARM_UP_EDGE_DAYS] if len(overlap_dates) > WARM_UP_EDGE_DAYS else overlap_dates[0]
        )
        interior_dates = overlap_dates[overlap_dates < edge_start]
        print(f"Truncated window: {len(trunc_X_raw)} rows total, {len(interior_dates)} interior dates")

        # Compare the two scalers' fitted stats directly - the "what changed" view.
        cmp = stats_df.copy()
        cmp["trunc_mean_"] = trunc_scaler.scaler.mean_
        cmp["trunc_scale_"] = trunc_scaler.scaler.scale_
        cmp["mean_shift"] = cmp["mean_"] - cmp["trunc_mean_"]
        cmp["scale_ratio (full/trunc)"] = cmp["scale_"] / cmp["trunc_scale_"]
        print("\nScaler comparison (full-sample vs. this truncated fit):")
        print(cmp.round(4).to_string())

        # Now the actual per-date question: same raw interior dates, standardized
        # two ways. This is what the clustering algorithm actually sees in each case.
        raw_interior = X_raw_full.loc[interior_dates]
        std_under_trunc = transform_with(raw_interior, trunc_scaler, trunc_clipper)
        std_under_full = transform_with(raw_interior, full_scaler, full_clipper)

        diff = std_under_full - std_under_trunc
        per_date_euclidean = np.sqrt((diff ** 2).sum(axis=1))

        print(f"\nPer-date Euclidean distance between the two standardizations "
              f"(interior dates only, n={len(interior_dates)}):")
        print(f"  mean:   {per_date_euclidean.mean():.4f}")
        print(f"  median: {per_date_euclidean.median():.4f}")
        print(f"  max:    {per_date_euclidean.max():.4f}  (on {per_date_euclidean.idxmax().date()})")
        print(f"  min:    {per_date_euclidean.min():.4f}")

        print("\nPer-feature mean absolute difference (full-standardization minus trunc-standardization):")
        print(diff.abs().mean().round(4).sort_values(ascending=False).to_string())

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = pd.concat(
            [std_under_trunc.add_suffix("_under_trunc"), std_under_full.add_suffix("_under_full")],
            axis=1,
        )
        out["euclidean_distance"] = per_date_euclidean
        out.to_csv(RESULTS_DIR / f"standardization_drift_{cutoff}.csv")

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
