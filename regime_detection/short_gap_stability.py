"""Short-gap pairwise-ARI follow-up, flagged in the recalibration design (item 4
plan, "trigger" section): does a 3-month or 6-month gap show meaningfully better
stability than the 1.5-3.6 year gaps already tested? Directly informs the fixed-
cadence floor with real evidence instead of a guess.

Reuses the exact same machinery as rolling_window_stability_consecutive.py (fit_jm,
fit_hmm, align_states, same interior/edge split relative to A's cutoff, same ARI
bars) - only the pairing structure is new. Four reference starting points (the same
cutoffs already characterized: 2016-12-31, 2018-12-31, 2020-06-30, 2022-12-31), each
tested against a +3-month and +6-month B, rather than against the next stress-event
cutoff - giving a direct, apples-to-apples comparison against the longer gaps already
measured from those same starting points.

Interpretability (dominant-regime-for-a-known-window) is intentionally NOT checked
here - most of these short-gap endpoints don't land on a clean stress-window
boundary, and forcing that mapping would be more confusing than informative. This is
an ARI-focused diagnostic, per the specific ask, not a full six-point re-run.

Tested on JM k=3 and HMM k=3, same scope as the other quick diagnostics this session.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from data_agent.db import get_connection
from regime_detection.features import build_feature_matrix
from regime_detection.jump_model_fit import RESULTS_DIR, prepare_fit_data
from regime_detection.rolling_window_stability import (
    EDGE_ARI_BAR,
    INTERIOR_ARI_BAR,
    WARM_UP_EDGE_DAYS,
    align_states,
    fit_hmm,
    fit_jm,
)

K = 3
REFERENCE_CUTOFFS = ["2016-12-31", "2018-12-31", "2020-06-30", "2022-12-31"]
GAPS = {"3mo": pd.DateOffset(months=3), "6mo": pd.DateOffset(months=6)}


def slice_data(X, cutoff):
    return X if cutoff is None else X.loc[:cutoff]


def nearest_trading_date(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    """The last available trading date on or before target (target itself may be a
    weekend/holiday)."""
    candidates = index[index <= target]
    return candidates[-1]


def evaluate_pair(X_raw_full, ret_ser_full, cutoff_a: str, cutoff_b: pd.Timestamp, k: int, model_kind: str) -> dict:
    X_a = slice_data(X_raw_full, cutoff_a)
    X_b = X_raw_full.loc[:cutoff_b]
    ret_a = slice_data(ret_ser_full.to_frame("r"), cutoff_a)["r"]
    ret_b = ret_ser_full.to_frame("r").loc[:cutoff_b]["r"]

    if model_kind == "jm":
        model_a, scaler_a, _ = fit_jm(X_a, ret_a, k)
        model_b, scaler_b, _ = fit_jm(X_b, ret_b, k)
        centers_a, centers_b = model_a.centers_, model_b.centers_
        labels_a = pd.Series(model_a.labels_, index=X_a.index)
        labels_b_full = pd.Series(model_b.labels_, index=X_b.index)
    else:
        model_a, scaler_a, clipper_a = fit_hmm(X_a, k)
        model_b, scaler_b, clipper_b = fit_hmm(X_b, k)
        centers_a, centers_b = model_a.means_, model_b.means_
        labels_a = pd.Series(model_a.predict(scaler_a.transform(clipper_a.transform(X_a)).to_numpy()), index=X_a.index)
        labels_b_full = pd.Series(model_b.predict(scaler_b.transform(clipper_b.transform(X_b)).to_numpy()), index=X_b.index)

    labels_b_on_a = labels_b_full.loc[labels_a.index]
    overlap_dates = labels_a.index
    edge_start = overlap_dates[-WARM_UP_EDGE_DAYS] if len(overlap_dates) > WARM_UP_EDGE_DAYS else overlap_dates[0]
    interior_mask = overlap_dates < edge_start
    edge_mask = ~interior_mask

    interior_ari = adjusted_rand_score(labels_a[interior_mask], labels_b_on_a[interior_mask]) if interior_mask.sum() > 1 else None
    edge_ari = adjusted_rand_score(labels_a[edge_mask], labels_b_on_a[edge_mask]) if edge_mask.sum() > 1 else None

    # state alignment computed for completeness/consistency with other checks, even
    # though no interpretability comparison is made here.
    _ = align_states(centers_a, scaler_a, centers_b, scaler_b)

    passed = (
        (interior_ari is None or interior_ari >= INTERIOR_ARI_BAR)
        and (edge_ari is None or edge_ari >= EDGE_ARI_BAR)
    )
    return {"interior_ari": interior_ari, "edge_ari": edge_ari, "passed": bool(passed)}


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw_full, ret_ser_full = prepare_fit_data(df)
    results = []

    for model_kind in ["jm", "hmm"]:
        print(f"\n=== {model_kind.upper()} k={K} ===")
        for cutoff_a in REFERENCE_CUTOFFS:
            for gap_label, offset in GAPS.items():
                target = pd.Timestamp(cutoff_a) + offset
                cutoff_b = nearest_trading_date(X_raw_full.index, target)
                r = evaluate_pair(X_raw_full, ret_ser_full, cutoff_a, cutoff_b, K, model_kind)
                r.update({"model": model_kind, "cutoff_a": cutoff_a, "gap": gap_label, "cutoff_b": str(cutoff_b.date())})
                results.append(r)
                print(f"  {cutoff_a} +{gap_label} (-> {cutoff_b.date()}): "
                      f"interior_ari={r['interior_ari']}, edge_ari={r['edge_ari']}, passed={r['passed']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_DIR / "short_gap_stability.csv", index=False)

    print("\n=== Summary by gap size ===")
    df_res = pd.DataFrame(results)
    for gap in ["3mo", "6mo"]:
        sub = df_res[df_res["gap"] == gap]
        print(f"  {gap}: {sub['passed'].sum()}/{len(sub)} passed, "
              f"mean interior ARI={sub['interior_ari'].mean():.4f}, "
              f"mean edge ARI={sub['edge_ari'].mean():.4f}")

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
