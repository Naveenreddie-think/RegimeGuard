"""VIX ablation: does removing VIX-derived features change panic-day classification?

Standing open question since the first JM fit (informal z-score read suggested return
features carried real weight, not just VIX, but that wasn't a real test). Run at
**k=3**, not k=4 - the refit-stability check found k=4/k=5 have real seed-dependent
instability (JM k=4 min ARI 0.78, k=5 min ARI 0.50), while k=3 is essentially
perfectly stable (mean/min ARI = 1.0). Running this ablation on an unstable k would
confound "does VIX drive classification" with "did we land on a different unstable
seed" - deliberately avoided. lambda=50 anchor carried over unchanged.

Both the full (11-feature) and ablated (9-feature, VIX dropped) fits use the
*identical* set of trading days - the 3900-date set the full feature matrix already
settles on via its VIX inner-join (see features.py). The ablated model is NOT given
its naturally-larger available sample (Nifty alone doesn't need VIX and could cover 3
more dates) - that would confound "removing VIX" with "a different sample", which
this deliberately avoids by restricting both fits to the same date index.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from sklearn.metrics import adjusted_rand_score

from data_agent.db import get_connection
from regime_detection.features import build_feature_matrix
from regime_detection.jump_model_fit import (
    RESULTS_DIR,
    KNOWN_STRESS_WINDOWS,
    interpretability_report,
    prepare_fit_data,
)

K = 3
LAMBDA = 50.0

NIFTY_ONLY_COLUMNS = [
    "ret_5", "DD_log_5", "sortino_5",
    "ret_20", "DD_log_20", "sortino_20",
    "ret_60", "DD_log_60", "sortino_60",
]
FULL_COLUMNS = NIFTY_ONLY_COLUMNS + ["vix_log", "vix_chg_5"]


def fit_at(X_raw: pd.DataFrame, ret_ser: pd.Series, columns: list[str]) -> JumpModel:
    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    X = scaler.fit_transform(clipper.fit_transform(X_raw[columns]))
    jm = JumpModel(n_components=K, jump_penalty=LAMBDA, cont=False, random_state=0)
    jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
    return jm, X


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw, ret_ser = prepare_fit_data(df)
    print(f"Fit set: {len(X_raw)} rows, identical for both models (full and ablated)")

    jm_full, X_full = fit_at(X_raw, ret_ser, FULL_COLUMNS)
    jm_ablated, X_ablated = fit_at(X_raw, ret_ser, NIFTY_ONLY_COLUMNS)

    labels_full = pd.Series(jm_full.labels_, index=X_raw.index)
    labels_ablated = pd.Series(jm_ablated.labels_, index=X_raw.index)

    ari = adjusted_rand_score(labels_full.to_numpy(), labels_ablated.to_numpy())
    print(f"\nOverall ARI between full (11-feat) and ablated (9-feat, no VIX) labels: {ari:.4f}")

    print("\n=== Full model (11 features) state centers ===")
    full_centers = pd.DataFrame(jm_full.centers_, columns=FULL_COLUMNS).round(3)
    print(full_centers.to_string())

    print("\n=== Ablated model (9 features, no VIX) state centers ===")
    ablated_centers = pd.DataFrame(jm_ablated.centers_, columns=NIFTY_ONLY_COLUMNS).round(3)
    print(ablated_centers.to_string())

    print("\n=== State counts ===")
    print("full:", labels_full.value_counts().sort_index().to_dict())
    print("ablated:", labels_ablated.value_counts().sort_index().to_dict())

    report_full = interpretability_report(labels_full)
    report_ablated = interpretability_report(labels_ablated)
    print("\n=== Known stress-window regime assignment: full vs. ablated ===")
    for name in KNOWN_STRESS_WINDOWS:
        f, a = report_full[name], report_ablated[name]
        print(f"  {name}:")
        print(f"    full:    state {f['dominant_regime']} ({f['dominant_share']:.2%}) - {f['regime_breakdown']}")
        print(f"    ablated: state {a['dominant_regime']} ({a['dominant_share']:.2%}) - {a['regime_breakdown']}")

    # Direct question: for the worst (lowest-cumret) state specifically, how much do
    # the two models' membership sets overlap? This is the sharpest version of "is
    # VIX driving panic-day classification" - not just overall ARI, which averages
    # across all states.
    worst_full = set(labels_full[labels_full == 0].index)
    worst_ablated = set(labels_ablated[labels_ablated == 0].index)
    overlap = worst_full & worst_ablated
    jaccard = len(overlap) / len(worst_full | worst_ablated) if (worst_full | worst_ablated) else float("nan")
    print(f"\nWorst-state (panic-like) membership overlap:")
    print(f"  full worst-state days: {len(worst_full)}, ablated worst-state days: {len(worst_ablated)}")
    print(f"  intersection: {len(overlap)}, Jaccard similarity: {jaccard:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_centers.to_csv(RESULTS_DIR / "vix_ablation_full_centers.csv")
    ablated_centers.to_csv(RESULTS_DIR / "vix_ablation_ablated_centers.csv")
    summary = {
        "k": K, "lambda": LAMBDA,
        "overall_ari": float(ari),
        "worst_state_jaccard": float(jaccard),
        "worst_state_full_n": len(worst_full),
        "worst_state_ablated_n": len(worst_ablated),
        "stress_windows": {
            name: {"full": report_full[name], "ablated": report_ablated[name]}
            for name in KNOWN_STRESS_WINDOWS
        },
    }
    with open(RESULTS_DIR / "vix_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
