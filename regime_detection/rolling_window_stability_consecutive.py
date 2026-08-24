"""Rolling-window stability, consecutive-pair variant - Phase 2 plan item 4 #4,
additional diagnostic per review.

The original rolling_window_stability.py compares each truncated fit against the
FULL-SAMPLE fit, which has hindsight (COVID, 2022) no real point-in-time system would
have had at the truncated fit's own moment. This variant compares each truncated fit
against the *next* truncated fit in the chain instead - both genuinely point-in-time-
honest at the moment each was fit. This is additional evidence, not a replacement:
the original full-sample comparison still answers a real, different question ("how
wrong would an early call have been against full hindsight") and is kept as a
separate finding, not discarded.

Chain: 2016-12-31 -> 2018-12-31 -> 2020-06-30 -> 2022-12-31 -> full-sample. Each
consecutive pair (A, B) is compared on A's own date range (A being the earlier,
less-informed fit). Per review:
- interior/edge split is measured relative to A's cutoff (A's own last 90 trading
  days = edge), not B's - the question is whether A's own honest call held up.
- the hindsight gap between pairs is NOT constant (computed and reported per pair,
  not assumed) - flagged as a possible confound if pass/fail trends with gap size,
  not corrected for here.

Same bars, same disqualification logic, same k grid, same lambda, same state-
alignment mechanism as the original check (now using B's standardized space as the
common reference instead of the full-sample's).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_agent.db import get_connection
from regime_detection.features import build_feature_matrix
from regime_detection.jump_model_fit import RESULTS_DIR, KNOWN_STRESS_WINDOWS, prepare_fit_data
from regime_detection.rolling_window_stability import (
    EDGE_ARI_BAR,
    INTERIOR_ARI_BAR,
    K_GRID,
    WARM_UP_EDGE_DAYS,
    align_states,
    dominant_regime,
    fit_hmm,
    fit_jm,
)
from sklearn.metrics import adjusted_rand_score

CHAIN = ["2016-12-31", "2018-12-31", "2020-06-30", "2022-12-31", None]  # None = full sample
PAIR_WINDOWS = {
    "2016-12-31": "2016 demonetization",
    "2018-12-31": "2018 IL&FS stress",
    "2020-06-30": "2020 COVID crash",
    "2022-12-31": "2022 rate-hike volatility",
}
PAIRS = list(zip(CHAIN[:-1], CHAIN[1:]))  # [(2016,2018), (2018,2020-06), (2020-06,2022), (2022,None)]


def slice_data(X_raw_full: pd.DataFrame, cutoff: str | None) -> pd.DataFrame:
    return X_raw_full if cutoff is None else X_raw_full.loc[:cutoff]


def gap_years(cutoff_a: str, cutoff_b: str | None, last_date: pd.Timestamp) -> float:
    end = last_date if cutoff_b is None else pd.Timestamp(cutoff_b)
    return (end - pd.Timestamp(cutoff_a)).days / 365.25


def evaluate_pair(X_raw_full: pd.DataFrame, ret_ser_full: pd.Series, cutoff_a: str, cutoff_b: str | None, k: int, model_kind: str) -> dict:
    X_a = slice_data(X_raw_full, cutoff_a)
    X_b = slice_data(X_raw_full, cutoff_b)
    ret_a = slice_data(ret_ser_full.to_frame("r"), cutoff_a)["r"]
    ret_b = slice_data(ret_ser_full.to_frame("r"), cutoff_b)["r"]

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
        Xa_std = scaler_a.transform(clipper_a.transform(X_a))
        Xb_std = scaler_b.transform(clipper_b.transform(X_b))
        labels_a = pd.Series(model_a.predict(Xa_std.to_numpy()), index=X_a.index)
        labels_b_full = pd.Series(model_b.predict(Xb_std.to_numpy()), index=X_b.index)

    labels_b_on_a = labels_b_full.loc[labels_a.index]  # B's labels, restricted to A's dates

    overlap_dates = labels_a.index
    edge_start = overlap_dates[-WARM_UP_EDGE_DAYS] if len(overlap_dates) > WARM_UP_EDGE_DAYS else overlap_dates[0]
    interior_mask = overlap_dates < edge_start
    edge_mask = ~interior_mask

    interior_ari = (
        adjusted_rand_score(labels_a[interior_mask], labels_b_on_a[interior_mask])
        if interior_mask.sum() > 1 else None
    )
    edge_ari = (
        adjusted_rand_score(labels_a[edge_mask], labels_b_on_a[edge_mask])
        if edge_mask.sum() > 1 else None
    )

    # Align A's states into B's standardized space (B is the reference - "the next,
    # more-informed point-in-time state of knowledge").
    state_map = align_states(centers_a, scaler_a, centers_b, scaler_b)

    window_name = PAIR_WINDOWS[cutoff_a]
    a_dom_raw = dominant_regime(labels_a, *KNOWN_STRESS_WINDOWS[window_name])
    a_dom_aligned = state_map.get(a_dom_raw) if a_dom_raw is not None else None
    b_dom = dominant_regime(labels_b_on_a, *KNOWN_STRESS_WINDOWS[window_name])
    interp_pass = (a_dom_aligned == b_dom) if b_dom is not None else None

    passed = (
        (interior_ari is None or interior_ari >= INTERIOR_ARI_BAR)
        and (edge_ari is None or edge_ari >= EDGE_ARI_BAR)
        and (interp_pass in (True, None))
    )

    return {
        "cutoff_a": cutoff_a, "cutoff_b": cutoff_b or "full-sample",
        "gap_years": round(gap_years(cutoff_a, cutoff_b, X_raw_full.index[-1]), 2),
        "window_tested": window_name,
        "interior_ari": interior_ari, "edge_ari": edge_ari,
        "a_dominant_state_raw": a_dom_raw, "a_dominant_state_aligned_to_b": a_dom_aligned,
        "b_dominant_state": b_dom, "interpretability_pass": interp_pass,
        "passed": bool(passed),
    }


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw_full, ret_ser_full = prepare_fit_data(df)
    print(f"Full-sample fit set: {len(X_raw_full)} rows")
    print(f"Pairs: {PAIRS}")

    all_results = []
    for model_kind in ["jm", "hmm"]:
        for k in K_GRID:
            print(f"\n=== {model_kind.upper()} k={k} ===")
            pair_results = []
            for cutoff_a, cutoff_b in PAIRS:
                r = evaluate_pair(X_raw_full, ret_ser_full, cutoff_a, cutoff_b, k, model_kind)
                pair_results.append(r)
                print(f"  {r['cutoff_a']} -> {r['cutoff_b']} (gap={r['gap_years']}y, window={r['window_tested']}): "
                      f"interior_ari={r['interior_ari']}, edge_ari={r['edge_ari']}, "
                      f"interp_pass={r['interpretability_pass']}, passed={r['passed']}")
            overall_pass = all(r["passed"] for r in pair_results)
            print(f"  overall_pass: {overall_pass}")
            all_results.append({"model": model_kind, "k": k, "overall_pass": bool(overall_pass), "pairs": pair_results})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rolling_window_stability_consecutive.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary ===")
    for r in all_results:
        print(f"  {r['model'].upper()} k={r['k']}: {'PASS' if r['overall_pass'] else 'FAIL'}")

    print("\n=== Gap-size vs. pass-rate check (possible confound, per review) ===")
    gap_pass = {}
    for r in all_results:
        for p in r["pairs"]:
            gap_pass.setdefault(p["gap_years"], []).append(p["passed"])
    for gap in sorted(gap_pass):
        passes = gap_pass[gap]
        print(f"  gap={gap}y: {sum(passes)}/{len(passes)} individual pair-checks passed")

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
