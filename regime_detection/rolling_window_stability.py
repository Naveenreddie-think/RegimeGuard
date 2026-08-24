"""Rolling-window stability - Phase 2 plan item 4, #4. The most expensive check, and
the one that settles (k, lambda)/k, per direction after the design review.

Design (reviewed and confirmed before running):
- 4 expanding-window cutoffs, each anchored right after a known stress event ends:
  2016-12-31 (demonetization), 2018-12-31 (IL&FS), 2020-06-30 (COVID), 2022-12-31
  (2022 rate-hike vol). Fixed start (2010-07-19), truncated end - expanding, not a
  sliding fixed-length window, matching how a real point-in-time system accumulates
  history.
- JM: k in {3,4,5}, lambda=50 fixed, single top-level random_state=0 per cutoff
  (JM's internal n_init=10 already runs regardless of the top-level seed, so this
  doesn't conflate with the separately-tested seed-instability finding).
- HMM: k in {3,4,5}, best-of-10-restarts per cutoff (same methodology used
  everywhere else for HMM) - caveat: this bakes a small amount of seed-selection
  variability into HMM's numbers that isn't separately controlled for, unlike JM's.
- Each truncated fit gets its OWN point-in-time-correct standardization (clip+scale
  fit only on data through that cutoff) - never the full-sample scaler.
- Pass/fail, set before running:
  - interior ARI >= 0.85 (dates >90 trading days before the cutoff)
  - near-edge ARI >= 0.60 (last 90 trading days before the cutoff)
  - the dominant regime for the nearest known stress window must match the
    full-sample fit's dominant regime for that window, after STATE ALIGNMENT (see
    align_states() - raw label numbers aren't comparable across separately-fit
    models, per review)
  - failing ANY criterion on ANY cutoff disqualifies that (k, lambda)/k
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.hmm_fit import N_ITER, TOL
from regime_detection.jump_model_fit import (
    RESULTS_DIR,
    KNOWN_STRESS_WINDOWS,
    prepare_fit_data,
)

K_GRID = [3, 4, 5]
JM_LAMBDA = 50.0
WARM_UP_EDGE_DAYS = 90

CUTOFFS = {
    "2016-12-31": "2016 demonetization",
    "2018-12-31": "2018 IL&FS stress",
    "2020-06-30": "2020 COVID crash",
    "2022-12-31": "2022 rate-hike volatility",
}

INTERIOR_ARI_BAR = 0.85
EDGE_ARI_BAR = 0.60


def fit_jm(X_raw: pd.DataFrame, ret_ser: pd.Series, k: int) -> tuple[JumpModel, StandardScalerPD, DataClipperStd]:
    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    X = scaler.fit_transform(clipper.fit_transform(X_raw))
    jm = JumpModel(n_components=k, jump_penalty=JM_LAMBDA, cont=False, random_state=0)
    jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
    return jm, scaler, clipper


def fit_hmm(X_raw: pd.DataFrame, k: int, n_restarts: int = 10) -> tuple[GaussianHMM, StandardScalerPD, DataClipperStd]:
    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    X = scaler.fit_transform(clipper.fit_transform(X_raw))
    candidates = []
    for seed in range(n_restarts):
        m = GaussianHMM(n_components=k, covariance_type="diag", random_state=seed, n_iter=N_ITER, tol=TOL)
        m.fit(X.to_numpy())
        if not m.monitor_.converged:
            print(f"    WARNING: HMM k={k} seed={seed} on {len(X_raw)}-row window did NOT converge "
                  f"(n_iter={N_ITER}) - excluded from best-of-{n_restarts} selection")
            continue
        candidates.append((m.score(X.to_numpy()), m))
    if not candidates:
        raise RuntimeError(f"HMM k={k}: none of {n_restarts} restarts converged on this window")
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1], scaler, clipper


def align_states(
    trunc_centers: np.ndarray, trunc_scaler: StandardScalerPD,
    full_centers: np.ndarray, full_scaler: StandardScalerPD,
) -> dict[int, int]:
    """Map truncated-fit state index -> full-sample-fit state index, by bringing
    both sets of centroids into the full-sample's standardized coordinate space
    and solving optimal one-to-one assignment. See module docstring."""
    raw_centers = trunc_scaler.scaler.inverse_transform(trunc_centers)
    trunc_in_full_space = full_scaler.transform(pd.DataFrame(raw_centers, columns=FEATURE_COLUMNS)).to_numpy()
    dist = np.linalg.norm(trunc_in_full_space[:, None, :] - full_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist)
    return dict(zip(row_ind.tolist(), col_ind.tolist()))


def dominant_regime(labels: pd.Series, start: str, end: str) -> int | None:
    window = labels.loc[start:end]
    if window.empty:
        return None
    return int(window.value_counts().idxmax())


def evaluate_config(df: pd.DataFrame, X_raw_full: pd.DataFrame, ret_ser_full: pd.Series, k: int, model_kind: str) -> dict:
    if model_kind == "jm":
        full_model, full_scaler, _ = fit_jm(X_raw_full, ret_ser_full, k)
        full_centers = full_model.centers_
        full_labels = pd.Series(full_model.labels_, index=X_raw_full.index)
    else:
        full_model, full_scaler, full_clipper = fit_hmm(X_raw_full, k)
        full_centers = full_model.means_
        full_X = full_scaler.transform(full_clipper.transform(X_raw_full))
        full_labels = pd.Series(full_model.predict(full_X.to_numpy()), index=X_raw_full.index)
        # NOTE: means_ order for HMM is EM's arbitrary order; no relabeling applied
        # here since align_states() handles cross-fit correspondence directly via
        # centroid matching, not via label convention.

    cutoff_results = {}
    for cutoff, window_name in CUTOFFS.items():
        trunc_X_raw = X_raw_full.loc[:cutoff]
        trunc_ret = ret_ser_full.loc[:cutoff]
        if len(trunc_X_raw) < 500:  # sanity floor, not expected to trigger
            continue

        if model_kind == "jm":
            trunc_model, trunc_scaler, _ = fit_jm(trunc_X_raw, trunc_ret, k)
            trunc_centers = trunc_model.centers_
            trunc_labels = pd.Series(trunc_model.labels_, index=trunc_X_raw.index)
        else:
            trunc_model, trunc_scaler, trunc_clipper = fit_hmm(trunc_X_raw, k)
            trunc_centers = trunc_model.means_
            trunc_labels = pd.Series(
                trunc_model.predict(trunc_scaler.transform(trunc_clipper.transform(trunc_X_raw)).to_numpy()),
                index=trunc_X_raw.index,
            )

        overlap_dates = trunc_labels.index
        full_on_overlap = full_labels.loc[overlap_dates]

        edge_start = overlap_dates[-WARM_UP_EDGE_DAYS] if len(overlap_dates) > WARM_UP_EDGE_DAYS else overlap_dates[0]
        interior_mask = overlap_dates < edge_start
        edge_mask = ~interior_mask

        interior_ari = (
            adjusted_rand_score(trunc_labels[interior_mask], full_on_overlap[interior_mask])
            if interior_mask.sum() > 1 else None
        )
        edge_ari = (
            adjusted_rand_score(trunc_labels[edge_mask], full_on_overlap[edge_mask])
            if edge_mask.sum() > 1 else None
        )

        state_map = align_states(trunc_centers, trunc_scaler, full_centers, full_scaler)
        trunc_dom_raw = dominant_regime(trunc_labels, *KNOWN_STRESS_WINDOWS[window_name])
        trunc_dom_aligned = state_map.get(trunc_dom_raw) if trunc_dom_raw is not None else None
        full_dom = dominant_regime(full_labels, *KNOWN_STRESS_WINDOWS[window_name])

        interpretability_pass = (trunc_dom_aligned == full_dom) if full_dom is not None else None

        passed = (
            (interior_ari is None or interior_ari >= INTERIOR_ARI_BAR)
            and (edge_ari is None or edge_ari >= EDGE_ARI_BAR)
            and (interpretability_pass in (True, None))
        )

        cutoff_results[cutoff] = {
            "window_tested": window_name,
            "interior_ari": interior_ari,
            "edge_ari": edge_ari,
            "trunc_dominant_state_raw": trunc_dom_raw,
            "trunc_dominant_state_aligned_to_full": trunc_dom_aligned,
            "full_dominant_state": full_dom,
            "interpretability_pass": interpretability_pass,
            "passed": bool(passed),
        }

    overall_pass = all(r["passed"] for r in cutoff_results.values())
    return {"k": k, "model": model_kind, "overall_pass": bool(overall_pass), "cutoffs": cutoff_results}


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw_full, ret_ser_full = prepare_fit_data(df)
    print(f"Full-sample fit set: {len(X_raw_full)} rows")

    all_results = []
    for model_kind in ["jm", "hmm"]:
        for k in K_GRID:
            print(f"\n=== Evaluating {model_kind.upper()} k={k} ===")
            result = evaluate_config(df, X_raw_full, ret_ser_full, k, model_kind)
            all_results.append(result)
            print(f"  overall_pass: {result['overall_pass']}")
            for cutoff, r in result["cutoffs"].items():
                print(f"  {cutoff} ({r['window_tested']}): interior_ari={r['interior_ari']}, "
                      f"edge_ari={r['edge_ari']}, interp_pass={r['interpretability_pass']}, "
                      f"passed={r['passed']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rolling_window_stability.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)

    print("\n=== Summary ===")
    for r in all_results:
        print(f"  {r['model'].upper()} k={r['k']}: {'PASS' if r['overall_pass'] else 'FAIL'}")
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
