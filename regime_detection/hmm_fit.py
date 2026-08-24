"""Gaussian HMM baseline: same feature set, same k grid, same reporting discipline
as the Statistical Jump Model (jump_model_fit.py), per the approved Phase 2 plan §3.

Reuses prepare_fit_data() and preprocess() from jump_model_fit.py rather than
duplicating the warm-up exclusion / clip+standardize logic - the whole point of a
"named baseline for comparison" is that differences between JM and HMM results
reflect the modeling approach, not different data handling.

Two real asymmetries versus JM, both called out here because they change how the
fitting code has to work, not just what numbers come out:
- hmmlearn's GaussianHMM has no built-in multi-restart (JumpModel has n_init=10).
  Multi-restart is done manually here: N=10 fits per k with different random_state,
  best selected by .score() (log-likelihood).
- hmmlearn's default n_iter=10 is too low for EM to reliably converge on financial
  return data. Every fit uses n_iter=1000, tol=1e-6, and checks
  model.monitor_.converged explicitly - an unconverged fit is not silently accepted
  just because it produced labels.
- hmmlearn has no sort_by="cumret" like JumpModel. States are relabeled post-fit by
  mean short-horizon return, ascending, so state numbering is comparable to JM's
  convention (0 = worst, k-1 = best) rather than the EM algorithm's arbitrary order.

This is the same full-sample exploratory pass as the JM first fit - not yet the
leakage-safe walk-forward-validated series (item 4).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.jump_model_fit import (
    RESULTS_DIR,
    KNOWN_STRESS_WINDOWS,
    interpretability_report,
    prepare_fit_data,
    preprocess,
)

K_GRID = [2, 3, 4, 5]
N_RESTARTS = 10
N_ITER = 1000
TOL = 1e-6


def _relabel_by_mean_return(labels: np.ndarray, ret_ser: pd.Series, n_components: int) -> np.ndarray:
    """Relabel states 0..k-1 by ascending mean return, matching JM's sort_by='cumret'
    convention, so state numbering is comparable across models."""
    mean_ret_per_state = {s: ret_ser.to_numpy()[labels == s].mean() for s in range(n_components)}
    order = sorted(mean_ret_per_state, key=mean_ret_per_state.get)  # ascending
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[s] for s in labels]), order


def _run_lengths(labels: np.ndarray) -> np.ndarray:
    s = pd.Series(labels)
    changes = s.ne(s.shift()).cumsum()
    return s.groupby(changes).size().to_numpy()


def fit_best_of_n(X: pd.DataFrame, k: int, n_restarts: int = N_RESTARTS) -> tuple[GaussianHMM, list[np.ndarray]]:
    """Fit N times with different random_state, keep the best by log-likelihood.
    Returns the best model and the raw (pre-relabel) label arrays from every
    restart, for the initialization/refit-stability check (item 4) later."""
    candidates = []
    for seed in range(n_restarts):
        model = GaussianHMM(
            n_components=k, covariance_type="diag", random_state=seed,
            n_iter=N_ITER, tol=TOL,
        )
        model.fit(X.to_numpy())
        if not model.monitor_.converged:
            print(f"  WARNING: k={k} seed={seed} did NOT converge (n_iter={N_ITER})")
        candidates.append((model.score(X.to_numpy()), model))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_model = candidates[0]
    all_labels = [m.predict(X.to_numpy()) for _, m in candidates]
    return best_model, all_labels


def run_grid(X: pd.DataFrame, ret_ser: pd.Series) -> tuple[pd.DataFrame, dict]:
    records = []
    fitted = {}
    n = len(X)
    for k in K_GRID:
        print(f"Fitting HMM k={k} ({N_RESTARTS} restarts)...")
        best_model, all_labels = fit_best_of_n(X, k)
        raw_labels = best_model.predict(X.to_numpy())
        labels, state_order = _relabel_by_mean_return(raw_labels, ret_ser, k)

        # Free parameters, respecting the sum-to-1 constraints: (k-1) startprob +
        # k*(k-1) transmat (each row sums to 1) + k*n_features means +
        # k*n_features diag covariances.
        n_features = X.shape[1]
        n_params = (k - 1) + k * (k - 1) + 2 * k * n_features
        bic = -2 * best_model.score(X.to_numpy()) + n_params * np.log(n)

        diag = np.diag(best_model.transmat_)
        run_lengths = _run_lengths(labels)
        records.append(
            {
                "k": k,
                "log_likelihood": float(best_model.score(X.to_numpy())),
                "bic": float(bic),
                "converged": bool(best_model.monitor_.converged),
                "n_switches": int((pd.Series(labels).diff().fillna(0) != 0).sum()),
                "mean_self_persistence": float(np.mean(diag)),
                "min_self_persistence": float(np.min(diag)),
                "median_run_length": float(np.median(run_lengths)),
                "min_run_length": int(np.min(run_lengths)),
                "state_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
            }
        )
        fitted[k] = {"model": best_model, "labels": labels, "state_order": state_order, "all_restarts": all_labels}
    return pd.DataFrame(records), fitted


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw, ret_ser = prepare_fit_data(df)
    print(f"Fit set: {len(X_raw)} rows ({df['warm_up'].sum()} warm-up rows hard-excluded)")
    X = preprocess(X_raw)

    grid, fitted = run_grid(X, ret_ser)
    pd.set_option("display.width", 160)
    print("\n=== Grid results ===")
    print(grid.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS_DIR / "hmm_grid.csv", index=False)

    # Same anchor-not-decision framing as the JM run: k=4 for direct comparability,
    # not because it's been selected.
    k = 4
    info = fitted[k]
    model = info["model"]
    print(f"\n=== Interpretability check: k={k} (comparability anchor, not decided) ===")

    # means_ are in the EM's original state order; reorder to match the relabeled
    # (ascending-return) state numbering used in labels_/reports below.
    means_reordered = model.means_[info["state_order"]]
    centers_df = pd.DataFrame(means_reordered, columns=FEATURE_COLUMNS).round(3)
    print("State means (standardized feature space):")
    print(centers_df.to_string())
    centers_df.to_csv(RESULTS_DIR / f"hmm_k{k}_means.csv")

    labels = pd.Series(info["labels"], index=X.index)
    report = interpretability_report(labels)
    print("\nKnown stress-window regime assignment:")
    for name, res in report.items():
        print(f"  {name}: {res}")
    with open(RESULTS_DIR / f"hmm_k{k}_interpretability.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
