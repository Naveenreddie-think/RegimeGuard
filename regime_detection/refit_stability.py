"""Initialization / refit stability check - Phase 2 plan item 4, #3 - for both JM
and HMM. First of the four remaining item-4 checks, run as a gating check per your
sequencing: if regime assignments aren't stable across random seeds, nothing else
(VIX ablation, HMM-vs-JM agreement, rolling-window stability) is worth trusting yet.

JM: JumpModel's n_init=10 handles restarts *within* one fit call (returns only the
best-of-10 solution's labels_). This check refits N=10 times with different top-level
random_state values (0-9) and compares the resulting label sequences via
sklearn.metrics.adjusted_rand_score - ARI near 1.0 means the overall solution is
stable across outer randomness, not just locally optimal within one run's internal
restarts. ARI is permutation-invariant (compares pairwise groupings, not literal
label values), so no relabeling is needed before comparing.

HMM: hmmlearn has no built-in multi-restart, so the N=10 restarts hmm_fit.py already
performs per k (to select the best-by-log-likelihood "official" fit) ARE the
refit-stability data - reused directly here via hmm_fit.run_grid(), not refit again.

Scope: k in {2,3,4,5} for both models. JM checked at the lambda=50 anchor only, not
the full lambda grid - keeps this proportionate to a "cheap gating check"; the
deferred rolling-window stability check is where the real (k, lambda) decision gets
made anyway, and can expand this check across lambda then if warranted.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.metrics import adjusted_rand_score

from data_agent.db import get_connection
from regime_detection.features import build_feature_matrix
from regime_detection.hmm_fit import K_GRID
from regime_detection.hmm_fit import run_grid as hmm_run_grid
from regime_detection.jump_model_fit import RESULTS_DIR, prepare_fit_data, preprocess

JM_LAMBDA_ANCHOR = 50.0
N_SEEDS = 10


def jm_refit_labels(X: pd.DataFrame, ret_ser: pd.Series, k: int, lam: float, n_seeds: int = N_SEEDS) -> list[np.ndarray]:
    label_sets = []
    for seed in range(n_seeds):
        jm = JumpModel(n_components=k, jump_penalty=lam, cont=False, random_state=seed)
        jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
        labels = jm.labels_
        label_sets.append(np.asarray(labels.to_numpy() if hasattr(labels, "to_numpy") else labels))
    return label_sets


def pairwise_ari(label_sets: list[np.ndarray]) -> list[float]:
    return [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw, ret_ser = prepare_fit_data(df)
    X = preprocess(X_raw)

    print(f"=== JM refit stability: {N_SEEDS} external random_state values, "
          f"lambda={JM_LAMBDA_ANCHOR} anchor ===")
    jm_records = []
    for k in K_GRID:
        label_sets = jm_refit_labels(X, ret_ser, k, JM_LAMBDA_ANCHOR)
        scores = pairwise_ari(label_sets)
        jm_records.append(
            {
                "k": k, "lambda": JM_LAMBDA_ANCHOR,
                "mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)),
                "max_ari": float(np.max(scores)),
            }
        )
        print(f"  k={k}: mean ARI={np.mean(scores):.4f}  min={np.min(scores):.4f}  max={np.max(scores):.4f}")
    jm_df = pd.DataFrame(jm_records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jm_df.to_csv(RESULTS_DIR / "jm_refit_stability.csv", index=False)

    print(f"\n=== HMM refit stability: {N_SEEDS} restarts per k (reusing hmm_fit.py's "
          f"own best-of-N selection data, not refitting) ===")
    _, hmm_fitted = hmm_run_grid(X, ret_ser)
    hmm_records = []
    for k in K_GRID:
        label_sets = hmm_fitted[k]["all_restarts"]
        scores = pairwise_ari(label_sets)
        hmm_records.append(
            {
                "k": k,
                "mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)),
                "max_ari": float(np.max(scores)),
            }
        )
        print(f"  k={k}: mean ARI={np.mean(scores):.4f}  min={np.min(scores):.4f}  max={np.max(scores):.4f}")
    hmm_df = pd.DataFrame(hmm_records)
    hmm_df.to_csv(RESULTS_DIR / "hmm_refit_stability.csv", index=False)

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
