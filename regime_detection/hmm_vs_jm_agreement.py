"""Formal HMM-vs-JM agreement - Phase 2 plan item 4, #6.

Turns the earlier qualitative comparison (hmm_fit.py's working-notes entry) into
actual numbers: ARI, NMI, and a transition-timing comparison between the two models'
label sequences.

Run at **k=3**, not k=4 - same reasoning as the VIX ablation: the refit-stability
check found k=4/k=5 seed-dependent for both models, and comparing two models at an
unstable k would confound "do JM and HMM agree" with "which unstable local optimum
did each happen to land on". k=3 is essentially perfectly stable for JM (ARI 1.0) and
comparatively stable for HMM (mean ARI 0.885, better than k=4/k=5). Full 11-feature
set for both (not the VIX-ablated set) - this checks agreement between the two
models as they're actually meant to run, not a variant.

ARI and NMI are both permutation-invariant (they compare partition structure, not
literal label values), so no relabeling is needed before comparing JM's and HMM's
label sequences, even though the two models number their states independently.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.hmm_fit import N_ITER, TOL
from regime_detection.jump_model_fit import RESULTS_DIR, prepare_fit_data, preprocess
from hmmlearn.hmm import GaussianHMM

K = 3
LAMBDA = 50.0


def transition_dates(labels: pd.Series) -> pd.DatetimeIndex:
    changed = labels.ne(labels.shift())
    changed.iloc[0] = False  # the first row isn't a "transition", it's the start
    return labels.index[changed]


def nearest_distances(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> np.ndarray:
    """For each date in a, the calendar-day distance to the nearest date in b."""
    if len(b) == 0:
        return np.array([])
    b_arr = np.asarray(b, dtype="datetime64[ns]")
    distances = [
        float(np.abs(b_arr - d).min() / np.timedelta64(1, "D"))
        for d in np.asarray(a, dtype="datetime64[ns]")
    ]
    return np.array(distances)


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw, ret_ser = prepare_fit_data(df)
    X = preprocess(X_raw)
    print(f"Fit set: {len(X)} rows (identical for both models)")

    print(f"\nFitting JM: k={K}, lambda={LAMBDA}...")
    jm = JumpModel(n_components=K, jump_penalty=LAMBDA, cont=False, random_state=0)
    jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
    jm_labels = pd.Series(jm.labels_, index=X.index)

    print(f"Fitting HMM: k={K}, best of 10 restarts...")
    candidates = []
    for seed in range(10):
        model = GaussianHMM(n_components=K, covariance_type="diag", random_state=seed, n_iter=N_ITER, tol=TOL)
        model.fit(X.to_numpy())
        candidates.append((model.score(X.to_numpy()), model))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    hmm_model = candidates[0][1]
    hmm_labels = pd.Series(hmm_model.predict(X.to_numpy()), index=X.index)

    ari = adjusted_rand_score(jm_labels.to_numpy(), hmm_labels.to_numpy())
    nmi = normalized_mutual_info_score(jm_labels.to_numpy(), hmm_labels.to_numpy())
    print(f"\n=== Label agreement ===")
    print(f"ARI:  {ari:.4f}")
    print(f"NMI:  {nmi:.4f}")

    jm_state_counts = jm_labels.value_counts().sort_index().to_dict()
    hmm_state_counts = hmm_labels.value_counts().sort_index().to_dict()
    print(f"\nJM state counts:  {jm_state_counts}")
    print(f"HMM state counts: {hmm_state_counts}")

    jm_trans = transition_dates(jm_labels)
    hmm_trans = transition_dates(hmm_labels)
    print(f"\n=== Transition timing ===")
    print(f"JM transitions: {len(jm_trans)}, HMM transitions: {len(hmm_trans)}")

    jm_to_hmm = nearest_distances(jm_trans, hmm_trans)
    hmm_to_jm = nearest_distances(hmm_trans, jm_trans)
    print(f"JM transition -> nearest HMM transition: median {np.median(jm_to_hmm):.1f} days, "
          f"mean {np.mean(jm_to_hmm):.1f} days")
    print(f"HMM transition -> nearest JM transition: median {np.median(hmm_to_jm):.1f} days, "
          f"mean {np.mean(hmm_to_jm):.1f} days")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "k": K, "lambda_jm": LAMBDA,
        "ari": float(ari), "nmi": float(nmi),
        "jm_state_counts": {int(k): int(v) for k, v in jm_state_counts.items()},
        "hmm_state_counts": {int(k): int(v) for k, v in hmm_state_counts.items()},
        "jm_n_transitions": len(jm_trans),
        "hmm_n_transitions": len(hmm_trans),
        "jm_to_hmm_median_days": float(np.median(jm_to_hmm)) if len(jm_to_hmm) else None,
        "hmm_to_jm_median_days": float(np.median(hmm_to_jm)) if len(hmm_to_jm) else None,
    }
    with open(RESULTS_DIR / "hmm_vs_jm_agreement.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
