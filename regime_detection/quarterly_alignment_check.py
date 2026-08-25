"""Diagnostic (not a fix): does sort_by="cumret" actually anchor state identity
consistently across the 63 independently-fit quarterly models in
quarterly_walk.py, or does naive concatenation-by-date silently patchwork together
labels that don't mean the same thing from quarter to quarter?

Re-fits every quarterly cutoff (deterministic - same random_state=0, same data as
quarterly_walk.py) to recover each quarter's (centers_, clipper, scaler) - these
were never persisted (model_versions only stores clipper/scaler bounds, not the
fitted JM centers_) - then applies the SAME centroid-based alignment
(regime_detection.rolling_window_stability.align_states, linear_sum_assignment) used
throughout Phase 2 for exactly this kind of cross-fit comparison, to every
consecutive pair of quarterly fits.

Isolating the question only - not fixing anything here, per the standing convention
for diagnostic scripts in this project.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.quarterly_walk import K, JUMP_PENALTY, generate_quarterly_cutoffs
from regime_detection.rolling_window_stability import align_states

warnings.filterwarnings("ignore")


def main() -> None:
    conn = get_connection()
    df = build_feature_matrix(conn)
    conn.close()

    fit_df = df.loc[~df["warm_up"]].copy()
    X_raw_full = fit_df[FEATURE_COLUMNS]
    ret_ser_full = fit_df["ret_5"]
    dates = X_raw_full.index

    cutoffs = generate_quarterly_cutoffs(dates)
    print(f"Re-fitting {len(cutoffs)} quarterly cutoffs to recover centers_ (not persisted)...")

    fits = []
    n_degenerate = 0
    for cutoff in cutoffs:
        X_raw = X_raw_full.loc[:cutoff]
        ret_ser = ret_ser_full.loc[:cutoff]
        clipper = DataClipperStd(mul=3.0)
        scaler = StandardScalerPD()
        X = scaler.fit_transform(clipper.fit_transform(X_raw))
        jm = JumpModel(n_components=K, jump_penalty=JUMP_PENALTY, cont=False, random_state=0)
        jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
        if np.isnan(jm.centers_).any():
            n_degenerate += 1
            print(f"  SKIPPING {cutoff.date()} (n={len(X_raw)}): degenerate fit, an empty state "
                  f"(NaN centroid) - too few rows for a meaningful k=3 fit at this cutoff. "
                  f"labels={pd.Series(jm.labels_).value_counts().to_dict()}")
            continue
        fits.append({"cutoff": cutoff, "centers": jm.centers_, "scaler": scaler, "ret_": jm.ret_})

    print(f"\n{n_degenerate} degenerate cutoff(s) excluded. "
          f"Checking alignment for {len(fits)-1} consecutive pairs among the remaining {len(fits)} fits.\n")

    identity = {0: 0, 1: 1, 2: 2}
    n_non_identity = 0
    rows = []
    for i in range(len(fits) - 1):
        a, b = fits[i], fits[i + 1]
        mapping = align_states(a["centers"], a["scaler"], b["centers"], b["scaler"])
        is_identity = mapping == identity
        n_non_identity += not is_identity
        rows.append({
            "cutoff_a": a["cutoff"].date(), "cutoff_b": b["cutoff"].date(),
            "mapping": mapping, "is_identity": is_identity,
            "ret_a": np.round(a["ret_"], 5).tolist(), "ret_b": np.round(b["ret_"], 5).tolist(),
        })
        marker = "" if is_identity else "  <-- NON-IDENTITY"
        print(f"  {a['cutoff'].date()} -> {b['cutoff'].date()}: mapping={mapping}{marker}")

    print(f"\n{n_non_identity} / {len(fits)-1} consecutive pairs have a NON-identity optimal alignment")
    print(f"({n_non_identity/(len(fits)-1):.1%} of quarter-to-quarter transitions)")

    pd.DataFrame(rows).to_csv("regime_detection/results/quarterly_alignment_check.csv", index=False)
    print("\nArtifact written to regime_detection/results/quarterly_alignment_check.csv")


if __name__ == "__main__":
    main()
