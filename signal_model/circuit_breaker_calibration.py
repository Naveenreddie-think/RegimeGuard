"""Phase 4, step 1: calibrate the M2 circuit-breaker drift threshold (`BREAKER`).

The confidence/abstention layer (docs/phase4_design_proposal.md, §2) has a hard
circuit-breaker: when tier-1 standardization drift is severe enough, the system
abstains regardless of regime. Phase 2 only ever computed the standardization-drift
metric at two rolling-window cutoffs - 2016-12-31 (mean drift 0.827, JM k=3 interior
ARI 0.44) and 2020-06-30 (0.327, interior ARI 0.56) - and BOTH are failing configs,
so there is no spread to place a hard cutoff against yet.

This spike closes that gap. It reuses `standardization_drift.py`'s exact metric
(per-date Euclidean distance in standardized space between two clip+scale pipelines,
interior dates only, clustering never runs) and computes it at the windows it was
never run on:

- **Full-sample comparison** (truncated fit vs. full-sample fit, same as the Phase 2
  script) at the 2018-12-31 and 2022-12-31 cutoffs. 2022-12-31 is the key addition:
  its JM k=3 interior ARI was 0.97 (a clean pass), so its drift is the first
  *passing* reference point this metric has ever had.
- **Consecutive-pair comparison** (fit_A's pipeline vs. fit_B's pipeline, on fit_A's
  interior dates) for all four Phase 2 consecutive pairs, in particular
  2020-06-30 -> 2022-12-31, where JM k=3 interior ARI was 0.55 and edge ARI -0.08.

2016-12-31 and 2020-06-30 are recomputed too, as a reproduction check against the
Phase 2 numbers (0.827 / 0.327).

Output: `signal_model/results/circuit_breaker_calibration.csv` (every drift value
paired with its already-known interior-ARI outcome) plus a printed recommendation
for `BREAKER` per the rule in design §2 ("at or just below the lowest drift among
windows whose interior ARI fell to near-chance, <= ~0.5"), or the documented 0.827
fallback if the points give no clean separation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.jump_model_fit import prepare_fit_data
from regime_detection.rolling_window_stability import WARM_UP_EDGE_DAYS
from regime_detection.standardization_drift import fit_pipeline, transform_with

RESULTS_DIR = Path(__file__).parent / "results"

# JM k=3 interior ARI already established in Phase 2 for each window - not recomputed
# here (this spike is about the drift metric only), pulled from
# regime_detection/results/rolling_window_stability{,_consecutive}.json.
FULL_SAMPLE_INTERIOR_ARI = {
    "2016-12-31": 0.4381,
    "2018-12-31": 0.5878,
    "2020-06-30": 0.5594,
    "2022-12-31": 0.9687,
}
CONSECUTIVE_INTERIOR_ARI = {
    ("2016-12-31", "2018-12-31"): 0.9828,
    ("2018-12-31", "2020-06-30"): 0.9527,
    ("2020-06-30", "2022-12-31"): 0.5484,
    ("2022-12-31", "full-sample"): 0.9687,
}
NEAR_CHANCE_ARI = 0.50  # design §2: "interior ARI fell to near-chance (<= ~0.5)"


def _interior_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Same >90-trading-day-before-cutoff interior definition as
    rolling_window_stability.py / standardization_drift.py."""
    if len(index) <= WARM_UP_EDGE_DAYS:
        return index[:0]
    edge_start = index[-WARM_UP_EDGE_DAYS]
    return index[index < edge_start]


def _drift(raw_interior: pd.DataFrame, pipe_a, pipe_b) -> dict:
    """Per-date Euclidean distance between the same raw rows standardized two ways."""
    std_a = transform_with(raw_interior, *pipe_a)
    std_b = transform_with(raw_interior, *pipe_b)
    per_date = np.sqrt(((std_b - std_a) ** 2).sum(axis=1))
    return {
        "n_interior": int(len(raw_interior)),
        "mean_drift": float(per_date.mean()),
        "median_drift": float(per_date.median()),
        "max_drift": float(per_date.max()),
    }


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw_full, _ = prepare_fit_data(df)
    full_pipe = fit_pipeline(X_raw_full)  # (scaler, clipper)
    rows = []

    # --- Full-sample comparison: truncated fit vs. full-sample fit ---
    print("=" * 78)
    print("FULL-SAMPLE COMPARISON  (truncated-fit standardization vs. full-sample fit)")
    print("=" * 78)
    for cutoff, known_ari in FULL_SAMPLE_INTERIOR_ARI.items():
        trunc_raw = X_raw_full.loc[:cutoff]
        trunc_pipe = fit_pipeline(trunc_raw)
        interior = _interior_dates(trunc_raw.index)
        d = _drift(X_raw_full.loc[interior], trunc_pipe, full_pipe)
        note = "reproduction check" if cutoff in ("2016-12-31", "2020-06-30") else "NEW"
        rows.append({
            "comparison": "full_sample", "window": cutoff, "vs": "full-sample",
            "jm_k3_interior_ari": known_ari, **d, "note": note,
        })
        print(f"\n  {cutoff}  ({note})")
        print(f"    interior dates            : {d['n_interior']}")
        print(f"    JM k=3 interior ARI (P2)  : {known_ari:.4f}  "
              f"({'PASS' if known_ari >= 0.85 else 'FAIL'} vs 0.85 bar)")
        print(f"    mean / median / max drift : {d['mean_drift']:.4f} / "
              f"{d['median_drift']:.4f} / {d['max_drift']:.4f}")

    # --- Consecutive-pair comparison: fit_A pipeline vs. fit_B pipeline ---
    print("\n" + "=" * 78)
    print("CONSECUTIVE-PAIR COMPARISON  (fit_A standardization vs. fit_B, on fit_A interior)")
    print("=" * 78)
    for (cutoff_a, cutoff_b), known_ari in CONSECUTIVE_INTERIOR_ARI.items():
        raw_a = X_raw_full.loc[:cutoff_a]
        pipe_a = fit_pipeline(raw_a)
        pipe_b = full_pipe if cutoff_b == "full-sample" else fit_pipeline(X_raw_full.loc[:cutoff_b])
        interior = _interior_dates(raw_a.index)
        d = _drift(X_raw_full.loc[interior], pipe_a, pipe_b)
        rows.append({
            "comparison": "consecutive", "window": cutoff_a, "vs": cutoff_b,
            "jm_k3_interior_ari": known_ari, **d,
            "note": "COVID->2022 (design-flagged)" if (cutoff_a, cutoff_b) == ("2020-06-30", "2022-12-31") else "",
        })
        print(f"\n  {cutoff_a} -> {cutoff_b}")
        print(f"    interior dates            : {d['n_interior']}")
        print(f"    JM k=3 interior ARI (P2)  : {known_ari:.4f}  "
              f"({'PASS' if known_ari >= 0.85 else 'FAIL'} vs 0.85 bar)")
        print(f"    mean / median / max drift : {d['mean_drift']:.4f} / "
              f"{d['median_drift']:.4f} / {d['max_drift']:.4f}")

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "circuit_breaker_calibration.csv", index=False)

    # --- Recommendation, per design §2 ---
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    near_chance = out[out["jm_k3_interior_ari"] <= NEAR_CHANCE_ARI]
    passing = out[out["jm_k3_interior_ari"] >= 0.85]
    print(f"\n  windows with interior ARI <= {NEAR_CHANCE_ARI} (near-chance):")
    if near_chance.empty:
        print("    (none)")
    else:
        for _, r in near_chance.iterrows():
            print(f"    {r['window']:>12} vs {r['vs']:<12}  ARI={r['jm_k3_interior_ari']:.3f}  "
                  f"mean drift={r['mean_drift']:.4f}")
    print(f"\n  windows with interior ARI >= 0.85 (passing - drift 'safe' reference):")
    if passing.empty:
        print("    (none)")
    else:
        for _, r in passing.iterrows():
            print(f"    {r['window']:>12} vs {r['vs']:<12}  ARI={r['jm_k3_interior_ari']:.3f}  "
                  f"mean drift={r['mean_drift']:.4f}")

    fail_band = out[(out["jm_k3_interior_ari"] > NEAR_CHANCE_ARI) & (out["jm_k3_interior_ari"] < 0.85)]
    print(f"\n  windows in the 0.5-0.85 'failing but not near-chance' band:")
    for _, r in fail_band.iterrows():
        print(f"    {r['window']:>12} vs {r['vs']:<12}  ARI={r['jm_k3_interior_ari']:.3f}  "
              f"mean drift={r['mean_drift']:.4f}")

    print(f"\n  max drift among passing windows      : "
          f"{passing['mean_drift'].max():.4f}" if not passing.empty else "  (no passing windows)")
    print(f"  min drift among near-chance windows  : "
          f"{near_chance['mean_drift'].min():.4f}" if not near_chance.empty else "  (no near-chance windows)")
    print("\n  -> see docs/phase4_working_notes.md for the chosen BREAKER value and rationale.")
    print(f"\nArtifact: {RESULTS_DIR / 'circuit_breaker_calibration.csv'}")


if __name__ == "__main__":
    main()
