"""Statistical Jump Model fitting: k/lambda grid, persistence diagnostics, and an
early interpretability read against known stress windows.

This is the exploratory checkpoint (per the approved Phase 2 plan, §2) - a full-
sample fit to see whether the approach produces sane regimes and narrow down
candidate (k, lambda) values. It is explicitly NOT yet the leakage-safe, walk-
forward-validated regime series - that's the rolling-window stability check (Phase 2
item 4), which refits per-window with its own standardization, never a single
global fit like this one. Flagged here, not silently conflated.

Warm-up rows (the first 90 trading days, where the longest EWM halflife hasn't
accumulated enough history) are a HARD exclusion: dropped from the DataFrame before
clipping, standardization, or fitting ever see them - not a flag carried through
that something downstream has to remember to honor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix

K_GRID = [2, 3, 4, 5]
LAMBDA_GRID = [0.0, 10.0, 30.0, 50.0, 100.0, 200.0]  # anchored on the reference
# implementation's own choice of lambda=50 for a 2-state discrete JM (see the
# jumpmodels package's Nasdaq-100 example) - not picked blind.

KNOWN_STRESS_WINDOWS = {
    "2013 taper tantrum": ("2013-05-22", "2013-09-15"),
    "2016 demonetization": ("2016-11-08", "2016-12-31"),
    "2018 IL&FS stress": ("2018-09-01", "2018-10-31"),
    "2020 COVID crash": ("2020-02-20", "2020-04-15"),
    "2022 rate-hike volatility": ("2022-04-01", "2022-10-31"),
}


def prepare_fit_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Hard-drop warm_up rows before anything else touches the data. Returns the
    raw (unstandardized) feature matrix and the aligned return series."""
    fit_df = df.loc[~df["warm_up"]].copy()
    assert not fit_df["warm_up"].any(), "warm_up rows leaked into the fit set"
    X_raw = fit_df[FEATURE_COLUMNS]
    ret_ser = fit_df["ret_5"]  # short-horizon return series for state sorting/stats
    return X_raw, ret_ser


def preprocess(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Clip at 3 std devs, then standardize. Full-sample fit for this exploratory
    pass (see module docstring) - the rolling-window check refits this per-window."""
    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    return scaler.fit_transform(clipper.fit_transform(X_raw))


def run_grid(X: pd.DataFrame, ret_ser: pd.Series) -> pd.DataFrame:
    records = []
    fitted = {}
    for k in K_GRID:
        for lam in LAMBDA_GRID:
            jm = JumpModel(n_components=k, jump_penalty=lam, cont=False, random_state=0)
            jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
            labels = jm.labels_
            diag = np.diag(jm.transmat_)
            run_lengths = _run_lengths(labels)
            records.append(
                {
                    "k": k,
                    "lambda": lam,
                    "n_switches": int((labels.diff().fillna(0) != 0).sum()),
                    "mean_self_persistence": float(np.mean(diag)),
                    "min_self_persistence": float(np.min(diag)),
                    "median_run_length": float(np.median(run_lengths)),
                    "min_run_length": int(np.min(run_lengths)),
                    "state_counts": labels.value_counts().sort_index().to_dict(),
                }
            )
            fitted[(k, lam)] = jm
    return pd.DataFrame(records), fitted


def _run_lengths(labels: pd.Series) -> np.ndarray:
    changes = labels.ne(labels.shift()).cumsum()
    return labels.groupby(changes).size().to_numpy()


def interpretability_report(labels: pd.Series) -> dict[str, dict]:
    report = {}
    for name, (start, end) in KNOWN_STRESS_WINDOWS.items():
        window = labels.loc[start:end]
        if window.empty:
            report[name] = {"note": "no data in this window"}
            continue
        counts = window.value_counts(normalize=True).sort_values(ascending=False)
        report[name] = {
            "dominant_regime": int(counts.index[0]),
            "dominant_share": float(counts.iloc[0]),
            "n_days": int(len(window)),
            "regime_breakdown": {int(k): float(v) for k, v in counts.to_dict().items()},
        }
    return report


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()

    X_raw, ret_ser = prepare_fit_data(df)
    print(f"Fit set: {len(X_raw)} rows ({df['warm_up'].sum()} warm-up rows hard-excluded)")
    X = preprocess(X_raw)

    print(f"\nFitting grid: k={K_GRID} x lambda={LAMBDA_GRID} ({len(K_GRID)*len(LAMBDA_GRID)} fits)...")
    grid, fitted = run_grid(X, ret_ser)
    pd.set_option("display.width", 160)
    print("\n=== Grid results ===")
    print(grid.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS_DIR / "jm_grid.csv", index=False)

    # Anchor point only - NOT a decided (k, lambda). The reference paper used
    # lambda=50 for a 2-state model; k=4 is the proposal's illustrative count. Both
    # are starting points for the formal sweep in item 4, not a conclusion reached
    # here.
    k, lam = 4, 50.0
    jm = fitted[(k, lam)]
    print(f"\n=== Interpretability check: k={k}, lambda={lam} (anchor point, not decided) ===")
    centers_df = pd.DataFrame(jm.centers_, columns=FEATURE_COLUMNS).round(3)
    print("State centers (standardized feature space):")
    print(centers_df.to_string())
    centers_df.to_csv(RESULTS_DIR / f"jm_k{k}_lambda{int(lam)}_centers.csv")
    print("\nTransition matrix (rows sum to 1, diagonal = self-persistence):")
    transmat = np.round(jm.transmat_, 4)
    print(transmat)
    pd.DataFrame(transmat).to_csv(RESULTS_DIR / f"jm_k{k}_lambda{int(lam)}_transmat.csv", index=False)

    labels = pd.Series(jm.labels_, index=X.index)
    report = interpretability_report(labels)
    print("\nKnown stress-window regime assignment:")
    for name, info in report.items():
        print(f"  {name}: {info}")
    with open(RESULTS_DIR / f"jm_k{k}_lambda{int(lam)}_interpretability.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
