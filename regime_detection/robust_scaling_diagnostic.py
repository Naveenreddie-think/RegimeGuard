"""Quick diagnostic: does a robust (median/IQR) scaler shrink the interior-ARI
failures at the pairs touching/following COVID, compared to StandardScaler?

Scoped narrowly per direction: swap only the standardization step (mean/std ->
median/IQR via sklearn's RobustScaler, default quantile_range=(25,75) = true IQR).
Clipping (DataClipperStd, 3-sigma winsorization) is left unchanged - it wasn't what
was named, and changing two things at once would muddy a "quick, cheap" test.

Tested on JM k=3 and HMM k=3 (the two best-characterized, most-analyzed configs so
far) across all 4 consecutive pairs from the chain already established - not the
full 3k x 2model grid, keeping this proportionate to "quick diagnostic".

RobustScalerPD below matches StandardScalerPD's interface (.scaler, .transform())
exactly, so align_states() and the rest of the existing comparison logic work
unchanged - no forking of the alignment mechanism for this test.

Expectation stated before running, per direction: not expected to fully resolve the
COVID-adjacent failures (a robust scaler neutralizes a few extreme outlier days, but
COVID looks like a sustained shift in the whole distribution, which a location/scale
change of this kind doesn't fix) - reporting the actual before/after regardless of
which way it comes out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

from data_agent.db import get_connection
from regime_detection.features import build_feature_matrix
from regime_detection.hmm_fit import N_ITER, TOL
from regime_detection.jump_model_fit import KNOWN_STRESS_WINDOWS, prepare_fit_data
from regime_detection.rolling_window_stability import (
    EDGE_ARI_BAR,
    INTERIOR_ARI_BAR,
    WARM_UP_EDGE_DAYS,
    align_states,
    dominant_regime,
)
from hmmlearn.hmm import GaussianHMM

JM_LAMBDA = 50.0
K = 3
PAIRS_WITH_WINDOWS = [
    ("2016-12-31", "2018-12-31", "2016 demonetization"),
    ("2018-12-31", "2020-06-30", "2018 IL&FS stress"),
    ("2020-06-30", "2022-12-31", "2020 COVID crash"),
    ("2022-12-31", None, "2022 rate-hike volatility"),
]


class RobustScalerPD:
    """Matches StandardScalerPD's interface (.scaler, .fit_transform, .transform)
    so align_states() and the rest of the comparison logic work unchanged."""

    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.scaler.fit_transform(X)
        return pd.DataFrame(arr, index=X.index, columns=X.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.scaler.transform(X)
        return pd.DataFrame(arr, index=X.index, columns=X.columns)


def slice_data(X, cutoff):
    return X if cutoff is None else X.loc[:cutoff]


def fit_jm_robust(X_raw, ret_ser, k):
    clipper = DataClipperStd(mul=3.0)
    scaler = RobustScalerPD()
    X = scaler.fit_transform(clipper.fit_transform(X_raw))
    jm = JumpModel(n_components=k, jump_penalty=JM_LAMBDA, cont=False, random_state=0)
    jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
    return jm, scaler, clipper


def fit_hmm_robust(X_raw, k, n_restarts=10):
    clipper = DataClipperStd(mul=3.0)
    scaler = RobustScalerPD()
    X = scaler.fit_transform(clipper.fit_transform(X_raw))
    candidates = []
    for seed in range(n_restarts):
        m = GaussianHMM(n_components=k, covariance_type="diag", random_state=seed, n_iter=N_ITER, tol=TOL)
        m.fit(X.to_numpy())
        if m.monitor_.converged:
            candidates.append((m.score(X.to_numpy()), m))
    candidates.sort(key=lambda p: p[0], reverse=True)
    return candidates[0][1], scaler, clipper


def evaluate_pair_robust(X_raw_full, ret_ser_full, cutoff_a, cutoff_b, k, model_kind):
    X_a, X_b = slice_data(X_raw_full, cutoff_a), slice_data(X_raw_full, cutoff_b)
    ret_a = slice_data(ret_ser_full.to_frame("r"), cutoff_a)["r"]
    ret_b = slice_data(ret_ser_full.to_frame("r"), cutoff_b)["r"]

    if model_kind == "jm":
        model_a, scaler_a, _ = fit_jm_robust(X_a, ret_a, k)
        model_b, scaler_b, _ = fit_jm_robust(X_b, ret_b, k)
        centers_a, centers_b = model_a.centers_, model_b.centers_
        labels_a = pd.Series(model_a.labels_, index=X_a.index)
        labels_b_full = pd.Series(model_b.labels_, index=X_b.index)
    else:
        model_a, scaler_a, clipper_a = fit_hmm_robust(X_a, k)
        model_b, scaler_b, clipper_b = fit_hmm_robust(X_b, k)
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

    state_map = align_states(centers_a, scaler_a, centers_b, scaler_b)
    return interior_ari, edge_ari, state_map


def main():
    conn = get_connection()
    try:
        df = build_feature_matrix(conn)
    finally:
        conn.close()
    X_raw_full, ret_ser_full = prepare_fit_data(df)

    print(f"{'Model':6} {'Pair':30} {'StdScaler interior':>19} {'Robust interior':>16} {'StdScaler edge':>15} {'Robust edge':>12}")

    # StandardScaler baseline numbers, taken directly from the already-run
    # consecutive-pair check (rolling_window_stability_consecutive.py results),
    # reproduced here for the JM k=3 / HMM k=3 rows only, for a direct side-by-side.
    baseline = {
        ("jm", "2016-12-31", "2018-12-31"): (0.9827698807944847, 1.0),
        ("jm", "2018-12-31", "2020-06-30"): (0.9526568067578349, 0.36789023049588776),
        ("jm", "2020-06-30", "2022-12-31"): (0.5483723457355558, -0.08330518444317613),
        ("jm", "2022-12-31", None): (0.9686515848216394, 0.35591991166928477),
        ("hmm", "2016-12-31", "2018-12-31"): (0.8707165858691978, 0.8277817641679076),
        ("hmm", "2018-12-31", "2020-06-30"): (0.891212322923558, 0.9288577321828247),
        ("hmm", "2020-06-30", "2022-12-31"): (0.8231865474663718, 0.17891825162578973),
        ("hmm", "2022-12-31", None): (0.6482725926526567, 0.7624267319549316),
    }

    results = []
    for model_kind in ["jm", "hmm"]:
        for cutoff_a, cutoff_b, window_name in PAIRS_WITH_WINDOWS:
            interior_ari, edge_ari, _ = evaluate_pair_robust(X_raw_full, ret_ser_full, cutoff_a, cutoff_b, K, model_kind)
            base_interior, base_edge = baseline[(model_kind, cutoff_a, cutoff_b)]
            pair_label = f"{cutoff_a}->{cutoff_b or 'full'}"
            print(f"{model_kind.upper():6} {pair_label:30} {base_interior:>10.4f} -> {interior_ari:>6.4f}  "
                  f"{base_edge:>13.4f} -> {edge_ari:>6.4f}")
            results.append({
                "model": model_kind, "pair": pair_label, "window": window_name,
                "std_interior_ari": base_interior, "robust_interior_ari": interior_ari,
                "std_edge_ari": base_edge, "robust_edge_ari": edge_ari,
                "interior_pass_std": base_interior >= INTERIOR_ARI_BAR,
                "interior_pass_robust": interior_ari >= INTERIOR_ARI_BAR if interior_ari is not None else None,
                "edge_pass_std": base_edge >= EDGE_ARI_BAR,
                "edge_pass_robust": edge_ari >= EDGE_ARI_BAR if edge_ari is not None else None,
            })

    print("\n=== Pass/fail flip summary ===")
    for r in results:
        flips = []
        if r["interior_pass_std"] != r["interior_pass_robust"]:
            flips.append(f"interior {r['interior_pass_std']}->{r['interior_pass_robust']}")
        if r["edge_pass_std"] != r["edge_pass_robust"]:
            flips.append(f"edge {r['edge_pass_std']}->{r['edge_pass_robust']}")
        if flips:
            print(f"  {r['model'].upper()} {r['pair']}: {', '.join(flips)}")
    if not any((r["interior_pass_std"] != r["interior_pass_robust"]) or (r["edge_pass_std"] != r["edge_pass_robust"]) for r in results):
        print("  No pass/fail flips in either direction, for any tested pair.")


if __name__ == "__main__":
    main()
