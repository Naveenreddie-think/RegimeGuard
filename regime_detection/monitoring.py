"""Two-tier drift monitoring - recalibration design §1 (approved).

Tier 1 (cheap, continuous): compares the live model_version's stored scaler stats
against a scaler freshly fit on current data - the same standardization-drift
metric already used diagnostically in standardization_drift.py, now operationalized
as a live check rather than a backtest. No refit required.

Tier 2 (expensive, confirmatory): only run when tier 1 fires. Shadow-fits the same
(model_kind, k, jump_penalty) on current full data and compares its labels against
the live model_version's own already-saved labels (from regime_labels - the
point-in-time-honest record, not a re-derivation) on their shared interior dates
(>90 trading days before the live version's fit_end_date, same threshold used
throughout this investigation). Reuses align_states() for cross-fit state
correspondence, same mechanism as every other check this session.

A recalibration is flagged for review only if tier 2 confirms tier 1 - avoids
acting on tier-1 noise alone, per the approved design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from sklearn.metrics import adjusted_rand_score

from regime_detection.features import build_feature_matrix
from regime_detection.jump_model_fit import prepare_fit_data
from regime_detection.regime_db import load_model_version
from regime_detection.rolling_window_stability import (
    INTERIOR_ARI_BAR,
    WARM_UP_EDGE_DAYS,
    align_states,
)

# TIER1_DRIFT_THRESHOLD is grounded in standardization_drift.py's real measurements
# (docs/phase2_working_notes.md, "standardization drift" entry), not a guess. Both of
# its reference points turned out to be failing configurations, not a bad/fine pair:
# the 2016-12-31 cutoff (drift 0.827) had the worst interior ARI in the whole rolling-
# window table (0.44), and the 2020-06-30 cutoff (drift 0.327) was less catastrophic
# but still failed the same 0.85 interior-ARI bar (ARI 0.56). There is no validated
# *passing* drift level from this metric to anchor a low end against, so the threshold
# is set below the milder of the two known-bad points - it should fire even on the
# less-severe failure case, not only the catastrophic one. Still a first cut, and
# still worth revisiting once real operational drift observations accumulate.
TIER1_DRIFT_THRESHOLD = 0.3  # mean per-date Euclidean distance, standardized space
TIER2_ARI_THRESHOLD = INTERIOR_ARI_BAR  # reuse the already-established bar


def _reconstruct_pipeline(model_version: dict) -> tuple[DataClipperStd, StandardScalerPD]:
    """Rebuild a (non-refit) clipper/scaler pair from a stored model_version's
    saved bounds/stats, so live data can be transformed exactly as that version's
    training data was, without needing the original raw data again."""
    clipper = DataClipperStd(mul=3.0)
    clipper.lb = model_version["clipper_lb"]
    clipper.ub = model_version["clipper_ub"]

    scaler = StandardScalerPD()
    scaler.scaler = scaler.init_scaler()
    scaler.scaler.mean_ = model_version["scaler_mean"]
    scaler.scaler.scale_ = model_version["scaler_scale"]
    scaler.scaler.n_features_in_ = len(model_version["scaler_mean"])
    return clipper, scaler


def compute_tier1_drift(conn, model_version: dict) -> dict:
    """Fit a fresh clip+scale pipeline on all current data and compare its stats
    against the stored model_version's - no refit of the actual regime model."""
    df = build_feature_matrix(conn)
    X_raw_current, _ = prepare_fit_data(df)
    feature_columns = model_version["feature_columns"]

    fresh_clipper = DataClipperStd(mul=3.0)
    fresh_scaler = StandardScalerPD()
    fresh_scaler.fit_transform(fresh_clipper.fit_transform(X_raw_current[feature_columns]))

    stored_mean = model_version["scaler_mean"]
    stored_scale = model_version["scaler_scale"]
    fresh_mean = fresh_scaler.scaler.mean_
    fresh_scale = fresh_scaler.scaler.scale_

    # Same per-date Euclidean-distance metric as standardization_drift.py, applied
    # to the most recent WARM_UP_EDGE_DAYS of current data (the freshest evidence
    # of whether "extreme" still means what the live model thinks it means).
    stored_clipper, stored_scaler = _reconstruct_pipeline(model_version)
    recent_raw = X_raw_current[feature_columns].iloc[-WARM_UP_EDGE_DAYS:]
    std_under_stored = stored_scaler.transform(stored_clipper.transform(recent_raw))
    std_under_fresh = fresh_scaler.transform(fresh_clipper.transform(recent_raw))
    per_date_distance = np.sqrt(((std_under_fresh - std_under_stored) ** 2).sum(axis=1))

    return {
        "mean_drift": float(per_date_distance.mean()),
        "max_drift": float(per_date_distance.max()),
        "scale_ratio": dict(zip(feature_columns, (fresh_scale / stored_scale).round(4).tolist())),
        "fires": bool(per_date_distance.mean() >= TIER1_DRIFT_THRESHOLD),
    }


def compute_tier2_shadow_fit(conn, model_version: dict) -> dict:
    """Shadow-fit the same config on current data, compare against the live
    model_version's OWN saved labels (not a re-derivation) on shared interior dates."""
    df = build_feature_matrix(conn)
    X_raw_current, ret_ser_current = prepare_fit_data(df)
    feature_columns = model_version["feature_columns"]

    live_labels = pd.read_sql_query(
        "SELECT trade_date, regime FROM regime_labels WHERE model_version_id = ? AND superseded_by IS NULL",
        conn, params=(model_version["id"],), parse_dates=["trade_date"],
    ).set_index("trade_date")["regime"]

    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    X_current = scaler.fit_transform(clipper.fit_transform(X_raw_current[feature_columns]))

    if model_version["model_kind"] == "jm":
        shadow = JumpModel(
            n_components=model_version["k"], jump_penalty=model_version["jump_penalty"],
            cont=False, random_state=0,
        )
        shadow.fit(X_current, ret_ser=ret_ser_current, sort_by="cumret")
        shadow_labels = pd.Series(shadow.labels_, index=X_current.index)
        shadow_centers = shadow.centers_
    else:
        raise NotImplementedError("HMM shadow-fit not wired up yet - JM only for this first pass")

    shared_dates = live_labels.index.intersection(shadow_labels.index)
    fit_end = pd.Timestamp(model_version["fit_end_date"])
    interior_dates = shared_dates[shared_dates < fit_end - pd.Timedelta(days=int(WARM_UP_EDGE_DAYS * 1.4))]
    # ~1.4x calendar-day buffer for WARM_UP_EDGE_DAYS trading days, consistent with
    # how the interior/edge split is approximated elsewhere when working in
    # calendar-day terms rather than a precomputed trading-day index slice.

    if len(interior_dates) < 2:
        return {"interior_ari": None, "confirmed": None, "note": "not enough interior overlap to evaluate"}

    interior_ari = adjusted_rand_score(
        live_labels.loc[interior_dates].to_numpy(), shadow_labels.loc[interior_dates].to_numpy()
    )
    return {
        "interior_ari": float(interior_ari),
        "n_interior_dates": len(interior_dates),
        "confirmed": bool(interior_ari < TIER2_ARI_THRESHOLD),
    }


def check_recalibration_trigger(conn, model_version_id: int) -> dict:
    model_version = load_model_version(conn, model_version_id)
    tier1 = compute_tier1_drift(conn, model_version)
    result = {"model_version_id": model_version_id, "tier1": tier1}
    if tier1["fires"]:
        tier2 = compute_tier2_shadow_fit(conn, model_version)
        result["tier2"] = tier2
        result["recalibration_recommended"] = bool(tier2.get("confirmed"))
    else:
        result["tier2"] = None
        result["recalibration_recommended"] = False
    return result
