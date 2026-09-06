"""Point-in-time regime label generation via quarterly recalibration - the required
Phase 3 follow-up (see docs/phase3_working_notes.md). Simulates what a live system,
recalibrating on Phase 2's own validated quarterly floor, would actually have
classified at each historical date - not the hindsight-informed final/best-available
labels used in the first Phase 3 pass (model_version 1).

Reviewed design (docs/phase3_working_notes.md, "point-in-time label generation -
reviewed design" entry) - two points confirmed explicitly, not left implicit:

1. Point-in-time-label recovery is scoped to ONLY this walk's model_version_ids.
   model_version 1 (the standalone hindsight fit) is deliberately excluded - its
   regime_labels rows have smaller ids than this walk's for any overlapping date, so
   an unscoped "smallest id per date" query would silently pick up HINDSIGHT labels
   for a large chunk of history instead of point-in-time ones. See
   `load_point_in_time_labels` below.

2. Predicting forward into the gap before the next cutoff transforms the new raw
   feature data using ONLY the already-fitted clipper/scaler's .transform() - never
   .fit_transform(). The clipper/scaler objects reused below are literally the same
   ones just fit in step 1 of the same iteration; nothing is refit on post-cutoff
   data. This is exactly the leakage point the original standardization-drift
   finding (Phase 2) came from.

At each quarterly cutoff:
1. Full-history refit (JM k=3/jump_penalty=50.0) through that cutoff - identical
   logic to fit_and_register.py. Registered as a new model_version; its own
   labels_ saved for every date through the cutoff.
2. Predict-forward for the gap strictly between this cutoff and the next one
   (exclusive of the next cutoff itself - that date gets its own, more direct
   fit-based label from the following iteration, not a stale predict-forward guess).
   Uses the SAME fitted model's already-fitted, FIXED centers_/jump_penalty_mx (not
   refit) and its own clipper/scaler's .transform() only. Labels are saved under the
   SAME model_version_id, since they're that model's own out-of-sample call.

JumpModel.predict() runs a full Viterbi decode over whatever sequence it's given,
using fixed (already-fitted) parameters - confirmed directly via source inspection.
Predicting a whole gap at once means an early date's label can be informed by later
dates within that SAME gap, but never by anything from a later quarter or a later
recalibration - a real, named, reviewed and approved compromise (batch decode within
one live decision window), not a true day-by-day causal filter.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.regime_db import ensure_schema, save_model_version, save_regime_labels
from regime_detection.rolling_window_stability import WARM_UP_EDGE_DAYS

K = 3
JUMP_PENALTY = 50.0


def generate_quarterly_cutoffs(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Nearest trading day <= each calendar quarter-end, from the first quarter-end
    after `dates` starts through the last quarter-end on or before `dates` ends."""
    dates = pd.DatetimeIndex(sorted(dates))
    quarter_ends = pd.date_range(start=dates.min(), end=dates.max(), freq="QE")

    cutoffs = []
    for q in quarter_ends:
        candidates = dates[dates <= q]
        if len(candidates) == 0:
            continue
        cutoffs.append(candidates[-1])
    return sorted(set(cutoffs))


def run_quarterly_walk(conn: sqlite3.Connection, k: int = K, jump_penalty: float = JUMP_PENALTY) -> list[int]:
    """Runs the fit-then-predict-forward walk described in the module docstring.
    Returns the list of model_version_ids created, in chronological order - callers
    MUST use this list (not a blanket query) to scope point-in-time-label recovery."""
    ensure_schema(conn)

    df = build_feature_matrix(conn)
    fit_df = df.loc[~df["warm_up"]].copy()
    X_raw_full = fit_df[FEATURE_COLUMNS]
    ret_ser_full = fit_df["ret_5"]
    dates = X_raw_full.index

    cutoffs = generate_quarterly_cutoffs(dates)
    version_ids: list[int] = []

    for i, cutoff in enumerate(cutoffs):
        X_raw = X_raw_full.loc[:cutoff]
        ret_ser = ret_ser_full.loc[:cutoff]

        clipper = DataClipperStd(mul=3.0)
        scaler = StandardScalerPD()
        X = scaler.fit_transform(clipper.fit_transform(X_raw))  # fit ONLY on data through this cutoff

        jm = JumpModel(n_components=k, jump_penalty=jump_penalty, cont=False, random_state=0)
        jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
        labels_fit = pd.Series(jm.labels_, index=X_raw.index)

        version_id = save_model_version(
            conn, model_kind="jm", k=k, jump_penalty=jump_penalty,
            fit_start_date=X_raw.index[0].date(), fit_end_date=X_raw.index[-1].date(),
            clipper=clipper, scaler=scaler, feature_columns=FEATURE_COLUMNS,
            notes=f"Phase 3 point-in-time quarterly walk: full-history refit at cutoff {cutoff.date()}",
        )
        save_regime_labels(conn, version_id, labels_fit)
        version_ids.append(version_id)

        next_cutoff = cutoffs[i + 1] if i + 1 < len(cutoffs) else None
        predict_end = next_cutoff if next_cutoff is not None else dates[-1]
        gap_mask = (dates > cutoff) & (dates <= predict_end)
        if next_cutoff is not None:
            gap_mask &= dates < next_cutoff  # next_cutoff's own date gets the NEXT iteration's fit-based label
        gap_dates = dates[gap_mask]
        if len(gap_dates) == 0:
            continue

        X_raw_predict_window = X_raw_full.loc[:predict_end]
        # .transform() only, on the SAME clipper/scaler fit above - never refit on
        # post-cutoff data. Confirmed explicitly per the reviewed design.
        X_predict_window = scaler.transform(clipper.transform(X_raw_predict_window))
        pred_labels_full = pd.Series(jm.predict(X_predict_window), index=X_raw_predict_window.index)
        pred_labels_gap = pred_labels_full.loc[gap_dates]

        save_regime_labels(conn, version_id, pred_labels_gap)

    return version_ids


def load_point_in_time_labels(conn: sqlite3.Connection, quarterly_version_ids: list[int]) -> pd.Series:
    """The point-in-time-honest label per date: the FIRST label ever registered for
    that date, among ONLY the given model_version_ids. Deliberately excludes
    model_version 1 (the standalone hindsight fit) and any other model_version not
    in `quarterly_version_ids` - an unscoped query would pick up hindsight labels
    for most of history instead (see module docstring, point 1).

    Uses MIN(id) as "first registered" - acceptable because regime_labels.id is an
    autoincrement primary key and this walk always inserts in chronological
    registration order. This is a real assumption, stated explicitly: if regime_labels
    were ever bulk-backfilled or reordered, this would need an explicit provenance
    column instead of relying on insertion order."""
    if not quarterly_version_ids:
        raise ValueError("quarterly_version_ids must not be empty")

    placeholders = ",".join("?" * len(quarterly_version_ids))
    query = f"""
        SELECT trade_date, regime FROM regime_labels
        WHERE model_version_id IN ({placeholders})
          AND id IN (
              SELECT MIN(id) FROM regime_labels
              WHERE model_version_id IN ({placeholders})
              GROUP BY trade_date
          )
        ORDER BY trade_date
    """
    params = list(quarterly_version_ids) * 2
    result = pd.read_sql_query(query, conn, params=params, parse_dates=["trade_date"])
    return result.set_index("trade_date")["regime"]


def point_in_time_regime_label(conn: sqlite3.Connection, model_version: dict, as_of) -> dict:
    """The point-in-time regime label for `as_of` under a single registered
    model_version - the fit-fixed predict-forward step from `run_quarterly_walk`,
    factored out so `decide_todays_call` can run it against whatever version is
    active for an arbitrary date.

    The JM's fitted `centers_` / `jump_penalty_mx` are not persisted in
    `model_versions`, so this deterministically RE-FITS the same config through the
    version's own `fit_end_date` (random_state=0, ~1s) to recover them - a
    reconstruction of an already-registered version, not a new model. It asserts the
    reconstructed clip+scale stats match the version's stored ones, and (where a
    stored label exists for `as_of`) that the reconstructed label matches it, same
    verify-don't-assume discipline as Phase 3's alignment check.

    Then, using ONLY that fit's fixed parameters and its own scaler's `.transform()`,
    it decodes the label sequence through `as_of` (batch Viterbi, the same reviewed
    compromise as the quarterly walk: an early date in the gap can be informed by a
    later date in the same gap, never by a later recalibration).

    Returns the current regime, its trailing run length in trading days, and the
    M6 out-of-distribution check inputs (distance from `as_of`'s standardized vector
    to the nearest state centroid, vs. the max such distance over the fit's own
    interior dates).
    """
    df = build_feature_matrix(conn)
    fit_df = df.loc[~df["warm_up"]]
    X_raw_full = fit_df[FEATURE_COLUMNS]
    ret_ser_full = fit_df["ret_5"]

    as_of = X_raw_full.index[-1] if as_of is None else pd.Timestamp(as_of)
    available = X_raw_full.index[X_raw_full.index <= as_of]
    if len(available) == 0:
        raise ValueError(f"as_of {as_of.date()} precedes the first feature row {X_raw_full.index[0].date()}")
    as_of = available[-1]

    fit_end = pd.Timestamp(model_version["fit_end_date"])
    k = int(model_version["k"])
    jump_penalty = float(model_version["jump_penalty"])

    X_raw_fit = X_raw_full.loc[:fit_end]
    ret_fit = ret_ser_full.loc[:fit_end]

    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    X_fit = scaler.fit_transform(clipper.fit_transform(X_raw_fit))

    stored_mean = np.asarray(model_version["scaler_mean"], dtype=float)
    if not np.allclose(scaler.scaler.mean_, stored_mean, atol=1e-8):
        raise AssertionError(
            f"reconstructed scaler stats differ from model_version {model_version['id']}'s stored stats"
        )

    jm = JumpModel(n_components=k, jump_penalty=jump_penalty, cont=False, random_state=0)
    jm.fit(X_fit, ret_ser=ret_fit, sort_by="cumret")

    X_thru = scaler.transform(clipper.transform(X_raw_full.loc[:as_of]))
    labels = pd.Series(jm.predict(X_thru), index=X_thru.index)
    current = int(labels.loc[as_of])

    td_since_fit = int(((X_raw_full.index > fit_end) & (X_raw_full.index <= as_of)).sum())

    values = labels.to_numpy()
    run_length = 1
    for i in range(len(values) - 2, -1, -1):
        if values[i] == values[-1]:
            run_length += 1
        else:
            break

    centers = np.asarray(jm.centers_, dtype=float)
    z_as_of = X_thru.loc[as_of].to_numpy(dtype=float)
    min_dist_to_centroid = float(np.min(np.linalg.norm(centers - z_as_of, axis=1)))

    interior_idx = X_fit.index[:-WARM_UP_EDGE_DAYS] if len(X_fit) > WARM_UP_EDGE_DAYS else X_fit.index
    z_interior = X_fit.loc[interior_idx].to_numpy(dtype=float)
    interior_min_dists = np.min(
        np.linalg.norm(z_interior[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    ood_threshold = float(interior_min_dists.max())

    stored = conn.execute(
        "SELECT regime FROM regime_labels WHERE trade_date = ? AND model_version_id = ? "
        "AND superseded_by IS NULL",
        (as_of.date().isoformat(), model_version["id"]),
    ).fetchone()
    if stored is not None and int(stored[0]) != current:
        raise AssertionError(
            f"reconstructed PIT label {current} != stored label {stored[0]} for {as_of.date()} "
            f"(model_version {model_version['id']})"
        )

    return {
        "regime": current,
        "run_length_td": run_length,
        "as_of": as_of.date().isoformat(),
        "td_since_fit": td_since_fit,
        "source": "in_sample" if as_of <= fit_end else "predict_forward",
        "n_fit_rows": len(X_raw_fit),
        "min_dist_to_centroid": min_dist_to_centroid,
        "ood_threshold": ood_threshold,
        "is_ood": bool(min_dist_to_centroid > ood_threshold),
    }
