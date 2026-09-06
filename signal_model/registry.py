"""Versioned, point-in-time-honest signal-model storage - Phase 4 design §5 step 2.

Mirrors `regime_detection/regime_db.py` deliberately (same append-only /
`superseded_by` pattern, same "one row per fitted model + a full stored output set +
a current_* view" shape), so the two model families are audited the same way and a
`todays_call_log` row can join cleanly against either.

- `signal_model_versions` - one row per fitted LightGBM model. A new version is
  created ONLY by an explicit `fit_and_register_signal.py` run (never on demand
  inside a prediction request - Phase 4 review decision 5). Stores the fit window,
  the target definition, the purge/embargo used, the feature list, and the LGBM
  params - everything needed to re-fit the exact model deterministically (fixed
  seeds throughout, established in Phase 3), so the model blob itself is not stored,
  matching regime_db's "store what downstream needs, not the whole object" stance.
- `signal_predictions` - one row per (trade_date, signal_model_version), append-only.
  Registering a version stores its out-of-sample predictions for every trading day
  from just after its training window through its coverage horizon; a later
  registration supersedes any overlapping non-superseded prediction, never mutates.
- `current_signal_predictions` - the latest non-superseded prediction per date.

Coverage horizon: a version registered as-of date C stores predictions through
`C + SIGNAL_COVERAGE_TD` trading days (or the last available feature row, whichever
comes first). Beyond that horizon the version is considered stale and
`decide_todays_call` reports `signal_model_stale` rather than reading a very old
model's guess. SIGNAL_COVERAGE_TD is anchored to the quarterly recalibration floor
(same cadence anchor as the regime side); the actual signal-model refit cadence is
the later Model Training Agent's decision, not fixed here.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

import pandas as pd

SIGNAL_COVERAGE_TD = 63  # ~one quarter of trading days; see module docstring

SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_model_versions (
    id INTEGER PRIMARY KEY,
    model_kind TEXT NOT NULL,
    target_horizon_days INTEGER NOT NULL,
    target_flat_bps REAL NOT NULL,
    fit_start_date TEXT NOT NULL,
    fit_end_date TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    purge_days INTEGER NOT NULL,
    embargo_days INTEGER NOT NULL,
    coverage_end_date TEXT NOT NULL,
    feature_columns TEXT NOT NULL,
    lgbm_params TEXT NOT NULL,
    best_iteration INTEGER,
    n_train_rows INTEGER NOT NULL,
    fitted_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS signal_predictions (
    id INTEGER PRIMARY KEY,
    trade_date TEXT NOT NULL,
    signal_model_version_id INTEGER NOT NULL REFERENCES signal_model_versions(id),
    pred_direction INTEGER NOT NULL,
    prob_down REAL NOT NULL,
    prob_flat REAL NOT NULL,
    prob_up REAL NOT NULL,
    predicted_at TEXT NOT NULL,
    superseded_by INTEGER REFERENCES signal_predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_signal_predictions_date ON signal_predictions(trade_date);
CREATE INDEX IF NOT EXISTS idx_signal_predictions_version ON signal_predictions(signal_model_version_id);

CREATE VIEW IF NOT EXISTS current_signal_predictions AS
    SELECT * FROM signal_predictions WHERE superseded_by IS NULL;
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def save_signal_model_version(
    conn: sqlite3.Connection,
    *,
    model_kind: str,
    target_horizon_days: int,
    target_flat_bps: float,
    fit_start_date: date,
    fit_end_date: date,
    as_of_date: date,
    purge_days: int,
    embargo_days: int,
    coverage_end_date: date,
    feature_columns: list[str],
    lgbm_params: dict,
    best_iteration: int | None,
    n_train_rows: int,
    notes: str | None = None,
) -> int:
    """Record a new signal-model version, marking any prior active version of the
    same (model_kind, target_horizon_days, target_flat_bps) as superseded - same
    supersede-on-key behaviour as regime_db.save_model_version."""
    conn.execute(
        "UPDATE signal_model_versions SET status = 'superseded' "
        "WHERE model_kind = ? AND target_horizon_days = ? AND target_flat_bps = ? AND status = 'active'",
        (model_kind, target_horizon_days, target_flat_bps),
    )
    cur = conn.execute(
        """
        INSERT INTO signal_model_versions
            (model_kind, target_horizon_days, target_flat_bps, fit_start_date, fit_end_date,
             as_of_date, purge_days, embargo_days, coverage_end_date, feature_columns,
             lgbm_params, best_iteration, n_train_rows, fitted_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_kind, target_horizon_days, target_flat_bps,
            fit_start_date.isoformat(), fit_end_date.isoformat(), as_of_date.isoformat(),
            purge_days, embargo_days, coverage_end_date.isoformat(),
            json.dumps(list(feature_columns)), json.dumps(lgbm_params, default=str),
            best_iteration, n_train_rows,
            datetime.now(timezone.utc).isoformat(), notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def load_signal_model_version(conn: sqlite3.Connection, version_id: int) -> dict:
    row = conn.execute(
        "SELECT * FROM signal_model_versions WHERE id = ?", (version_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no signal_model_version with id={version_id}")
    cols = [d[0] for d in conn.execute("SELECT * FROM signal_model_versions LIMIT 0").description]
    rec = dict(zip(cols, row))
    rec["feature_columns"] = json.loads(rec["feature_columns"])
    rec["lgbm_params"] = json.loads(rec["lgbm_params"])
    return rec


def get_active_signal_model_version(
    conn: sqlite3.Connection, model_kind: str, target_horizon_days: int, target_flat_bps: float
) -> dict | None:
    row = conn.execute(
        "SELECT id FROM signal_model_versions WHERE model_kind = ? AND target_horizon_days = ? "
        "AND target_flat_bps = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
        (model_kind, target_horizon_days, target_flat_bps),
    ).fetchone()
    return load_signal_model_version(conn, row[0]) if row else None


def save_signal_predictions(conn: sqlite3.Connection, version_id: int, preds: pd.DataFrame) -> int:
    """Insert a prediction set for this version, superseding any prior non-superseded
    prediction for the same dates. `preds` is indexed by trade_date with columns
    [pred_direction, prob_down, prob_flat, prob_up] - same append-only supersede
    logic as regime_db.save_regime_labels."""
    required = {"pred_direction", "prob_down", "prob_flat", "prob_up"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"preds is missing columns: {sorted(missing)}")

    predicted_at = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for trade_date, r in preds.iterrows():
        trade_date_str = pd.Timestamp(trade_date).date().isoformat()
        prior = conn.execute(
            "SELECT id FROM signal_predictions WHERE trade_date = ? AND superseded_by IS NULL",
            (trade_date_str,),
        ).fetchone()
        cur = conn.execute(
            "INSERT INTO signal_predictions "
            "(trade_date, signal_model_version_id, pred_direction, prob_down, prob_flat, prob_up, predicted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trade_date_str, version_id, int(r["pred_direction"]),
                float(r["prob_down"]), float(r["prob_flat"]), float(r["prob_up"]), predicted_at,
            ),
        )
        if prior:
            conn.execute(
                "UPDATE signal_predictions SET superseded_by = ? WHERE id = ?", (cur.lastrowid, prior[0])
            )
        inserted += 1
    conn.commit()
    return inserted


def load_signal_prediction(conn: sqlite3.Connection, as_of: date) -> dict | None:
    """The current (non-superseded) signal prediction for `as_of`, joined to its
    version's identity, or None if no version covers that date. This is the only
    read `decide_todays_call` needs - it never fits a model."""
    row = conn.execute(
        """
        SELECT p.trade_date, p.pred_direction, p.prob_down, p.prob_flat, p.prob_up,
               p.signal_model_version_id, v.as_of_date, v.fit_end_date, v.coverage_end_date
        FROM signal_predictions p
        JOIN signal_model_versions v ON v.id = p.signal_model_version_id
        WHERE p.trade_date = ? AND p.superseded_by IS NULL
        """,
        (pd.Timestamp(as_of).date().isoformat(),),
    ).fetchone()
    if row is None:
        return None
    keys = ["trade_date", "pred_direction", "prob_down", "prob_flat", "prob_up",
            "signal_model_version_id", "version_as_of_date", "version_fit_end_date",
            "version_coverage_end_date"]
    return dict(zip(keys, row))
