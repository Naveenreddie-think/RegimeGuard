"""Versioned, point-in-time-honest regime label storage - recalibration design §2
(approved, see docs/phase2_working_notes.md).

Directly reuses the append-only/superseded_by pattern already established for
daily_bars (data_agent/db.py) rather than inventing a new one:

- `model_versions` — one row per fitted model (one per recalibration event),
  mirroring ingestion_runs' audit-trail role. Stores the fitted preprocessing
  pipeline (clipper bounds, scaler mean/scale) alongside the model's own
  hyperparameters, so a later drift check can reconstruct exactly what that version
  saw without needing to re-derive it.
- `regime_labels` — one row per (trade_date, model_version), append-only. A
  recalibration event adds a full new label set under a new model_version_id and
  marks the previous version's labels as superseded; it never mutates history. This
  is a necessity, not tidiness: the whole rolling-window stability investigation
  this session ran depends on being able to reconstruct "what did the model
  believe with only data through date X" after the fact - silently overwriting
  history would destroy exactly the evidence that investigation exists to examine.
- `current_regime_labels` — the latest non-superseded label per date, the "what do
  we believe now" convenience view, analogous to `current_bars`.

Full-history refit on every recalibration (not a partial/incremental relabel) is a
deliberate scoping decision appropriate to the current data horizon - every fit this
session ran in low single-digit minutes at 15 years of daily data - not a permanent
constant. Worth revisiting (e.g. bounded-window refits) purely on compute-cost
grounds if this system is ever still running decades from now.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY,
    model_kind TEXT NOT NULL,
    k INTEGER NOT NULL,
    jump_penalty REAL,
    fit_start_date TEXT NOT NULL,
    fit_end_date TEXT NOT NULL,
    fitted_at TEXT NOT NULL,
    clipper_lb TEXT NOT NULL,
    clipper_ub TEXT NOT NULL,
    scaler_mean TEXT NOT NULL,
    scaler_scale TEXT NOT NULL,
    feature_columns TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS regime_labels (
    id INTEGER PRIMARY KEY,
    trade_date TEXT NOT NULL,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    regime INTEGER NOT NULL,
    labeled_at TEXT NOT NULL,
    superseded_by INTEGER REFERENCES regime_labels(id)
);

CREATE INDEX IF NOT EXISTS idx_regime_labels_date ON regime_labels(trade_date);
CREATE INDEX IF NOT EXISTS idx_regime_labels_version ON regime_labels(model_version_id);

CREATE VIEW IF NOT EXISTS current_regime_labels AS
    SELECT * FROM regime_labels WHERE superseded_by IS NULL;
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def save_model_version(
    conn: sqlite3.Connection,
    model_kind: str,
    k: int,
    jump_penalty: float | None,
    fit_start_date: date,
    fit_end_date: date,
    clipper: DataClipperStd,
    scaler: StandardScalerPD,
    feature_columns: list[str],
    notes: str | None = None,
) -> int:
    """Record a new model version and mark any prior active version of the same
    (model_kind, k, jump_penalty) as superseded - mirrors how a daily_bars
    correction supersedes the row it replaces, not an UPDATE in place."""
    conn.execute(
        "UPDATE model_versions SET status = 'superseded' "
        "WHERE model_kind = ? AND k = ? AND (jump_penalty IS ? OR jump_penalty = ?) AND status = 'active'",
        (model_kind, k, jump_penalty, jump_penalty),
    )
    cur = conn.execute(
        """
        INSERT INTO model_versions
            (model_kind, k, jump_penalty, fit_start_date, fit_end_date, fitted_at,
             clipper_lb, clipper_ub, scaler_mean, scaler_scale, feature_columns, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_kind, k, jump_penalty, fit_start_date.isoformat(), fit_end_date.isoformat(),
            datetime.now(timezone.utc).isoformat(),
            json.dumps(clipper.lb.tolist()), json.dumps(clipper.ub.tolist()),
            json.dumps(scaler.scaler.mean_.tolist()), json.dumps(scaler.scaler.scale_.tolist()),
            json.dumps(list(feature_columns)), notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def load_model_version(conn: sqlite3.Connection, model_version_id: int) -> dict:
    row = conn.execute(
        "SELECT * FROM model_versions WHERE id = ?", (model_version_id,)
    ).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM model_versions LIMIT 0").description]
    rec = dict(zip(cols, row))
    rec["clipper_lb"] = np.array(json.loads(rec["clipper_lb"]))
    rec["clipper_ub"] = np.array(json.loads(rec["clipper_ub"]))
    rec["scaler_mean"] = np.array(json.loads(rec["scaler_mean"]))
    rec["scaler_scale"] = np.array(json.loads(rec["scaler_scale"]))
    rec["feature_columns"] = json.loads(rec["feature_columns"])
    return rec


def get_active_model_version(conn: sqlite3.Connection, model_kind: str, k: int, jump_penalty: float | None) -> dict | None:
    row = conn.execute(
        "SELECT id FROM model_versions WHERE model_kind = ? AND k = ? AND "
        "(jump_penalty IS ? OR jump_penalty = ?) AND status = 'active' ORDER BY id DESC LIMIT 1",
        (model_kind, k, jump_penalty, jump_penalty),
    ).fetchone()
    return load_model_version(conn, row[0]) if row else None


def save_regime_labels(conn: sqlite3.Connection, model_version_id: int, labels: pd.Series) -> int:
    """Insert a full label set for this model version, superseding any prior
    non-superseded label for the same dates (from an earlier model_version)."""
    labeled_at = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for trade_date, regime in labels.items():
        trade_date_str = pd.Timestamp(trade_date).date().isoformat()
        prior = conn.execute(
            "SELECT id FROM regime_labels WHERE trade_date = ? AND superseded_by IS NULL",
            (trade_date_str,),
        ).fetchone()
        cur = conn.execute(
            "INSERT INTO regime_labels (trade_date, model_version_id, regime, labeled_at) "
            "VALUES (?, ?, ?, ?)",
            (trade_date_str, model_version_id, int(regime), labeled_at),
        )
        new_id = cur.lastrowid
        if prior:
            conn.execute("UPDATE regime_labels SET superseded_by = ? WHERE id = ?", (new_id, prior[0]))
        inserted += 1
    conn.commit()
    return inserted
