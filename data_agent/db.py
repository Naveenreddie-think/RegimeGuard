"""Point-in-time SQLite store for RegimeGuard.

Design (see docs/Project_Proposal_RegimeGuard.md and the compiled Phase 1 plan):
daily_bars is append-only — a correction inserts a new row and marks the old one
superseded via `superseded_by`, it is never UPDATEd in place. `current_bars` selects
the latest non-superseded row per (instrument, date). This is what makes an
as-of-date-T query unambiguous and reproducible for later purged/embargoed
walk-forward validation.

`calendar_days` (holiday/outage/unexplained-gap classification) has its own schema in
calendar_days.py, added via that module's ensure_schema().

`ingestion_runs` is the shared audit trail for both the daily_bars fetch pipeline and
calendar_days rebuilds — start_ingestion_run()/finish_ingestion_run() below are used
by both, so any operation that (re)writes derived data leaves a record of when it ran
and what changed, per the proposal's audit-trail principle.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "regimeguard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    source TEXT NOT NULL,
    date_range_requested TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS daily_bars (
    id INTEGER PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    trade_date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    raw_file_ref TEXT NOT NULL,
    ingestion_run_id INTEGER REFERENCES ingestion_runs(id),
    is_corrected INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER REFERENCES daily_bars(id)
);

CREATE INDEX IF NOT EXISTS idx_daily_bars_instrument_date
    ON daily_bars(instrument_id, trade_date);

CREATE VIEW IF NOT EXISTS current_bars AS
    SELECT * FROM daily_bars WHERE superseded_by IS NULL;
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def get_or_create_instrument(
    conn: sqlite3.Connection, symbol: str, display_name: str, source: str
) -> int:
    row = conn.execute(
        "SELECT id FROM instruments WHERE symbol = ?", (symbol,)
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO instruments (symbol, display_name, source) VALUES (?, ?, ?)",
        (symbol, display_name, source),
    )
    conn.commit()
    return cur.lastrowid


def start_ingestion_run(conn: sqlite3.Connection, source: str, start: date, end: date) -> int:
    cur = conn.execute(
        "INSERT INTO ingestion_runs (started_at, source, date_range_requested, status) "
        "VALUES (?, ?, ?, 'running')",
        (datetime.now(timezone.utc).isoformat(), source, f"{start.isoformat()}..{end.isoformat()}"),
    )
    conn.commit()
    return cur.lastrowid


def finish_ingestion_run(conn: sqlite3.Connection, run_id: int, status: str, notes: str) -> None:
    conn.execute(
        "UPDATE ingestion_runs SET finished_at = ?, status = ?, notes = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), status, notes, run_id),
    )
    conn.commit()


def insert_bar(
    conn: sqlite3.Connection,
    instrument_id: int,
    trade_date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float | None,
    source: str,
    raw_file_ref: str,
    ingestion_run_id: int,
) -> bool:
    """Insert one daily bar, skipping it if a non-superseded row for this
    (instrument, date) already exists. Shared by every ingestion path
    (fetch_niftyindices.py, the VIX manual loader) so "already loaded" means
    the same thing everywhere and every bar's provenance is recorded the same
    way, regardless of how it got here. Returns True if inserted, False if skipped.
    """
    exists = conn.execute(
        "SELECT 1 FROM daily_bars "
        "WHERE instrument_id = ? AND trade_date = ? AND superseded_by IS NULL",
        (instrument_id, trade_date),
    ).fetchone()
    if exists:
        return False
    conn.execute(
        """
        INSERT INTO daily_bars
            (instrument_id, trade_date, open, high, low, close, volume,
             source, ingested_at, raw_file_ref, ingestion_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            instrument_id, trade_date, open_, high, low, close, volume,
            source, datetime.now(timezone.utc).isoformat(), raw_file_ref, ingestion_run_id,
        ),
    )
    return True
