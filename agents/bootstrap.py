"""One-time privileged schema setup for the agent layer - Phase 5 design §4.3 / §6.

Run once against `data/regimeguard.db` with an UNSCOPED connection (agent connections
deny all DDL). Idempotent. Does three things:

1. Ensures every table exists (data, calendar, regime, signal, decision, audit).
2. Applies the additive migration that gives `todays_call_log` its `trace_id`,
   `prev_hash`, `row_hash` columns (Phase 4 created it without them).
3. Sets `PRAGMA journal_mode = WAL` on the database file so the four MCP server
   processes + the orchestrator can hold concurrent read connections with a single
   writer; agent connections then only ever *read* journal_mode.

`python -m agents.bootstrap` prints what it did and runs `verify_chain` on both
chained tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from data_agent.calendar_days import ensure_schema as ensure_calendar_schema
from data_agent.db import DB_PATH, get_connection
from regime_detection.regime_db import ensure_schema as ensure_regime_schema
from signal_model.registry import ensure_schema as ensure_signal_schema
from signal_model.todays_call import ensure_log_schema

from agents import audit

_TODAYS_CALL_LOG_ADDED_COLUMNS = (
    ("trace_id", "TEXT"),
    ("prev_hash", "TEXT"),
    ("row_hash", "TEXT"),
)


def _migrate_todays_call_log(conn: sqlite3.Connection) -> list[str]:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(todays_call_log)")}
    added = []
    for col, decl in _TODAYS_CALL_LOG_ADDED_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE todays_call_log ADD COLUMN {col} {decl}")
            added.append(col)
    conn.commit()
    return added


def bootstrap(db_path: Path | str = DB_PATH) -> dict:
    conn = get_connection(db_path) if db_path == DB_PATH else sqlite3.connect(db_path)
    try:
        # 1. every schema
        ensure_calendar_schema(conn)
        ensure_regime_schema(conn)
        ensure_signal_schema(conn)
        ensure_log_schema(conn)
        audit.ensure_schema(conn)
        conn.commit()

        # 2. additive migration
        added = _migrate_todays_call_log(conn)

        # 3. WAL
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    finally:
        conn.close()
    return {"journal_mode": mode, "todays_call_log_columns_added": added}


def main() -> None:
    result = bootstrap()
    print(f"journal_mode: {result['journal_mode']}")
    print(f"todays_call_log columns added: {result['todays_call_log_columns_added'] or '(none)'}")
    conn = sqlite3.connect(DB_PATH)
    try:
        for table in ("agent_call_log", "todays_call_log"):
            print(f"verify_chain({table}): {audit.verify_chain(conn, table)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
