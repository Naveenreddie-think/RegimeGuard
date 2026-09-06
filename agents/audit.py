"""Broker-written, hash-chained audit trail - Phase 5 design §3.

Two append-only, SHA-256-hash-chained tables share one mechanism (`chain_append` /
`verify_chain`):

- `agent_call_log` - one row per tool invocation, written by the broker (never by a
  tool), including denied calls. This is what makes "which agent called what, in what
  order, with what result" reconstructable.
- `todays_call_log` - the Phase 4 decision log, now also chained. Phase 4's
  `append_to_log` routes through `chain_append` so both the direct path and the
  orchestrator path produce identical, chained rows.

Chain: `row_hash = sha256(prev_hash || canonical_json(business fields))`, `prev_hash`
being the previous row's `row_hash` (genesis = 64 hex zeros). `verify_chain` walks a
table in id order and reports the first broken link. This is **tamper-evident, not
tamper-proof**: any edit, reorder or deletion breaks the chain from that point for
anyone who cannot also recompute every subsequent `row_hash`. No HMAC, no key
management - that matches the honest scope (design §3.3).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Callable

GENESIS_HASH = "0" * 64
_CHAINED_TABLES = frozenset({"agent_call_log", "todays_call_log"})

AGENT_CALL_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_call_log (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_call_id INTEGER REFERENCES agent_call_log(id),
    seq INTEGER NOT NULL,
    ts_start TEXT NOT NULL,
    ts_end TEXT NOT NULL,
    caller TEXT NOT NULL,
    callee_agent TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_summary_json TEXT,
    status TEXT NOT NULL,
    denial_reason TEXT,
    grant_digest TEXT NOT NULL,
    code_rev TEXT,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_call_log_trace ON agent_call_log(trace_id, seq);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(AGENT_CALL_LOG_SCHEMA)

# Business columns per chained table, in a fixed order - the exact set fed to the
# hash (everything the table stores except the autoincrement id and row_hash itself).
_BUSINESS_COLUMNS: dict[str, tuple[str, ...]] = {
    "agent_call_log": (
        "trace_id", "parent_call_id", "seq", "ts_start", "ts_end", "caller",
        "callee_agent", "tool_name", "args_json", "result_summary_json", "status",
        "denial_reason", "grant_digest", "code_rev", "prev_hash",
    ),
    "todays_call_log": (
        "as_of_date", "generated_at", "disposition", "reliability_tier", "actionable",
        "regime_pit_id", "regime_model_version_id", "signal_model_version_id",
        "directional_lean", "abstention_reasons", "tier1_drift_mean",
        "trading_days_since_fit", "record_json", "code_rev", "trace_id", "prev_hash",
    ),
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _row_hash(table: str, fields: dict[str, Any]) -> str:
    ordered = {c: fields.get(c) for c in _BUSINESS_COLUMNS[table]}
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def grant_digest(grant) -> str:
    """Stable digest of a Grant, so a later policy change is visible in the trail."""
    payload = {
        "db_read": sorted(grant.db_read),
        "db_write": sorted(grant.db_write),
        "fs_write_prefixes": list(grant.fs_write_prefixes),
        "net_hosts": sorted(grant.net_hosts),
        "may_call": sorted(a.value for a in grant.may_call),
        "reject_dates_within_embargo_of_last_bar": grant.reject_dates_within_embargo_of_last_bar,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _retry_on_locked(fn: Callable[[], Any], attempts: int = 5) -> Any:
    """Bounded exponential backoff on SQLITE_BUSY ('database is locked'). WAL's write
    lock is database-file-wide, so even role-partitioned writers can collide (design
    §4.3). PRAGMA busy_timeout handles most of it; this is the belt-and-braces layer."""
    delay = 0.05
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.8)


def _last_row_hash(conn: sqlite3.Connection, table: str) -> str:
    row = conn.execute(f"SELECT row_hash FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else GENESIS_HASH


def chain_append(conn: sqlite3.Connection, table: str, fields: dict[str, Any]) -> int:
    """Append one row to a hash-chained table. `fields` supplies every business
    column except `prev_hash` (added here) and `row_hash` (computed here). Returns
    the new row id. The INSERT is retried on SQLITE_BUSY."""
    if table not in _CHAINED_TABLES:
        raise ValueError(f"{table} is not a chained table")
    business = set(_BUSINESS_COLUMNS[table]) - {"prev_hash"}
    missing = business - set(fields)
    if missing:
        raise ValueError(f"chain_append({table}) missing fields: {sorted(missing)}")

    def _do() -> int:
        prev = _last_row_hash(conn, table)
        full = {**fields, "prev_hash": prev}
        full["row_hash"] = _row_hash(table, full)
        cols = list(_BUSINESS_COLUMNS[table]) + ["row_hash"]
        placeholders = ",".join("?" * len(cols))
        cur = conn.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            tuple(full.get(c) for c in cols),
        )
        conn.commit()
        return cur.lastrowid

    return _retry_on_locked(_do)


def verify_chain(conn: sqlite3.Connection, table: str) -> dict:
    """Walk `table` in id order; recompute each row_hash and check the prev_hash link.
    Returns {"ok": True, "n": N} or {"ok": False, ...first break...}."""
    if table not in _CHAINED_TABLES:
        raise ValueError(f"{table} is not a chained table")
    cols = _BUSINESS_COLUMNS[table] + ("row_hash",)
    rows = conn.execute(
        f"SELECT id,{','.join(cols)} FROM {table} ORDER BY id"
    ).fetchall()

    expected_prev = GENESIS_HASH
    for r in rows:
        rid = r[0]
        stored = dict(zip(cols, r[1:]))
        if stored["prev_hash"] != expected_prev:
            return {"ok": False, "table": table, "id": rid, "reason": "prev_hash mismatch",
                    "stored_prev": stored["prev_hash"], "expected_prev": expected_prev}
        recomputed = _row_hash(table, stored)
        if recomputed != stored["row_hash"]:
            return {"ok": False, "table": table, "id": rid, "reason": "row_hash mismatch",
                    "stored": stored["row_hash"], "recomputed": recomputed}
        expected_prev = stored["row_hash"]
    return {"ok": True, "table": table, "n": len(rows)}
