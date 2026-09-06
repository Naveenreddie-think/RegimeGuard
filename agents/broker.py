"""The enforcement layer - Phase 5 design §2.

Nothing here trusts an agent to behave. Each mechanism reads the policy from
`agents.capabilities` and blocks disallowed actions regardless of the calling code:

- `scoped_connection(agent)` - a real sqlite3 connection with `set_authorizer` wired
  from that agent's `Grant`. SQLite invokes the authorizer while *compiling* every
  statement; a denied table read/write fails before execution, however the SQL was
  built. DDL is denied for every agent (schema is a bootstrap-only privilege).
- `restrict_network(hosts)` - blocks outbound sockets to any host not in `hosts`
  (empty => all blocked). Installed permanently by non-Data MCP servers at startup.
- `ScopedFS` - rejects file writes outside an agent's `fs_write_prefixes`.
- `brokered_call(...)` - the one path a tool is invoked through: checks the call is
  permitted by `CALL_GRAPH`, applies Validation's no-recent-dates rule, runs the tool
  with a scoped connection, and writes the `agent_call_log` row (including on denial),
  so the audit trail is produced by the enforcer, not the enforced.
"""

from __future__ import annotations

import os
import re
import socket
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from data_agent.db import DB_PATH as _DEFAULT_DB_PATH


def db_path() -> str:
    """The DB every scoped connection opens. `REGIMEGUARD_DB` overrides the default
    (`data/regimeguard.db`) - used to point the whole agent layer at an isolated
    copy for tests without touching the real store."""
    return os.environ.get("REGIMEGUARD_DB") or str(_DEFAULT_DB_PATH)

from agents import audit
from agents.capabilities import (
    ALLOWED_PRAGMAS,
    BUSY_TIMEOUT_MS,
    CALL_GRAPH,
    EMBARGO_DAYS,
    GRANTS,
    AgentName,
)

_ALWAYS_READABLE = frozenset({
    "sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_temp_schema",
    "current_bars", "current_regime_labels", "current_signal_predictions",
})
_DDL_ACTIONS = frozenset({
    sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX, sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_REINDEX, sqlite3.SQLITE_ANALYZE, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


class ScopeError(PermissionError):
    """Raised by the broker for a call/date-range denial (not a raw SQLite denial)."""


def _is_authorizer_denial(exc: sqlite3.DatabaseError) -> bool:
    """SQLite phrases an authorizer DENY as 'not authorized' for writes/DDL and
    'access to <t>.<c> is prohibited' for reads - accept either."""
    m = str(exc).lower()
    return "not authorized" in m or "prohibited" in m


# --------------------------------------------------------------------------- DB

def _make_authorizer(agent: AgentName) -> Callable:
    grant = GRANTS[agent]

    def authorizer(action, arg1, arg2, db_name, trigger_or_view):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT,
                      sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = arg1
            if table in _ALWAYS_READABLE or table in grant.db_read:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            return sqlite3.SQLITE_OK if arg1 in grant.db_write else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            name = (arg1 or "").lower()
            if name not in ALLOWED_PRAGMAS:
                return sqlite3.SQLITE_DENY
            if name == "journal_mode" and arg2 is not None:
                return sqlite3.SQLITE_DENY  # reading ok, changing it is bootstrap-only
            return sqlite3.SQLITE_OK
        if action in _DDL_ACTIONS:
            return sqlite3.SQLITE_DENY  # schema is a bootstrap-only privilege, for everyone
        return sqlite3.SQLITE_DENY  # fail closed on anything unrecognised

    return authorizer


def scoped_connection(agent: AgentName, db_file: Path | str | None = None) -> sqlite3.Connection:
    """A sqlite3 connection scoped to `agent`'s Grant. The authorizer is installed
    before the handle is returned, and this is the only DB handle a tool receives -
    a tool cannot obtain an unscoped one."""
    conn = sqlite3.connect(db_file or db_path())
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.set_authorizer(_make_authorizer(agent))
    return conn


@contextmanager
def scoped_db(agent: AgentName, db_file: Path | str | None = None):
    conn = scoped_connection(agent, db_file)
    try:
        yield conn
    finally:
        conn.set_authorizer(None)
        conn.close()


# ----------------------------------------------------------------------- network

_real_create_connection = socket.create_connection
_real_socket_connect = socket.socket.connect


def _host_of(address) -> str | None:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return None


@contextmanager
def restrict_network(allowed_hosts: frozenset[str]):
    """Block outbound sockets to any host not in `allowed_hosts` (empty => all
    blocked). Covers `socket.create_connection` (urllib3/requests) and raw
    `socket.socket.connect`."""
    def guarded_create_connection(address, *a, **kw):
        host = _host_of(address)
        if host not in allowed_hosts:
            raise ScopeError(f"outbound network blocked: {host!r} not in {sorted(allowed_hosts)}")
        return _real_create_connection(address, *a, **kw)

    def guarded_socket_connect(self, address, *a, **kw):
        host = _host_of(address)
        if host is not None and host not in allowed_hosts:
            raise ScopeError(f"outbound network blocked: {host!r} not in {sorted(allowed_hosts)}")
        return _real_socket_connect(self, address, *a, **kw)

    socket.create_connection = guarded_create_connection
    socket.socket.connect = guarded_socket_connect
    try:
        yield
    finally:
        socket.create_connection = _real_create_connection
        socket.socket.connect = _real_socket_connect


def enforce_network_at_startup(agent: AgentName) -> None:
    """Permanent (non-context-managed) network restriction for an MCP server process."""
    guard = restrict_network(GRANTS[agent].net_hosts)
    guard.__enter__()  # never exited - the process lives with the restriction


# -------------------------------------------------------------------------- FS

class ScopedFS:
    """Path-prefix write guard. Reads are unrestricted (the repo is not secret);
    writes outside the agent's prefixes are refused."""

    def __init__(self, agent: AgentName, repo_root: Path | None = None):
        self.agent = agent
        self.prefixes = GRANTS[agent].fs_write_prefixes
        self.repo_root = (repo_root or Path(__file__).resolve().parent.parent).resolve()

    def check_write(self, path: Path | str) -> Path:
        p = Path(path)
        resolved = p if p.is_absolute() else (self.repo_root / p)
        resolved = resolved.resolve()
        try:
            rel = resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            raise ScopeError(f"{self.agent.value} may not write outside the repo: {resolved}")
        if not any(rel == pre.rstrip("/") or rel.startswith(pre) for pre in self.prefixes):
            raise ScopeError(
                f"{self.agent.value} may not write {rel!r}; allowed prefixes: {list(self.prefixes)}"
            )
        return resolved

    def open_write(self, path: Path | str, mode: str = "w", **kw):
        if "r" in mode and "+" not in mode:
            return open(path, mode, **kw)
        return open(self.check_write(path), mode, **kw)


# ------------------------------------------------------------------ brokered call

def _iter_date_strings(args: dict) -> list[str]:
    out = []
    for v in args.values():
        if isinstance(v, (date, datetime)):
            out.append(v.isoformat()[:10])
        elif isinstance(v, str) and _ISO_DATE.match(v):
            out.append(v[:10])
    return out


def _reject_recent_dates(args: dict, db_file: Path | str) -> str | None:
    """Validation's 'historical only' rule: reject any date arg with fewer than
    EMBARGO_DAYS non-superseded bars strictly after it (i.e. inside the last
    EMBARGO_DAYS of history). The broker does a raw read here for policy evaluation -
    it is the enforcer, not a tool."""
    dates = _iter_date_strings(args)
    if not dates:
        return None
    ro = sqlite3.connect(db_file)
    try:
        for d in dates:
            n_after = ro.execute(
                "SELECT COUNT(*) FROM daily_bars WHERE trade_date > ? AND superseded_by IS NULL", (d,)
            ).fetchone()[0]
            if n_after < EMBARGO_DAYS:
                return (f"date argument {d} is within EMBARGO_DAYS={EMBARGO_DAYS} of the last "
                        f"bar (only {n_after} bars after it); Validation is historical-only")
    finally:
        ro.close()
    return None


def _default_summary(result: Any) -> Any:
    """Compact result for the audit row - never a full DataFrame."""
    try:
        import pandas as pd
        if isinstance(result, pd.DataFrame):
            return {"_dataframe": True, "shape": list(result.shape), "columns": list(result.columns)}
        if isinstance(result, pd.Series):
            return {"_series": True, "len": int(result.size), "name": result.name}
    except Exception:
        pass
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None)))}
    if isinstance(result, (list, tuple)):
        return {"_seq": True, "len": len(result)}
    if isinstance(result, (str, int, float, bool, type(None))):
        return result
    return {"_type": type(result).__name__}


def brokered_call(
    *,
    broker_conn: sqlite3.Connection,
    trace_id: str,
    seq: int,
    caller: AgentName,
    callee: AgentName,
    tool_name: str,
    args: dict,
    tool_fn: Callable[[sqlite3.Connection, dict], Any],
    parent_call_id: int | None = None,
    summarize: Callable[[Any], Any] = _default_summary,
    db_file: Path | str | None = None,
    code_rev: str | None = None,
) -> dict:
    """Invoke `tool_fn` on behalf of `caller -> callee.tool_name`, enforcing the
    call graph and Validation's date rule, then record the outcome (ok/denied/error)
    to the hash-chained `agent_call_log`. Returns
    {status, result, call_id, denial_reason}. Raises ScopeError on a policy denial."""
    db_file = db_file or db_path()
    ts_start = datetime.now(timezone.utc).isoformat()
    status, denial_reason, result = "ok", None, None

    if callee not in CALL_GRAPH.get(caller, frozenset()):
        status, denial_reason = "denied", f"{caller.value} -> {callee.value} not permitted by CALL_GRAPH"
    if status == "ok" and GRANTS[callee].reject_dates_within_embargo_of_last_bar:
        why = _reject_recent_dates(args, db_file)
        if why:
            status, denial_reason = "denied", why

    if status == "ok":
        try:
            with scoped_db(callee, db_file) as sconn:
                result = tool_fn(sconn, args)
        except sqlite3.DatabaseError as exc:
            status = "denied" if _is_authorizer_denial(exc) else "error"
            denial_reason = str(exc)
        except ScopeError as exc:
            status, denial_reason = "denied", str(exc)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised below
            status, denial_reason = "error", f"{type(exc).__name__}: {exc}"

    ts_end = datetime.now(timezone.utc).isoformat()
    call_id = audit.chain_append(broker_conn, "agent_call_log", {
        "trace_id": trace_id,
        "parent_call_id": parent_call_id,
        "seq": seq,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "caller": caller.value,
        "callee_agent": callee.value,
        "tool_name": tool_name,
        "args_json": audit.canonical_json(args),
        "result_summary_json": audit.canonical_json(_safe_summary(summarize, result)),
        "status": status,
        "denial_reason": denial_reason,
        "grant_digest": audit.grant_digest(GRANTS[callee]),
        "code_rev": code_rev,
    })

    if status == "denied":
        raise ScopeError(denial_reason)
    if status == "error":
        raise RuntimeError(f"tool {callee.value}.{tool_name} failed: {denial_reason}")
    return {"status": status, "result": result, "call_id": call_id, "denial_reason": None}


def _safe_summary(summarize: Callable[[Any], Any], result: Any) -> Any:
    try:
        return summarize(result)
    except Exception as exc:  # noqa: BLE001
        return {"_summary_error": f"{type(exc).__name__}: {exc}"}
