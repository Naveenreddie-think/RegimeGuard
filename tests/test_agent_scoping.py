"""Deny-path suite for the Phase 5 permission layer - design §2.5.

This is the deliverable that makes "tool-permission scoping" a fact rather than a
claim: for each agent it builds the *real* scoped handle and asserts the things
outside its grant actually fail - at the SQLite engine, the socket layer, and the
filesystem guard - regardless of the calling code.

Run: `python -m pytest tests/test_agent_scoping.py -q`
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import date, timedelta

import pytest

from agents import audit, bootstrap
from agents.broker import ScopedFS, ScopeError, brokered_call, restrict_network, scoped_db
from agents.capabilities import ALL_TABLES, CALL_GRAPH, GRANTS, TRADE_VOCABULARY, AgentName
from data_agent.db import DB_PATH

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "regimeguard.db"
    shutil.copy2(DB_PATH, p)
    bootstrap.bootstrap(p)
    return p


def _raw(db):
    c = sqlite3.connect(db)
    return c


# --------------------------------------------------------------- DB: denials

@pytest.mark.parametrize("agent, sql", [
    (AgentName.VALIDATION, "SELECT * FROM todays_call_log LIMIT 1"),
    (AgentName.VALIDATION, "SELECT * FROM agent_call_log LIMIT 1"),
    (AgentName.VALIDATION, "INSERT INTO signal_predictions (trade_date, signal_model_version_id, "
                           "pred_direction, prob_down, prob_flat, prob_up, predicted_at) "
                           "VALUES ('2020-01-01', 1, 0, 0.3, 0.4, 0.3, 'x')"),
    (AgentName.REGIME, "INSERT INTO signal_model_versions (model_kind, target_horizon_days, "
                       "target_flat_bps, fit_start_date, fit_end_date, as_of_date, purge_days, "
                       "embargo_days, coverage_end_date, feature_columns, lgbm_params, "
                       "n_train_rows, fitted_at) VALUES ('lgbm',1,20,'a','b','c',1,240,'d','[]','{}',1,'e')"),
    (AgentName.REGIME, "SELECT * FROM signal_model_versions LIMIT 1"),
    (AgentName.TRAINING, "INSERT INTO model_versions (model_kind, k, jump_penalty, fit_start_date, "
                         "fit_end_date, fitted_at, clipper_lb, clipper_ub, scaler_mean, scaler_scale, "
                         "feature_columns) VALUES ('jm',3,50,'a','b','c','[]','[]','[]','[]','[]')"),
    (AgentName.DATA, "INSERT INTO todays_call_log (as_of_date, generated_at, disposition, "
                     "reliability_tier, actionable, record_json) VALUES ('2020-01-01','x','ABSTAIN','none',0,'{}')"),
    (AgentName.DATA, "SELECT * FROM model_versions LIMIT 1"),
    (AgentName.ORCHESTRATOR, "INSERT INTO model_versions (model_kind, k, jump_penalty, fit_start_date, "
                             "fit_end_date, fitted_at, clipper_lb, clipper_ub, scaler_mean, scaler_scale, "
                             "feature_columns) VALUES ('jm',3,50,'a','b','c','[]','[]','[]','[]','[]')"),
])
def test_db_denied(db, agent, sql):
    with scoped_db(agent, db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized|prohibited"):
            conn.execute(sql)


@pytest.mark.parametrize("agent", list(AgentName))
@pytest.mark.parametrize("ddl", [
    "CREATE TABLE evil (x INTEGER)",
    "DROP TABLE todays_call_log",
    "ALTER TABLE todays_call_log ADD COLUMN evil TEXT",
    "CREATE INDEX evil_ix ON daily_bars(trade_date)",
])
def test_ddl_denied_for_everyone(db, agent, ddl):
    with scoped_db(agent, db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized|prohibited"):
            conn.execute(ddl)


def test_data_cannot_read_regime_view(db):
    # current_regime_labels is always-readable as a name, but its underlying
    # regime_labels read is still checked - and DATA cannot read regime_labels.
    with scoped_db(AgentName.DATA, db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized|prohibited"):
            conn.execute("SELECT * FROM current_regime_labels LIMIT 1")


# --------------------------------------------------------------- DB: allowances (positive controls)

def test_db_allowed_paths(db):
    with scoped_db(AgentName.DATA, db) as conn:
        conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
        conn.execute("INSERT INTO ingestion_runs (started_at, source, date_range_requested) "
                     "VALUES ('t','selftest','x')")
    with scoped_db(AgentName.REGIME, db) as conn:
        conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
        conn.execute("SELECT COUNT(*) FROM current_bars").fetchone()  # view via allowed underlying table
    with scoped_db(AgentName.TRAINING, db) as conn:
        conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()
    with scoped_db(AgentName.VALIDATION, db) as conn:
        conn.execute("SELECT COUNT(*) FROM signal_predictions").fetchone()
    with scoped_db(AgentName.ORCHESTRATOR, db) as conn:
        conn.execute("SELECT COUNT(*) FROM todays_call_log").fetchone()
        conn.execute("SELECT COUNT(*) FROM agent_call_log").fetchone()


# --------------------------------------------------------------- network

def test_network_blocked_when_no_hosts():
    import socket
    with restrict_network(frozenset()):
        with pytest.raises(ScopeError):
            socket.create_connection(("example.com", 80), timeout=0.1)
        with pytest.raises(ScopeError):
            socket.socket().connect(("93.184.216.34", 80))


def test_network_blocks_disallowed_host_for_data():
    import socket
    with restrict_network(GRANTS[AgentName.DATA].net_hosts):
        with pytest.raises(ScopeError):
            socket.create_connection(("evil.example", 443), timeout=0.1)


def test_non_data_agents_have_no_net_hosts():
    for a in (AgentName.REGIME, AgentName.TRAINING, AgentName.VALIDATION, AgentName.ORCHESTRATOR):
        assert GRANTS[a].net_hosts == frozenset()


# --------------------------------------------------------------- filesystem

def test_fs_scope():
    tr = ScopedFS(AgentName.TRAINING)
    tr.check_write("signal_model/results/x.csv")           # ok
    with pytest.raises(ScopeError):
        tr.check_write("data/raw/x")
    with pytest.raises(ScopeError):
        tr.check_write("regime_detection/results/x")
    with pytest.raises(ScopeError):
        tr.check_write("../outside-the-repo")

    ScopedFS(AgentName.DATA).check_write("data/raw/nifty50/x.json")   # ok
    with pytest.raises(ScopeError):
        ScopedFS(AgentName.REGIME).check_write("signal_model/results/x")
    with pytest.raises(ScopeError):
        ScopedFS(AgentName.ORCHESTRATOR).check_write("anything")      # no prefixes at all


# --------------------------------------------------------------- call graph

def test_call_graph_shape():
    for a, callees in CALL_GRAPH.items():
        assert a not in callees
        assert AgentName.ORCHESTRATOR not in callees
        assert AgentName.BROKER not in callees
        if a is not AgentName.ORCHESTRATOR:
            assert not callees
    assert CALL_GRAPH[AgentName.ORCHESTRATOR] == frozenset(
        {AgentName.DATA, AgentName.REGIME, AgentName.TRAINING, AgentName.VALIDATION}
    )


def test_brokered_call_rejects_non_permitted_edge(db):
    broker = _raw(db)
    try:
        with pytest.raises(ScopeError, match="not permitted by CALL_GRAPH"):
            brokered_call(
                broker_conn=broker, trace_id="t1", seq=1,
                caller=AgentName.DATA, callee=AgentName.REGIME, tool_name="whatever",
                args={}, tool_fn=lambda c, a: None, db_file=db,
            )
        row = broker.execute("SELECT status, denial_reason FROM agent_call_log ORDER BY id DESC LIMIT 1").fetchone()
        assert row[0] == "denied" and "CALL_GRAPH" in row[1]
        assert audit.verify_chain(broker, "agent_call_log")["ok"]
    finally:
        broker.close()


def test_brokered_call_permitted_edge_runs_and_logs(db):
    broker = _raw(db)
    try:
        out = brokered_call(
            broker_conn=broker, trace_id="t2", seq=1,
            caller=AgentName.ORCHESTRATOR, callee=AgentName.REGIME, tool_name="count_bars",
            args={}, tool_fn=lambda c, a: c.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0],
            db_file=db,
        )
        assert out["status"] == "ok" and isinstance(out["result"], int)
        row = broker.execute("SELECT status FROM agent_call_log ORDER BY id DESC LIMIT 1").fetchone()
        assert row[0] == "ok"
        assert audit.verify_chain(broker, "agent_call_log")["ok"]
    finally:
        broker.close()


def test_brokered_call_records_tool_authorizer_denial(db):
    """A tool that tries an out-of-scope query is denied by the scoped handle, and
    the broker records it as status='denied' rather than letting it through."""
    broker = _raw(db)
    try:
        with pytest.raises(ScopeError):
            brokered_call(
                broker_conn=broker, trace_id="t3", seq=1,
                caller=AgentName.ORCHESTRATOR, callee=AgentName.VALIDATION, tool_name="peek_decisions",
                args={}, tool_fn=lambda c, a: c.execute("SELECT * FROM todays_call_log").fetchall(),
                db_file=db,
            )
        row = broker.execute("SELECT status, denial_reason FROM agent_call_log ORDER BY id DESC LIMIT 1").fetchone()
        assert row[0] == "denied"
        assert "not authorized" in row[1].lower() or "prohibited" in row[1].lower()
    finally:
        broker.close()


# --------------------------------------------------------------- Validation date rule

def test_validation_rejects_recent_date_arg(db):
    broker = _raw(db)
    last_bar = broker.execute("SELECT MAX(trade_date) FROM daily_bars").fetchone()[0]
    try:
        with pytest.raises(ScopeError, match="historical-only"):
            brokered_call(
                broker_conn=broker, trace_id="t4", seq=1,
                caller=AgentName.ORCHESTRATOR, callee=AgentName.VALIDATION, tool_name="eval_at",
                args={"as_of": last_bar}, tool_fn=lambda c, a: "should not run", db_file=db,
            )
        # an old date is fine
        out = brokered_call(
            broker_conn=broker, trace_id="t4", seq=2,
            caller=AgentName.ORCHESTRATOR, callee=AgentName.VALIDATION, tool_name="eval_at",
            args={"start": "2016-01-04"}, tool_fn=lambda c, a: "ran", db_file=db,
        )
        assert out["result"] == "ran"
    finally:
        broker.close()


# --------------------------------------------------------------- no trade capability (by construction)

def test_no_trade_capability_anywhere():
    assert TRADE_VOCABULARY == ()
    for t in ALL_TABLES:
        assert not any(w in t for w in ("order", "trade", "execution", "position", "fill"))


def test_no_execution_sdk_imported():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    banned = ("ccxt", "kiteconnect", "alpaca", "ib_insync", "ibapi", "alice_blue",
              "smartapi", "fyers", "upstox", "robin_stocks")
    hits = []
    for py in root.rglob("*.py"):
        if "/.venv/" in py.as_posix() or "/node_modules/" in py.as_posix():
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for b in banned:
            if f"import {b}" in text or f"from {b}" in text:
                hits.append((py.name, b))
    assert not hits, f"execution/broker SDK referenced: {hits}"


# --------------------------------------------------------------- audit hash chain

def test_chain_detects_tampering(db):
    broker = _raw(db)
    try:
        for i in range(3):
            audit.chain_append(broker, "agent_call_log", {
                "trace_id": "tamper", "parent_call_id": None, "seq": i, "ts_start": "a",
                "ts_end": "b", "caller": "operator", "callee_agent": "regime",
                "tool_name": "x", "args_json": "{}", "result_summary_json": "null",
                "status": "ok", "denial_reason": None, "grant_digest": "d", "code_rev": None,
            })
        assert audit.verify_chain(broker, "agent_call_log")["ok"]
        broker.execute("UPDATE agent_call_log SET status='TAMPERED' WHERE id=2")
        broker.commit()
        v = audit.verify_chain(broker, "agent_call_log")
        assert v["ok"] is False and v["id"] == 2
    finally:
        broker.close()
