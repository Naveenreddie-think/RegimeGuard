"""regimeguard-orchestrator - the MCP client that sequences the four agent servers
and owns `todays_call` end to end (Phase 5 design §1.5).

`todays_call` gathers the same inputs `signal_model.todays_call.gather_inputs`
gathers, but via MCP tool calls to scoped subprocess servers, then feeds them to the
identical pure `assemble_regimeguard_call`. So the decision logic has exactly one
implementation and the two paths produce identical `RegimeGuardCall` records.

Deterministic sequencing - no LLM planner. Each tool call is recorded (by the
servers' broker) to the hash-chained `agent_call_log` under one `trace_id`;
`explain(as_of)` reads the decision row plus its full call tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from contextlib import AsyncExitStack
from datetime import date, datetime, timezone

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.broker import db_path, scoped_connection
from agents.capabilities import AgentName
from signal_model.todays_call import (
    RegimeGuardCall,
    _code_rev,
    append_to_log,
    assemble_regimeguard_call,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_MODULES = {
    "regime": "agents.servers.regime_server",
    "training": "agents.servers.training_server",
    "validation": "agents.servers.validation_server",
    "data": "agents.servers.data_server",
}


def _server_params(module: str) -> StdioServerParameters:
    env = {**os.environ, "PYTHONPATH": _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return StdioServerParameters(command=sys.executable, args=["-m", module], env=env)


def _unwrap(res):
    """Every tool result crosses MCP wrapped as {"data": ...} by _common.run_tool."""
    if getattr(res, "isError", False):
        raise RuntimeError(res.content[0].text if res.content else "MCP tool error")
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict) and "data" in sc:
        return sc["data"]
    if res.content:
        parsed = json.loads(res.content[0].text)
        return parsed["data"] if isinstance(parsed, dict) and "data" in parsed else parsed
    return None


class _Sessions:
    """Open MCP client sessions to all four servers for the life of one request."""

    def __init__(self):
        self._stack = AsyncExitStack()
        self.session: dict[str, ClientSession] = {}

    async def __aenter__(self):
        for name, module in SERVER_MODULES.items():
            read, write = await self._stack.enter_async_context(stdio_client(_server_params(module)))
            s = await self._stack.enter_async_context(ClientSession(read, write))
            await s.initialize()
            self.session[name] = s
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def call(self, server: str, tool: str, **args):
        return _unwrap(await self.session[server].call_tool(tool, args))


async def _gather_via_mcp(sx: _Sessions, trace_id: str, as_of: str | None,
                          regime_version_id: int | None, code_rev: str | None) -> dict:
    seq = iter(range(1, 10_000))
    common = lambda: {"trace_id": trace_id, "seq": next(seq)}

    if regime_version_id is None:
        rv = await sx.call("regime", "get_active_version", **common())
    else:
        rv = await sx.call("regime", "load_version", regime_version_id=regime_version_id, **common())

    reg = await sx.call("regime", "point_in_time_label", as_of=as_of, regime_version_id=rv["id"], **common())
    resolved_as_of = reg["as_of"]  # snapped to the last trading day <= as_of

    drift = await sx.call("regime", "check_drift", regime_version_id=rv["id"], **common())
    vix_pct = await sx.call("regime", "trailing_vix_pct", as_of=resolved_as_of, **common())
    tiers = await sx.call("validation", "get_reliability_tiers", **common())
    signal_pred = await sx.call("training", "get_signal_prediction", as_of=resolved_as_of, **common())

    tier2 = drift.get("tier2")
    return {
        "as_of_date": resolved_as_of,
        "regime_version": {"id": int(rv["id"]), "fit_end_date": str(rv["fit_end_date"])},
        "reg": {k: reg[k] for k in (
            "regime", "run_length_td", "source", "n_fit_rows",
            "min_dist_to_centroid", "ood_threshold", "is_ood",
        )},
        "monitoring": {
            "tier1_mean": float(drift["tier1"]["mean_drift"]),
            "tier1_fires": bool(drift["tier1"]["fires"]),
            "tier2_ari": (tier2.get("interior_ari") if tier2 else None),
        },
        "vix_pct": float(vix_pct),
        "td_since_fit": int(reg["td_since_fit"]),
        "tiers_records": tiers,
        "signal_pred": signal_pred,
    }


async def _todays_call_async(as_of: str | None, regime_version_id: int | None, log: bool) -> RegimeGuardCall:
    trace_id = f"orch-{uuid.uuid4().hex[:12]}"
    code_rev = _code_rev()
    async with _Sessions() as sx:
        inputs = await _gather_via_mcp(sx, trace_id, as_of, regime_version_id, code_rev)
    record = assemble_regimeguard_call(
        inputs, generated_at=datetime.now(timezone.utc).isoformat(), code_rev=code_rev
    )
    record.audit["trace_id"] = trace_id
    if log:
        conn = scoped_connection(AgentName.ORCHESTRATOR)
        try:
            record.audit["log_id"] = append_to_log(conn, record, trace_id=trace_id)
        finally:
            conn.close()
    return record


def todays_call(as_of: date | str | None = None, regime_version_id: int | None = None,
                log: bool = True) -> RegimeGuardCall:
    """Same result as `signal_model.todays_call.decide_todays_call`, gathered over MCP."""
    as_of_str = as_of.isoformat() if isinstance(as_of, date) else as_of
    return asyncio.run(_todays_call_async(as_of_str, regime_version_id, log))


def explain(as_of: str, trace_id: str | None = None) -> dict:
    """The decision row for `as_of` plus its full ordered agent-call trace. With no
    `trace_id`, picks the most recent decision for `as_of` that has an agent-call
    trace (i.e. an orchestrator run), falling back to the most recent of any kind."""
    conn = scoped_connection(AgentName.ORCHESTRATOR)
    try:
        if trace_id is not None:
            row = conn.execute(
                "SELECT id, trace_id, disposition, reliability_tier, record_json "
                "FROM todays_call_log WHERE trace_id = ? ORDER BY id DESC LIMIT 1", (trace_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, trace_id, disposition, reliability_tier, record_json FROM todays_call_log "
                "WHERE as_of_date = ? "
                "ORDER BY (trace_id IN (SELECT DISTINCT trace_id FROM agent_call_log)) DESC, id DESC "
                "LIMIT 1", (as_of,)
            ).fetchone()
        if row is None:
            return {"as_of": as_of, "found": False}
        trace_id = row[1]
        calls = conn.execute(
            "SELECT seq, callee_agent, tool_name, status, denial_reason, result_summary_json "
            "FROM agent_call_log WHERE trace_id = ? ORDER BY seq", (trace_id,)
        ).fetchall()
        return {
            "as_of": as_of, "found": True, "trace_id": trace_id,
            "disposition": row[2], "reliability_tier": row[3],
            "record": json.loads(row[4]),
            "trace": [dict(zip(
                ("seq", "callee_agent", "tool_name", "status", "denial_reason", "result_summary"), c
            )) for c in calls],
        }
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="RegimeGuard orchestrator (MCP client)")
    sub = p.add_subparsers(dest="cmd", required=True)
    tc = sub.add_parser("todays-call")
    tc.add_argument("--as-of", default=None)
    tc.add_argument("--regime-version-id", type=int, default=None)
    tc.add_argument("--no-log", action="store_true")
    ex = sub.add_parser("explain")
    ex.add_argument("as_of")
    args = p.parse_args()

    if args.cmd == "todays-call":
        rec = todays_call(args.as_of, args.regime_version_id, log=not args.no_log)
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(json.dumps(explain(args.as_of), indent=2))


if __name__ == "__main__":
    main()
