"""Shared plumbing for the four MCP servers - Phase 5 §4.

`run_tool` is the single path a tool body takes: open a BROKER-scoped connection
for the audit write, call `brokered_call` (which enforces the call graph + the
Validation date rule, runs `tool_fn` on a *callee*-scoped connection, and appends
the `agent_call_log` row), and hand back the result. A denial raises out of the
server as an MCP error, but the audit row is written first.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from agents.broker import brokered_call, enforce_network_at_startup, scoped_connection
from agents.capabilities import AgentName


def make_server(agent: AgentName) -> FastMCP:
    """Build the FastMCP server for `agent`, installing the permanent network
    restriction from that agent's Grant before any tool can run."""
    enforce_network_at_startup(agent)
    return FastMCP(f"regimeguard-{agent.value}")


def _jsonable(value: Any) -> Any:
    """Coerce a tool result to something MCP can serialise (numpy scalars, etc.)."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "item", lambda: str(o))()))


def run_tool(
    agent: AgentName,
    tool_name: str,
    trace_id: str,
    seq: int,
    args: dict,
    tool_fn: Callable[[Any, dict], Any],
    parent_call_id: int | None = None,
    code_rev: str | None = None,
) -> Any:
    broker_conn = scoped_connection(AgentName.BROKER)
    try:
        out = brokered_call(
            broker_conn=broker_conn,
            trace_id=trace_id,
            seq=seq,
            parent_call_id=parent_call_id,
            caller=AgentName.ORCHESTRATOR,   # the MCP client is the only permitted caller
            callee=agent,
            tool_name=tool_name,
            args=args,
            tool_fn=tool_fn,
            code_rev=code_rev,
        )
        # brokered_call already wrote the audit row (incl. call_id). Wrap the result
        # in a dict so FastMCP always has a structured object to serialise, whatever
        # the tool returned (list / scalar / None included).
        return {"data": _jsonable(out["result"])}
    finally:
        broker_conn.close()
