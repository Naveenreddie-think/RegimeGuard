"""The four RegimeGuard MCP stdio servers - Phase 5 design §1, §4.

Each server exposes only its own agent's tools. Every tool routes through
`_common.run_tool`, which runs the wrapped Phase 1-4 function on a
capability-scoped connection (the SQLite authorizer is the real enforcement) and
records the call - including on denial - to the hash-chained `agent_call_log`.

Run one directly for debugging: `python -m agents.servers.regime_server`.
The `agents.orchestrator` MCP client spawns all four as subprocesses.
"""
