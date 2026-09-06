"""regimeguard-data MCP server - Phase 5 §1.1. Wraps data_agent/.

Tools run on a DATA-scoped connection: may write instruments / ingestion_runs /
daily_bars / calendar_days, read the same, nothing else. This is the only server
allowed outbound network (niftyindices.com only), and the only one that may write
under data/raw/. No DDL.

`ingest_prices` / `load_vix_csv` are deferred to a follow-up (network fetch and a
`conn`-parameter refactor of `load_india_vix_csv` respectively); the read tools and
gap check below are what the orchestrator needs.
"""

from __future__ import annotations

from datetime import date

from agents.capabilities import AgentName
from agents.servers._common import make_server, run_tool
from data_agent.calendar_days import find_unexplained_gaps

mcp = make_server(AgentName.DATA)
_A = AgentName.DATA


@mcp.tool()
def get_bars(trace_id: str, seq: int, symbol: str, start: str, end: str):
    """Point-in-time-correct daily OHLC for `symbol` (NIFTY50 / BANKNIFTY / INDIAVIX)
    over [start, end], from the non-superseded `current_bars` view."""
    def _fn(c, a):
        rows = c.execute(
            "SELECT b.trade_date, b.open, b.high, b.low, b.close FROM current_bars b "
            "JOIN instruments i ON i.id = b.instrument_id "
            "WHERE i.symbol = ? AND b.trade_date BETWEEN ? AND ? ORDER BY b.trade_date",
            (a["symbol"], a["start"], a["end"]),
        ).fetchall()
        return [dict(zip(("trade_date", "open", "high", "low", "close"), r)) for r in rows]
    return run_tool(_A, "get_bars", trace_id, seq,
                    {"symbol": symbol, "start": start, "end": end}, _fn)


@mcp.tool()
def find_gaps(trace_id: str, seq: int, symbol: str, start: str, end: str):
    """Trading days / special sessions in [start, end] with no `daily_bars` row for
    `symbol` - the real silent-gap check (calendar says data should exist, we have none)."""
    def _fn(c, a):
        row = c.execute("SELECT id FROM instruments WHERE symbol = ?", (a["symbol"],)).fetchone()
        if row is None:
            raise ValueError(f"unknown symbol {a['symbol']!r}")
        return find_unexplained_gaps(c, row[0], date.fromisoformat(a["start"]), date.fromisoformat(a["end"]))
    return run_tool(_A, "find_gaps", trace_id, seq,
                    {"symbol": symbol, "start": start, "end": end}, _fn)


if __name__ == "__main__":
    mcp.run()
