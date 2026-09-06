"""regimeguard-validation MCP server - Phase 5 §1.4. Wraps the signal_model/ eval side.

Tools run on a VALIDATION-scoped connection: READ-ONLY on daily_bars + regime tables
+ signal tables; DENIED todays_call_log / agent_call_log; no network; no DDL. The
broker additionally rejects any tool call whose date arguments fall within
EMBARGO_DAYS of the last available bar ("historical only" as an enforced boundary).
"""

from __future__ import annotations

import pandas as pd

from agents.capabilities import AgentName
from agents.servers._common import make_server, run_tool
from signal_model.regime_reliability_tiers import TIERS_CSV, compute_tiers

mcp = make_server(AgentName.VALIDATION)
_A = AgentName.VALIDATION


@mcp.tool()
def get_reliability_tiers(trace_id: str, seq: int):
    """The current per-regime reliability tier table (`monitored` / `none`) with its
    supporting numbers - read from the build artifact CSV."""
    return run_tool(_A, "get_reliability_tiers", trace_id, seq, {},
                    lambda c, a: pd.read_csv(TIERS_CSV).to_dict("records"))


@mcp.tool()
def regenerate_reliability_tiers(trace_id: str, seq: int):
    """CONSEQUENTIAL (artifact) - recompute the tier table from the frozen OOF
    predictions + point-in-time regime labels and rewrite the CSV. Read-only on the
    DB; writes only under signal_model/results/."""
    def _fn(c, a):
        df = compute_tiers(c)
        df.to_csv(TIERS_CSV, index=False)
        return df.to_dict("records")
    return run_tool(_A, "regenerate_reliability_tiers", trace_id, seq, {}, _fn)


if __name__ == "__main__":
    mcp.run()
