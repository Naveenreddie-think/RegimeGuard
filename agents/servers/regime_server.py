"""regimeguard-regime MCP server - Phase 5 §1.2. Wraps regime_detection/.

Tools run on a REGIME-scoped connection: may write model_versions / regime_labels,
read daily_bars + the regime tables, nothing else; no network; no DDL.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from agents.capabilities import AgentName
from agents.servers._common import make_server, run_tool
from regime_detection.monitoring import check_recalibration_trigger
from regime_detection.quarterly_walk import point_in_time_regime_label, run_quarterly_walk
from regime_detection.regime_db import get_active_model_version, load_model_version
from signal_model.todays_call import trailing_vix_percentile

mcp = make_server(AgentName.REGIME)
_A = AgentName.REGIME


def _slim_version(mv: dict) -> dict:
    return {
        "id": int(mv["id"]), "model_kind": mv["model_kind"], "k": int(mv["k"]),
        "jump_penalty": float(mv["jump_penalty"]), "fit_start_date": str(mv["fit_start_date"]),
        "fit_end_date": str(mv["fit_end_date"]), "status": mv["status"],
    }


@mcp.tool()
def get_active_version(trace_id: str, seq: int):
    """The currently-active JM k=3 / lambda=50 regime model_version (slim view)."""
    return run_tool(_A, "get_active_version", trace_id, seq, {},
                    lambda c, a: _slim_version(get_active_model_version(c, "jm", 3, 50.0)))


@mcp.tool()
def load_version(trace_id: str, seq: int, regime_version_id: int):
    """A specific regime model_version by id (slim view)."""
    return run_tool(_A, "load_version", trace_id, seq, {"regime_version_id": regime_version_id},
                    lambda c, a: _slim_version(load_model_version(c, a["regime_version_id"])))


@mcp.tool()
def point_in_time_label(trace_id: str, seq: int, as_of: str | None = None, regime_version_id: int | None = None):
    """Point-in-time regime label for `as_of` under the active (or given) version:
    regime id, trailing run length, trading-days-since-fit, and the M6 OOD inputs."""
    def _fn(c, a):
        mv = (get_active_model_version(c, "jm", 3, 50.0) if a["regime_version_id"] is None
              else load_model_version(c, a["regime_version_id"]))
        return point_in_time_regime_label(c, mv, a["as_of"])
    return run_tool(_A, "point_in_time_label", trace_id, seq,
                    {"as_of": as_of, "regime_version_id": regime_version_id}, _fn)


@mcp.tool()
def check_drift(trace_id: str, seq: int, regime_version_id: int | None = None):
    """Two-tier drift snapshot (`check_recalibration_trigger`) for the active (or
    given) version. Shadow-fits in memory; writes nothing."""
    def _fn(c, a):
        mvid = a["regime_version_id"] or get_active_model_version(c, "jm", 3, 50.0)["id"]
        return check_recalibration_trigger(c, int(mvid))
    return run_tool(_A, "check_drift", trace_id, seq, {"regime_version_id": regime_version_id}, _fn)


@mcp.tool()
def trailing_vix_pct(trace_id: str, seq: int, as_of: str):
    """Expanding-window percentile rank of the India-VIX close at `as_of` (M1 stress
    proxy). Point-in-time: trailing data only."""
    return run_tool(_A, "trailing_vix_pct", trace_id, seq, {"as_of": as_of},
                    lambda c, a: trailing_vix_percentile(c, pd.Timestamp(a["as_of"])))


@mcp.tool()
def fit_and_register(trace_id: str, seq: int, cutoff: str | None = None, k: int = 3,
                     jump_penalty: float = 50.0, notes: str | None = None):
    """CONSEQUENTIAL - fit a JM config through `cutoff` and register it as a new
    (active) model_version with its point-in-time labels. Operator-initiated only."""
    from regime_detection.fit_and_register import fit_and_register as _far

    def _fn(c, a):
        vid, n = _far(c, date.fromisoformat(a["cutoff"]) if a["cutoff"] else None,
                      a["k"], a["jump_penalty"], a["notes"])
        return {"regime_version_id": int(vid), "n_labels": int(n)}
    return run_tool(_A, "fit_and_register", trace_id, seq,
                    {"cutoff": cutoff, "k": k, "jump_penalty": jump_penalty, "notes": notes}, _fn)


@mcp.tool()
def quarterly_walk(trace_id: str, seq: int):
    """CONSEQUENTIAL - the full quarterly point-in-time recalibration backfill."""
    return run_tool(_A, "quarterly_walk", trace_id, seq, {},
                    lambda c, a: {"regime_version_ids": [int(i) for i in run_quarterly_walk(c)]})


if __name__ == "__main__":
    mcp.run()
