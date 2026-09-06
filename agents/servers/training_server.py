"""regimeguard-training MCP server - Phase 5 §1.3. Wraps the signal_model/ fit side.

Tools run on a TRAINING-scoped connection: may write signal_model_versions /
signal_predictions, read daily_bars + regime tables + the signal tables, nothing
else; no network; no DDL.
"""

from __future__ import annotations

from datetime import date

from agents.capabilities import AgentName
from agents.servers._common import make_server, run_tool
from signal_model.registry import (
    get_active_signal_model_version,
    load_signal_model_version,
    load_signal_prediction,
)

mcp = make_server(AgentName.TRAINING)
_A = AgentName.TRAINING


def _slim(v: dict) -> dict:
    return {
        "id": int(v["id"]), "model_kind": v["model_kind"],
        "target_horizon_days": int(v["target_horizon_days"]),
        "target_flat_bps": float(v["target_flat_bps"]),
        "as_of_date": str(v["as_of_date"]), "fit_end_date": str(v["fit_end_date"]),
        "coverage_end_date": str(v["coverage_end_date"]), "status": v["status"],
    }


@mcp.tool()
def get_active_signal_version(trace_id: str, seq: int):
    """The currently-active LightGBM signal_model_version (slim view), or null."""
    def _fn(c, a):
        v = get_active_signal_model_version(c, "lgbm", 1, 20.0)
        return _slim(v) if v else None
    return run_tool(_A, "get_active_signal_version", trace_id, seq, {}, _fn)


@mcp.tool()
def load_signal_version(trace_id: str, seq: int, version_id: int):
    """A specific signal_model_version by id (slim view)."""
    return run_tool(_A, "load_signal_version", trace_id, seq, {"version_id": version_id},
                    lambda c, a: _slim(load_signal_model_version(c, a["version_id"])))


@mcp.tool()
def get_signal_prediction(trace_id: str, seq: int, as_of: str):
    """The current (non-superseded) registered signal prediction for `as_of`, or
    null if no version covers that date (== signal_model_stale)."""
    return run_tool(_A, "get_signal_prediction", trace_id, seq, {"as_of": as_of},
                    lambda c, a: load_signal_prediction(c, date.fromisoformat(a["as_of"])))


@mcp.tool()
def fit_and_register_signal(trace_id: str, seq: int, as_of: str | None = None,
                            coverage_td: int = 63, notes: str | None = None):
    """CONSEQUENTIAL - fit a LightGBM model as-of `as_of` and register it as a new
    (active) signal_model_version with its stored predictions. Operator-initiated only."""
    from signal_model.fit_and_register_signal import fit_and_register_signal as _fars

    def _fn(c, a):
        vid, n = _fars(c, date.fromisoformat(a["as_of"]) if a["as_of"] else None,
                       a["coverage_td"], a["notes"])
        return {"signal_model_version_id": int(vid), "n_predictions": int(n)}
    return run_tool(_A, "fit_and_register_signal", trace_id, seq,
                    {"as_of": as_of, "coverage_td": coverage_td, "notes": notes}, _fn)


if __name__ == "__main__":
    mcp.run()
