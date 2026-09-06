"""RegimeGuard dashboard - Phase 6, proposal §5.4. Minimal, read-only, honest.

Three sections, exactly as scoped in docs/phase6_design_proposal.md §5.1:
  A. Today's Call    - latest todays_call_log row: regime, tier, disposition,
                       INFORMATIONAL pattern (always with its not-an-edge disclaimer)
                       or ABSTAIN reasons, monitoring strip, staleness banner.
  B. Why             - the agent-call trace for that decision (mirrors
                       agents.orchestrator.explain) + verify_chain on both hash chains.
  C. Recent History  - last ~20 decisions.

Reads a local snapshot of the SQLite db (copied from the Modal Volume mount by
db_sync.read_only_snapshot, or the in-repo db when run locally). No writes, no
agent-layer imports beyond the stdlib-only hash-chain verifier.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# --- locate the database -----------------------------------------------------------
_VOLUME_DB = Path("/data/regimeguard.db")                                  # on Modal
_REPO_DB = Path(__file__).resolve().parent.parent / "data" / "regimeguard.db"  # local
_SNAPSHOT = Path(tempfile.gettempdir()) / "regimeguard_dashboard.db"


def _source_db() -> Path:
    """The db to snapshot from: an explicit override (local dev), else the Modal
    Volume mount, else the in-repo copy."""
    override = os.environ.get("REGIMEGUARD_DASHBOARD_DB")
    if override:
        return Path(override)
    return _VOLUME_DB if _VOLUME_DB.exists() else _REPO_DB

TIER_COLOR = {"monitored": "#1a7f37", "suppressed": "#9a6700", "none": "#8b8b8b"}
DISPOSITION_COLOR = {"INFORMATIONAL": "#1a7f37", "ABSTAIN": "#9a6700"}


@st.cache_data(ttl=300, show_spinner=False)
def _load_snapshot_mtime() -> float:
    """Refresh the Volume (best effort) and take a consistent local copy every 5 min.
    Returns the snapshot mtime so downstream caches key off it."""
    src = _source_db()
    if _VOLUME_DB.exists():
        try:
            import modal  # only present/meaningful on Modal

            modal.Volume.from_name("regimeguard-data").reload()
        except Exception:
            pass
    _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, _SNAPSHOT)
    return _SNAPSHOT.stat().st_mtime


def _db() -> Path:
    _load_snapshot_mtime()
    return _SNAPSHOT if _SNAPSHOT.exists() else _source_db()


def _connect(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=300, show_spinner=False)
def load_latest(_mtime: float) -> dict | None:
    with _connect(_db()) as conn:
        row = conn.execute(
            "SELECT id, as_of_date, generated_at, disposition, reliability_tier, "
            "trace_id, record_json FROM todays_call_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["record"] = json.loads(out.pop("record_json"))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_trace(_mtime: float, trace_id: str) -> list[dict]:
    """The ordered agent-call trace for one decision - the same query
    agents.orchestrator.explain runs against agent_call_log."""
    if not trace_id:
        return []
    with _connect(_db()) as conn:
        rows = conn.execute(
            "SELECT seq, caller, callee_agent, tool_name, status, denial_reason, "
            "result_summary_json FROM agent_call_log WHERE trace_id = ? ORDER BY seq",
            (trace_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(ttl=300, show_spinner=False)
def load_history(_mtime: float, n: int = 20) -> pd.DataFrame:
    with _connect(_db()) as conn:
        df = pd.read_sql_query(
            "SELECT as_of_date, generated_at, reliability_tier, disposition, "
            "directional_lean, regime_pit_id, trading_days_since_fit, trace_id "
            "FROM todays_call_log ORDER BY id DESC LIMIT ?",
            conn, params=(n,),
        )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def verify_chains(_mtime: float) -> dict:
    try:
        from agents.audit import verify_chain

        with _connect(_db()) as conn:
            return {
                "agent_call_log": verify_chain(conn, "agent_call_log"),
                "todays_call_log": verify_chain(conn, "todays_call_log"),
            }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# --- helpers ---------------------------------------------------------------------

def _staleness_days(as_of_date: str) -> int:
    try:
        return (date.today() - date.fromisoformat(as_of_date)).days
    except Exception:
        return 0


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except Exception:
        return ts


# --- page ----------------------------------------------------------------------

st.set_page_config(page_title="RegimeGuard", page_icon="📉", layout="centered")
st.title("RegimeGuard")
st.caption(
    "Regime-aware, leakage-safe signal validation. The system states the conditions "
    "under which its read can be trusted, and abstains outside them. "
    "**Never an actionable trade call** (`actionable = false`, by construction)."
)

mtime = _load_snapshot_mtime()
latest = load_latest(mtime)

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("↻ refresh"):
        st.cache_data.clear()
        st.rerun()

if latest is None:
    st.warning(
        "No decision has been recorded yet. Run one with "
        "`modal run deploy/modal_app.py` (or the weekday schedule will produce one)."
    )
    st.stop()

rec = latest["record"]
regime = rec["regime"]
mon = rec["monitoring"]
disp = rec["disposition"]
tier = rec["reliability_tier"]
stale = _staleness_days(latest["as_of_date"])

# ---------------------------------------------------------------- A. Today's Call
st.header("Today's Call")

if stale > 4:
    st.warning(
        f"**Data is {stale} calendar days behind.** Latest decision is *as of* "
        f"{latest['as_of_date']}. India VIX ingestion is a manual step (NSE blocks "
        f"automated access), so the store only advances when an operator refreshes it."
    )

c1, c2, c3 = st.columns(3)
c1.metric("As of", latest["as_of_date"])
c2.metric(
    "Regime",
    f"{regime['point_in_time_id']} · {regime['character']}",
    help=f"run length {regime['run_length_trading_days']} trading days · "
         f"label source {regime['label_source']} · fit rows {regime['fit_rows']}",
)
c3.metric("Reliability tier", tier)

badge = DISPOSITION_COLOR.get(disp, "#8b8b8b")
st.markdown(
    f"<div style='padding:0.6rem 1rem;border-radius:8px;background:{badge}22;"
    f"border:1px solid {badge};font-size:1.15rem'><b>{disp}</b> &nbsp;·&nbsp; "
    f"generated {_fmt_ts(rec['generated_at'])} &nbsp;·&nbsp; code_rev "
    f"<code>{rec['audit'].get('code_rev') or 'n/a'}</code></div>",
    unsafe_allow_html=True,
)
st.write("")

pattern = rec.get("regime_pattern") or {}
if disp == "INFORMATIONAL":
    st.subheader("Regime pattern")
    st.markdown(pattern.get("statement", "_none_"))
    if pattern.get("not_an_edge"):
        st.info(f"**Not an edge.** {pattern['not_an_edge']}")

    lean = rec.get("directional_lean") or {}
    st.subheader("Directional lean")
    st.caption(lean.get("note", "INFORMATIONAL ONLY - not a trade recommendation"))
    if lean.get("stale_reason"):
        st.write(f"_No current signal model_ (`{lean['stale_reason']}`).")
    elif lean.get("direction"):
        scores = lean.get("raw_class_scores") or {}
        st.write(
            f"**{lean['direction'].upper()}** — "
            + ", ".join(f"{k} {v:.2f}" for k, v in scores.items())
            + f"  ·  signal model v{lean.get('signal_model_version_id')} "
            f"(as of {lean.get('signal_model_as_of')})"
        )
else:  # ABSTAIN
    ab = rec.get("abstention") or {}
    st.subheader("Why it abstains")
    st.write(ab.get("plain_language", ""))
    if ab.get("reasons"):
        st.write("Reason codes: " + ", ".join(f"`{r}`" for r in ab["reasons"]))
    if pattern.get("statement"):
        st.caption(pattern["statement"])

st.subheader("Monitoring")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trading days since fit", mon["trading_days_since_fit"],
          help=f"past quarterly floor: {mon['past_quarterly_floor']}")
m2.metric("Tier-1 drift (mean)", f"{mon['tier1_drift_mean']:.3f}",
          help=f"fires: {mon['tier1_fires']}")
m3.metric("Tier-2 shadow ARI",
          "n/a" if mon["tier2_shadow_ari"] is None else f"{mon['tier2_shadow_ari']:.3f}")
m4.metric("Trailing VIX pct", f"{mon['trailing_vix_percentile']:.2f}")
flags = []
if mon.get("circuit_breaker"):
    flags.append("🛑 circuit breaker")
if mon.get("recalibration_flag"):
    flags.append(f"recalibration: {mon['recalibration_flag']}")
if flags:
    st.write(" · ".join(flags))

with st.expander("Full RegimeGuardCall record (JSON)"):
    st.json(rec)

# ------------------------------------------------------------------------ B. Why
st.header("Why — audit trail")
st.caption(
    f"trace `{latest['trace_id']}` · the ordered agent-call tree that produced this "
    f"decision (mirrors `orchestrator.explain`). Every row was written by the broker "
    f"on a capability-scoped connection, into a SHA-256 hash chain."
)

trace = load_trace(mtime, latest["trace_id"] or "")
if trace:
    tdf = pd.DataFrame(trace)
    # keep result_summary as a compact string - shapes vary per tool and Arrow
    # cannot infer a mixed struct/None column
    tdf["result_summary"] = tdf["result_summary_json"].apply(
        lambda s: json.dumps(json.loads(s), separators=(",", ":")) if s else ""
    )
    st.dataframe(
        tdf[["seq", "caller", "callee_agent", "tool_name", "status", "denial_reason",
             "result_summary"]],
        hide_index=True, width="stretch",
    )
else:
    st.write(
        "_No agent-call trace for this decision._ It was produced by the direct "
        "(non-MCP) path — only the MCP orchestrator path writes `agent_call_log`."
    )

chains = verify_chains(mtime)
if "error" in chains:
    st.caption(f"chain verification unavailable: {chains['error']}")
else:
    for tbl, res in chains.items():
        if res.get("ok"):
            st.success(f"`{tbl}` hash chain intact — {res['n']} rows.")
        else:
            st.error(f"`{tbl}` hash chain BROKEN at id {res.get('id')}: {res.get('reason')}")

# -------------------------------------------------------------- C. Recent History
st.header("Recent history")
hist = load_history(mtime, 20)
if hist.empty:
    st.write("_No prior decisions._")
else:
    show = hist.copy()
    show["generated_at"] = show["generated_at"].map(_fmt_ts)
    st.dataframe(
        show.rename(columns={
            "as_of_date": "as of", "generated_at": "generated",
            "reliability_tier": "tier", "directional_lean": "lean",
            "regime_pit_id": "regime", "trading_days_since_fit": "td since fit",
        }),
        hide_index=True, width="stretch",
    )

st.caption(
    f"snapshot {datetime.fromtimestamp(mtime, timezone.utc):%Y-%m-%d %H:%M UTC} "
    f"· auto-refreshes every 5 min · research project — operational face only, not a product"
)
