"""Phase 4 design §3/§4: the single combined confidence + abstention decision.

`decide_todays_call(conn, as_of)` produces one `RegimeGuardCall` record - the current
point-in-time regime, a reliability tier, and either an INFORMATIONAL read (regime
pattern + a non-actionable directional lean) or an ABSTAIN, with machine-readable
reasons. It NEVER emits an actionable trade call this phase (design decision 1) - the
top-level `actionable` field is a hard-coded `False`; §1.4 of the design states the
single condition (a costed-P&L edge) that would ever flip it.

Monitoring is not a side system: `check_recalibration_trigger` (Phase 2) feeds the
same function that assigns the tier, via modifiers M1-M6. Each modifier can only
downgrade (`monitored` -> `suppressed` -> ABSTAIN); none can raise a `none` regime.

The signal model is read, never fit here: the directional lean comes from a
registered `signal_model_version`; if none covers `as_of`, the lean is null with
`stale_reason="signal_model_stale"` and - per design decision 3 - the disposition is
unaffected (it is governed solely by the regime-side gates).

Every record is appended to `todays_call_log`, joinable against `model_versions` and
`signal_model_versions` for audit queries.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_agent.db import get_connection
from regime_detection.features import build_feature_matrix
from regime_detection.monitoring import TIER1_DRIFT_THRESHOLD, TIER2_ARI_THRESHOLD, check_recalibration_trigger
from regime_detection.quarterly_walk import point_in_time_regime_label
from regime_detection.regime_db import get_active_model_version, load_model_version
from regime_detection.rolling_window_stability import WARM_UP_EDGE_DAYS
from signal_model.regime_reliability_tiers import REGIME_CHARACTER, TIERS_CSV
from signal_model.registry import load_signal_prediction
from signal_model.target import FLAT_THRESHOLD, HORIZON_DAYS

# --- thresholds (all first-cut, each grounded in a Phase 2/3 result; see the design
#     doc and docs/phase4_working_notes.md - revisit with real operational data) ---
EDGE_ZONE_TD = WARM_UP_EDGE_DAYS            # 90 - Phase 2 regime-edge-instability zone
M1_VIX_PERCENTILE = 0.80                    # trailing expanding-window VIX rank, stress proxy
BREAKER = 0.80                              # M2 hard circuit-breaker (calibrated 2026-09-06)
TIER2_BREAKER_ARI = 0.50                    # "barely better than chance" (Phase 2)
QUARTERLY_FLOOR_TD = 63                     # one recalibration floor
STALENESS_HARD_TD = 126                     # two floors - last Phase-2-evidenced stable point
STALENESS_DRIFT_COTRIGGER = 0.20            # drift level that makes 1-2 quarters stale unsafe
MIN_RUN_TD = 20                             # < JM k=3 min historical run length (23 td)
MIN_REGIME_FIT_ROWS = 250                   # 2.5x the empirical k=3 degeneracy ceiling (~100)

DIRECTION_WORD = {-1: "down", 0: "flat", 1: "up"}
LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS todays_call_log (
    id INTEGER PRIMARY KEY,
    as_of_date TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    disposition TEXT NOT NULL,
    reliability_tier TEXT NOT NULL,
    actionable INTEGER NOT NULL,
    regime_pit_id INTEGER,
    regime_model_version_id INTEGER REFERENCES model_versions(id),
    signal_model_version_id INTEGER REFERENCES signal_model_versions(id),
    directional_lean TEXT,
    abstention_reasons TEXT,
    tier1_drift_mean REAL,
    trading_days_since_fit INTEGER,
    record_json TEXT NOT NULL,
    code_rev TEXT
);
"""


def ensure_log_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(LOG_SCHEMA)


@dataclass
class RegimeGuardCall:
    as_of_date: str
    generated_at: str
    regime: dict
    monitoring: dict
    actionable: bool
    reliability_tier: str
    disposition: str
    regime_pattern: dict
    directional_lean: dict | None
    abstention: dict | None
    caveats: list
    audit: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _code_rev() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        return None


def _trailing_vix_percentile(conn, as_of: pd.Timestamp) -> float:
    vix = build_feature_matrix(conn)["vix_close"]
    trailing = vix.loc[:as_of]
    return float((trailing <= trailing.iloc[-1]).mean())


def _trading_days_between(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int(((index > start) & (index <= end)).sum())


def _monitored_pattern_text(tiers_df: pd.DataFrame, regime_id: int) -> dict:
    row = tiers_df.loc[tiers_df["regime"] == regime_id].iloc[0]
    weakest = tiers_df["pit_oof_accuracy"].min()
    return {
        "statement": (
            f"regime_{regime_id} ({row['character']}) shows a statistically significant "
            f"directional-accuracy pattern (regime-spread permutation p={row['perm_p_value']}, "
            f"survives BH-FDR): {row['pit_oof_accuracy']:.1%} point-in-time OOF accuracy vs "
            f"{weakest:.1%} in the weakest regime (lead {row['acc_lead_over_weakest']:.3f} > "
            f"null p95 spread {row['perm_null_p95_spread']:.3f})."
        ),
        "not_an_edge": (
            "This is an accuracy pattern only. No regime's standalone mean P&L is "
            "distinguishable from zero (every per-regime p >= 0.44; risk-off p = 0.85), "
            "and no transaction costs are modelled. Not a demonstrated profitable edge."
        ),
    }


def decide_todays_call(
    conn: sqlite3.Connection,
    as_of: date | None = None,
    regime_model_version_id: int | None = None,
    log: bool = True,
) -> RegimeGuardCall:
    """Combined confidence/abstention decision for `as_of` (default: latest feature row).

    `regime_model_version_id` overrides which registered regime version is treated as
    active - default (None) uses `get_active_model_version` (the live path). Passing a
    historical version reconstructs "what a live system would have decided then", for
    testing and for after-the-fact audit.

    `log=True` (default) appends the record to `todays_call_log` - every real call is
    logged; pass `log=False` for dry runs.
    """
    feat = build_feature_matrix(conn)
    dates = feat.loc[~feat["warm_up"]].index
    as_of_ts = dates[-1] if as_of is None else pd.Timestamp(as_of)
    eligible = dates[dates <= as_of_ts]
    if len(eligible) == 0:
        raise ValueError(f"as_of {as_of_ts.date()} precedes the first feature row {dates[0].date()}")
    as_of_ts = eligible[-1]

    if regime_model_version_id is None:
        mv = get_active_model_version(conn, "jm", 3, 50.0)
        if mv is None:
            raise RuntimeError("no active jm k=3 lambda=50 model_version - run fit_and_register.py first")
    else:
        mv = load_model_version(conn, regime_model_version_id)

    # --- (2) point-in-time regime + run length + OOD inputs ---
    reg = point_in_time_regime_label(conn, mv, as_of_ts)
    regime_id = reg["regime"]
    fit_end = pd.Timestamp(mv["fit_end_date"])
    td_since_fit = _trading_days_between(dates, fit_end, as_of_ts)
    in_edge_zone = 0 <= td_since_fit <= EDGE_ZONE_TD

    # --- (3) monitoring snapshot ---
    trig = check_recalibration_trigger(conn, mv["id"])
    tier1 = trig["tier1"]
    tier2 = trig.get("tier2")
    tier2_ari = tier2.get("interior_ari") if tier2 else None
    tier1_mean = float(tier1["mean_drift"])
    tier1_fires = bool(tier1["fires"])

    vix_pct = _trailing_vix_percentile(conn, as_of_ts)

    # --- (5) base tier ---
    tiers_df = pd.read_csv(TIERS_CSV)
    base_tier = dict(zip(tiers_df["regime"].astype(int), tiers_df["tier"])).get(regime_id, "none")

    # --- (6) modifiers M1-M6 (downgrade-only) ---
    modifiers: list[tuple[str, str]] = []
    if in_edge_zone and (vix_pct >= M1_VIX_PERCENTILE or tier1_mean >= TIER1_DRIFT_THRESHOLD):
        modifiers.append((
            "M1_edge_zone_stress",
            f"regime label is inside the {EDGE_ZONE_TD}-td edge zone ({td_since_fit} td since fit) "
            f"and stress is elevated (trailing VIX pct {vix_pct:.2f}, tier-1 drift {tier1_mean:.3f}) "
            f"- point-in-time regime labels are unreliable here (COVID edge ARI was -0.08).",
        ))
    breaker = tier1_mean >= BREAKER or (tier1_fires and tier2_ari is not None and tier2_ari < TIER2_BREAKER_ARI)
    if breaker:
        cause = (f"tier-1 drift {tier1_mean:.3f} >= {BREAKER}" if tier1_mean >= BREAKER
                 else f"tier-1 drift fired and tier-2 shadow-fit ARI {tier2_ari:.3f} < {TIER2_BREAKER_ARI}")
        modifiers.append((
            "M2_circuit_breaker",
            f"severe standardization drift ({cause}) - hard abstain regardless of regime.",
        ))
    drift_confirmed = tier1_fires and tier2_ari is not None and tier2_ari < TIER2_ARI_THRESHOLD
    if drift_confirmed and not breaker:
        modifiers.append((
            "M3_drift_confirmed",
            f"tier-1 drift fired ({tier1_mean:.3f}) and tier-2 shadow-fit ARI {tier2_ari:.3f} "
            f"< {TIER2_ARI_THRESHOLD} - recalibration recommended for review.",
        ))
    if td_since_fit > STALENESS_HARD_TD or (td_since_fit > QUARTERLY_FLOOR_TD and tier1_mean > STALENESS_DRIFT_COTRIGGER):
        modifiers.append((
            "M4_stale",
            f"regime model is {td_since_fit} td past its fit end "
            f"(hard limit {STALENESS_HARD_TD}; {QUARTERLY_FLOOR_TD}+drift co-trigger at "
            f"{STALENESS_DRIFT_COTRIGGER}, drift {tier1_mean:.3f}) - recalibration overdue.",
        ))
    if reg["run_length_td"] < MIN_RUN_TD:
        modifiers.append((
            "M5_fresh_regime",
            f"current regime run is {reg['run_length_td']} td, shorter than any JM k=3 historical "
            f"run (min 23 td) - the label is likely to be revised by the next refit.",
        ))
    thin_fit = reg["n_fit_rows"] < MIN_REGIME_FIT_ROWS
    if thin_fit or reg["is_ood"]:
        why = []
        if thin_fit:
            why.append(f"active regime fit has only {reg['n_fit_rows']} rows (floor {MIN_REGIME_FIT_ROWS})")
        if reg["is_ood"]:
            why.append(f"today is out-of-distribution (dist-to-nearest-centroid "
                       f"{reg['min_dist_to_centroid']:.2f} > in-sample interior max {reg['ood_threshold']:.2f})")
        modifiers.append(("M6_ood_or_thin_fit", " and ".join(why) + "."))

    # --- (7) tier + disposition ---
    if base_tier == "monitored" and not modifiers:
        reliability_tier, disposition = "monitored", "INFORMATIONAL"
    elif base_tier == "monitored":
        reliability_tier, disposition = "suppressed", "ABSTAIN"
    else:
        reliability_tier, disposition = "none", "ABSTAIN"

    # --- regime_pattern (always present) ---
    if reliability_tier in ("monitored", "suppressed"):
        pat = _monitored_pattern_text(tiers_df, regime_id)
        regime_pattern = {"applies_now": disposition == "INFORMATIONAL", **pat}
    else:
        regime_pattern = {
            "applies_now": False,
            "statement": "no established directional-accuracy pattern for the current regime",
        }

    # --- (8) directional lean (INFORMATIONAL only; signal staleness never changes disposition) ---
    directional_lean = None
    if disposition == "INFORMATIONAL":
        pred = load_signal_prediction(conn, as_of_ts.date())
        if pred is None:
            directional_lean = {
                "note": "INFORMATIONAL ONLY - not a trade recommendation",
                "direction": None, "signal_model_version_id": None,
                "raw_class_scores": None, "stale_reason": "signal_model_stale",
            }
        else:
            directional_lean = {
                "note": "INFORMATIONAL ONLY - not a trade recommendation",
                "direction": DIRECTION_WORD[int(pred["pred_direction"])],
                "signal_model_version_id": int(pred["signal_model_version_id"]),
                "signal_model_as_of": pred["version_as_of_date"],
                "raw_class_scores": {
                    "up": round(float(pred["prob_up"]), 4),
                    "flat": round(float(pred["prob_flat"]), 4),
                    "down": round(float(pred["prob_down"]), 4),
                },
                "stale_reason": None,
            }

    # --- abstention block ---
    abstention = None
    if disposition == "ABSTAIN":
        reasons = [c for c, _ in modifiers]
        if base_tier != "monitored":
            reasons = ["none_no_reliability"] + reasons
        abstention = {
            "reasons": reasons,
            "plain_language": " ".join(t for _, t in modifiers) or
            (f"current regime is regime_{regime_id} ({REGIME_CHARACTER.get(regime_id, '?')}), "
             f"which showed no directional-accuracy pattern under point-in-time validation."),
        }

    recal_flag = None
    if breaker or drift_confirmed:
        recal_flag = "recommended"
    elif any(c == "M4_stale" for c, _ in modifiers):
        recal_flag = "overdue"

    record = RegimeGuardCall(
        as_of_date=as_of_ts.date().isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        regime={
            "point_in_time_id": regime_id,
            "character": REGIME_CHARACTER.get(regime_id, "unknown"),
            "active_model_version_id": mv["id"],
            "model_fit_through": mv["fit_end_date"],
            "run_length_trading_days": reg["run_length_td"],
            "label_source": reg["source"],
            "fit_rows": reg["n_fit_rows"],
            "in_label_edge_zone": in_edge_zone,
        },
        monitoring={
            "trading_days_since_fit": td_since_fit,
            "past_quarterly_floor": td_since_fit > QUARTERLY_FLOOR_TD,
            "tier1_drift_mean": round(tier1_mean, 4),
            "tier1_fires": tier1_fires,
            "tier2_shadow_ari": round(tier2_ari, 4) if tier2_ari is not None else None,
            "trailing_vix_percentile": round(vix_pct, 4),
            "circuit_breaker": bool(breaker),
            "recalibration_flag": recal_flag,
        },
        actionable=False,
        reliability_tier=reliability_tier,
        disposition=disposition,
        regime_pattern=regime_pattern,
        directional_lean=directional_lean,
        abstention=abstention,
        caveats=[
            "no actionable direction call is emitted in any regime this phase (blocked on a costed-P&L series; design §1.4)",
            "point-in-time regime labelling is not walk-forward stable by the project's pre-registered bar (0/6); the monitored-regime finding survives only because it is a regime-spread result, not a level",
            "point-in-time vs hindsight regime labels agree at ARI 0.59 on the evaluation window",
        ],
        audit={
            "thresholds": {
                "edge_zone_td": EDGE_ZONE_TD, "m1_vix_percentile": M1_VIX_PERCENTILE,
                "tier1_fire": TIER1_DRIFT_THRESHOLD, "circuit_breaker": BREAKER,
                "tier2_breaker_ari": TIER2_BREAKER_ARI, "tier2_review_ari": TIER2_ARI_THRESHOLD,
                "quarterly_floor_td": QUARTERLY_FLOOR_TD, "staleness_hard_td": STALENESS_HARD_TD,
                "staleness_drift_cotrigger": STALENESS_DRIFT_COTRIGGER, "min_run_td": MIN_RUN_TD,
                "min_regime_fit_rows": MIN_REGIME_FIT_ROWS,
            },
            "target": {"horizon_days": HORIZON_DAYS, "flat_bps": FLAT_THRESHOLD * 10000},
            "regime_ood": {
                "min_dist_to_centroid": round(reg["min_dist_to_centroid"], 4),
                "in_sample_max": round(reg["ood_threshold"], 4),
                "is_ood": reg["is_ood"],
            },
            "regime_reliability_tiers_ref": str(TIERS_CSV.name),
            "code_rev": _code_rev(),
        },
    )
    if log:
        record.audit["log_id"] = append_to_log(conn, record)
    return record


def append_to_log(conn: sqlite3.Connection, record: RegimeGuardCall) -> int:
    ensure_log_schema(conn)
    d = record.to_dict()
    lean = d["directional_lean"]["direction"] if d["directional_lean"] else None
    cur = conn.execute(
        """
        INSERT INTO todays_call_log
            (as_of_date, generated_at, disposition, reliability_tier, actionable,
             regime_pit_id, regime_model_version_id, signal_model_version_id,
             directional_lean, abstention_reasons, tier1_drift_mean, trading_days_since_fit,
             record_json, code_rev)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d["as_of_date"], d["generated_at"], d["disposition"], d["reliability_tier"],
            int(d["actionable"]), d["regime"]["point_in_time_id"], d["regime"]["active_model_version_id"],
            (d["directional_lean"] or {}).get("signal_model_version_id"),
            lean, json.dumps(d["abstention"]["reasons"]) if d["abstention"] else None,
            d["monitoring"]["tier1_drift_mean"], d["monitoring"]["trading_days_since_fit"],
            json.dumps(d), d["audit"]["code_rev"],
        ),
    )
    conn.commit()
    return cur.lastrowid


def main() -> None:
    parser = argparse.ArgumentParser(description="RegimeGuard - today's call (confidence + abstention)")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--regime-version-id", type=int, default=None,
                        help="Override active regime model_version (for historical replay / audit).")
    parser.add_argument("--no-log", action="store_true", help="Do not append to todays_call_log.")
    args = parser.parse_args()

    conn = get_connection()
    try:
        record = decide_todays_call(conn, args.as_of, args.regime_version_id, log=not args.no_log)
    finally:
        conn.close()
    print(json.dumps(record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
