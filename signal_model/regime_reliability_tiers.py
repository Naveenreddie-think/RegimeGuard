"""Phase 4 design §1.3: derive the per-regime reliability tier table from the
point-in-time regime-stratified evaluation - a reproducible build artifact, not a
constant baked into code.

A regime is `monitored` iff ALL of:
  1. the regime-permutation test survives Benjamini-Hochberg FDR correction
     (the verified Phase 3 headline: PIT p = 0.0005), AND
  2. it has the highest point-in-time OOF accuracy of the k regimes, AND
  3. its accuracy lead over the weakest regime exceeds the permutation null's p95
     spread (the gap is not within chance range for this partition).
Otherwise the regime is `none`. If condition 1 fails, NO regime is `monitored` and
`decide_todays_call` abstains everywhere - the honest fallback.

Reuses the exact out-of-fold predictions from `signal_model/results/oof_predictions.csv`
and the point-in-time regime labels from `regime_detection.quarterly_walk`, same as
`run_pit_evaluation.py` - only the output (a tier table) differs.

Output: `signal_model/results/regime_reliability_tiers.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_agent.db import get_connection
from regime_detection.quarterly_walk import load_point_in_time_labels
from signal_model import significance
from signal_model.run_pit_evaluation import QUARTERLY_VERSION_ID_RANGE
from signal_model.run_signal_model import RESULTS_DIR

TIERS_CSV = RESULTS_DIR / "regime_reliability_tiers.csv"

# Human-readable character per regime id, from the per-state mean daily return in
# regime_detection/results/quarterly_alignment_check.csv (identity-aligned across
# every quarterly fit from 2014-06 on - see docs/phase3_working_notes.md). sort_by
# "cumret": state 0 = highest mean return, state 2 = lowest.
REGIME_CHARACTER = {
    0: "calm uptrend",
    1: "volatile rally",
    2: "risk-off / drawdown",
}


def compute_tiers(conn) -> pd.DataFrame:
    oof = pd.read_csv(RESULTS_DIR / "oof_predictions.csv", index_col=0, parse_dates=True)
    y_true, oof_pred, fwd_ret = oof["y_true"], oof["oof_pred"], oof["fwd_ret"]

    pit_regime = load_point_in_time_labels(conn, list(QUARTERLY_VERSION_ID_RANGE)).reindex(oof_pred.index)
    correct = (oof_pred == y_true).astype(float)
    pnl = oof_pred * fwd_ret

    # per-regime accuracy on the evaluation window
    acc = correct.groupby(pit_regime).mean()
    n = pit_regime.value_counts().sort_index()

    # condition 1: regime-permutation test survives BH across the PIT test family
    perm = significance.regime_permutation_test(pit_regime, correct)
    p_values = {"regime_spread_pit": perm["p_value"]}
    for reg in sorted(pit_regime.dropna().unique()):
        mask = pit_regime == reg
        if mask.sum() < 30:
            continue
        p_values[f"regime_{int(reg)}_mean_pnl_pit"] = significance.block_bootstrap_metric(
            pnl[mask], np.mean
        )["p_value"]
    bh = significance.benjamini_hochberg(p_values)
    perm_survives_bh = bool(bh.loc[bh["test"] == "regime_spread_pit", "reject_null"].iloc[0])

    # condition 2 + 3
    best_regime = int(acc.idxmax())
    acc_lead = float(acc.max() - acc.min())  # == perm["observed_spread"] by construction
    lead_exceeds_null_p95 = acc_lead > perm["null_p95"]

    rows = []
    for reg in sorted(pit_regime.dropna().unique()):
        reg = int(reg)
        is_monitored = (
            perm_survives_bh and reg == best_regime and lead_exceeds_null_p95
        )
        rows.append({
            "regime": reg,
            "character": REGIME_CHARACTER.get(reg, "unknown"),
            "n_eval": int(n.get(reg, 0)),
            "pit_oof_accuracy": round(float(acc.get(reg, float("nan"))), 4),
            "is_highest_accuracy": reg == best_regime,
            "perm_p_value": perm["p_value"],
            "perm_survives_bh": perm_survives_bh,
            "acc_lead_over_weakest": round(acc_lead, 4),
            "perm_null_p95_spread": round(float(perm["null_p95"]), 4),
            "lead_exceeds_null_p95": lead_exceeds_null_p95,
            "tier": "monitored" if is_monitored else "none",
        })
    return pd.DataFrame(rows)


def load_reliability_tiers(path: Path = TIERS_CSV) -> dict[int, str]:
    """{regime_id: 'monitored'|'none'} - the read side used by decide_todays_call."""
    df = pd.read_csv(path)
    return dict(zip(df["regime"].astype(int), df["tier"]))


def main() -> None:
    conn = get_connection()
    try:
        tiers = compute_tiers(conn)
    finally:
        conn.close()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tiers.to_csv(TIERS_CSV, index=False)
    print(tiers.to_string(index=False))
    print(f"\nWritten to {TIERS_CSV}")
    monitored = tiers.loc[tiers["tier"] == "monitored", "regime"].tolist()
    print(f"\nmonitored regimes: {monitored or '(none - system abstains everywhere)'}")


if __name__ == "__main__":
    main()
