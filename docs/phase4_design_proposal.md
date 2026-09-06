# Phase 4 Design Proposal — Confidence-Aware Abstention

Status: **finalized after two review rounds (2026-09-06). Cleared for implementation.**
Covers proposal §4 ("Confidence-Aware, Abstention-Capable Prediction") and the
recalibration design §3 (staleness/drift as a confidence input, plus a hard
circuit-breaker), approved in Phase 2 but never wired into anything live.

Build order: (1) drift-calibration spike for the M2 circuit breaker, (2) `signal_model_versions`
registry + `fit_and_register_signal.py`, (3) `decide_todays_call()`. Real output reviewed
before any piece is considered done.

Everything below is anchored to verified Phase 2/3 output, not designed in the abstract.
The grounding facts are collected first so every later choice can point back to one.

### Decisions locked in the 2026-09-06 review

1. **The system never emits an actionable direction call this phase.** The favourable
   regime (regime_2) is reframed as **informational / monitored only**: the system may
   report "regime_2 shows a statistically significant accuracy pattern, not yet a
   demonstrated P&L edge," and may surface the model's directional *lean* explicitly
   marked non-actionable — it must never output anything that reads as "take this trade."
   The top-level decision states are renamed so this is unambiguous without reading a
   caveat field (§1, §2, §4). This is reversible: once the costed-P&L backlog item is
   done, if a real edge is found, loosening this is a well-evidenced next step.
2. **Circuit-breaker threshold is not settled here.** Only two Phase-2 windows
   (2016-12-31, 2020-06-30) ever had the standardization-drift metric computed (0.827 and
   0.327). Placing a hard-breaker number requires running that metric on the other
   known-bad windows first — a Phase-4 calibration prerequisite, not a pick between two
   points (§2 M2, §5).
3. **M4 staleness stays at 126 td / 63 td + drift** — it is the actual Phase-2-tested
   boundary. No change.
4. **M1 drops the `regime == regime_2` self-reference** in favour of an independent stress
   proxy (trailing VIX percentile or tier-1 drift magnitude) (§2 M1).
5. **No on-demand model refitting.** The directional lean comes from a **registered,
   versioned** signal-model version, created through the same reviewed registration path
   as regime recalibration — never fit inside the "ask for today's call" invocation (§5).

---

## 0. What is actually verified (the only basis Phase 4 is allowed to use)

### 0.1 The production regime model

JM, k=3, λ=50 (`jm_grid.csv`, `regime_db` model_versions). One global fit = `model_version 1`
(hindsight); the 63-cutoff quarterly walk = `model_versions 2–64` (point-in-time).

- 31 state switches over 15 years, mean self-persistence 0.992, **median run length 89.5
  trading days, minimum historical run length 23 td** (`jm_grid.csv`, k=3/λ=50 row).
- State characters, from the per-state mean daily return in `quarterly_alignment_check.csv`
  (identity-aligned across 57/61 consecutive quarterly fits, and **every** transition from
  2014-06-30 onward is identity-aligned — verified, `docs/phase3_working_notes.md`):
  - **regime_0 — calm uptrend.** mean ≈ +0.12 % to +0.14 %/day. Largest state (~1753/3900).
  - **regime_1 — volatile rally.** mean ≈ 0 to +0.25 %/day, smallest state (~415/3900),
    most variable centre across fits.
  - **regime_2 — risk-off / drawdown.** mean ≈ −0.21 %/day (~1732/3900).
- Regime labelling is **not walk-forward stable by the project's own pre-registered bar**:
  0/6 configs passed (`rolling_window_stability.json`). Interior ARI frequently 0.4–0.7;
  **JM k=3 edge ARI during the COVID cutoff = −0.08** (worse than chance). BUT: state
  *identity* held up far better — 22/24 (91.7 %) of interpretability checks passed
  full-sample; the two clearest stress windows (2018, 2020) were correctly characterised
  by every single config.
- Short-gap follow-up: at a 3-month / 6-month recalibration cadence, **interior ARI is
  0.978 / 0.965** — interior drift is essentially solved at a quarterly floor. Every
  residual failure is an **edge-zone** failure, and every failing edge zone sits inside an
  unusually extreme stretch (COVID, 2022). Cadence cannot fix edge-zone risk during a live
  crisis — that is explicitly what the confidence/abstention layer is for
  (`docs/phase2_working_notes.md`, "short-gap follow-up").
- Point-in-time vs. hindsight regime labels disagree materially: **ARI 0.590** on the 2,880
  evaluation dates.

### 0.2 The signal model's regime-stratified performance (the confidence basis)

`signal_model/results/model_regime_stratified_pit.csv` (point-in-time regime labels — the
verified headline, `significance_bh_pit.csv`):

| regime (character) | n | model accuracy | model mean P&L (bps/day) | standalone P&L p-value |
|---|---|---|---|---|
| aggregate | 2880 | 33.5 % | −0.72 | 0.533 (n.s.) |
| regime_0 — calm uptrend | 1394 | **30.1 %** | −0.42 | 0.443 (n.s.) |
| regime_1 — volatile rally | 871 | 34.2 % | −1.86 | 0.541 (n.s.) |
| regime_2 — risk-off / drawdown | 615 | **40.2 %** | +0.20 | 0.848 (n.s.) |

- **Regime-permutation test (point-in-time labels): uncorrected p = 0.0005, and it SURVIVES
  Benjamini-Hochberg** (threshold 0.0125). This is the verified headline finding: regime
  membership — determined the way a live system would determine it — carries statistically
  significant information about *where this model's directional calls are more vs. less
  accurate*. Under hindsight labels the same test did **not** survive (p = 0.0275).
- The separation runs **calm uptrend (30.1 %) → volatile rally (34.2 %) → risk-off
  (40.2 %)**. The model's directional edge concentrates in the **risk-off / drawdown**
  regime, consistent with the target-design finding that the only exploitable structure in
  this data is volatility clustering (squared-return autocorr 0.184), which is
  concentrated in drawdowns.
- **No regime shows a standalone edge that is distinguishable from zero.** Every per-regime
  mean-P&L p-value is ≥ 0.44. regime_2's +0.20 bps/day has p = 0.848.
- The only significant *level* result in Phase 3 is model-vs-momentum-baseline **accuracy**:
  +4.13 pp, 95 % CI [1.0, 6.8], p = 0.008, survives BH (`significance_bh.csv`). Aggregate
  model P&L vs zero: p = 0.533.
- Sanity ceiling: "always predict up" scores 42.4 % on the same evaluation window. The
  model (33.5 %) and baseline (29.4 %) are both below it — `class_weight="balanced"` was
  deliberate; accuracy vs. macro-F1 vs. P&L answer different questions
  (`docs/phase3_working_notes.md`).
- No transaction costs or slippage anywhere in Phase 3.

### 0.3 The monitoring infrastructure (`regime_detection/monitoring.py`, already built)

- **Tier 1** (cheap, continuous, no refit): per-date Euclidean distance in standardized
  space between the live version's stored scaler and a fresh scaler on current data, over
  the last `WARM_UP_EDGE_DAYS = 90` td. `TIER1_DRIFT_THRESHOLD = 0.3`.
- **Tier 2** (only if tier 1 fires): shadow-fit the same config, compare its labels against
  the live version's own stored labels on shared interior dates, `TIER2_ARI_THRESHOLD =
  0.85`. Returns `confirmed = interior_ari < 0.85`.
- `check_recalibration_trigger()` returns `{tier1, tier2, recalibration_recommended}`.
- Known measured drift: 0.335 (model 6 years stale) vs. 0.161 (recent). Diagnostic
  reference points: 0.827 (2016 cutoff → interior ARI 0.44) and 0.327 (2020-06 cutoff →
  interior ARI 0.56). **Neither reference point is a passing configuration** — there is no
  validated "safe" drift level, which is why the 0.3 threshold sits just below the milder
  known-bad point.
- Gaps: tier 2 is **JM-only** (`NotImplementedError` for HMM); the quarterly walk has **no
  minimum-sample-size floor** (degenerate fit at n = 27 on the first cutoff — harmless to
  every result so far, logged as backlog).

---

## 1. Regime-conditioned confidence

### 1.1 What "confidence" means here: a **documented reliability tier**, not a calibrated probability, and never an actionable signal this phase

Recommendation: confidence is a **discrete, evidence-bound reliability tier per regime**,
carried with the human-readable reason it was assigned. It is **not** a calibrated
probability that the direction call is correct, it is **not** the LightGBM class
probability, and — per the review — **no tier makes the output actionable**. The best a
regime can currently earn is "watch this; the pattern is real but unpriced."

Reasoning — a calibrated probability would be false precision the verified results cannot
support:

1. **No regime has a P&L edge distinguishable from zero** (all p ≥ 0.44). A number like
   "62 % chance this call is correct / +X bps expected" would assert an edge the
   significance tests explicitly failed to find.
2. **Raw accuracy is below the majority-class baseline (42.4 %) in every regime.** A
   probability calibrated on 30–40 % 3-class accuracy would mostly encode "this is worse
   than guessing up," which is not a useful confidence signal to hand a downstream user.
3. **The regime label itself is uncertain.** PIT-vs-hindsight ARI is 0.590; walk-forward
   regime stability is 0/6. Any per-regime P(correct) is conditioned on a regime
   assignment that is itself provisional — stacking a calibrated probability on that is
   more precision than the pipeline earns.
4. **What *is* verified is ordinal and comparative**, not a level: regime membership
   significantly ranks where the model is more vs. less accurate (permutation p = 0.0005,
   survives BH). A tier captures exactly that and nothing more.

The LightGBM class scores are still **reported** in the output for transparency, but
labelled as raw model scores, never as calibrated confidence, and only inside a block
explicitly flagged non-actionable.

### 1.2 The tiers

- **`monitored`** (was "T2"). The model's directional calls in this regime are more
  accurate than the momentum baseline, and the regime-spread result establishes this
  regime as the favourable side of a statistically significant separation. **Currently:
  regime_2 (risk-off / drawdown) only.** This tier is **informational**: it authorises the
  system to *report* the pattern and to surface the model's directional lean as
  non-actionable context — it does **not** authorise a trade call. Every `monitored`
  output permanently carries the caveat: *higher directional accuracy than a momentum
  baseline (regime-spread p = 0.0005, BH-significant); NOT a demonstrated positive P&L
  edge (regime_2 standalone P&L p = 0.85); no transaction costs modelled.*
- **`none`** (was "T0"). No verified evidence the model's directional call carries
  information in this regime. **Currently: regime_0 (calm uptrend) and regime_1 (volatile
  rally).** Resolves to disposition `ABSTAIN`.
- **`suppressed`.** Assigned by the Phase-2 modifiers in §2/§3 regardless of the base tier
  — the regime call or the model inputs are currently untrustworthy, so the system will
  not even report the `monitored` pattern as currently-applicable. Resolves to `ABSTAIN`.

### 1.3 Tier assignment is reproducible, not hand-set

A regime is **`monitored`** iff, on the latest `run_pit_evaluation.py` output:

1. the regime-permutation test survives BH-FDR (currently p = 0.0005 ✓), **and**
2. that regime has the highest point-in-time OOF accuracy of the k regimes (regime_2:
   40.2 % ✓), **and**
3. its accuracy exceeds the weakest regime's by more than the permutation null's p95
   spread (i.e. the gap is not within chance range for this partition).

Otherwise the regime is **`none`**. If condition 1 fails on a future re-run, **no regime
is `monitored`** and the system abstains everywhere — that is the honest fallback, stated
up front. The tier table is regenerated whenever the signal model or the regime
infrastructure is re-run; it is a build artifact
(`signal_model/results/regime_reliability_tiers.csv`), not a constant baked into code.

### 1.4 What flips this to actionable later (not this phase)

The single blocker is proposal §4.5 / Phase 3's logged backlog item: **a costed P&L
series** (transaction costs + slippage). If, after that, a regime shows a P&L edge
distinguishable from zero under the same block-bootstrap + BH-FDR discipline, a new tier
above `monitored` can authorise an actionable call, and a calibrated per-regime P(correct)
becomes defensible to layer on. Until then the ceiling is `monitored`.

---

## 2. Abstention logic

Two dispositions: **`ABSTAIN`** and **`INFORMATIONAL`**. There is no actionable disposition
this phase (§1, review decision 1). `INFORMATIONAL` is reached **only** when the current
regime's base tier is `monitored` **and** none of the modifiers below fire; everything
else is `ABSTAIN`. Each modifier is grounded in a specific Phase 2/3 result and can only
downgrade (`monitored` → `suppressed` → `ABSTAIN`); none can raise a `none` regime.

| # | Trigger | Grounding | Action |
|---|---|---|---|
| **M1** | Today is inside the label edge zone (≤ 90 td after the active version's `fit_end_date`) **and** an **independent** stress proxy is elevated: trailing India-VIX percentile ≥ **80th** (expanding window, point-in-time) **or** tier-1 drift ≥ **0.3** | JM k=3 **edge ARI = −0.08** during the COVID cutoff — a freshly-computed edge-zone label is least trustworthy during stress; short-gap follow-up: every edge-zone failure sat inside an extreme stretch. Stress is judged from VIX / drift, **not** from the regime label itself (removes the self-reference flagged in review) | SUPPRESS |
| **M2** | Circuit breaker: tier-1 drift ≥ **0.80** (`BREAKER`, calibrated 2026-09-06 — see below), **or** (tier-1 fires ≥ 0.3 **and** tier-2 shadow-fit ARI < **0.5**) | ARI < 0.5 is the "barely better than chance" line used for k=5 refit instability in Phase 2. `BREAKER = 0.80` sits just below the one window whose interior ARI fell to near-chance (2016-12-31: ARI 0.44, drift 0.827) and above the highest drift on any *passing* window (0.728) — a blunt catastrophic-only trip, not a tuned boundary | SUPPRESS (hard, regardless of regime) + flag recalibration |
| **M3** | Drift confirmed: tier-1 fires (≥ 0.3) **and** tier-2 ARI < 0.85 (= `recalibration_recommended` from `check_recalibration_trigger`) | The approved two-tier design's own "recalibration for review" condition | SUPPRESS + flag recalibration for review |
| **M4** | Staleness: trading days since `fit_end_date` > **126** (two quarterly floors), **or** > 63 **and** tier-1 drift > 0.2 | Short-gap follow-up: interior ARI 0.978 at 3 mo, 0.965 at 6 mo — pure staleness inside ~2 quarters with low drift is tolerable; beyond that, or combined with rising drift, it is not | SUPPRESS + flag recalibration overdue |
| **M5** | Current regime run length < **20 td** | JM k=3 minimum historical run length is 23 td, median 89.5 — a run shorter than any the model has ever produced is very likely to be revised by the next refit | SUPPRESS |
| **M6** | Active regime version fit on < the minimum-sample floor, **or** today's standardized feature vector's distance to the nearest state centroid exceeds the in-sample interior maximum | Degenerate n=27 fit (Phase 3 backlog); the standardization-drift finding — an out-of-distribution day is one the fitted geometry cannot place | SUPPRESS |

### M2 circuit-breaker calibration — done 2026-09-06 (`signal_model/circuit_breaker_calibration.py`)

Ran `standardization_drift.py`'s metric on the windows it was never run on — the
2018-12-31 and 2022-12-31 rolling-window cutoffs (full-sample comparison) and all four
Phase-2 consecutive pairs, in particular 2020-06-30→2022-12-31. The two Phase-2 reference
points reproduced to the decimal (0.827, 0.327).

**Finding: drift is a weak separator.** The highest-drift *passing* window (2016→2018
consecutive, drift 0.728, interior ARI 0.983) sits just below the one near-chance
*failing* window (2016-12-31, drift 0.827, interior ARI 0.44); everything else clusters
in 0.26–0.36 where drift and interior ARI are uncorrelated (that band holds both clean
passes and fails, some fails at *lower* drift than the 2022 pass). Full table:
`docs/phase4_working_notes.md`, 2026-09-06 entry.

**`BREAKER = 0.80`** — just below the single near-chance point (0.827), above the highest
passing-window drift (0.728). It is a blunt "something is catastrophically wrong" trip,
not a tuned boundary; the ambiguous 0.3–0.75 middle is M3's job, and the calibration
empirically confirms tier-1 drift alone cannot adjudicate there — which is the whole
reason the two-tier structure exists. Same first-cut status as `TIER1_DRIFT_THRESHOLD =
0.3`; revisit with real operational readings. Not characterised for HMM.

### Notes

- **The edge zone (90 td) is longer than the quarterly recalibration interval (~63 td)**,
  so a live "today" label is *always* formally inside the current fit's edge zone. That is
  why M1 does **not** abstain on edge-zone alone (it would abstain 100 % of the time) — it
  fires only when the edge zone coincides with an independently-measured stress reading,
  where Phase 2 shows the label genuinely breaks down. In calm conditions the edge-zone
  label is carried with whatever base tier the regime already has.
- **Consequence, stated plainly:** the favourable regime is regime_2 (risk-off), and M1
  suppresses it whenever VIX/drift say we are actually in stress while still inside the
  edge zone. So the system reaches `INFORMATIONAL` — reporting the regime_2 accuracy
  pattern plus a non-actionable lean — **only during a persistent, already-established
  risk-off regime that has outlasted the edge zone and is not in an acute-stress
  reading**, never at the onset of a drawdown. Combined with decision 1 (no actionable
  output at all this phase), the live behaviour is: abstain by default, and at most say
  "you are in the regime we are watching; here is the historical pattern and the model's
  non-actionable lean." That is proposal §3's "say 'I don't know' and be right about it."
- **M1 keeps both triggers (VIX percentile OR tier-1 drift), either sufficient** (review
  round 2). They plausibly catch different failure shapes — a fast volatility shock shows
  up in the VIX percentile before the scaler stats move; a slower structural distribution
  shift shows up as tier-1 drift without a VIX spike. The project has consistently
  preferred caution over precision on abstention, so the redundancy is deliberate.
- M6's OOD centroid check and the minimum-sample floor both need a one-off empirical
  calibration when built (pick the floor from where k=3 stops leaving a state near-empty;
  take the OOD threshold from the in-sample interior distance distribution). The M1 VIX
  percentile (80th) is likewise a starting value, to confirm against the COVID / 2018 /
  2022 windows during calibration. Flagged, not guessed here.

---

## 3. Integration with Phase 2 monitoring — one combined decision, not two systems

There is a **single entry point** that produces one decision. Monitoring is not a
side-channel; its outputs are inputs to the same function that assigns the tier and
resolves the modifiers.

```
decide_todays_call(conn, as_of=None) -> RegimeGuardCall
  1. resolve active regime model_version   get_active_model_version(conn,"jm",3,50.0)
  2. current PIT regime + run length       predict-forward with the active version's
                                           FIXED params (quarterly_walk mechanism,
                                           factored into a helper), .transform() only
  3. monitoring snapshot                    check_recalibration_trigger(conn, version_id)
                                             -> tier1{mean,fires}, tier2{ari,confirmed},
                                                recalibration_recommended
  4. staleness                              as_of - fit_end_date, in trading days
  5. base tier                              lookup regime in regime_reliability_tiers.csv
                                             -> "monitored" | "none"
  6. apply M1..M6 using (2),(3),(4), the feature vector from (2), and the
     trailing VIX percentile at as_of        -> tier stays "monitored" or drops to "suppressed"
  7. disposition = INFORMATIONAL iff base tier == "monitored" and no modifier fired;
     else ABSTAIN.  (No actionable disposition exists this phase.)
  8. if INFORMATIONAL: attach the non-actionable directional lean from the registered
     signal-model version covering as_of (see §5); if none covers it, lean = null with
     reason "signal_model_stale" and disposition stays INFORMATIONAL (pattern still reported)
  9. assemble the record in §4, including every threshold and input used
```

Wiring specifics:

- **Tier-1 drift** feeds M2 (magnitude vs. `BREAKER`) and M4 (magnitude, 0.2 co-trigger
  with staleness). It is also reported raw in the output so a reviewer sees it rising
  before it trips anything.
- **Tier-2** is only computed when tier-1 fires (unchanged from the built code). Its ARI
  feeds M2 (< 0.5 → hard breaker) and M3 (< 0.85 → recalibration-for-review). M3 mirrors
  exactly what `check_recalibration_trigger` already returns as
  `recalibration_recommended`.
- **The hard circuit-breaker (M2)** is the only modifier that ignores the regime and the
  tier entirely — a severe-drift day produces `ABSTAIN` even in a well-established risk-off
  regime.
- **M1's stress proxy** (trailing VIX percentile) is computed from the India-VIX close
  series already in `build_feature_matrix`, as an expanding-window percentile rank at
  `as_of` — no lookahead, no dependence on the JM fit.
- **Staleness (M4)** is the time-since-recalibration input from recalibration design §3,
  now concrete: 63 td = one quarterly floor, 126 td = the point beyond which Phase 2 has
  no evidence of interior stability.
- **HMM gap:** the whole path is JM-only, matching `monitoring.py`'s existing
  `NotImplementedError`. Not closed in Phase 4.
- **Recalibration is flagged, never automatic.** M2/M3/M4 set a `recalibration` flag in
  the output for human review; Phase 4 does not swap the live model. Consistent with the
  project's human-in-the-loop stance.

---

## 4. What a single "ask for today's call" invocation returns

One call → one structured record (`RegimeGuardCall`). JSON-serialisable; this is also the
object the future signed audit trail (proposal §5.3, later phase) will store verbatim.

```jsonc
{
  "as_of_date": "2026-09-04",           // last trading day with a complete feature row
  "generated_at": "2026-09-06T09:12:00Z",

  "regime": {
    "point_in_time_id": 2,
    "label": "risk-off / drawdown",
    "active_model_version_id": 64,
    "model_fit_through": "2026-06-30",
    "run_length_trading_days": 41,
    "in_label_edge_zone": true,          // <= 90 td after model_fit_through
    "pit_vs_hindsight_ari_context": 0.59 // static, documented caveat, not recomputed
  },

  "monitoring": {
    "trading_days_since_fit": 46,
    "past_quarterly_floor": false,       // > 63 td
    "tier1_drift_mean": 0.18,
    "tier1_fires": false,                // >= 0.30
    "tier2_shadow_ari": null,            // only computed when tier1 fires
    "circuit_breaker": false,
    "recalibration_flag": null           // "recommended" | "overdue" | null
  },

  "actionable": false,                   // constant this phase; see §1.4 for what flips it

  "reliability_tier": "monitored",       // "monitored" | "none" | "suppressed"

  "disposition": "INFORMATIONAL",        // "INFORMATIONAL" | "ABSTAIN"  — never a trade call
                                         //   governed ONLY by the regime-side gates M1-M6;
                                         //   signal-model staleness never changes it

  "directional_lean": {                  // null when disposition == ABSTAIN, OR when no
                                         //   registered signal-model version covers as_of
    "note": "INFORMATIONAL ONLY — not a trade recommendation",
    "direction": "down",                 // up | flat | down  (+-20bps band)
    "signal_model_version_id": 7,        // a registered, versioned fit — never fit on demand
    "signal_model_fit_through": "2025-10-31",
    "raw_class_scores": {"up": 0.29, "flat": 0.22, "down": 0.49}, // raw LGBM, NOT calibrated
    "stale_reason": null                 // "signal_model_stale" -> the other lean fields are
                                         //   null; disposition is unaffected (see design points)
  },

  "abstention": {                        // null when disposition == INFORMATIONAL
    "reasons": ["M1_edge_zone_stress"],  // machine codes, may be several
    "plain_language": "In a risk-off regime, but the current regime model was refit only 46 trading days ago (still inside the 90-day edge zone) and trailing VIX is in its 88th percentile — the point-in-time regime label is unreliable under these conditions."
  },

  "regime_pattern": {                    // ALWAYS present, shape depends on reliability_tier
    "applies_now": false,                // true only on INFORMATIONAL; false if tier=="none"
                                         //   or a modifier suppressed a "monitored" regime
    // --- when reliability_tier is "monitored" (or "suppressed" from it): ---
    "statement": "regime_2 (risk-off) shows a statistically significant directional-accuracy pattern (regime-spread permutation p=0.0005, survives BH-FDR): 40.2% vs 30.1% in the weakest regime.",
    "not_an_edge": "regime_2 standalone mean P&L is +0.20 bps/day, NOT distinguishable from zero (p=0.85). No transaction costs modelled. This is not a demonstrated profitable edge."
    // --- when reliability_tier is "none": the two fields above are replaced by: ---
    // "statement": "no established directional-accuracy pattern for the current regime"
  },

  "caveats": [
    "the system emits no actionable direction call in any regime this phase (costed-P&L work is the blocker; see design §1.4)",
    "point-in-time regime labelling is not walk-forward stable by the project's pre-registered bar (0/6); the monitored-regime finding survives only because it is a regime-spread result, not a level",
    "point-in-time vs hindsight regime labels agree at ARI 0.59 on the evaluation window"
  ],

  "audit": {
    "thresholds": {"tier1_fire": 0.30, "circuit_breaker": 0.80,
                   "tier2_review": 0.85, "tier2_breaker": 0.50, "edge_zone_td": 90,
                   "quarterly_floor_td": 63, "staleness_hard_td": 126, "min_run_td": 20,
                   "m1_vix_percentile": 80},
    "feature_row_hash": "…",
    "regime_reliability_tiers_ref": "signal_model/results/regime_reliability_tiers.csv@<hash>",
    "code_rev": "<git sha>"
  }
}
```

Design points:

- **`actionable` is a hard-coded `false` this phase** and sits at the top level, so no
  consumer can mistake `INFORMATIONAL` for a trade signal. §1.4 states the single condition
  that flips it.
- **`disposition` is binary (`INFORMATIONAL` | `ABSTAIN`); `reliability_tier` is the "why".**
  There is deliberately no `PREDICT` value anywhere in the schema.
- **`disposition` is governed only by the regime-side gates (M1–M6).** A stale signal
  model is a separate uncertainty source: it nulls the `directional_lean` fields and sets
  `stale_reason = "signal_model_stale"`, but it does **not** flip an `INFORMATIONAL`
  regime read to `ABSTAIN` — collapsing the two would blur what the "why" fields are for.
  A stale signal model does not make the regime information untrustworthy.
- **`regime_pattern` is always present.** For a `monitored` regime it carries the
  `statement` + `not_an_edge` pair (even when a modifier suppressed it — `applies_now`
  then `false` — so a reviewer sees what was suppressed). For a `none` regime it carries
  only a short generic `statement` ("no established directional-accuracy pattern for the
  current regime") — regime_2's specific numbers are never repeated outside a
  `monitored`/`suppressed` context.
- **`abstention.reasons` carries machine codes** (`M1_edge_zone_stress`,
  `M2_circuit_breaker`, `none_no_reliability`, …) so the audit trail can answer "why did it
  abstain on date X" mechanically, per proposal §5.3.
- The record is returned *and* appended to the `todays_call_log` **DB table** (not a JSONL
  file), so it is joinable against `model_versions` / `signal_model_versions` for audit
  queries — consistent with `ingestion_runs`, `model_versions`, `regime_labels`. Signing /
  tamper-evidence is a later security phase.

---

## 5. Scope boundary for Phase 4

**In scope (what gets built after this review):**

1. `regime_reliability_tiers.csv` generator — derives the §1.3 tier table from
   `run_pit_evaluation.py` output. Emits `monitored` / `none` per regime, with the
   supporting numbers.
2. A point-in-time "regime for today" helper — factor the fit-fixed predict-forward step
   out of `quarterly_walk.py` so it can run against the active `model_version` for an
   arbitrary `as_of` date. Also the trailing-VIX-percentile helper for M1.
3. **A minimal versioned signal-model registry** (review decision 5) — a
   `signal_model_versions` table + stored per-date predictions, mirroring
   `regime_db.save_model_version` / `save_regime_labels`, and a `fit_and_register_signal.py`
   CLI that fits LightGBM through a given cutoff (honouring `PURGE`/`EMBARGO`) and stores
   its predictions. New model versions are created **only** by an explicit run of this CLI
   — the same reviewed, versioned discipline as regime recalibration. `decide_todays_call`
   *reads* the registered predictions; it never fits.
4. The combined `decide_todays_call()` function (§3) and the `RegimeGuardCall` record (§4),
   including appending each record to a plain `todays_call_log` table.
5. One-off empirical calibration, documented the same way every other threshold in this
   project is:
   - **M2 `BREAKER`** — run `standardization_drift.py` on the 2018 / 2022 / COVID→2022
     windows first (§2), then place the threshold per the stated rule.
   - **M5 min-run** (start 20 td) and **M6 min-sample floor + OOD centroid distance**.
   - **M1 VIX percentile** (start 80th) — confirm against the COVID / 2018 / 2022 windows.
6. A short CLI (`python -m signal_model.todays_call`) that prints the record — the
   single-invocation interface, not a service.

**Explicitly not in scope (later phases):**

- MCP agent wrapping, orchestration, tool-permission scoping (proposal §5.2–5.3).
- Signed / tamper-evident audit storage — Phase 4 emits the record and appends it to a
  table; cryptographic signing is the security phase.
- Live data ingestion cadence, deployment, dashboard (proposal §5.4).
- **Signal-model retraining cadence and the Model Training Agent** — Phase 4 builds the
  registry and one registered version; deciding *when* to register new ones on a schedule
  or a trigger is the later Model Training Agent's job. Phase 4 consumes whatever
  registered version currently covers `as_of`.
- Any actionable / trade-signal output, and any calibrated P(correct) — blocked on the
  costed-P&L backlog item (§1.4).
- HMM shadow-fit support in tier 2 (`monitoring.py` gap, unchanged).
- Closing the quarterly-walk minimum-sample-size backlog item beyond the M6 floor.

---

## 6. Review decisions — all resolved

**Round 1 (§1–§5, folded in):** (1) no actionable output this phase — `monitored` is
informational-only, `disposition ∈ {INFORMATIONAL, ABSTAIN}`, no `PREDICT` in the schema;
(2) M2 `BREAKER` left unset pending a drift-calibration spike; (3) M4 staleness unchanged
(126 td / 63 td + drift); (4) M1 uses an independent stress proxy, not the regime label;
(5) no on-demand refit — a versioned signal-model registry.

**Round 2:**

1. **`regime_pattern` is always present**, but a `none` regime gets only a short generic
   `statement` ("no established directional-accuracy pattern for the current regime") —
   regime_2's specific numbers are never repeated outside a `monitored`/`suppressed`
   context (§4).
2. **M1 keeps both triggers** (trailing VIX percentile ≥ 80th **or** tier-1 drift ≥ 0.3),
   either sufficient — caution over precision, and the two catch different failure shapes
   (fast shock vs. slower structural drift) (§2 notes).
3. **`signal_model_stale` is a flag on `directional_lean` only**, kept conceptually
   separate from regime-model health. `disposition` is governed solely by M1–M6; a stale
   signal model nulls the lean fields with `stale_reason = "signal_model_stale"` and does
   not force `ABSTAIN` (§4 design points).
4. **`todays_call_log` is a DB table**, joinable against `model_versions` /
   `signal_model_versions` for audit queries — consistent with every other audit-trail
   table in the project (§4).

## 7. Implementation order (this phase)

1. **Drift-calibration spike** — ✅ done 2026-09-06. `signal_model/circuit_breaker_calibration.py`,
   results in `signal_model/results/circuit_breaker_calibration.csv`, rationale in
   `docs/phase4_working_notes.md`. Outcome: drift is a weak separator; `BREAKER = 0.80` as
   a blunt catastrophic-only trip.
2. **`signal_model_versions` registry + `fit_and_register_signal.py`** — ✅ done 2026-09-06.
   `signal_model/registry.py` + `signal_model/fit_and_register_signal.py`, verified on an
   isolated DB copy (schema, per-key version supersede, per-date prediction supersede,
   `current_signal_predictions`, `load_signal_prediction` → `None` == the stale signal,
   live path, guards). Details: `docs/phase4_working_notes.md`, step 2 entry.
3. **`decide_todays_call()` + `RegimeGuardCall` + `todays_call_log`** — ✅ done 2026-09-06.
   `signal_model/regime_reliability_tiers.py`, `point_in_time_regime_label()` in
   `regime_detection/quarterly_walk.py`, `signal_model/todays_call.py`. Every disposition
   branch and M1/M4/M5/M6 exercised on an isolated DB copy; M2/M3 (tier-2-driven) verified
   by predicate — no catastrophic-drift event exists in the data to trigger them live.
   Two limitations logged: historical-replay monitoring magnitudes are not point-in-time
   (`compute_tier1_drift` is always "vs today"; the live path and M1's VIX clause are
   correct), and tier-2 can't run against a superseded regime version. Details:
   `docs/phase4_working_notes.md`, step 3 entry.

Real output reviewed before any piece is considered done. No commits or pushes.
