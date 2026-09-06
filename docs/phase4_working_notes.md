# Phase 4 Working Notes — Confidence-Aware Abstention

Running, dated, append-only evidence log for Phase 4, same purpose and standard as
`docs/phase2_working_notes.md` / `docs/phase3_working_notes.md`: raw material for the
eventual `FINDINGS.md`, not the polished writeup. The approved design is
`docs/phase4_design_proposal.md`; this logs what actually happened building it.

---

## 2026-09-06 — Step 1: M2 circuit-breaker calibration — drift is a WEAK separator, `BREAKER` set coarse at 0.80

Script: `signal_model/circuit_breaker_calibration.py`. Full output:
`signal_model/results/circuit_breaker_calibration.csv`. Reuses
`standardization_drift.py`'s exact metric (per-date Euclidean distance in standardized
space between two clip+scale pipelines, interior dates only, clustering never runs) —
no new metric invented.

### Reproduction check passed

The two Phase-2 reference points reproduce to the decimal: 2016-12-31 mean drift
**0.8270** (Phase 2: 0.827), 2020-06-30 mean drift **0.3266** (Phase 2: 0.327). The
metric and the interior-date split are being computed the same way as the original
diagnostic.

### The new points — and the finding: drift does not cleanly separate pass from fail

| comparison | window | JM k=3 interior ARI (Phase 2) | mean drift | note |
|---|---|---|---|---|
| full-sample | 2016-12-31 | **0.438** (near-chance) | **0.827** | Phase 2 |
| consecutive | 2016-12-31 → 2018-12-31 | 0.983 (PASS) | **0.728** | NEW |
| full-sample | 2018-12-31 | 0.588 (fail) | 0.291 | NEW |
| consecutive | 2018-12-31 → 2020-06-30 | 0.953 (PASS) | 0.260 | NEW |
| full-sample | 2020-06-30 | 0.559 (fail) | 0.327 | Phase 2 |
| consecutive | 2020-06-30 → 2022-12-31 | 0.548 (fail) | 0.351 | NEW (design-flagged) |
| full-sample | 2022-12-31 | 0.969 (PASS) | 0.358 | NEW |
| consecutive | 2022-12-31 → full-sample | 0.969 (PASS) | 0.358 | NEW |

**Read straight:**

1. **The single highest-drift *passing* window (2016→2018 consecutive, drift 0.728) sits
   just below the single near-chance *failing* window (2016-12-31, drift 0.827).** Those
   two points are only 0.10 apart, and everything else is piled into the 0.26–0.36 band.
2. **In the 0.26–0.36 band, drift is uninformative about interior ARI.** It contains two
   clean passes (0.953 at 0.260, 0.969 at 0.358) *and* three fails (0.588 at 0.291, 0.559
   at 0.327, 0.548 at 0.351). Two of those fails have *lower* drift than the 2022 pass.
3. The Phase-2 note that "there is no validated *passing* drift level to anchor against"
   is now resolved but not in the hoped-for way: there are four passing drift levels
   (0.728, 0.260, 0.358, 0.358) and they overlap the failing ones almost entirely.

**Conclusion:** standardization drift, as this metric measures it, is a **coarse screen,
not a precise instrument** for regime-labelling breakdown. It cannot support a
finely-tuned threshold. This is itself a useful result — it empirically confirms the
two-tier monitoring structure rather than letting tier-1 drift masquerade as an
adjudicator: the cheap tier-1 metric is only good enough to flag "look closer," and
tier-2 shadow-fit ARI is what actually decides.

### `BREAKER` = 0.80 (design §2 M2), justified as a catastrophic-only trip

- **Just below the one near-chance data point** (2016-12-31: interior ARI 0.438, drift
  0.827). At drift ≥ 0.80 the metric has, once, coincided with a near-total labelling
  breakdown.
- **Above the highest drift on any interior-ARI-passing window** (2016→2018 consecutive,
  drift 0.728). The hard breaker will not fire on a configuration Phase 2 showed was
  actually stable.
- The 0.728 / 0.827 gap is narrow, so **0.80 is explicitly a blunt "something is
  catastrophically wrong" trip**, not a tuned boundary. The ambiguous 0.3–0.75 middle is
  M3's job (tier-1 fires ≥ 0.3 **and** tier-2 shadow-fit ARI < 0.85 → recalibration for
  review), and the calibration confirms drift alone genuinely cannot do better there.
- **Metric consistency:** `BREAKER` is compared against `standardization_drift.py`'s
  mean-per-date-Euclidean metric. Live tier-1 (`monitoring.compute_tier1_drift`) computes
  a methodologically different version (stored-vs-fresh scaler on the last 90 days, not
  truncated-vs-full-sample on interior dates) — Phase 2 measured the two within ~2.5 % of
  each other at the one comparable point (0.335 vs 0.327 at the 2020-06-30 cutoff), close
  enough to carry the calibration across, but worth re-checking once real operational
  tier-1 readings accumulate.
- Still a first cut, same status as `TIER1_DRIFT_THRESHOLD = 0.3` — revisit when there
  are real live drift observations, and if HMM is ever monitored the same way (its
  drift/ARI relationship is not characterised here at all).
- **Sample size for the anchor: exactly one.** `BREAKER = 0.80` is calibrated against a
  single near-chance case (2016-12-31, interior ARI 0.44). It is a first-cut value, same
  status as `TIER1_DRIFT_THRESHOLD = 0.3`, and specifically worth revisiting if/when a
  real catastrophic-drift event occurs in live operation — to confirm the breaker
  actually fires when it should, rather than assuming a one-point calibration generalises.

**Design doc updated:** §2 M2 `BREAKER` placeholder replaced with 0.80 and this rationale.

**Next:** step 2 — `signal_model_versions` registry + `fit_and_register_signal.py`.

---

## 2026-09-06 — Step 2: `signal_model_versions` registry + `fit_and_register_signal.py`, verified on an isolated DB copy

New: `signal_model/registry.py` (schema + save/load, mirrors `regime_detection/regime_db.py`
1:1) and `signal_model/fit_and_register_signal.py` (CLI, mirrors
`regime_detection/fit_and_register.py`). Verified against a **copy** of
`data/regimeguard.db` in the scratchpad — the real DB was never written to (confirmed:
it still has no `signal_model_versions` table), same discipline as the Phase 2/3
verification runs.

### Schema — mirrors regime_db deliberately

- `signal_model_versions` — one row per fitted LGBM model. Stores the target definition
  (H=1, ±20bps), the fit window, `as_of_date`, the purge/embargo used, `coverage_end_date`,
  `feature_columns`, `lgbm_params`, `best_iteration`, `n_train_rows`. **No model blob** —
  re-fitting is deterministic (fixed seeds, Phase 3), and the row records everything needed
  to reproduce it, matching regime_db's "store what downstream needs" stance.
- `signal_predictions` — one row per (trade_date, version), append-only, `superseded_by`.
- `current_signal_predictions` — latest non-superseded per date, analogous to
  `current_regime_labels` / `current_bars`.

### Train/predict separation mirrors `walk_forward.py` exactly

`train_end = as_of − PURGE_DAYS(1) − EMBARGO_DAYS(240)` positionally on the feature-date
index; predictions run from `as_of` forward for `SIGNAL_COVERAGE_TD = 63` trading days
(one quarterly floor — same cadence anchor as the regime side; the real signal-model
refit cadence is the later Model Training Agent's call) or to the last available row,
whichever comes first. So a version's predictions near its `as_of` are produced under
the same purge+embargo discipline the walk-forward evaluation used.

### Verified behaviour (isolated DB copy)

Registered v1 as-of 2016-06-30, then v2 as-of 2016-09-30:

- v1: train 2010-11-24 → 2015-07-08 (1,148 labelled rows, ~1 yr embargo gap before
  as_of, correct), best_iteration 47, 64 predictions 2016-06-30 → 2016-10-03.
- Registering v2 **superseded v1**: `signal_model_versions` v1 → `status='superseded'`,
  v2 → `'active'`; `get_active_signal_model_version("lgbm",1,20.0)` resolves to v2.
- Prediction supersede is **per-date and exact**: v1's 2 predictions on dates also
  covered by v2 (2016-09-30, 2016-10-03) flipped to `superseded_by` set; v1's other 62
  stayed current. `current_signal_predictions` = 126 rows across 126 distinct dates (no
  date carries two current predictions).
- `load_signal_prediction` resolves overlap dates to the newer version (v2) and returns
  `None` for an uncovered date (2020-01-02) — that `None` is exactly the
  `signal_model_stale` signal `decide_todays_call` will key on.
- **Coverage is a ceiling, not a freshness guarantee — confirmed explicitly.** With a
  single version registered as-of 2018-03-28 (coverage_end 2018-06-28), querying
  `coverage_end + 1 td`, `+ 30 td`, `+ 200 td`, years later, and any date *before* the
  as_of all return `None`. `load_signal_prediction` does an exact `trade_date` match on
  non-superseded rows — it never falls back to a nearest/latest prediction, so a system
  where nobody has re-run `fit_and_register_signal.py` recently enough will correctly
  report `signal_model_stale` rather than serve a stale guess.
- Probabilities sum to 1.0 to 6 dp for all 126 rows; `pred_direction == argmax(prob)`
  for every row.
- **Live path** (`--as-of` omitted → last feature row, 2026-08-21): train through
  2025-08-29, 3,658 rows, one prediction (for 2026-08-21), direction `flat` with
  near-uniform probs (0.323 / 0.342 / 0.335) — consistent with Phase 3's "almost no
  learnable signal at this horizon" finding, not a bug.
- **Guards:** an as-of date with < 241 td of history raises a clear `ValueError`; a
  weekend as-of (2019-06-29 Sat) snaps back to the last trading day (2019-06-28).

**Next:** step 3 — `decide_todays_call()` + `RegimeGuardCall` + `todays_call_log`.

---

## 2026-09-06 — Step 3: `decide_todays_call()` + `RegimeGuardCall` + `todays_call_log`, every branch exercised on an isolated DB copy

New:
- `signal_model/regime_reliability_tiers.py` — regenerates the §1.3 tier table from the
  point-in-time evaluation. Output `signal_model/results/regime_reliability_tiers.csv`.
- `regime_detection/quarterly_walk.py` — added `point_in_time_regime_label()`, the
  fit-fixed predict-forward step factored out for an arbitrary `as_of` against any
  registered version.
- `signal_model/todays_call.py` — `RegimeGuardCall`, `decide_todays_call()`,
  `todays_call_log` schema + append, CLI (`python -m signal_model.todays_call`).

### Tier table reproduces the Phase 3 headline

`regime_reliability_tiers.csv`: regime_2 (risk-off / drawdown) → **`monitored`**
(40.16 % PIT OOF accuracy, highest; regime-spread perm p = 0.0005 survives BH-FDR;
accuracy lead 0.1003 > permutation null p95 spread 0.059). regime_0, regime_1 →
**`none`**. All three §1.3 conditions checked, all reproduced from the saved OOF
predictions + the quarterly-walk PIT labels — not hand-set.

### `point_in_time_regime_label()` — reconstruction, not a new fit

The JM's `centers_` aren't persisted, so the helper deterministically re-fits the
same config (random_state=0, ~1 s) through the registered version's own
`fit_end_date` to recover them, then decodes forward with those fixed params and the
version's own `.transform()`. Two asserts, matching Phase 3's verify-don't-assume
discipline: the reconstructed clip+scale stats must match the version's stored ones,
and where a stored label exists for `as_of` the reconstructed label must match it.
Both held across every version tested (active v64 and historical v3, v24, v34, v38).

### Every disposition branch exercised (isolated DB copy, 6+ records)

| scenario | regime | tier → disposition | key trigger |
|---|---|---|---|
| **live path** (as_of=latest, active v64) | 2 risk-off | `monitored` → **INFORMATIONAL** | no modifier; VIX 4th pctile, drift 0.016; lean = `flat` from signal v-latest |
| live path, no recent signal version | 2 risk-off | `monitored` → **INFORMATIONAL** | `directional_lean.direction = null`, `stale_reason = "signal_model_stale"` — disposition unchanged (design decision 3) |
| **COVID onset 2020-03-25** | 2 risk-off | `monitored` → `suppressed` → **ABSTAIN** | **M1** (edge zone + VIX 99.96th pctile) — the one usable regime, correctly suppressed at crash onset |
| early COVID 2020-03-11 | 2 risk-off | `suppressed` → **ABSTAIN** | **M5** (reconstructed run 10 td < 20) + M1 |
| 2016-06-30 | 1 volatile rally | `none` → **ABSTAIN** | `none_no_reliability` (+ M1/M2 also listed) |
| 2019-10-18 vs a 2018-12-31 fit | 1 | `none` → **ABSTAIN** | **M4** (196 td stale > 126), `recalibration_flag = "overdue"` |
| 2011-06-30 vs a 60-row fit | 0 calm | `none` → **ABSTAIN** | **M6** thin-fit (fit_rows < 250) |

`regime_pattern` is always present: the regime_2 `statement` + `not_an_edge` pair on
`monitored`/`suppressed` (with `applies_now` = whether it's currently INFORMATIONAL),
a short generic line on `none`. `actionable` is a hard-coded `false` in every record.
Output is deterministic across repeated calls (same `as_of` → identical record bar
`generated_at`).

### `todays_call_log`

6 rows written, every column populated, `record_json` round-trips, `actionable = 0`
throughout. Joins cleanly to `model_versions` (via `regime_model_version_id`) and
`signal_model_versions` (via `signal_model_version_id`) — the audit-query shape the
review asked for. Real `data/regimeguard.db` never written to (verified: no
`todays_call_log` table there).

### M2 / M3 (tier-2-driven) verified by predicate, not by a live trigger

Neither can fire naturally with the current data: there is no catastrophic-drift
event in 15 years of history, and historical-replay versions have all their
`regime_labels` superseded by the active version, so `compute_tier2_shadow_fit`
returns `None` for them (a Phase 2 known limitation, "a superseded version has no
live labels to check by construction"). Direct evaluation of the predicates confirms:
tier-1 ≥ 0.80 → M2; tier-1 fires + tier-2 ARI < 0.50 → M2; tier-1 fires + tier-2 ARI
in [0.50, 0.85) → M3; tier-1 fires + tier-2 ARI ≥ 0.85 → neither. The Phase 2
monitoring verification already exercised `compute_tier2_shadow_fit` itself against a
live version (ARI 0.9989).

### Known limitation: historical-replay monitoring magnitudes are not point-in-time

`monitoring.compute_tier1_drift` always compares a version's stored scaler against a
fresh scaler on the **last 90 days of currently-available data** — so in a historical
`--regime-version-id` replay, `tier1_drift_mean` reflects "drift vs. today," not
"drift as of that historical date," and M1's drift-OR-clause fires readily in replay.
M1's **VIX-percentile clause is point-in-time correct** (`_trailing_vix_percentile`
uses `vix.loc[:as_of]`), and the **live path is fully correct** (tier1 = 0.016). Replay
still exercises the regime/tier/disposition logic faithfully; only the monitoring
*magnitudes* in replay records are not point-in-time. Making `compute_tier1_drift`
as-of-aware is a Phase 2 change, out of scope here — logged.

### First-cut thresholds carried in every record's `audit.thresholds`

`edge_zone_td=90`, `m1_vix_percentile=0.80`, `tier1_fire=0.30`, `circuit_breaker=0.80`,
`tier2_breaker_ari=0.50`, `tier2_review_ari=0.85`, `quarterly_floor_td=63`,
`staleness_hard_td=126`, `staleness_drift_cotrigger=0.20`, `min_run_td=20`,
`min_regime_fit_rows=250`. `min_regime_fit_rows` is anchored to a quick empirical
check (k=3 JM leaves a state empty / NaN centroid at n ≤ 75, all states populated from
n ≥ 100; 250 is ~2.5× that ceiling with every state ≥ ~20 members). The rest are
grounded in the design doc's §2 table; all first-cut, all in the record so any call
can be re-audited against the thresholds it actually used.
