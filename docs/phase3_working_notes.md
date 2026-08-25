# Phase 3 Working Notes — Signal Model, Leakage-Safe Validation, Regime-Stratified Evaluation

Running, dated, append-only evidence log for Phase 3, same purpose and standard as
`docs/phase2_working_notes.md`: raw material for the eventual `FINDINGS.md`, not the
polished writeup itself.

---

## 2026-08-24 — Design: signal model, target, and purged/embargoed walk-forward CV

### Model choice (approved): LightGBM (gradient-boosted trees), not a neural net

Reasoning: ~3,900 total rows is far too few for a neural net's extra capacity to pay
off over tree ensembles on tabular data - that extra capacity would mostly buy
overfitting, especially once folds and regime-strata each shrink the effective
training set further. Native feature-importance / SHAP explainability also matches
proposal §4.3's "kept honest and explainable" requirement directly. Reuses Phase 2's
11 already-validated, point-in-time-safe features as-is - no new feature engineering
in this phase.

### Prediction target (approved): H=1 trading day, ±20bps flat band

- **Horizon H=1** (next-trading-day direction), chosen after checking real return
  structure rather than assuming it: lag-1 autocorrelation of the raw 1-day return is
  **0.006** (essentially zero - the honest, hardest form of the target, not routed
  around). Squared-return autocorrelation is **0.184** (real vol clustering) - the
  project's actual hypothesis is that regime-conditional structure may exist where
  raw direction shows none. A longer horizon was checked and rejected: 5-day forward
  returns showed **autocorr(1)=0.81** between adjacent labels (mechanical, from
  sharing 4/5 days) - H=1 avoids this overlapping-label complication entirely.
- **Flat band = ±20bps** on the 1-day log return, chosen from real class-balance
  evidence: 42.5/21.3/36.1 (up/flat/down) overall, and non-degenerate across all
  three already-detected regimes (flat class 17-26% in every regime, never collapsed
  to near-zero).

### Purged, embargoed walk-forward CV (approved, after one correction)

**PURGE_DAYS = 1**, exactly matching the H=1 label horizon - the only training
sample whose label could reach into a test period is the single day before it.

**EMBARGO_DAYS = 240** - re-derived after an initial mistake was caught. First draft
reused `WARM_UP_EDGE_DAYS=90` from Phase 2's regime-edge-instability finding, which
was flagged as measuring the wrong mechanism (how long regime classification takes to
stabilize near a fitting cutoff, not feature-memory serial correlation across a
train/test boundary). Re-derivation:
1. An impulse-response test on `.ewm(halflife=60)` showed the naive `0.5^(k/hl)`
   formula is *also* wrong (it ignores pandas' EWM weight normalization) - caught by
   verifying directly rather than trusting either formula from memory.
2. The right question isn't "how fast does one raw observation's weight decay" but
   whether the *feature series itself* (a smoothed, persistent process) stays
   autocorrelated near a boundary. Measured real autocorrelation of the halflife-60
   feature family directly: `DD_log_60` (the slowest to decay) is still **0.56**
   autocorrelated at lag 90 - nowhere near negligible - and doesn't drop below an
   |ACF|<0.10 bar until **lag 240**. `ret_60`/`sortino_60` clear that bar earlier
   (lag 180); 240 is the correctly binding number across the family.
3. Placement follows the mechanism precisely: each fold's training set excludes its
   own most recent 240 trading days (not an idle gap between test folds). Test folds
   stay full, consecutive, annual blocks - no loss of evaluation coverage, only a
   training-eligibility lag. Real, stated cost: each fold's usable training set
   trails about one fold-cycle behind a naive expanding-window walk-forward.

Implementation: `signal_model/walk_forward.py`.

### Regime-stratified evaluation, baselines, significance testing (approved)

- **#3**: per-regime accuracy, macro precision/recall/F1, Sharpe-like ratio, max
  drawdown, using regime as a conditioning variable on pooled out-of-fold predictions.
- **#4a**: not a separate model - the aggregate vs. regime-stratified view of the
  *same* predictions, which is the comparison itself.
- **#4b**: rule-based baseline = sign of the already-built `ret_5` feature against
  the same ±20bps band ("predict continuation of recent short-term momentum").
- **#5**: block bootstrap (not iid) throughout, given real serial correlation in the
  data; a paired block bootstrap for model-vs-baseline; a **regime-permutation test**
  (block-permutes regime labels, preserving real frequency, to ask how much
  regime-to-regime spread would arise from partitioning ~3,900 days into persistent
  buckets by chance) as the instrument aimed specifically at the project's thesis;
  Benjamini-Hochberg FDR correction across the resulting test family.

### Open item, logged explicitly - required before any deployment claim

Regime stratification (#3) uses **final/best-available** regime labels (the
full-sample JM k=3 fit, `model_version 1`), not point-in-time labels. **Approved for
this phase specifically** - "does the model's edge genuinely vary by regime at all"
is a research/diagnostic question that deserves the cleanest available regime
classification, same reasoning already used in Phase 2 when the full-sample fit was
kept as a valid, separate reference point from the point-in-time-honest checks.

**This is not a closed decision.** Before any deployment claim, this evaluation must
be rerun using point-in-time regime labels - what a live `model_version`, active at
each historical date, would actually have classified at the time - using the
versioned `model_versions`/`regime_labels` infrastructure Phase 2 built for exactly
this purpose. If the apparent regime-dependent edge doesn't survive that stricter,
hindsight-free test, that is itself an important, honest finding to report, not a
reason to skip running it. **Required follow-up, not optional.**

---

## 2026-08-24 — Implementation: walk-forward LightGBM, regime-stratified results, baselines, significance

Built `signal_model/` (`target.py`, `walk_forward.py`, `lgbm_model.py`,
`baselines.py`, `evaluate.py`, `significance.py`, `run_signal_model.py`) and ran the
full pipeline end to end against real data. Registered the real (non-demo)
full-sample JM k=3/λ=50 regime fit as `model_version 1` (3,900 labels) to provide the
final/best-available regime labels for stratification, per the approved judgment call
above.

### Fold structure - real numbers

12 annual folds (2015 through 2026 partial), purge=1d, embargo=240d. Training set
size grows from 778 rows (fold 2015) to 3,501 rows (fold 2026), consistent with the
expanding-window design and the stated embargo cost (fold 2015's ~1,030-day initial
window is trimmed to 778 usable training rows once purge+embargo is applied).
2,880 total out-of-fold predictions, all with a current regime label. Full detail:
`signal_model/results/fold_summary.csv`.

### best_iteration_=1 on 4/12 folds - verified genuine, not a bug

Folds 2015, 2020, 2025, 2026 stopped after a single boosting round. Checked directly
before trusting it (same standard as every other suspicious result this project has
investigated): confirmed no label-encoding bug (no NaNs, reasonable class counts in
both the training-proper and internal early-stopping validation slices), then
inspected the actual validation-loss curve for fold 2015. First-iteration
`multi_logloss` = **1.0985** - within 0.0001 of **ln(3) = 1.0986**, the exact loss of
uniform random 3-class guessing - and it climbs monotonically from there as more
trees are added. This is a clean, independent confirmation of the design phase's
own finding (lag-1 return autocorrelation ≈ 0.006, essentially zero): there is
genuinely almost no learnable signal in this target at this horizon, and additional
boosting rounds purely overfit noise rather than finding real structure. Not an
implementation defect.

### Results — real, and honestly mixed to negative in aggregate, per §4.5's framing

| regime | n | model accuracy | baseline accuracy | model mean PnL (bps/day) |
|---|---|---|---|---|
| aggregate | 2880 | 33.5% | 29.4% | -0.72 |
| regime_0 | 1389 | 30.2% | 28.9% | -1.30 |
| regime_1 | 295 | 30.8% | 37.3% | -0.36 |
| regime_2 | 1196 | 38.0% | 28.0% | -0.14 |

Full detail: `signal_model/results/model_regime_stratified.csv`,
`baseline_regime_stratified.csv`.

**Important caveat, stated plainly, not buried**: raw accuracy for both the model
(33.5%) and the momentum baseline (29.4%) is *below* the trivial "always predict up"
strategy, which scores **42.4%** on this same held-out period (up is the plurality
class, 42.4% of days in 2015-2026). This isn't a bug - `class_weight="balanced"` was
used deliberately (§ design notes) to stop the model from just calling every day
"up," trading raw accuracy for macro-balance across the minority flat/down classes
(model macro-F1 = 0.332 vs. baseline's 0.286, vs. what "always predict up" would
score on macro-F1, which is far worse still since it never predicts flat/down at
all). Whether accuracy or macro-F1 is the right lens depends on the actual use case
(a real trading rule cares about P&L, not macro-F1) - flagging this explicitly as an
interpretive choice made in evaluation design, not hiding it behind the metric that
looks better.

### Significance (#5)

- **Model vs. rule-based baseline**: paired block bootstrap on per-day correctness
  difference, point estimate **+4.13pp** (model above baseline), 95% CI [1.0pp,
  6.8pp], **p=0.008**. Survives BH-FDR correction (reject_null=True). The model is
  measurably, significantly more accurate than the simple momentum rule under this
  validation scheme - a real, if modest, result.
- **Overall model P&L vs. zero**: point estimate -0.7bps/day, 95% CI [-4.1bps,
  +1.9bps], **p=0.533**. Not distinguishable from zero. No significant aggregate
  trading edge - consistent with the near-zero raw-return autocorrelation established
  in the design phase, and an honest negative result, not something to route around.
- **Regime-permutation test**: observed regime-to-regime accuracy spread = 0.079,
  vs. a null mean of 0.035 (null p95 = 0.072), **uncorrected p=0.0275**. On its own
  this looks like a real signal that regime matters. **After BH-FDR correction across
  the full test family, it does not survive (reject_null=False)** - full table in
  `signal_model/results/significance_bh.csv`. This is exactly the shape of finding
  §4.5 describes as legitimate and valuable: an apparent regime-dependent effect that
  looks interesting in isolation but is not distinguishable from chance once properly
  corrected for how many things were tested. Not a failure to hide - the honest
  answer to the question this whole validation layer exists to ask.

### Honest caveat: "more accurate" is not "better trading outcome" - stated explicitly, not left implicit in the table

The significance section above reports the model as significantly more accurate than
the rule-based baseline (p=0.008). **That is a narrower claim than "the model is
better," and the P&L numbers directly contradict the naive reading of it**: the
baseline's P&L is *better* than the model's in aggregate (+1.30 vs. -0.72 bps/day)
and dramatically better in regime_1 (+6.61 vs. -0.36 bps/day) - the one regime with
the fewest observations (n=295) but the clearest gap. This is a real, mechanistic
consequence of `class_weight="balanced"`: it was chosen specifically to stop the
model from just predicting the plurality "up" class, which improves macro-balanced
accuracy (correctly calling more flat/down days) but says nothing about whether those
additional correct calls are on days where the *magnitude* of the move is large
enough to matter for P&L, versus small in-band moves the accuracy metric treats as
equally important as a large one. The rule-based momentum baseline, despite lower
raw accuracy, apparently gets its calls right on days that matter more in return
terms. **Bottom line: accuracy and P&L are answering different questions here, and
this backtest's honest answer is that the model wins on one and loses on the other -
not that "the model won."**

### Documented limitations (not fixed in this pass)

- **No transaction costs or slippage are modeled** in any P&L series reported above
  (model, baseline, or aggregate). At the weak signal levels found throughout this
  phase (near-zero raw-return autocorrelation, no significant aggregate edge even
  before costs), real transaction costs could plausibly erase an apparent edge in
  either direction - or, in principle, could also matter less than expected if actual
  realized moves on the model's correct-call days are large relative to typical
  index-fund/futures spread costs. This is a real, unresolved gap, not a rounding
  concern - any live-trading claim needs a costed P&L series before it means anything
  operationally.
- Regime-stratified evaluation (above) uses final/best-available regime labels, not
  point-in-time ones - see the open item logged in the design section above, now
  being addressed in the next entry.

### Bottom line for this pass

Real, verified output, not "it runs": the model shows a significant edge over the
simple rule-based baseline *in accuracy*, but not in P&L, no significant aggregate
trading edge either way, and the apparent regime-dependent performance variation does
not survive correction for multiple comparisons. **This result is provisional**
pending the required follow-up already logged above - re-running the regime-stratified
view with point-in-time regime labels rather than the final/best-available fit used
here - and pending a costed P&L series before any operational claim.

---

## 2026-08-25 — Required follow-up: point-in-time regime-stratified re-run

### Design review caught a real look-ahead issue before anything was built

First draft of the point-in-time label generator proposed fitting at each quarterly
cutoff, then calling `JumpModel.predict()` once over the entire following quarter in
a single batch. Flagged on review: `predict()` runs a full Viterbi decode over
whatever sequence it's given, using the model's fixed, already-fitted parameters -
verified directly via source inspection (`predict_proba` calls `do_E_step(X_arr,
self.centers_, self.jump_penalty_mx, ...)` over the whole input array). That means an
*early* date's label within a quarter could still be informed by *later* dates in
that same quarter - not genuinely live/causal, even though the model's parameters
themselves were fixed at the prior cutoff.

**Resolved as a reviewed, accepted design, not by building an incremental Viterbi
filter**: batch-decode each gap once, using the previous cutoff's fixed parameters -
so an early-in-quarter date can be informed by later-in-quarter dates, but *never* by
a later quarter or a later recalibration. This is a real, named compromise (a decode
made once per live decision window, not a true day-by-day causal filter), reviewed
and explicitly accepted rather than left as an implicit assumption. A single
full-history JM fit takes ~1.3s, so an incremental alternative was not required on
cost grounds either - the choice was purely about what's methodologically defensible
for this exercise.

### Two further points confirmed explicitly before running

1. **Point-in-time-label recovery scoped to exclude `model_version 1`.** The
   "first label ever registered per date" recovery query (`MIN(id)` per `trade_date`)
   is restricted to `WHERE model_version_id IN (<63 quarterly-walk ids>)`. Without
   this, `model_version 1`'s rows (registered before the walk, so lower ids for any
   overlapping date) would have silently won the `MIN(id)` comparison for a large
   chunk of history, defeating the entire point of the exercise. Implemented and
   documented directly in `regime_detection/quarterly_walk.py`'s
   `load_point_in_time_labels`, including the explicit caveat that `MIN(id)` relies
   on regime_labels being insert-ordered chronologically (true here, but a real
   assumption, not a schema guarantee - would need an explicit provenance column if
   that ever changed).
2. **Predict-forward transforms with `.transform()` only.** Confirmed by
   construction, not by inspection after the fact: the clipper/scaler objects reused
   in the predict-forward step are literally the same Python objects just fit in
   step 1 of that same loop iteration - there is no code path that could call
   `.fit_transform()` on post-cutoff data. Documented explicitly in the module
   docstring as the exact leakage point the original standardization-drift finding
   came from.

### Execution: 63 quarterly cutoffs, 62 seconds, real output

`regime_detection/quarterly_walk.py` generated 63 quarterly cutoffs (2010-12-31
through 2026-06-30, nearest-trading-day per calendar quarter-end) and ran the full
fit-then-predict-forward walk in 62.4s, registering `model_versions` 2-64.

**A second real bug caught before trusting the comparison**: re-running the original
(hindsight) evaluation afterward gave slightly different regime counts than
originally reported (regime_0: 1389→1390 rows). Traced to source: the quarterly
walk's model_versions share the same `(model_kind="jm", k=3, jump_penalty=50.0)`
config as `model_version 1`, so `current_regime_labels` (a "whatever's currently
non-superseded" view) silently drifted to reflect the quarterly walk's *last* cutoff
instead of the original standalone fit once the walk ran. Fixed by pinning the
hindsight comparison to `model_version_id = 1` explicitly
(`run_signal_model.HINDSIGHT_MODEL_VERSION_ID`) rather than relying on
`current_regime_labels` - confirmed this restores the exact originally-reported
numbers before proceeding.

### Point-in-time vs. hindsight regime labels: a real, substantial divergence

ARI between the two labelings on the 2,880 evaluation dates: **0.590** (moderate
agreement, real disagreement - not a rubber-stamp of the hindsight labels). Regime
sizes shift meaningfully: regime_1 grows from 295 (hindsight) to 871 (point-in-time);
regime_2 shrinks from 1,196 to 615. Consistent with everything Phase 2 already found
about instability in this fit under a hindsight-limited information set.

| | hindsight (model_version 1) | point-in-time (quarterly walk) |
|---|---|---|
| regime_0 | n=1389, acc=30.2%, PnL=-1.30bps/day | n=1394, acc=30.1%, PnL=-0.42bps/day |
| regime_1 | n=295, acc=30.8%, PnL=-0.36bps/day | n=871, acc=34.2%, PnL=-1.86bps/day |
| regime_2 | n=1196, acc=38.0%, PnL=-0.14bps/day | n=615, acc=40.2%, PnL=+0.20bps/day |
| regime-permutation p-value (uncorrected) | 0.0275 | **0.0005** |
| regime-permutation p-value survives BH-FDR? | **No** | **Yes** |

Full detail: `signal_model/results/model_regime_stratified_pit.csv`,
`significance_bh_pit.csv`.

### The headline finding, stated plainly and without overclaiming a mechanism

**Under the stricter, hindsight-free point-in-time labeling, the apparent
regime-dependent performance variation is not weaker - it is stronger, and it
survives Benjamini-Hochberg correction that the hindsight version failed.** This is
the opposite of the naive expectation (removing lookahead usually *weakens* an
apparent effect, not strengthens it), and it is reported exactly as computed, not
adjusted toward what might have seemed more plausible going in.

**What this does and doesn't establish**: this is real, verified evidence that
regime membership - when regime is itself determined the way a live system would
actually have determined it - carries statistically significant information about
this model's per-date performance, correcting for how many things were tested. It
does **not** by itself explain *why* the effect strengthens under point-in-time
labeling (e.g., whether point-in-time regimes happen to carve up history in a way
that aligns more sharply with when this particular model does well or poorly, versus
hindsight regimes averaging that structure away) - that would need further,
targeted diagnosis and is not asserted here. Also unresolved, same as the hindsight
pass: P&L tells a different, weaker story than accuracy in some regimes (e.g.
regime_1's point-in-time P&L is *worse*, at -1.86bps/day, despite that regime's
higher accuracy), and no transaction costs are modeled anywhere in this analysis -
both caveats from the hindsight pass apply here unchanged.

### Status (superseded by the verification below - see next entry before trusting this)

The required follow-up is complete: the regime-stratified evaluation has now been
run both ways, side by side, using the project's own versioned model_versions/
regime_labels infrastructure exactly as it was built for. The open item from the
design section above is resolved - with a result that makes the regime-stratification
finding *more* credible under scrutiny, not less, which is itself worth treating
carefully rather than as a convenient conclusion to stop investigating at.

---

## 2026-08-25 — Verification: does state identity actually hold across the 63 independently-fit quarterly models?

Flagged on review, before accepting the headline result above: each quarterly cutoff
is a fresh, independent JM fit. Cluster label indices (0/1/2) are arbitrary per fit
unless anchored - `sort_by="cumret"` is used (confirmed directly from
`quarterly_walk.py`), but unlike the rolling-window stability check (which explicitly
centroid-aligned states via `linear_sum_assignment` before comparing across fits),
the quarterly walk concatenates 63 fits' labels by date alone. If `sort_by="cumret"`
isn't a strong enough anchor, the composite sequence could be a patchwork where
"regime 1" means different things in different stretches - which alone could produce
exactly the kind of size redistribution seen above (regime_1: 295→871), independent
of any genuine finding. This needed checking before the headline result meant
anything.

### What `sort_by="cumret"` actually does - verified from source, not assumed

`JumpModel.fit()` → `sort_states_from_ret()`: computes each fit's own k states' mean
return *within that fit's own data*, then sorts states by that value. This is a
**purely relative, rank-based ordering** (this fit's best-return state, middle,
worst-return state) - it provides no guarantee that "state 1" represents the same
actual market conditions across two fits on different data windows. Confirms the
concern was well-founded in principle; the question is whether it holds in practice
for this specific data.

### Empirical check: centroid-alignment on all 61 consecutive quarterly pairs

Re-fit every quarterly cutoff (deterministic - same data/seed as the original walk;
`centers_` was never persisted, so this required re-fitting rather than reading from
storage) and applied `regime_detection.rolling_window_stability.align_states` - the
exact same centroid-matching/`linear_sum_assignment` mechanism already validated
throughout Phase 2 - to every consecutive pair. Script:
`regime_detection/quarterly_alignment_check.py`; full detail in
`regime_detection/results/quarterly_alignment_check.csv`.

**Result: 57/61 (93.4%) of consecutive pairs have identity alignment (0→0, 1→1,
2→2).** The 4 non-identity pairs are: 2012-06-29→2012-09-28, 2012-09-28→2012-12-31,
2012-12-31→2013-03-28, 2014-03-31→2014-06-30 - all in the earliest, smallest-sample
part of history. **Every single transition from 2014-06-30 through 2026-06-30 - a
span that fully covers the entire 2015-2026 evaluation period - is identity-aligned,
with zero exceptions.** Since composing a chain of identity permutations is itself
the identity, this means the composite point-in-time label sequence actually used in
the Phase 3 comparison above is verified semantically consistent across its full
length, not merely assumed to be. `sort_by="cumret"` is a strong enough anchor here -
confirmed empirically, not by argument.

### A second, genuine, separate issue surfaced by this check

The very first quarterly cutoff (2010-12-31, only 27 post-warm-up rows) produced a
**degenerate fit**: one of the three states received zero observations (NaN
centroid) - too little data to support a meaningful 3-state split. This does not
corrupt anything already stored (`labels_` has no NaN, it simply never uses state 2;
`model_versions` doesn't persist `centers_` at all, so nothing NaN reached the DB),
and its predict-forward gap (Jan-Mar 2011) falls far outside the 2015-2026 evaluation
window, so **it does not affect the reported comparison**. It's a real methodological
wart nonetheless - the quarterly cadence currently has no minimum-sample-size floor
before its first cutoff - worth fixing (e.g., skip or defer cutoffs below a minimum
row count) before this infrastructure is reused for anything beyond this evaluation,
but not blocking here.

### Conclusion: the headline result stands, verified rather than assumed

No re-derivation with explicit re-alignment is needed - the check that would have
triggered one came back clean for the region that matters. **The point-in-time
regime-stratified result reported above (effect stronger than hindsight, survives
BH-FDR where hindsight didn't) is confirmed to not be an artifact of cross-quarter
label-identity drift.** This is now real, verified evidence, not a result that
happened to look right before anyone checked the one thing that could have quietly
invalidated it.

The earlier caveats stay attached to this result unchanged: the accuracy-vs-P&L
disconnect (the model is more accurate than the rule-based baseline but not
better-performing in P&L, and this varies by regime), and no transaction costs or
slippage modeled anywhere in this analysis.

### Backlog (not urgent, not blocking)

- **Minimum-sample-size floor for quarterly-style refits.** The degenerate
  first-cutoff issue above (empty state / NaN centroid at n=27) should not be able to
  recur if `regime_detection/quarterly_walk.py`'s pattern is reused later - e.g., for
  intraday granularity, where the same "recalibrate on a fixed cadence starting near
  the very beginning of available history" structure would likely hit the same
  failure mode again, possibly less harmlessly. Fix: skip or defer cutoffs below a
  minimum row count (a reasonable starting point: enough rows that k=3 is very
  unlikely to leave a state empty - worth grounding in evidence rather than picking a
  round number, same standard as everything else in this project, when it's actually
  built).
