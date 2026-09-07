# RegimeGuard — Findings

A regime-aware, leakage-safe signal-validation system for daily Indian equity-index
direction. This document reports what was built, what was tested, and what held up.
Every number below is drawn from the project's own result artifacts
(`regime_detection/results/`, `signal_model/results/`) and dated working notes
(`docs/phase2_working_notes.md` … `docs/phase6_working_notes.md`); where a figure is
approximate or first-cut it is said to be.

**Headline result.** On this data and feature set, a gradient-boosted direction model
has no aggregate trading edge (mean P&L −0.72 bps/day, p = 0.53) and no regime shows a
standalone P&L edge distinguishable from zero (every per-regime p ≥ 0.44). What *does*
survive scrutiny is an *ordinal* result: **which regime a day belongs to carries
statistically significant information about where the model's directional calls are more
versus less accurate** — regime-spread permutation test p = 0.0005, surviving
Benjamini-Hochberg correction. This result is *stronger* under point-in-time regime
labels (the way a live system would assign them) than under hindsight labels, where the
same test fails correction (p = 0.0275). That reversal is the opposite of the usual
effect of removing look-ahead, and it is reported exactly as computed. The system built
around this finding never emits an actionable trade call; its operating behaviour is to
abstain by default and, at most, report a monitored historical pattern with an explicit
non-actionable disclaimer.

---

## 1. The origin problem

Predictive models on non-stationary time series are usually judged by a single
aggregate performance number computed across an entire historical test period. That
number can be misleading in three distinct ways, each well known in principle and each
rarely built into a project as a first-class, provable component:

1. **Regime blending.** Markets move through persistent behavioural states — calm,
   volatile, trending, drawdown. An edge that is real in one state can be absent or
   negative in another. An aggregate metric averages these together, so a model that
   does well across many calm days and badly across a few stressed days can still show a
   good mean while being unreliable exactly when conditions change.

2. **Information leakage.** If a feature or a validation split ever uses information from
   after the moment being predicted — including subtle leakage across a train/test
   boundary via slow-moving, autocorrelated features — backtested performance looks
   better than anything achievable live.

3. **Multiple-testing / selection bias.** Try enough variants of a strategy and some
   will look good by chance. Without correcting for how many things were tried, an
   apparently strong result may not be real.

The goal here is a system that does not merely predict but **states the conditions
under which its predictions can be trusted, and proves that statistically rather than
assuming it** — one that can say "I don't know" and be right about that. Concretely:
discover regimes from data; build a direction model under strict point-in-time
discipline; evaluate it regime-by-regime with purged/embargoed validation and
significance testing that corrects for the test family; and turn whatever survives into
an explicit confidence tier and abstention behaviour rather than a confident-looking
number.

---

## 2. Regime detection and the validation of regime stability

### 2.1 Data

Nifty 50 and Bank Nifty daily OHLC and India VIX (index level), **2010-07-19 →
2026-08-21** — ~3,993 equity-index sessions, ~3,990 for VIX. 2010-07-19 is a **hard
floor**: NSE's historical VIX report serves nothing earlier, so the whole system is
gated to this ~15-year window (a deliberate choice over a dual VIX-covered/VIX-free
pipeline). India VIX is ingested **by hand** — automated NSE access is bot-blocked
(plain HTTP, headless browser, NSEpy all fail). Three sessions have no usable VIX and
were handled explicitly, not interpolated: 2024-03-02 (a SEBI disaster-recovery drill —
two short sessions, no VIX published though both indices closed) and 2021-02-12 /
2021-03-30 (zero-rows in NSE's own export, skipped after confirming both indices traded
normally). `daily_bars` is append-only (`superseded_by`, never updated in place), so any
as-of-date-T query is reproducible for later purged/embargoed validation.

**Features (11, all point-in-time):** log return, downside deviation, and a Sortino-like
ratio at 5/20/60-trading-day horizons (9; the 60-day family exp-weighted, 60-day
halflife), plus log VIX level and its 5-day change (2). A **90-trading-day warm-up** is
dropped before any fit, leaving **3,900 rows**.

### 2.2 Method: Statistical Jump Model, k = 3, λ = 50

Primary model: a **Statistical Jump Model** (JM; `jumpmodels`), chosen for its explicit
persistence (jump) penalty λ, which targets the flickering-state failure mode plain HMMs
show on imbalanced financial series. A **Gaussian HMM** (`hmmlearn`, 10 restarts/k, all
fits converged) is the named baseline.

Grid k ∈ {2,3,4,5} × λ ∈ {0,10,30,50,100,200}:

- **λ = 0 reproduces the flicker on this data**: 307 (k=2) to 683 (k=5) state switches,
  median run 3–5 days, min run 1 day at every k.
- **λ = 200, k ≥ 4 collapses**: one state falls to 59 observations at k=4 despite the
  highest aggregate persistence — the concrete reason persistence cannot be the selector.
- **λ = 50**: self-persistence 0.984–0.993 across k, median runs 66–112 days, no
  collapsed states.

**Selected k = 3, λ = 50** (active version fit through 2026-06-30): on the 3,900 rows,
31 regime switches over 15 years, mean self-persistence 0.992, median run 89.5 td, min
historical run 23 td, state sizes {1753, 415, 1732}. Per-state mean daily return
(identity-aligned across quarterly fits, §2.6):

| Regime | Character | Mean daily return / size |
|---|---|---|
| regime_0 | calm uptrend | ≈ +0.12 % to +0.14 %/day; largest (~1753/3900) |
| regime_1 | volatile rally | ≈ 0 to +0.25 %/day; smallest (~415/3900), most variable centre |
| regime_2 | risk-off / drawdown | only consistently negative state, ≈ −0.2 %/day in the full-sample fit (~1732/3900) |

HMM's persistence is meaningfully lower at every matched k — the own-data confirmation of
why JM was chosen: mean self-persistence (HMM vs JM λ=50) is 0.961 / 0.993 (k=2),
0.924 / 0.992 (k=3), 0.932 / 0.986 (k=4), 0.913 / 0.984 (k=5); HMM's minimum run length
is 1 day at k = 3, 4, 5.

### 2.3 Stability checklist (6 checks)

Regime detection was validated before anything downstream depended on it: state
duration/persistence, k-sweep, refit stability, rolling-window stability, a VIX
ablation, and HMM-vs-JM agreement. Duration and the k-sweep are covered by §2.2;
rolling-window stability is §2.4; the other three are below.

**Refit / seed stability — real instability at k = 4/5.** Ten seeds/config, pairwise
ARI:

| k | JM mean (min) | HMM mean (min) |
|---|---|---|
| 2 | 1.000 (1.000) | 0.991 (0.953) |
| 3 | **1.000 (1.000)** | 0.885 (0.676) |
| 4 | 0.875 (0.781) | 0.890 (0.794) |
| 5 | 0.696 (0.498) | 0.861 (0.727) |

JM k=2/k=3 are seed-invariant; JM k=5's min ARI 0.498 is barely above chance; HMM
degrades even at k=3. This fixed **k = 3**. Why k = 4/5 destabilise (a feature, a
historical stretch, or an intrinsic ~3-state ceiling for this data) was never
root-caused.

**VIX ablation — near-redundant for classification.** Full 11-feature vs. 9-feature
(no-VIX) fit on the identical 3,900-date set, k=3/λ=50: label ARI **0.967**; the 9
shared centres near-identical; stress-window assignment byte-identical for 4 of 5
windows (only 2013 taper shifts, 92.5 % → 86.25 %); **COVID 100 % risk-off in both**;
worst-state Jaccard 0.984. The return/downside-deviation/Sortino features do essentially
all the classification work once returns and realised vol are known. This does **not**
test whether VIX flags a shift *earlier* (a timing question, not pursued).

**HMM-vs-JM agreement — low but coherent.** k=3: ARI 0.310, NMI 0.335; JM 31 transitions
vs HMM 285 (~9×); each JM transition is a median 5 days from some HMM transition, each
HMM transition a median 43 days from the nearest JM one. JM catches a subset of real
shifts HMM also sees; HMM adds many with no JM counterpart — the quantified persistence
gap, not a contradiction.

### 2.4 Walk-forward stability: 0 of 6 configurations pass a pre-registered bar

The load-bearing check. Bar set **before running**: interior ARI ≥ 0.85, edge-zone ARI ≥
0.60 (edge zone = 90 td before the cutoff), plus a state-identity rule; one cutoff
failure disqualifies. Four expanding-window cutoffs (2016-12-31, 2018-12-31, 2020-06-30,
2022-12-31) × k ∈ {3, 4, 5} × 2 models. Separately-fit models are centroid-aligned
(`linear_sum_assignment`) before comparison.

| Model | k | 2016 int/edge | 2018 int/edge | 2020-COVID int/edge | 2022 int/edge |
|---|---|---|---|---|---|
| JM | 3 | 0.44 / 0.67 | 0.59 / 0.40 | 0.56 / **−0.08** | 0.97 / 0.36 |
| JM | 4 | 0.56 / 0.86 | 0.71 / 1.00 | 0.76 / 0.17 | 0.78 / 0.64 (interp) |
| JM | 5 | 0.70 / 1.00 | 0.53 / 0.85 | 0.77 / 0.21 | 0.41 / 0.00 (interp) |
| HMM | 3 | 0.60 / 0.57 | 0.60 / 0.54 | 0.63 / 0.80 | 0.65 / 0.76 (interp) |
| HMM | 4 | 0.81 / 0.68 | 0.66 / 0.67 | 0.48 / 0.17 | 0.74 / 0.83 (interp) |
| HMM | 5 | 0.59 / 0.51 | 0.70 / 0.66 | 0.49 / 0.34 | 0.79 / 0.52 (interp) |

- **JM k=3's edge ARI of −0.08 at the COVID cutoff** is the worst number: the truncated
  fit's labels for the ~90 days before it (COVID recovery, Apr–Jun 2020) *actively
  disagree* with the full-sample fit's, worse than chance.
- Not just an edge-zone problem — interior ARI is below 0.85 in nearly every cell, often
  0.4–0.7.
- **State identity holds far better than the ARI bars:** 22 of 24 (91.7 %) config×cutoff
  interpretability checks pass; no config fails at more than one cutoff; every config
  correctly characterises the two clearest windows (2018, 2020). The "0/6" is the
  numeric bars, not the models misreading what a regime is.

**Consecutive-pair variant** (each fit vs. the *next*, not the full-sample fit): still
0/6 overall, but the earliest, most point-in-time-honest pair (2016→2018) passes cleanly
for 3 of 6 (JM k=3, JM k=4, HMM k=3; interior ARI 0.86–0.98, edge 0.86–1.0). Every
config fails once the chain reaches the COVID-straddling pairs. Interpretability
agreement here is 14/24 (58.3 %) — each "next" fit is itself far less informed.

### 2.5 Mechanism: standardization drift and the COVID structural break

**Per-window standardization drift.** Euclidean distance between each interior date's
standardized feature vector under the truncated-fit vs. full-sample clip/scale pipeline
(clustering never runs — isolates whether the *inputs* already differ):

| Cutoff | COVID in training? | Mean per-date drift |
|---|---|---|
| 2016-12-31 | no | **0.827** (median 0.743, max 1.560) |
| 2020-06-30 | yes | **0.327** (median 0.318) |

The pre-COVID cutoff shows ~2.5× the drift. Scale ratios widen most for the volatility
family (`DD_log_60` 1.61×, `DD_log_20` 1.28×, `vix_log` 1.18×) — what a COVID-scale vol
spike would widen — and the 2016 cutoff had the worst interior ARI (0.44).

**A robust-scaler swap does not fix it.** For HMM the partition is mathematically
unchanged (ARI 1.0 between StandardScaler and RobustScaler fits of the same window) —
scaler choice *ruled out* for HMM. For JM it is net-negative: the COVID-adjacent
2020→2022 pair improves (interior 0.55 → 0.86) but the previously-clean 2016→2018 pair
collapses (interior 0.98 → 0.35, edge 1.00 → 0.25). Not adopted.

**Reading.** No pre-registered configuration is walk-forward stable at 15-year daily
granularity with this feature set. Per-window standardization is *part* of the mechanism
(a real, quantified artifact), but the consecutive-pair result shows two genuinely
point-in-time-honest states of knowledge — one before COVID, one after — still disagree
substantially on their shared 2010–2020 history. **The pre/post-COVID structural break
is the dominant driver, not a comparison-fairness artifact.**

### 2.6 What *is* stable: short cadence, and label identity

**Short-gap follow-up.** At a 3-month / 6-month recalibration cadence (4 reference points
× 2 gaps × 2 models, k=3): **5/8 pass at both; mean interior ARI 0.978 / 0.965**, every
one of the sixteen interior ARIs in 0.89–1.0 — versus 0.4–0.7 at 1.5–3.6-year gaps.
**Interior drift is essentially solved at a quarterly cadence.** Every residual failure
is edge-zone and sits inside an extreme stretch (COVID crash window: JM edge ARI 0.57,
HMM 0.19; 2022 near-tied window: JM 0.356, HMM passes at 1.0). No recalibration
*frequency* stabilises a fresh classification of the last 90 days when those 90 days
*are* the crisis — that is the abstention layer's job.

**Cross-quarter label identity.** The point-in-time evaluation (§4) concatenates 63
independently-fit quarterly models' labels by date; cluster indices are only anchored by
a within-fit rank ordering (`sort_by="cumret"`), which guarantees nothing across fits.
Centroid-alignment on all 61 consecutive pairs: **57/61 (93.4 %) identity-aligned**; the
4 exceptions are all 2012–2014 (smallest samples). **Every transition from 2014-06-30
onward — the entire 2015–2026 evaluation window — is identity-aligned, zero
exceptions**, so the composite label sequence is verified semantically consistent. (The
first cutoff, 2010-12-31, n=27 rows, gave a degenerate fit with one empty state —
outside the evaluation window, logged as backlog.)

### 2.7 Recalibration and drift monitoring (built)

- **Cadence:** a **quarterly floor**, from the 3-month (interior ARI 0.978) / 6-month
  (0.965) evidence specifically; 1-week–3-month and 6-month–1.5-year gaps are untested.
- **Storage:** append-only `model_versions` / `regime_labels` (`superseded_by` pattern),
  so every past label is reconstructable as it was known at the time — 64 versions:
  version 1 the hindsight full-sample fit, 2–64 the 63-cutoff quarterly walk.
- **Two-tier monitor:** a cheap continuous **tier-1** standardization-drift metric
  (`TIER1_DRIFT_THRESHOLD = 0.3`, set just below the milder known-bad point at 0.327 —
  there is no validated *safe* level) that, only when it fires, triggers an expensive
  **tier-2** shadow-fit ARI check (`TIER2_ARI_THRESHOLD = 0.85`) against the live
  version's stored labels. Measured tier-1 drift: 0.335 (model six years stale) vs.
  0.161 (recent). Tier-2 is **JM-only** (`NotImplementedError` for HMM). A trigger flags
  recalibration *for human review* — it never swaps the live model.

---

## 3. Leakage-safe validation, and why it was necessary

### 3.1 Target: H = 1 trading day, ±20 bps flat band

Chosen after checking the real return structure rather than assuming a tractable target:

- **Lag-1 autocorrelation of the raw 1-day return is 0.006** — essentially zero. This is
  the hardest, most honest form of the target; it was not routed around.
- **Squared-return autocorrelation is 0.184** — real volatility clustering. The
  project's actual hypothesis is that *regime-conditional* structure may exist where raw
  direction shows none.
- A 5-day horizon was checked and rejected: adjacent 5-day-forward labels share 4 of 5
  days, giving autocorr(1) ≈ 0.81 between labels — a mechanical overlap H = 1 avoids
  entirely.
- **±20 bps** flat band from real class balance: 42.5 / 21.3 / 36.1 (up/flat/down)
  overall, with the flat class 17–26 % in every regime (never collapsed).

### 3.2 Purge and embargo — including a self-corrected derivation

- **PURGE_DAYS = 1**, exactly the H = 1 label horizon: the only training sample whose
  label could reach into a test period is the single day before it.
- **EMBARGO_DAYS = 240**, re-derived after an initial mistake. The first draft reused
  Phase 2's `WARM_UP_EDGE_DAYS = 90` (regime-classification edge-zone length) — flagged
  as the wrong mechanism (that measures how long regime classification takes to settle
  near a cutoff, not feature-memory serial correlation across a train/test boundary). A
  second check found the naive `0.5^(k/halflife)` decay formula is *also* wrong (it
  ignores pandas' EWM weight normalization). The correct question — does the *feature
  series itself* stay autocorrelated near a boundary — was measured directly:
  `DD_log_60` is still **0.56 autocorrelated at lag 90** and does not fall below
  |ACF| < 0.10 until **lag 240** (`ret_60` / `sortino_60` clear that bar at lag 180; 240
  is the binding number across the family).
- **Placement follows the mechanism:** each fold's training set excludes its own most
  recent 240 trading days — a training-eligibility lag, not an idle gap between test
  folds. Test folds stay full, consecutive, annual blocks: no loss of evaluation
  coverage, only a training lag of about one fold-cycle.

### 3.3 Model and folds

- **LightGBM** (gradient-boosted trees), not a neural net: ~3,900 rows is far too few
  for extra network capacity to buy anything but overfitting once folds and regime
  strata shrink the effective training set. Reuses Phase 2's 11 validated features
  as-is; `class_weight="balanced"`.
- **12 annual folds (2015–2026), 2,880 out-of-fold predictions.** Training grows from
  778 rows (fold 2015, after purge + embargo trims a ~1,030-day window) to 3,501 rows
  (fold 2026).
- **`best_iteration = 1` on 4 of 12 folds** (2015, 2020, 2025, 2026) — verified genuine,
  not a bug: fold 2015's first-round `multi_logloss` is 1.0985, within 0.0001 of
  ln 3 = 1.0986 (the exact loss of uniform 3-class guessing), and it *climbs*
  monotonically as trees are added. Additional boosting rounds purely overfit noise —
  an independent confirmation of the ~0.006 return autocorrelation.

### 3.4 Point-in-time discipline

The regime-stratified evaluation was first run on **hindsight** regime labels (the
full-sample fit) — approved for a first pass as a research question ("does edge vary by
regime at all?"), explicitly logged as provisional and **not** a deployment basis. It
was then re-run on **point-in-time** labels from the versioned infrastructure: what a
`model_version` active at each historical date would actually have classified. Two
look-ahead risks were caught and closed on review:

1. The point-in-time decode is a **once-per-quarter batch Viterbi decode** using the
   prior cutoff's fixed parameters. An early-in-quarter date can therefore be informed
   by later-in-quarter dates — but never by a later quarter or a later recalibration.
   This is a named, reviewed compromise (a decode per live-decision window, not a true
   day-by-day causal filter), accepted rather than left implicit.
2. The predict-forward step uses `.transform()` only — confirmed by construction (the
   clip/scale objects are the same Python objects fit in the same loop iteration; no
   code path calls `.fit_transform()` on post-cutoff data).

---

## 4. Regime-stratified results, significance, and the negative/mixed findings

### 4.1 Aggregate — no edge

| Metric (2,880 OOF predictions) | Model | Momentum baseline |
|---|---|---|
| Accuracy | 33.5 % | 29.4 % |
| Macro-F1 | 0.332 | 0.286 |
| Mean P&L (bps/day) | −0.72 | +1.30 |

"Always predict up" scores **42.4 %** on the same window — above both the model and the
baseline. This is a direct consequence of `class_weight="balanced"`, chosen to stop the
model collapsing onto the plurality "up" class: it trades raw accuracy for macro-balance
across the minority flat/down classes. Whether accuracy, macro-F1, or P&L is the right
lens depends on the use case; this is an interpretive choice made in evaluation design,
stated rather than hidden behind the flattering metric.

- **Model vs. momentum baseline, accuracy:** paired block bootstrap on per-day
  correctness, **+4.13 pp**, 95 % CI [1.0, 6.8], **p = 0.008**, survives BH-FDR. The
  model is measurably more accurate than the simple rule.
- **Aggregate model P&L vs. zero:** −0.72 bps/day, 95 % CI [−4.1, +1.9], **p = 0.533**.
  Not distinguishable from zero.
- **"More accurate" is not "better."** The baseline's P&L is *higher* than the model's
  in aggregate (+1.30 vs −0.72 bps/day) and far higher in the smallest regime (regime_1:
  +6.61 vs −0.36). The model's extra correct calls land on smaller in-band moves;
  accuracy and P&L answer different questions, and the honest answer is that the model
  wins one and loses the other.

### 4.2 The point-in-time vs. hindsight reversal — the headline

Regime-stratified, model direction accuracy and P&L:

| | Hindsight labels (`model_version 1`) | Point-in-time labels (quarterly walk) |
|---|---|---|
| regime_0 | n = 1389, acc 30.2 %, P&L −1.30 | n = 1394, acc 30.1 %, P&L −0.42 |
| regime_1 | n = 295, acc 30.8 %, P&L −0.36 | n = 871, acc 34.2 %, P&L −1.86 |
| regime_2 | n = 1196, acc 38.0 %, P&L −0.14 | n = 615, acc **40.2 %**, P&L +0.20 |
| regime-permutation p (uncorrected) | 0.0275 | **0.0005** |
| survives Benjamini-Hochberg? | **No** (threshold 0.0167) | **Yes** (threshold 0.0125) |

Point-in-time and hindsight labels disagree materially: **ARI 0.590** on the 2,880
evaluation dates; regime_1 grows from 295 to 871 days, regime_2 shrinks from 1196 to
615.

**The finding, stated without overclaiming a mechanism.** Under the stricter,
hindsight-free point-in-time labeling, the apparent regime-dependent variation in the
model's directional accuracy is **not weaker — it is stronger, and it survives the
Benjamini-Hochberg correction that the hindsight version failed.** Removing look-ahead
usually weakens an apparent effect; here it strengthened it. This is reported exactly as
computed, not adjusted toward what seemed more plausible going in. The regime_2 accuracy
lead over the weakest regime is 0.1003, exceeding the permutation null's p95 spread of
0.059 — the gap is outside chance range for this partition. What this establishes:
regime membership, determined the way a live system would determine it, carries
statistically significant information about where this model's calls are more vs. less
accurate. What it does **not** establish: *why* the effect strengthens under
point-in-time labeling (whether point-in-time regimes happen to carve history in a way
that aligns more sharply with this model's good/bad stretches, versus hindsight regimes
averaging that structure away). That is not diagnosed and not asserted.

### 4.3 The alternative explanation, ruled out

A size redistribution like regime_1: 295 → 871 could in principle arise purely from
cross-quarter label-identity drift (each of the 63 fits being independent, indices only
rank-anchored) rather than from a genuine finding. This was checked before the headline
was trusted: centroid-alignment on all 61 consecutive quarterly pairs found **every
transition from 2014-06-30 onward identity-aligned** (§2.6). The composite label
sequence over the full evaluation window is semantically consistent; the strengthened
result is **not** an artifact of label drift.

### 4.4 What did not survive

- **No regime shows a standalone P&L edge distinguishable from zero.** Every per-regime
  mean-P&L p-value is ≥ 0.44 (regime_2 point-in-time: +0.20 bps/day, p = 0.848).
- Under **hindsight** labels the regime-spread result **fails** BH-FDR (p = 0.0275,
  threshold 0.0167) — exactly the shape of finding that looks interesting in isolation
  and dissolves once corrected for the test family. It is reported as such, not buried.
- **No transaction costs or slippage are modeled anywhere.** Every P&L number above is
  gross.

---

## 5. From finding to abstention behaviour

### 5.1 Confidence is a documented reliability tier, not a probability

A calibrated per-regime P(correct) would be false precision the results cannot support:
no regime has a P&L edge > 0; raw accuracy is below the 42.4 % majority baseline in
every regime; and the regime label itself is provisional (point-in-time vs. hindsight
ARI 0.590, walk-forward stability 0/6). What *is* verified is ordinal — regime
membership significantly ranks where the model is more vs. less accurate. A discrete
tier captures exactly that and nothing more. LightGBM class scores are still reported,
but labelled raw and only inside an explicitly non-actionable block.

**Tiers** (a regenerated build artifact, `signal_model/results/regime_reliability_tiers.csv`,
not a constant in code):

- **`monitored`** — the regime-permutation test survives BH-FDR, **and** this regime has
  the highest point-in-time OOF accuracy, **and** its accuracy lead over the weakest
  regime exceeds the permutation null's p95 spread. **Currently regime_2 only** (40.2 %
  point-in-time OOF accuracy; lead 0.1003 > null p95 0.059). This tier is
  **informational**: it authorises *reporting* the pattern and surfacing a non-actionable
  directional lean. It never authorises a trade call. Every `monitored` output
  permanently carries: *higher directional accuracy than a momentum baseline
  (regime-spread p = 0.0005, BH-significant); NOT a demonstrated positive P&L edge
  (regime_2 standalone P&L p = 0.85); no transaction costs modelled.*
- **`none`** — no verified evidence the model's call carries information in this regime.
  regime_0, regime_1. Resolves to `ABSTAIN`.
- **`suppressed`** — a `monitored` regime knocked down by a modifier (§5.2). Resolves to
  `ABSTAIN`.

**Two dispositions only: `INFORMATIONAL` and `ABSTAIN`.** There is deliberately no
`PREDICT` value anywhere in the schema, and `actionable` is a hard-coded `false` in
every record. **Why `monitored` never triggers an actionable call:** the only
statistically defensible result is a regime-spread in *accuracy*; there is no regime
with a P&L edge distinguishable from zero, and no costs are modelled. Emitting a trade
call would assert an edge the significance tests explicitly failed to find. The single
condition that would ever flip this: a **costed P&L series** (transaction costs +
slippage) showing a regime edge > 0 under the same block-bootstrap + BH-FDR discipline —
an open backlog item. If a future re-run's regime-permutation test fails BH, **no**
regime is `monitored` and the system abstains everywhere — the honest fallback, stated
up front.

### 5.2 The M1–M6 gates

`decide_todays_call(as_of)` resolves the active regime version, the point-in-time regime
and its run length, a monitoring snapshot, and staleness, looks up the base tier, then
applies six **downgrade-only** modifiers — each grounded in a specific Phase 2/3 result.
None can raise a `none` regime; any can push `monitored` → `suppressed` → `ABSTAIN`.

| Gate | Trigger | Grounding |
|---|---|---|
| **M1** edge-zone stress | within 90 td of the fit end **and** (trailing India-VIX percentile ≥ 80th, expanding-window / point-in-time **or** tier-1 drift ≥ 0.3) | JM k=3 edge ARI = −0.08 at the COVID cutoff; every short-gap edge-zone failure sat inside an extreme stretch. Stress is judged from VIX/drift, **not** the regime label (a self-reference removed on review). |
| **M2** circuit breaker | tier-1 drift ≥ 0.80, **or** (tier-1 fires **and** tier-2 shadow-fit ARI < 0.50) | ARI < 0.5 is the "barely better than chance" line from k=5 refit instability. `BREAKER = 0.80` sits just below the one window whose interior ARI fell to near-chance (2016-12-31: ARI 0.44, drift 0.827) and above the highest drift on any *passing* window (0.728). A blunt catastrophic-only trip. |
| **M3** drift confirmed | tier-1 fires **and** tier-2 ARI < 0.85 | the two-tier monitor's own "recalibration for review" condition. |
| **M4** staleness | > 126 td since fit, **or** > 63 td **and** tier-1 drift > 0.2 | short-gap follow-up: interior ARI 0.978 at 3 mo, 0.965 at 6 mo — staleness within ~2 quarters at low drift is tolerable; beyond that, or with rising drift, it is not. |
| **M5** fresh regime | current regime run < 20 td | JM k=3's minimum historical run length is 23 td (median 89.5) — a run shorter than any the model has produced is likely to be revised at the next refit. |
| **M6** OOD / thin fit | active regime fit on < 250 rows, **or** today's feature vector's distance to the nearest state centroid exceeds the in-sample interior maximum | k=3 leaves a state empty at n ≤ 75, all populated by n ≥ 100 (250 ≈ 2.5× that ceiling); the standardization-drift finding — an OOD day is one the fitted geometry cannot place. |

Signal-model staleness is kept conceptually separate: it nulls the directional lean
(`stale_reason = "signal_model_stale"`) but never changes the disposition, which is
governed solely by M1–M6.

### 5.3 Real scenarios (verified on isolated database copies)

| As-of | Regime | Disposition / tier | Trigger |
|---|---|---|---|
| 2026-08-21 (live, active v64) | regime_2 | **INFORMATIONAL / monitored** | no modifier; VIX 4th percentile, tier-1 drift 0.016 |
| 2020-03-25 (COVID onset, replay) | regime_2 | **ABSTAIN / suppressed** | **M1** — edge zone + VIX 99.96th percentile |
| 2020-03-11 (early COVID) | regime_2 | **ABSTAIN** | **M5** (reconstructed run 10 td < 20) + M1 |
| 2019-11-15 / 2016-06-30 (calm) | regime_0 / regime_1 | **ABSTAIN / none** | `none_no_reliability` |
| 2019-10-18 vs. a 2018-12-31 fit | regime_1 | **ABSTAIN** | **M4** — 196 td stale > 126; recalibration flag "overdue" |

The COVID-onset case is the intended behaviour in miniature: the one usable regime,
correctly suppressed at the exact moment a fresh edge-zone label is least trustworthy.
Net live behaviour: **abstain by default; at most, during an established, non-acute
risk-off regime that has outlasted the edge zone, report the historical accuracy
pattern plus a non-actionable lean.** That is the "say 'I don't know' and be right about
it" principle, operationalised.

---

## 6. Security design rationale

Five capability-scoped agents (Data, Regime Detection, Model Training, Validation) behind
MCP stdio tool boundaries, sequenced by a **deterministic** Orchestrator client (no LLM
planner — determinism keeps the decision record reproducible). Design stance: **the tool
layer is the control point, not agent judgement**; enforcement must hold even if a tool
is buggy or an LLM is later dropped in as orchestrator.

- **One declarative grant table** (`agents/capabilities.py`) — per-agent allowlists for
  table reads/writes, filesystem prefixes, outbound hosts, callable agents. Every
  mechanism below reads from it.
- **Database: SQLite's own statement authorizer** (`sqlite3.Connection.set_authorizer`),
  built per-agent from the grant. SQLite invokes it while *compiling* every statement; a
  denied read/write fails **before execution, regardless of how the SQL was built**. DDL
  is denied for every agent. This primitive has **no Postgres or managed-DB equivalent**
  — the concrete reason the store stayed a single SQLite file through deployment.
- **Network / filesystem / call graph:** non-Data servers install a permanent socket
  block (empty allowlist); Data may reach only `www.niftyindices.com`; each agent has a
  path-prefix write guard; only the Orchestrator may call anyone (acyclic, asserted at
  import).
- **No trade capability, by construction.** No trade tool, broker client, `orders` table,
  or trade verb in the grant vocabulary (`TRADE_VOCABULARY = ()`, kept empty by a test; a
  source scan asserts no execution SDK is imported). "Cannot trade" is not a rule that
  could be misconfigured — the capability does not exist to grant. Ties to Phase 4's
  `actionable: false`.
- **Tamper-evident audit.** Every tool call — **including denied ones** — is written **by
  the broker, not the tool**, into a SHA-256 hash-chained `agent_call_log`
  (`row_hash = sha256(prev_hash ‖ canonical_json(fields))`, genesis 64 zeros); the
  `todays_call_log` decision log is chained the same way. **Plain chain, no HMAC** —
  HMAC implies a key-management story this project lacks; the honest claim is
  **tamper-evident, not tamper-proof** (any edit, reorder or deletion breaks the chain
  for anyone who cannot also recompute every later hash). `orchestrator.explain`
  reconstructs a decision from its full ordered call tree.
- **The 47-test deny-path suite** makes the scoping a fact, not a claim: per agent it
  builds the real scoped handle and asserts the denials (e.g. Validation can't read the
  decision log; any `CREATE TABLE` raises; a non-Data server can't open a socket;
  tampering one row makes `verify_chain` report the break).
- **Deployment (Modal):** the daily job runs the real MCP path — four scoped subprocesses
  in one container — against one SQLite file on a Modal Volume; verified across ≥ 3
  separate container invocations that the append-only logs and both hash chains stay
  intact across container lifecycles.

**Scope limit.** This covers the `todays_call` **decision path**. Data ingestion and the
heavy validation entry points still run via the pre-existing CLIs on unscoped
connections — consistent with those being consequential operator-initiated stages, but
not full end-to-end coverage.

---

## 7. Honest limitations

**Data.**

- The India VIX floor of 2010-07-19 is hard. The system covers ~15 years; earlier
  regimes (2008) are entirely out of sample.
- India VIX is a **manual** NSE download — automated access is bot-blocked. Three
  sessions have no usable VIX (2024-03-02 DR-drill; 2021-02-12 and 2021-03-30 zero-rows
  in NSE's export). Deployment consequently **cannot be fully live**: the pipeline is
  operator-gated on that manual step, and the scheduled job re-runs on whatever data
  exists (correctly emitting an `ABSTAIN`/stale result when nobody has refreshed).

**Regime detection.**

- Regime labelling is **not walk-forward stable by the project's own pre-registered
  bar** — 0 of 6 configurations pass. The monitored-regime finding survives *only*
  because it is a regime-spread (ordinal) result, not a level; point-in-time vs.
  hindsight label ARI is 0.590.
- The pre/post-COVID structural break is **not solved** — it is *contained* by a
  quarterly recalibration floor (interior stability) plus the abstention layer
  (edge-zone risk during live crises). A robust-scaler fix was tried and rejected (it
  helped one COVID-adjacent pair and broke a previously-clean one).
- Regime coverage is uneven. regime_2 — the only `monitored` regime — has n = 615
  point-in-time evaluation days. COVID is a single event that dominates the most extreme
  behaviour in nearly every stability result; the milder stress windows (2013 taper,
  2016 demonetization, 2018 IL&FS) are only a few dozen days each.
- The k = 4 / k = 5 seed instability (JM min pairwise ARI 0.78 / 0.50) was never
  root-caused. VIX's potential *early-warning / timing* value — as distinct from
  same-day classification, where the ablation shows it near-redundant — was never
  tested.
- tier-2 shadow-fit monitoring is **JM-only**; HMM `model_versions` could not be
  monitored the same way.

**Signal model and evaluation.**

- No regime shows a P&L edge distinguishable from zero (every per-regime p ≥ 0.44).
- **No transaction costs or slippage are modelled anywhere.** A costed P&L series is the
  single open backlog item that blocks any actionable output. Until it exists, "edge" in
  this document means "directional-accuracy pattern" and nothing more.
- Aggregate model accuracy (33.5 %) and the momentum baseline (29.4 %) are both below
  "always predict up" (42.4 %) on the evaluation window — a deliberate consequence of
  `class_weight="balanced"`, but a real caveat on the accuracy framing.
- The point-in-time regime label is a **once-per-quarter batch Viterbi decode**, not a
  true day-by-day causal filter: an early-in-quarter date can be informed by
  later-in-quarter dates (never a later quarter). Reviewed and accepted, not fully
  causal.

**Abstention thresholds.**

- **M2 and M3 have never fired.** There is no catastrophic-drift event in 15 years of
  data to exercise the tier-2 / circuit-breaker paths end to end; they are verified only
  by predicate evaluation. `BREAKER = 0.80` is calibrated against **exactly one** data
  point (2016-12-31: interior ARI 0.438, drift 0.827).
- `TIER1_DRIFT_THRESHOLD = 0.3`, `BREAKER = 0.80`, M1's 80th-percentile VIX gate, M5's
  20 td, and M6's 250-row floor and OOD distance are all **first-cut** values; several
  are anchored to a single reference point and none has been confirmed against live
  operational readings.
- Historical-replay monitoring *magnitudes* are not point-in-time:
  `compute_tier1_drift` always compares against the last 90 days of currently-available
  data, so a `--regime-version-id` replay's `tier1_drift_mean` reflects "drift vs.
  today," not "drift as of that date." The live path and M1's VIX-percentile clause
  *are* point-in-time correct.

**Infrastructure.**

- The quarterly-walk infrastructure has **no minimum-sample-size floor** before its
  first cutoff (a degenerate n = 27 fit at 2010-12-31; harmless here — outside the
  evaluation window — but a real wart before reuse at finer, e.g. intraday, granularity).
- Security scoping and the hash chain cover the **decision path**, not the whole
  pipeline; ingestion and heavy validation still use unscoped connections.
- The deployment demonstrates persistence and the real MCP path but is single-operator,
  single-region; backup is an occasional volume snapshot plus the fact that the store is
  reproducible from raw data.

---

## Appendix — key figures at a glance

| Quantity | Value | Source |
|---|---|---|
| Data window | 2010-07-19 → 2026-08-21 | `data/regimeguard.db` |
| Rows entering each fit (post 90-day warm-up) | 3,900 | `jm_grid.csv` |
| Regime model | JM, k = 3, λ = 50 | `jm_grid.csv`, `model_versions` |
| Regime switches / median run / min run | 31 / 89.5 td / 23 td | `jm_grid.csv` |
| JM refit-stability ARI, k=3 / k=4 / k=5 | 1.000 / 0.875 / 0.696 | `jm_refit_stability.csv` |
| Walk-forward stability | 0 of 6 configs pass; JM k=3 COVID edge ARI −0.08 | `rolling_window_stability.json` |
| Standardization drift, 2016 / 2020-06 cutoff | 0.827 / 0.327 | `circuit_breaker_calibration.csv` |
| Short-cadence interior ARI, 3 mo / 6 mo | 0.978 / 0.965 | `short_gap_stability.csv` |
| VIX-ablation label ARI | 0.967 | `vix_ablation_summary.json` |
| Cross-quarter label identity (post-2014-06-30) | 100 % identity-aligned | `quarterly_alignment_check.csv` |
| Target | H = 1 td, ±20 bps; return autocorr(1) = 0.006 | `target.py`, `phase3_working_notes.md` |
| Purge / embargo | 1 td / 240 td | `walk_forward.py` |
| Walk-forward folds / OOF predictions | 12 (2015–2026) / 2,880 | `fold_summary.csv` |
| Aggregate accuracy / P&L | 33.5 % / −0.72 bps/day (p = 0.53) | `model_regime_stratified.csv`, `significance_bh.csv` |
| Model vs. baseline accuracy | +4.13 pp, p = 0.008, survives BH | `significance_bh.csv` |
| PIT vs. hindsight label ARI | 0.590 | `phase3_working_notes.md` |
| Regime-spread test, hindsight / point-in-time | p = 0.0275 (fails BH) / p = 0.0005 (survives BH) | `significance_bh.csv`, `significance_bh_pit.csv` |
| regime_2 point-in-time OOF accuracy | 40.2 % (lead 0.100 > null p95 0.059) | `regime_reliability_tiers.csv` |
| Per-regime standalone P&L | none distinguishable from zero (all p ≥ 0.44) | `significance_bh_pit.csv` |
| Deny-path security tests | 47 passing | `tests/test_agent_scoping.py` |
| Actionable trade calls emitted | 0 (by construction) | `signal_model/todays_call.py` |
