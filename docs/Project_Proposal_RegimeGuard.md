# Project Proposal: RegimeGuard
### A Regime-Aware, Leakage-Safe Signal Validation System

---

## 1. The Problem This Project Exists to Solve

Models trained and validated on market history are usually judged by a single aggregate performance number. That number can hide a dangerous truth: a model's edge is often real only under specific market conditions, and silently absent — or negative — under others. Since markets move through distinct behavioral regimes (calm, volatile, trending, choppy), a model can look statistically solid overall while actually being unreliable exactly when conditions change — which is precisely when it matters most.

This is a real, general failure mode in applied machine learning on any non-stationary system. It is worth solving properly, with real data, real validation rigor, and honest documentation of what does and doesn't hold up — not assumed away with a single backtest number.

---

## 2. The Problem, Precisely

Most predictive models on time-series data are validated with a single aggregate performance number computed across the entire historical test period. That number can be dangerously misleading:

- Markets move through distinct behavioral regimes — calm, volatile, trending, choppy, panic-driven.
- A pattern that produces genuine edge in one regime can produce no edge, or negative edge, in another.
- An aggregate metric blends all of these together. A model that performs very well across 300 calm days and poorly across 20 volatile days can still show a good *average* — hiding the fact that it is dangerous exactly when it matters most.
- This is compounded by a second, quieter failure mode: **information leakage**. If a model's features or validation setup ever "see" information from the future relative to the moment being predicted, its backtested performance will look better than it can ever actually achieve live.
- A third, rarely-checked failure mode: **multiple-testing / selection bias**. If enough variations of a strategy are tried, some will look good purely by chance. Without correcting for this, an apparently strong result may not be real at all.

Together, these three failure modes are the standard ways a quantitative ML project quietly lies to its own creator. They are well known in principle, but rarely built into a project as a first-class, provable component rather than an afterthought.

---

## 3. The Proposed Solution

Build a system that does not just predict — it **knows and states the conditions under which its predictions can be trusted**, and proves this statistically rather than assuming it.

Core behavioral principle: **the system should be able to say "I don't know" and be right about that**, rather than always producing a confident-looking number.

This is achieved through four connected components:

1. **Regime Detection** — discover market regimes from data rather than hand-picking thresholds.
2. **Leakage-Safe Signal Model** — a predictive model built under strict point-in-time discipline.
3. **Statistical Validation Layer** — regime-stratified, leakage-safe, significance-checked evaluation.
4. **Confidence-Aware, Abstention-Capable Prediction** — the model outputs a prediction *and* a regime-conditioned confidence, and abstains when outside proven conditions.

Wrapped around this: a multi-agent architecture (MCP-orchestrated), security-scoped tool permissions with an audit trail, and a live deployment — so the finding becomes an operating system, not just a report.

---

## 4. Research Design

### 4.1 Data

- Historical Nifty / Bank Nifty price and derivatives data (OHLCV, and where available, options chain / OI data for PCR).
- Volatility and macro context signals (India VIX, and broader regime-indicative series where relevant).
- Data organized into a proper point-in-time schema: every feature must be reconstructable *as it would have been known* at the timestamp it's used, with no forward-looking joins.

### 4.2 Regime Detection Methodology

- Rather than manually defining thresholds ("VIX above X = volatile"), use an unsupervised approach — e.g., a Hidden Markov Model or clustering over volatility, trend, and volume-derived features — to let regimes emerge from the data itself.
- Each historical period is labeled with a detected regime (e.g., calm-trending, calm-choppy, volatile-trending, volatile-choppy / panic).
- Regime detection is validated for stability (does it produce persistent, interpretable states, not noisy flickering) before being used downstream.

### 4.3 Signal Model

- A predictive model on engineered features (price action, volatility, macro context) targeting short-horizon direction or a related tradeable signal.
- Strict point-in-time feature construction — every feature audited to confirm it could only use information available at the time of prediction.
- Model choice kept honest and explainable (e.g., gradient-boosted trees or a small neural network) — complexity is not the goal, correctness of validation is.

### 4.4 Validation Methodology — the research core

- **Purged, embargoed walk-forward cross-validation**: standard time-series validation is extended with purging (removing training samples too close to the test window) and embargo periods, to prevent subtle leakage across the train/test boundary.
- **Regime-stratified reporting**: performance (accuracy, precision/recall as relevant, Sharpe-like risk-adjusted metrics, drawdown) is reported *separately per detected regime*, not only in aggregate.
- **Significance testing**: apparent differences between regimes, and the apparent edge of the model overall, are tested for statistical significance rather than accepted at face value — including an explicit check for whether results could plausibly be explained by chance given the number of variations tested.
- **Baseline comparison**: the regime-aware model is compared against (a) a naive aggregate-validated baseline, and (b) a simple rule-based baseline consistent with prior work, to make the value of regime-awareness concretely visible.

### 4.5 Expected Research Output

A finding of the shape: *"The model shows statistically meaningful edge in regime A. In regime B, its apparent edge disappears under proper validation. In regime C, performance is inconclusive given available data."* This is treated as a legitimate, valuable result — not a failure to hide. Negative and mixed results are documented with equal weight to positive ones, consistent with how every other project in this portfolio is documented.

---

## 5. From Research to System — Production Architecture

### 5.1 Operating Behavior

The deployed system's defining feature: for any given moment, it reports the **currently detected regime**, its **calibrated confidence** conditioned on that regime, and either a **prediction** or an explicit **abstention** when current conditions fall outside what has been validated as reliable.

### 5.2 Multi-Agent, MCP-Orchestrated Architecture

| Agent | Responsibility |
|---|---|
| Data Agent | Pulls and maintains historical + live market data, SQL-backed, point-in-time correct |
| Regime Detection Agent | Runs the regime model, labels current and historical conditions |
| Model Training Agent | Trains / retrains the signal model on a defined schedule |
| Validation Agent | Runs the regime-stratified, leakage-safe evaluation; produces performance-by-regime reports |
| Orchestrator | Sequences agent calls via MCP tool-calling; owns the end-to-end pipeline |

Each agent exposes its capability as an MCP tool rather than being called through hardcoded function logic — this is a deliberate architectural choice, not a cosmetic one, since it is what makes the system's components independently callable, testable, and auditable.

### 5.3 Security-by-Design

- **Tool-permission scoping**: each agent is restricted to only the tools/data it needs — e.g., the Data Agent cannot trigger trades, the Validation Agent cannot reach live trading endpoints, the Regime Detection Agent cannot modify the trained model.
- **Signed, tamper-evident audit trail**: every agent decision and tool call is logged in a way that lets any prediction or abstention be fully reconstructed and explained after the fact — "why did the system abstain at this moment" must always be answerable.
- This is a direct, natural extension of prior findings from the MCP-based Agent Security Testbed: model-level judgment alone is not sufficient; enforcement belongs at the tool layer.

### 5.4 Deployment

- Backend and orchestration deployed on Modal / Render, consistent with existing project infrastructure.
- SQL-backed historical and live data store with a schema designed explicitly to prevent lookahead (point-in-time tables, not simple flat snapshots).
- A live dashboard surfacing: current detected regime, model confidence, current prediction or abstention status, and a link into the audit trail for that decision.

---

## 6. Documentation

A single `FINDINGS.md`, written as a short research report rather than a build log, covering:

1. The origin problem and why it matters (Section 1–2 of this proposal, refined post-build).
2. Regime detection methodology and validation of regime stability.
3. Leakage-safe validation setup and why it was necessary.
4. Regime-stratified results, including statistical significance checks and negative/mixed findings.
5. How the finding was operationalized into the system's abstention behavior.
6. Security design rationale (tool scoping, audit trail).
7. Honest limitations: data availability constraints, what regimes were and weren't well covered, what would need more data or infrastructure to strengthen further.

---

## 7. Why This Matters

Most ML-on-markets work reports a single backtested performance number and stops there. This project is built around the opposite instinct: that number is only meaningful once you've checked *why* it's true, *whether* it holds under honest validation, and *when* it stops being true. The system's core feature — knowing when not to trust itself — is a direct, engineered answer to a real failure mode observed firsthand in prior work.

The combination of regime discovery, leakage-safe statistical validation, MCP-orchestrated multi-agent architecture, and security-scoped design is intentionally rare as a single coherent system, each piece grounded in an actual problem this project set out to solve rather than assembled to satisfy an external checklist.

---

## 8. Design Decisions — Resolved

- **Data granularity:** daily, starting point. Price (Nifty/Bank Nifty OHLCV) + India VIX. Intraday granularity is an explicit later phase, only after the daily version is complete and validated — not part of initial scope.
- **Data source:** NSE archives directly (price, VIX, and bhavcopy/F&O data as needed). India VIX confirmed manually against NSE's own historical report page as the primary source — automated access (headless browser, NSEpy) is hard-blocked by NSE's bot protection, so this series is pulled manually rather than via scripted evasion.
- **Prediction target:** direction (primary) — short-horizon up/down/flat on Nifty/Bank Nifty. Volatility is a secondary, regime-context target — it feeds the regime-detection layer and the confidence/abstention decision on the direction call, rather than standing as an independent prediction output.
- **Training/validation window:** 19 July 2010 – present (~15 years), confirmed as the earliest date NSE's own historical VIX report serves. This is a hard floor — India VIX does not exist as a usable series before this date, regardless of source.
- **Historical scope:** the entire system (Nifty and Bank Nifty both) is hard-gated to the 2010+ VIX-covered window. No separate VIX-free extended-history mode using Bank Nifty's longer independent history (since 2003) or Nifty's (since ~1996). This window already spans multiple genuinely distinct market regimes (2013 taper tantrum, 2016 demonetization, 2018 IL&FS stress, 2020 COVID crash, 2022 rate-hike volatility) — sufficient to prove or disprove the core regime-stratified-validation thesis without the added complexity of a dual-pipeline system. A VIX-free extended mode remains a possible future enhancement, not current scope.
- **Regime detection methodology:** Statistical Jump Model as primary (chosen over plain HMM/clustering for its explicit persistence penalty, which directly targets the flickering-state failure mode plain HMMs exhibit on imbalanced, limited-sample financial series). Gaussian HMM retained as a named baseline for comparison.

## 9. Open Questions Before Build Begins

- Decide retraining cadence and how "drift into a new, previously unseen regime" will be detected and handled operationally.
