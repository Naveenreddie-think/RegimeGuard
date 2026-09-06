# Phase 6 Design Proposal — Deployment

Status: **research + proposal, for review.** No code written. Covers proposal §5.4
(deployment: scheduled decision job, persistent SQL store, live dashboard).

This phase changes **no research finding, model, threshold, or decision logic**. It gives
the already-built Phase 1–5 system an operational home. Same discipline as every prior
phase: propose → review → build file-by-file. **No commits or pushes at any point.**

---

## 0. TL;DR recommendation

| Piece | Platform | Form |
|---|---|---|
| Persistent store | **Modal Volume** (`regimeguard-data`) holding the one `regimeguard.db` | SQLite stays; it is load-bearing for the Phase 5 security layer (see §3) |
| Daily decision job | **Modal**, `modal.Cron` weekdays + `modal run` on demand | runs the **MCP orchestrator path** (`agents.orchestrator.todays_call`) in one container; copy-DB-local → run → commit |
| Dashboard | **Modal** web endpoint (`@modal.web_server`, scale-to-zero) | minimal Streamlit or FastAPI, read-only mount of the same Volume |
| Data refresh (prices + manual VIX) | **Modal**, `modal run` operator-initiated | unchanged from Phase 5 review decision 5 — never auto-chained |

**One platform (Modal), one Volume, one SQLite file.** Render is a viable fallback for the
dashboard only, and it is *worse* here for a concrete reason (§4): it forces a second copy
of the database and a sync mechanism, for no benefit a portfolio project needs.

Cost: comfortably **$0** — inside Modal's $30/mo Starter credit, no card required. Realistic
usage is a ~1–2 min CPU job on weekdays plus a scale-to-zero web endpoint serving a handful
of page loads a day.

---

## 1. What actually needs to run in production — framing confirmed, with three corrections

The framing in the ask — *a scheduled job runs the decision once per trading day and writes
it to the DB; a separate lightweight dashboard reads the DB and displays it* — is
**essentially correct**. Three corrections, each material:

### 1.1 It cannot be fully "live" or fully scheduled — India VIX ingestion is a manual human step

This is not a limitation to design around; it is a documented, deliberate Phase 1 finding.
NSE-direct automated access to India VIX history is **hard-blocked** by Akamai bot
protection — a plain HTTP client, a headless browser, and NSEpy all failed
(`data_agent/load_india_vix_manual.py` module docstring; proposal §8). VIX is pulled **by
hand** from NSE's historical-VIX report page and dropped as a CSV for
`load_india_vix_csv()` to validate and load.

Consequences for deployment:

- A blind daily cron on "make today's decision" is only meaningful **after** someone has
  downloaded the VIX CSV and refreshed the store. Without that, the job re-decides on stale
  data — and correctly emits `ABSTAIN` with `M4_stale` once `td_since_fit` crosses the
  threshold. That is honest behaviour, not a bug, but it means the cron is a *staleness
  guard / dashboard-freshness keeper*, not the primary trigger.
- The **primary trigger is the operator**: download VIX → `modal volume put` the CSV →
  `modal run refresh` (prices + VIX + calendar + gap check) → `modal run decide`. This is
  exactly the "operator invokes each consequential stage explicitly, with review between"
  stance already locked in Phase 5 review decision 5. Deployment should not quietly reverse
  it.
- Price data (Nifty / Bank Nifty via `niftyindices.com`) *is* automatable and needs no
  human step — but it is not useful on its own without the matching VIX bar.

**Proposed shape:** operator-initiated `refresh` + `decide` entrypoints (`modal run`), plus
an **optional** `modal.Cron` weekday run of `decide` alone — which keeps the dashboard
honest (it will show `ABSTAIN / stale` if nobody refreshed) rather than silently frozen on
last week's call.

### 1.2 The daily job should run the **MCP orchestrator path**, not the direct path

There are two implementations of "today's call", by design (Phase 5 §5):

- `signal_model.todays_call.decide_todays_call` — the **direct** path. Gathers inputs via
  in-process imports, assembles the record, writes `todays_call_log`. **No `agent_call_log`,
  no capability-scoped connections, no hash-chained agent-call trace.**
- `agents.orchestrator.todays_call` — the **MCP** path. Spawns the four stdio servers,
  every input read goes through the broker on a callee-scoped connection, the full ordered
  call tree is written to the hash-chained `agent_call_log` under one `trace_id`. Produces a
  **byte-identical** `RegimeGuardCall` (verified in Phase 5 e2e).

The deployed system is the portfolio artifact for Phase 5's headline — *MCP multi-agent
orchestration + tool-scoped permissions + tamper-evident agent-call audit*. If the daily
job runs the direct path, the deployed system never exercises any of that, and
`orchestrator.explain(as_of)` has **no agent-call trace to show** (it falls back to the
decision row alone). The dashboard's "link into the audit trail" (proposal §5.4) is
substantially the agent-call tree.

**Cost of running the MCP path:** each of the four server subprocesses imports
lightgbm / sklearn / jumpmodels on startup → one `todays_call` over MCP takes **~1–2 min**
(Phase 5 working notes). For a once-per-weekday batch job this is a non-issue. It is not a
hot path and never will be.

**Recommendation:** the scheduled/triggered job runs `agents.orchestrator.todays_call`. Keep
`decide_todays_call` available as a fast local fallback / debugging tool, unchanged.

### 1.3 "Separate dashboard" — yes, but same file, not a replica

The dashboard should read the **same** `regimeguard.db` (read-only Volume mount), not a
copied/exported replica. A replica introduces a sync job and a staleness question for zero
gain at this scale. See §3 and §4.

### 1.4 What does *not* need to run in production

- No live trading, no order path — by construction there is no trade capability anywhere in
  the system (`TRADE_VOCABULARY = ()`, Phase 5 §2.4). Nothing to deploy there.
- No autonomous retraining. `regime.fit_and_register`, `training.fit_and_register_signal`,
  `regime.quarterly_walk` stay operator-initiated (`modal run`), one at a time, review
  between — Phase 5 review decision 5, unchanged.
- No always-on compute. Everything is either a short batch job or a scale-to-zero endpoint.

---

## 2. MCP architecture compatibility — verified against both platforms

The concern is real and worth checking rather than assuming: the orchestrator
(`agents/orchestrator.py`) spawns **four stdio subprocesses** via
`mcp.client.stdio.stdio_client`, each launched as `sys.executable -m agents.servers.<x>`,
and talks to them over stdin/stdout pipes.

### 2.1 Can Modal run this pattern as designed? **Yes.**

- A Modal Function runs your code in an ordinary Linux container. `subprocess.Popen`,
  pipes, and stdio work with no special handling. Modal's **own** Streamlit example runs its
  web server "in a background subprocess using `subprocess.Popen`" inside the container —
  subprocess spawning is a documented, first-class pattern, not something to work around.
- All four servers **plus** the orchestrator run inside **one** Modal container for the
  duration of one job. There is no cross-container / cross-host communication — it is
  parent process + four children on one machine, exactly the topology Phase 5 tested on
  (Windows, then implicitly Linux via the `mcp` SDK).
- The three non-Data servers install a permanent `restrict_network(frozenset())` at startup
  (all outbound sockets blocked). The `todays_call` MCP sequence calls only regime /
  validation / training — **it needs no network at all**. Nothing in Modal's sandbox
  interferes with an in-process `socket` monkeypatch.
- `PYTHONPATH` injection (`orchestrator._server_params`) works unchanged; the repo is on the
  container image.

**No adaptation required for Modal.** The only operational nuance is cold-start import time
(§1.2), which is acceptable for a batch job.

### 2.2 Can Render run this pattern? **Yes, mechanically** — but Render is not where this job should live

- Render services are also ordinary containers; `subprocess` + stdio work.
- Render's job primitive is a **Cron Job** service. It *can* run
  `python -m agents.orchestrator todays-call`. But a Render Cron Job **cannot mount a
  persistent disk** (verified — "You can't add a disk to a cron job service"). So the job
  would have nowhere durable to write `regimeguard.db`. The documented Render workaround is
  a **Background Worker** (which *can* have a disk) running its own scheduler (APScheduler /
  `schedule`), i.e. an always-on process for a once-a-day task — more moving parts and a
  non-zero idle cost, for a job Modal does more cleanly with `modal.Cron`.

### 2.3 Is the in-process broker fallback an option? **It does not exist as code.**

Phase 5's design named an "in-process broker" as the documented fallback *if the MCP wiring
overran scope*. It didn't overrun — real MCP shipped (`bf69380`), and the in-process broker
was **never built**. So today the choices are:

1. Deploy the real MCP subprocess path (works on Modal — §2.1). **Recommended.**
2. Build the in-process broker now as a Phase 6 sub-task (same `capabilities.py`, same
   `ScopedDB` authorizer, same hash chain; "agents" as objects holding scoped handles
   instead of subprocesses). Removes the 1–2 min cold start and the multi-process SQLite
   surface, at the cost of ~1 day of work and losing the literal protocol boundary in the
   deployed path.

Given the job runs at most a few times a day and 1–2 min is irrelevant there,
**option 1 is the right call** — build nothing new, deploy what exists and what Phase 5's
narrative is actually about. Option 2 is a reasonable future optimisation, not Phase 6
scope, unless review disagrees.

### 2.4 The one real technical care point: SQLite + WAL under four subprocesses

Phase 5 already handles this *within a single host*: `PRAGMA journal_mode = WAL` (set once
at bootstrap), `PRAGMA busy_timeout = 5000` on every connection, and a bounded
exponential-backoff `SQLITE_BUSY` retry in the broker write path. Phase 5's e2e wrote 24
`agent_call_log` rows from four concurrent server subprocesses with the chain verifying
clean.

WAL needs a real POSIX filesystem (the `-shm` shared-memory file, `mmap`, `fcntl` locks).
That is fine on a container's **local disk** (ext4/overlayfs). It is **not** guaranteed on a
Modal Volume mount (FUSE, "distributed file locking is not supported" — Modal docs). So the
job must **not** run SQLite directly against the Volume mount while writing. Handled in §3.

---

## 3. Database persistence — the critical piece

### 3.1 Why SQLite cannot simply be swapped for a managed Postgres

The Phase 5 security layer is **built on a SQLite-specific primitive**:
`sqlite3.Connection.set_authorizer` (`agents/broker.py::_make_authorizer`). SQLite invokes
the authorizer callback while *compiling every statement*, per table/column/action, and a
`SQLITE_DENY` fails the statement before execution regardless of how the SQL was built. This
is the whole enforcement mechanism for per-agent table read/write scoping and the
blanket-deny on DDL.

**Postgres (and every managed Postgres offering) has no equivalent API.** Reproducing the
grant table on Postgres would mean: a separate DB role per agent, `GRANT`/`REVOKE` per
table, `search_path` and `SET ROLE` juggling in the broker, and re-testing the entire
`tests/test_agent_scoping.py` deny-path suite against a different engine. That is a
rewrite of Phase 5's core deliverable, not a deployment step.

Add to that: `agents/audit.py` and `signal_model/todays_call.py` hash-chain rows with
SQLite semantics; `data_agent/db.py`'s point-in-time `superseded_by` model and the
`current_*` views; 130k `regime_labels`, 64 `model_versions`. All SQLite.

**Decision: SQLite stays.** The deployment question is therefore "where does the one
`.db` file live so it survives restarts and redeploys", not "which database service".

### 3.2 Modal Volumes — how persistence actually works

- A **Volume is independent of app code and app deploys.** Redeploying the app does **not**
  touch Volume contents. Data survives restarts and redeploys. (The only footgun: *deleting*
  a Volume and recreating one with the same name gives it a new ID; deployed apps keep
  pointing at the old ID until redeployed. Don't delete the Volume.)
- Writes are persisted by **background commits every few seconds** and a **final commit on
  container shutdown**, or an explicit `volume.commit()`. A reader container sees new data
  only after `volume.reload()` (or on a fresh container).
- Volumes are **not** a low-latency POSIX filesystem and **do not support distributed file
  locking** — so you don't run a live, write-active SQLite database *directly* on the mount.

### 3.3 The write pattern for the daily job (safe, simple, single-writer)

The job is a **once-a-day batch with exactly one writer**. No concurrency problem exists if
we don't create one:

```
1. mount Volume at /data  (contains the canonical regimeguard.db)
2. copy /data/regimeguard.db  ->  /tmp/regimeguard.db      (local container disk)
3. set REGIMEGUARD_DB=/tmp/regimeguard.db                  (env var already supported —
                                                            broker.db_path() reads it)
4. run agents.orchestrator.todays_call()  — 4 subprocesses + orchestrator, all against
   /tmp/regimeguard.db on local ext4, WAL + busy_timeout + retry exactly as Phase 5 tested
5. checkpoint WAL, copy /tmp/regimeguard.db  ->  /data/regimeguard.db
6. volume.commit()
```

`REGIMEGUARD_DB` already exists for precisely this indirection (Phase 5 working notes: "points
the whole agent layer at an isolated DB copy … without touching the real store"). The
copy-local / commit-back pattern is the standard Modal idiom for exactly this.

The `data refresh` job (prices + VIX + calendar) uses the identical pattern — copy local,
mutate, commit back.

### 3.4 The dashboard reads the Volume directly, read-only

The dashboard mounts the same Volume, opens `regimeguard.db` **read-only**
(`file:...?mode=ro`), and calls `volume.reload()` on a short TTL (or per request) so it
picks up the day's commit. Reads don't need locking cooperation with a writer that isn't
running. Between the daily job's `commit()` calls the file is static.

### 3.5 The git-tracked `data/regimeguard.db` (16 MB) becomes a *seed*, not the source of truth

The repo currently tracks `data/regimeguard.db`. After deployment, **the Volume is
canonical.** Proposed handling:

- One-time seed: `modal volume put regimeguard-data ./data/regimeguard.db /regimeguard.db`.
- From then on, treat the in-repo copy as a **point-in-time seed snapshot** (refreshed
  deliberately and occasionally), or stop tracking it and keep only a documented seed
  procedure. Either is fine; what matters is that nobody assumes `git pull` updates
  production state. Flag for review — this is a working-practice call, not a technical one.

### 3.6 Render's disk model, for completeness

- Render **persistent disks** are real block storage, survive deploys and restarts,
  auto-mounted on paid **web services** and **background workers**. **Not** available on
  free web services (ephemeral FS — SQLite would reset on every deploy/restart/spin-down)
  and **not** available on Cron Jobs at all.
- Render **Postgres** on the free plan has historically been time-limited (expires after
  ~30 days) — durable use is the paid tier. Moot here given §3.1, but worth stating: "just
  use Render Postgres" is neither free nor compatible with the security layer.

---

## 4. Modal vs Render — per piece, with reasoning

### 4.1 Persistent store → **Modal Volume**

Only real options are Modal Volume or a Render paid disk (§3.1 rules out Postgres). A
Render disk attaches only to an always-on paid service, which means paying for idle time to
hold a file that changes once a day. Modal Volume is free, deploy-independent, and the job
already has the `REGIMEGUARD_DB` seam to use it cleanly. **Modal Volume.**

### 4.2 Daily decision job → **Modal**

- `modal.Cron("30 12 * * 1-5", timezone="Asia/Kolkata")` (post-close IST, weekdays) for the
  scheduled staleness-guard run; `modal run decide` for the operator's real trigger.
- Spawns the four MCP subprocesses in one container — verified compatible (§2.1).
- Scale-to-zero: you pay for ~1–2 min of CPU per run and nothing else.
- Render's equivalent is a Cron Job with no disk (blocked) or an always-on Background Worker
  running its own scheduler — more parts, non-zero idle cost, worse fit.

**Modal**, decisively.

### 4.3 Dashboard → **Modal** (Render is the only defensible fallback, and it's still worse)

Modal `@modal.web_server` (Streamlit) or `@modal.asgi_app` (FastAPI):

- Mounts the **same Volume** read-only — single source of truth, no export/sync job.
- Scale-to-zero — $0 when nobody's looking; cold start a few seconds.
- One platform, one deploy, one set of credentials.

Render free web service as dashboard host:

- Ephemeral FS → **cannot hold the DB** → needs the daily job to *push* a snapshot
  (SQLite file or a JSON export) somewhere Render can pull → a sync mechanism + a
  second copy + "is the dashboard showing stale data" as a new failure mode.
- Spins down after 15 min idle → ~50 s cold start.
- Second platform, second deploy, second dashboard of credentials.
- The *only* thing it buys: a fixed always-the-same URL and a marginally more "normal web
  app" story. Not worth a database-replication problem for a research project.

**Modal.** Name Render as the fallback if Modal web endpoints prove awkward in practice, but
go in expecting Modal.

### 4.4 Data refresh (prices + manual VIX load) → **Modal**

`modal run refresh` — same container image, same Volume, same copy-local/commit pattern.
Needs outbound HTTPS to `www.niftyindices.com` only (Data agent's grant). The VIX CSV
arrives via `modal volume put` before the run. Keeping it on Modal means one image, one
store, one mental model.

### 4.5 Net: **Modal for all four pieces.** No mix.

The proposal's "Modal / Render" was an either/or placeholder from before the security layer
was SQLite-bound and before the VIX-manual constraint was load-bearing. With both facts in
hand, splitting across platforms only adds a DB-sync problem. Single platform is the honest
call.

---

## 5. Dashboard scope — minimal, honest, per proposal §5.4

Proposal §5.4 asks for: current detected regime, model confidence, current
prediction-or-abstention status, and a link into the audit trail for that decision.

### 5.1 One page, three sections

**A. Today's call** (from the latest `todays_call_log` row):
- `as_of_date`, `generated_at`, and a plain **"data is N trading days stale"** banner when
  `as_of_date` lags today — the single most important honesty signal on the page.
- Detected regime: `regime.point_in_time_id` + `character` (e.g. "regime 2 — risk-on / low-vol")
  and `run_length_trading_days`.
- Reliability tier: `monitored` / `suppressed` / `none`, and disposition:
  **`INFORMATIONAL`** or **`ABSTAIN`**, shown as the headline.
- If `INFORMATIONAL`: the `regime_pattern.statement` **and** the `not_an_edge` disclaimer,
  verbatim — never the pattern without the disclaimer. The directional lean shown only with
  its "INFORMATIONAL ONLY — not a trade recommendation" note, or "signal model stale" when
  null.
- If `ABSTAIN`: `abstention.reasons` (the M-codes) and `abstention.plain_language`.
- Monitoring strip: `tier1_drift_mean`, `tier1_fires`, `tier2_shadow_ari`,
  `trailing_vix_percentile`, `trading_days_since_fit`, `circuit_breaker`,
  `recalibration_flag`.

**B. Why — the audit trail** (`orchestrator.explain(as_of)`):
- The ordered agent-call trace: `regime.get_active_version → point_in_time_label →
  check_drift → trailing_vix_pct → validation.get_reliability_tiers →
  training.get_signal_prediction`, each with `status` and a compact `result_summary`.
- The `trace_id`, and the result of `verify_chain` on both `agent_call_log` and
  `todays_call_log` — a green "audit chain intact (N rows)" / red "broken at row K". This
  is the Phase 5 tamper-evidence claim made visible.

**C. Recent history** (small table): last ~20 `todays_call_log` rows — date, regime,
tier, disposition, lean — so the page shows the abstention behaviour over time, not just a
single snapshot.

### 5.2 Deliberately NOT in v1

- No date picker / historical replay UI. `explain(as_of)` is wired for it; add later if
  wanted. v1 shows "today".
- No charts. A regime-over-time strip is a nice-to-have, explicitly deferred.
- No auth / multi-user. Single public read-only page. Nothing sensitive — there is no trade
  capability and no PII.
- No write actions from the dashboard. Refresh and retraining stay `modal run`.
- No live polling. The data changes once a day; a manual refresh button + `volume.reload()`
  is enough.

### 5.3 Implementation

Streamlit (there is a project skill for it; lowest-effort path to a clean read-only page) or
a single FastAPI + one Jinja template. Either is ~150–200 lines reading the DB and calling
`orchestrator.explain`. Recommend **Streamlit** unless review prefers the smaller
dependency footprint of FastAPI. Read-only DB connection; no `agents`-layer writes.

---

## 6. Cost & complexity honesty

**Flagged as over-engineering if we did them — we won't:**

- *Always-on anything.* No 24/7 web service, no always-on worker. A daily batch + a
  scale-to-zero endpoint is the correct weight. Rejecting a Render paid web-service-with-disk
  for the dashboard is largely this point.
- *Managed Postgres / RDS.* §3.1 — breaks the security layer, isn't free, solves a problem
  we don't have (there is no concurrent-writer workload; it's one batch writer per day).
- *A real scheduler / queue / Airflow / Dagster.* One `modal.Cron` line. The pipeline is
  operator-gated by the manual VIX step anyway (§1.1).
- *An LLM orchestrator in the deployed path.* Explicitly out of scope since Phase 5; the
  deterministic client is the point.
- *Building the in-process broker just for deployment* (§2.3). The 1–2 min cold start it
  would save is irrelevant for a daily job.
- *Multi-region, HA, backups-as-a-service.* A `modal volume get` cron into cloud storage (or
  an occasional `git`-tracked seed refresh) is a proportionate backup for a single-operator
  research system. Anything more is theatre.

**Genuine cost:**

- Modal Starter: **$30/mo credit, no card.** Estimated real usage: ~5–20 min of CPU-minutes
  a day (one MCP decide run + occasional refresh/retrain) + a scale-to-zero endpoint serving
  tens of requests. Well inside the free credit. If it ever isn't, the fix is "run the
  direct path in the cron and keep MCP for on-demand" — a config change.

**Genuine added complexity (accepted, small):**

- The copy-local → run → commit-back pattern in two entrypoints (~20 lines each).
- `volume.reload()` discipline in the dashboard.
- The Volume-vs-git-DB source-of-truth clarification (§3.5) — a docs/working-practice note.
- A Modal image definition (`requirements.txt` is already fully pinned — this is
  straightforward).

---

## 7. Proposed architecture (concrete)

```
Modal app: regimeguard
├── Volume: regimeguard-data                 # canonical regimeguard.db  (survives all deploys)
│
├── image = Image.debian_slim(python="3.12")
│           .pip_install_from_requirements("requirements.txt")
│           .add_local_dir(".", "/app", ignore=[data/raw, *.db, .git, ...])
│
├── @app.function(volumes={"/data": vol}, schedule=Cron("30 12 * * 1-5", tz="Asia/Kolkata"))
│   def decide():                            # scheduled staleness-guard + operator `modal run`
│       cp /data/regimeguard.db /tmp/db ; REGIMEGUARD_DB=/tmp/db
│       rec = agents.orchestrator.todays_call(log=True)      # 4 MCP subprocesses, ~1–2 min
│       checkpoint WAL ; cp /tmp/db /data/regimeguard.db ; vol.commit()
│       print(rec.disposition, rec.reliability_tier)
│
├── @app.function(volumes={"/data": vol})    # operator-initiated only
│   def refresh(vix_csv_name: str | None):   # `modal run regimeguard::refresh`
│       cp local ; fetch_niftyindices (prices) ; load_india_vix_csv (from /data/raw/...) ;
│       rebuild_calendar ; find_gaps  → STOP on any unexplained gap ;
│       cp back ; vol.commit()
│
├── @app.function(volumes={"/data": vol})    # operator-initiated only, one at a time
│   def retrain_regime(cutoff=None) / retrain_signal(...) / run_validation(...)
│       # thin wrappers over existing CLIs; review between stages; each its own commit
│
└── @app.function(volumes={"/data": vol}, min_containers=0)
    @modal.web_server(8000)                  # or @modal.asgi_app()
    def dashboard():
        vol.reload() ; open /data/regimeguard.db read-only ;
        render §5.1 A/B/C  (latest todays_call_log + orchestrator.explain + verify_chain)
```

New files (all under a new `deploy/` package, nothing in Phases 1–5 changes):

- `deploy/modal_app.py` — the app, image, Volume, the functions above.
- `deploy/dashboard.py` — Streamlit (or FastAPI + `deploy/templates/call.html`).
- `deploy/db_sync.py` — the copy-local / checkpoint / commit-back helper (~30 lines), shared
  by every function.
- `docs/phase6_working_notes.md` — dated evidence log, per prior-phase standard.
- `requirements.txt` — add `modal`, and `streamlit` (or nothing extra if FastAPI, since
  `mcp` already pulls a Starlette/uvicorn stack — to verify).
- Possibly a one-line change: `load_india_vix_csv(csv_path, conn=None)` to accept an
  injected connection (Phase 5 already flagged this as the one signature gap; only needed if
  `refresh` routes it through the scoped path — otherwise it opens its own and is fine).

**Not changed:** every module in `data_agent/`, `regime_detection/`, `signal_model/`,
`agents/`. No research, model, threshold, or decision-logic edit. `tests/` unchanged; a
small `deploy/` smoke test added.

---

## 8. Open questions for review

1. **MCP path vs direct path in the scheduled job** — proposal recommends the MCP path so
   the deployed system exercises Phase 5 and `explain()` has a trace. Accept the ~1–2 min
   cold start, or prefer the fast direct path in cron with MCP only on `modal run`?
2. **The optional weekday `modal.Cron`** — worth having a scheduled `decide` that will often
   just re-emit `ABSTAIN / stale`, purely to keep the dashboard's staleness banner honest
   and catch "operator forgot"? Or make *everything* `modal run` and have no cron at all?
3. **Dashboard framework** — Streamlit (project skill exists, fastest to a clean page) vs
   FastAPI + one template (smaller dep surface)?
4. **The git-tracked `data/regimeguard.db`** (§3.5) — keep tracking it as an occasional seed
   snapshot, or drop it from git and document a seed procedure?
5. **`deploy/` vs `agents/` placement** — new top-level `deploy/` package proposed; any
   preference to nest it elsewhere?
6. **Backup** — is "occasional `modal volume get` + the git seed" enough, or do you want a
   scheduled Volume→object-storage dump as part of Phase 6?

No code until these are settled and the design is approved.
