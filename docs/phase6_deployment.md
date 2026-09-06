# Phase 6 — Deployment runbook

Approved design: `docs/phase6_design_proposal.md`. Evidence log:
`docs/phase6_working_notes.md`.

Everything runs on **Modal**: one app (`regimeguard`), one Volume
(`regimeguard-data`) holding the single SQLite `regimeguard.db`, one image. SQLite
stays because the Phase 5 security layer is built on `sqlite3`'s statement authorizer,
which has no Postgres equivalent — see the design proposal §3.1.

**No `git commit` / `git push` is ever run by this project's tooling or by Claude.**

---

## 0. Components

| Modal object | What it is |
|---|---|
| Volume `regimeguard-data` | canonical `/regimeguard.db` — survives every deploy, restart, and redeploy |
| function `decide_scheduled` | `modal.Cron("30 13 * * 1-5", tz="Asia/Kolkata")` — 19:00 IST weekdays. Runs the **MCP orchestrator path** (four scoped stdio subprocesses + hash-chained `agent_call_log`). |
| function `decide` | the same, on demand: `modal run deploy/modal_app.py [--as-of YYYY-MM-DD]` |
| web function `dashboard` | scale-to-zero Streamlit, read-only. `https://naveenreddie-think--regimeguard-dashboard.modal.run` |

The daily job is deliberately the real MCP path, not the fast in-process
`decide_todays_call`: the deployed system is the artifact for Phase 5, and
`orchestrator.explain` / the dashboard's "Why" panel need the `agent_call_log` trace.
Cost is ~1–2 min of CPU per run (four subprocesses each import lightgbm / jumpmodels /
sklearn) — irrelevant for a once-a-day batch.

---

## 1. Prerequisites

```bash
pip install -r requirements.txt          # includes modal==1.5.5, streamlit==1.59.2
modal token new                          # or have ~/.modal.toml with a token
modal profile current                    # should print your workspace
```

**Windows / Git Bash:** always export these for the `modal` CLI, or paths and the
Rich console break:

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # Rich prints ✓/… ; cp1252 console chokes
export MSYS_NO_PATHCONV=1                    # stop Git Bash rewriting /regimeguard.db
```

---

## 2. Seeding the Volume (one time)

`data/regimeguard.db` is **not** tracked in git (the Volume is the source of truth; a
stale tracked binary is worse than none). To stand up a fresh deployment you need a
`regimeguard.db` locally — build one from the pipeline (`agents.bootstrap` + the
Phase 1–4 ingestion/fit CLIs) or restore a backup — then:

```bash
modal volume create regimeguard-data                      # if it doesn't exist
modal volume put regimeguard-data ./data/regimeguard.db /regimeguard.db --force
modal volume ls regimeguard-data                          # expect exactly: regimeguard.db
```

The file must already have the Phase 5 schema (run `python -m agents.bootstrap`
against it first if unsure — it is idempotent and sets `journal_mode=WAL`).

---

## 3. Deploy

```bash
modal deploy deploy/modal_app.py
```

Builds the image (pins from `requirements.txt` + `libgomp1` for lightgbm), bakes the
current short git SHA into `REGIMEGUARD_CODE_REV` (so the container — built without
`.git` — still records provenance in the audit tables), registers the two `decide`
functions + the weekday cron, and publishes the dashboard URL.

Redeploy any time; the Volume is untouched by deploys.

---

## 4. Daily operation

The pipeline is **operator-gated** — India VIX is a manual NSE download (automated
access is hard-blocked; Phase 1). A blind cron cannot make the data fresh.

**Normal day (operator):**
1. Download the latest India VIX CSV from NSE's historical-VIX report page.
2. Refresh a local `regimeguard.db` with the existing CLIs
   (`data_agent/load_india_vix_manual.py`, `data_agent/fetch_niftyindices.py`,
   `data_agent/calendar_days.py` — stop on any unexplained gap), then
   `modal volume put regimeguard-data ./data/regimeguard.db /regimeguard.db --force`.
   *(Wrapping this as a scoped Modal `refresh` job is the obvious next increment — see
   §7. It was deliberately not in Phase 6's build scope.)*
3. `modal run deploy/modal_app.py` — produces today's decision on the fresh data.

**Any day (automatic):** the `decide_scheduled` cron fires at 19:00 IST Mon–Fri. If
nobody refreshed, it re-runs on the stale data and the decision correctly carries
`M4_stale` / an `ABSTAIN`, and the dashboard shows a "data is N days behind" banner.
That is the point — a silently frozen dashboard would be the real failure mode.

**Look at it:** the dashboard URL above. It snapshots the Volume every 5 minutes.

---

## 5. Backup

- Ad hoc: `modal volume get regimeguard-data /regimeguard.db ./backup/regimeguard-$(date +%F).db`
- The whole store is reproducible from `data/raw/**` + `data/manual_drops/**` (tracked)
  via the Phase 1–4 CLIs if ever truly lost.

A scheduled Volume→object-storage dump was considered and **rejected** for this stage
(design proposal §6) — disproportionate for a single-operator research system.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: No database on the Volume` | Volume not seeded — §2. |
| dashboard first hit times out | scale-to-zero cold start (image pull + heavy imports). Retry after ~1–2 min; subsequent hits are fast. |
| `modal` CLI crashes with `'charmap' codec` | set `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (§1). |
| `modal volume put` lands file at `C:/...` | Git Bash path conversion — set `MSYS_NO_PATHCONV=1` (§1). |
| decision has `code_rev` `unknown` | image built somewhere without git; harmless, or redeploy from a git checkout. |

---

## 7. Scope boundary (what Phase 6 deliberately did **not** build)

- A scoped Modal **`refresh`** job (prices + manual-VIX load + calendar rebuild).
  Operator uses the existing local CLIs + `modal volume put` for now. Additive.
- Scoped Modal **retraining** wrappers (`regime.fit_and_register`,
  `training.fit_and_register_signal`, validation runs). Still operator-initiated, one
  at a time, via the existing CLIs (Phase 5 review decision 5). Additive.
- Historical-replay UI in the dashboard (`explain(as_of)` is already wired for it).
- Any auth, any write path from the dashboard, any always-on compute, any managed
  database. All by design — see the design proposal §5.2 / §6.
