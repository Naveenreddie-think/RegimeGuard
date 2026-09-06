# Phase 6 Working Notes — Deployment

Dated, append-only evidence log, same standard as the earlier phase notes. Approved
design: `docs/phase6_design_proposal.md`. Runbook: `docs/phase6_deployment.md`.
This phase changes no research finding, model, threshold, or decision logic.

---

## 2026-09-06 — deploy/ built, deployed to Modal, verified end-to-end

### What was built

- `deploy/db_sync.py` — the copy-local / checkpoint / commit-back helper.
  `local_db_session()`: Volume file → local container disk, `REGIMEGUARD_DB` pointed
  at the copy (the four MCP subprocesses inherit it via `os.environ`), run the work,
  `PRAGMA wal_checkpoint(TRUNCATE)`, copy the single file back; caller commits the
  Volume. `read_only_snapshot()` for readers. No `modal` import — unit-testable.
- `deploy/modal_app.py` — one `modal.App`, one Volume `regimeguard-data`, one image
  (`debian_slim` + `libgomp1` + pinned `requirements.txt`; `workdir("/app")` + `.env`
  **before** `add_local_dir`, which must be last). Functions: `decide_scheduled`
  (`Cron("30 13 * * 1-5", tz="Asia/Kolkata")`), `decide` (on-demand + local
  entrypoint), `dashboard` (`@modal.web_server(8000)`, `min_containers=0`). The short
  git SHA is resolved client-side at deploy time and baked into
  `REGIMEGUARD_CODE_REV`.
- `deploy/dashboard.py` — Streamlit, read-only, three sections exactly as scoped
  (§5.1): **Today's Call** (regime, tier, disposition, INFORMATIONAL pattern always
  with its `not_an_edge` disclaimer / ABSTAIN reasons, monitoring strip, staleness
  banner), **Why** (the agent-call trace — same query as `orchestrator.explain` —
  plus `verify_chain` on both hash chains, shown green/red), **Recent history** (last
  20). Snapshots the Volume every 5 min; `REGIMEGUARD_DASHBOARD_DB` overrides the
  source for local dev.
- `signal_model/todays_call.py` — `_code_rev()` gains a `REGIMEGUARD_CODE_REV`
  fallback for when `git` is unavailable (deployed container). `import os` added.
  **Only Phase 1–5 file touched**; additive; 47/47 existing tests still green.
- `requirements.txt` — `modal==1.5.5`, `streamlit==1.59.2` added.
- `data/regimeguard.db` — **untracked** (`git rm --cached` + `.gitignore`). The Volume
  is the source of truth; seed procedure documented in the runbook.

### Verification — real runs, not "should work"

**Local (scratch DB copy, `REGIMEGUARD_DB` override):**
- `python -m agents.orchestrator todays-call` → `INFORMATIONAL / monitored`,
  `trace_id orch-924c2d14…`, wrote 1 `todays_call_log` + 6 `agent_call_log` rows,
  both chains `verify_chain` clean.
- `streamlit.testing.v1.AppTest` on `deploy/dashboard.py` → no exceptions, no
  `st.error`; headers / metrics / staleness warning / both "hash chain intact"
  successes / trace df (6×7) / history df all render. `not_an_edge` disclaimer present.

**Modal (deployed app `regimeguard`):**
- `modal deploy` — image built (lightgbm, jumpmodels, hmmlearn, mcp, streamlit all
  install clean on `debian_slim` + `libgomp1`); 3 functions + the weekday cron
  registered; dashboard at
  `https://naveenreddie-think--regimeguard-dashboard.modal.run`.
- `modal run deploy/modal_app.py` (three separate container invocations —
  latest, `--as-of 2020-03-25`, latest again) and
  `modal run …::decide_scheduled` once:

  | run | as_of | disposition / tier | trace_id | code_rev |
  |---|---|---|---|---|
  | 1 | 2026-08-21 | INFORMATIONAL / monitored | orch-10a144ef… | bf69380 |
  | 2 | 2020-03-25 | **ABSTAIN / suppressed** | orch-fd3444b6… | bf69380 |
  | 3 | 2026-08-21 | INFORMATIONAL / monitored | orch-b6e07abb… | bf69380 |
  | 4 (scheduled fn) | 2026-08-21 | INFORMATIONAL / monitored | orch-a680cfcf… | bf69380 |

- `code_rev` = `bf69380` came from the **baked env var** (git absent in the
  container) — the `_code_rev()` fallback works as intended. Precedence verified
  locally: when `.git` is present, git wins.
- **Persistence across restarts:** after runs 1–3, `modal volume get` → the Volume
  `regimeguard.db` has **3** `todays_call_log` rows (ids 1/2/3, append-only) and
  **18** `agent_call_log` rows (6 per run); both hash chains `verify_chain` clean;
  `journal_mode = wal`. The copy-local → run → checkpoint → commit cycle survives
  separate container lifecycles — the core Phase 6 requirement.
- MCP path in-container: server logs show `CallToolRequest` for all six tools
  (`get_active_version → point_in_time_label → check_drift → trailing_vix_pct →
  validation.get_reliability_tiers → training.get_signal_prediction`), i.e. the four
  scoped stdio subprocesses spawn and respond inside one Modal container. Wall time
  ~35–40 s with a warm image.
- **Dashboard:** `/_stcore/health` → 200 (after a ~1–2 min cold start); `GET /` →
  200; Streamlit `Uvicorn server started on 0.0.0.0:8000`, no tracebacks in logs.
  `AppTest` against the **pulled remote DB** (3 calls incl. the ABSTAIN) renders the
  3-row history, both "chain intact" (18 / 3 rows), staleness banner ("16 calendar
  days behind"), monitoring metrics, 6-row trace — clean.

### Fixes made during the run

- Image build order: `.env()` after `.add_local_dir()` is rejected by Modal
  ("build step after add_local_*"). Moved `workdir` + `env` before; `add_local_dir`
  last.
- Dashboard trace table: `result_summary` column mixed struct/None → pyarrow
  `ArrowInvalid`. Render it as a compact JSON **string** instead of nested objects.
- `use_container_width=True` → deprecated; switched to `width="stretch"`.
- Dashboard snapshot path: hardcoded `/tmp` → `tempfile.gettempdir()` (+ `mkdir`),
  so it works on Windows for local dev and `/tmp` in the container.
- Windows tooling: `modal` CLI needs `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (Rich
  console vs cp1252) and `MSYS_NO_PATHCONV=1` (`/regimeguard.db` was being rewritten
  to `C:/Program Files/Git/regimeguard.db`, creating a stray `/C:` dir on the Volume —
  removed with `modal volume rm -r`). Documented in the runbook.

### Scope boundary (deferred, not blocking — see runbook §7)

- No scoped Modal `refresh` job yet (prices + manual-VIX load + calendar). Operator
  uses the existing local CLIs + `modal volume put`.
- No scoped Modal retraining wrappers — operator-initiated via existing CLIs, one at
  a time (Phase 5 review decision 5, unchanged).
- Dashboard has no historical-replay picker (v1 shows "today"; `explain(as_of)` is
  already wired for it), no auth, no write path.
