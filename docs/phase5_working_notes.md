# Phase 5 Working Notes — MCP Multi-Agent Architecture + Security-by-Design

Running, dated, append-only evidence log for Phase 5, same standard as the earlier
phase notes: raw material for `FINDINGS.md`, not the polished writeup. Approved design:
`docs/phase5_design_proposal.md`. This phase repackages Phases 1–4 — it re-derives no
research finding.

---

## 2026-09-06 — Steps 1–5: capabilities + audit chain + broker + bootstrap + deny-path suite — 47/47 green

New (`agents/`):
- `capabilities.py` — the whole policy in one place: `AgentName`, `Grant`, `GRANTS`,
  `CALL_GRAPH`. Explicit allowlists (read/write per table), empty `net_hosts` for
  everyone but Data, per-agent `fs_write_prefixes`, `may_call` (only Orchestrator →
  {Data, Regime, Training, Validation}; nothing calls Orchestrator or Broker; no
  self-calls; asserted at import). `TRADE_VOCABULARY = ()` — intentionally empty, and
  a test keeps it empty.
- `audit.py` — `agent_call_log` schema + a generic SHA-256 hash chain
  (`chain_append` / `verify_chain`) shared by `agent_call_log` and `todays_call_log`.
  `row_hash = sha256(prev_hash ‖ canonical_json(business fields))`, genesis = 64 zeros.
  Plain chain, no HMAC (review decision 3). Bounded `SQLITE_BUSY` retry on the insert.
- `broker.py` — the enforcement layer:
  - `scoped_connection(agent)` — a real `sqlite3` connection with `set_authorizer`
    wired from the agent's `Grant`. `SQLITE_READ` on an unlisted table → `DENY`;
    `INSERT/UPDATE/DELETE` outside `db_write` → `DENY`; **all DDL → `DENY` for every
    agent** (schema is bootstrap-only); PRAGMA allowlist; `journal_mode` read-only for
    agents; fail-closed on any unrecognised action. `busy_timeout = 5000` +
    `foreign_keys = ON` on every handle.
  - `restrict_network(hosts)` — patches `socket.create_connection` and
    `socket.socket.connect`; blocks any host not in `hosts` (empty ⇒ all blocked).
  - `ScopedFS` — path-prefix write guard, also rejects escapes above the repo root.
  - `brokered_call(...)` — the single invocation path: enforces `CALL_GRAPH`, applies
    Validation's no-recent-dates rule, runs the tool on a scoped handle, and writes
    the `agent_call_log` row (**including on denial**) via the hash chain, then raises
    on denial/error. The audit record is produced by the enforcer, not the tool.
- `bootstrap.py` — one-time privileged setup (unscoped). Ensures every schema, applies
  the additive `todays_call_log` migration (`trace_id`, `prev_hash`, `row_hash`), sets
  `PRAGMA journal_mode = WAL`. Idempotent. **Ran once against the real
  `data/regimeguard.db`**: WAL set, 3 nullable columns added to the (empty)
  `todays_call_log`, `agent_call_log` created empty. No data touched; both chains
  verify (n=0).

### `tests/test_agent_scoping.py` — 47 passed

The deliverable that makes §2 a fact. Runs each agent's *real* scoped handle against
an isolated bootstrapped DB copy and asserts:

- **DB denials**: Validation `SELECT todays_call_log` / `SELECT agent_call_log` /
  any `INSERT` → denied; Regime `INSERT signal_model_versions` / `SELECT
  signal_model_versions` → denied; Training `INSERT model_versions` → denied; Data
  `INSERT todays_call_log` / `SELECT model_versions` → denied; Orchestrator `INSERT
  model_versions` → denied; Data reading `current_regime_labels` (view) → denied
  because the underlying `regime_labels` read is still checked.
- **DDL denied for all six identities** — `CREATE TABLE` / `DROP TABLE` / `ALTER
  TABLE … ADD COLUMN` / `CREATE INDEX` all raise (24 param cases).
- **Positive controls** — Data `INSERT ingestion_runs`, Regime read `daily_bars` +
  `current_bars`, Training read `model_versions`, Validation read `signal_predictions`,
  Orchestrator read+write `todays_call_log` / read `agent_call_log` all succeed (so
  it is scoping, not deny-all).
- **Network** — `restrict_network(∅)` blocks `create_connection` and raw
  `socket.connect`; Data's host set blocks `evil.example`; the other four agents have
  `net_hosts == ∅`.
- **Filesystem** — Training may write `signal_model/results/`, not `data/raw/` /
  `regime_detection/results/` / outside the repo; Data may write `data/raw/`;
  Orchestrator may write nowhere.
- **Call graph** — no self-calls, nothing calls Orchestrator/Broker, only Orchestrator
  has callees; `brokered_call(Data→Regime)` raises `ScopeError` and logs a `denied`
  row; `brokered_call(Orchestrator→Regime)` runs and logs an `ok` row; a tool that
  tries an out-of-scope query is denied by the scoped handle and recorded as `denied`.
- **Validation date rule** — `brokered_call(Orchestrator→Validation, {as_of: <last
  bar>})` → denied (`historical-only`); an old date (`2016-01-04`) runs.
- **No trade capability** — `TRADE_VOCABULARY == ()`; no table name contains
  order/trade/execution/position/fill; **no module anywhere imports a known
  broker/execution SDK** (ccxt, kiteconnect, alpaca, ib_insync, … — source scan).
- **Audit chain** — 3 appends verify; a single `UPDATE … SET status='TAMPERED'`
  makes `verify_chain` return `{ok: False, id: 2, reason: "row_hash mismatch"}`.

One fix during the run: SQLite phrases an authorizer READ denial as "access to
`<t>.<c>` is prohibited" (vs "not authorized" for writes/DDL). `broker._is_authorizer_denial`
now accepts either, and the tests match `not authorized|prohibited`.

**Next:** step 6 (extract `assemble_regimeguard_call`), then steps 7–8 (four MCP
stdio servers + the MCP-client Orchestrator).

---

## 2026-09-06 — Steps 6–8: shared assembler + four MCP servers + MCP-client orchestrator — direct and MCP paths produce byte-identical records

### Step 6 — `assemble_regimeguard_call` (the shared pure function)

`signal_model/todays_call.py` split into:
- `gather_inputs(conn, as_of, regime_version_id)` — the direct-path gather; returns a
  JSON-safe `inputs` dict (resolved as_of, slim regime version, PIT label + OOD inputs,
  `{tier1_mean, tier1_fires, tier2_ari}`, VIX percentile, `td_since_fit`, the tier
  records, the signal prediction).
- `assemble_regimeguard_call(inputs, *, generated_at, code_rev)` — **PURE**: the single
  implementation of the M1–M6 tier/disposition logic and the `RegimeGuardCall` record.
  No DB, no clock, no filesystem. Deterministic given its inputs (verified:
  `json.dumps` of two calls on the same inputs is identical).
- `decide_todays_call` = `gather_inputs` → `assemble_regimeguard_call` → optional log.
  Behaviour unchanged (LIVE → INFORMATIONAL/monitored, COVID → ABSTAIN/suppressed).
- `td_since_fit` moved into `point_in_time_regime_label`'s return (and its `as_of` is
  now an iso string; it also resolves `as_of=None` to the last feature row) so both
  gather paths get it identically.
- `append_to_log` now writes through `agents.audit.chain_append` (lazy import — the
  dependency runs `agents/` → `signal_model/`, never the reverse), so the direct path
  and the orchestrator path share **one** chained `todays_call_log` writer.
  `ensure_log_schema` is now existence-checked so it is a no-op (DDL-free) on a
  capability-scoped connection.

### Steps 7–8 — the MCP layer (`agents/servers/`, `agents/orchestrator.py`)

Real MCP (`mcp` 1.28.1). Four `FastMCP` stdio servers, one per agent; the orchestrator
is a real `ClientSession` MCP client that spawns them as subprocesses and calls tools
by name. Every tool body goes through `_common.run_tool` → `broker.brokered_call`, so
it runs on a **callee-scoped** connection (the authorizer is the enforcement) and the
call — ok, denied, or error — is written to the hash-chained `agent_call_log` by the
broker.

Tools registered: regime (7: `get_active_version`, `load_version`, `point_in_time_label`,
`check_drift`, `trailing_vix_pct`, `fit_and_register`, `quarterly_walk`), training (4:
`get_active_signal_version`, `load_signal_version`, `get_signal_prediction`,
`fit_and_register_signal`), validation (2: `get_reliability_tiers`,
`regenerate_reliability_tiers`), data (2: `get_bars`, `find_gaps`).

`REGIMEGUARD_DB` env var (read by `broker.db_path()`) points the whole agent layer at
an isolated DB copy for testing without touching the real store.

### End-to-end verification (isolated bootstrapped DB copy)

`orchestrator._gather_via_mcp` vs `gather_inputs`, then the same
`assemble_regimeguard_call(generated_at="FIXED", code_rev="REV")` on both:

| case | disposition / tier | inputs identical | **record byte-identical** |
|---|---|---|---|
| LIVE (as_of=latest, active version) | INFORMATIONAL / monitored | yes | **yes** |
| COVID 2020-03-25 (historical replay) | ABSTAIN / suppressed | yes | **yes** |
| calm 2019-11-15 (historical replay) | ABSTAIN / none | yes | **yes** |

- `orchestrator.todays_call` and `decide_todays_call` both logged; both
  `todays_call_log` chains and the `agent_call_log` chain (**24 rows written by 4
  concurrent server subprocesses**) `verify_chain` clean — WAL + `busy_timeout` +
  the broker's `SQLITE_BUSY` retry held under real multi-process writes.
- `explain(as_of, trace_id)` returns the decision row plus the ordered agent-call
  trace: `regime.get_active_version → point_in_time_label → check_drift →
  trailing_vix_pct → validation.get_reliability_tiers → training.get_signal_prediction`,
  all `ok`.
- Deny-path suite still 47/47 after the broker `db_file`/env changes.
- Real `data/regimeguard.db` untouched by the e2e (0 rows in both log tables); only
  the earlier one-time `bootstrap` migration is on it.

Fixes during the run: `point_in_time_label` accepts `as_of=None`; FastMCP rejects bare
`-> float` / `-> list` return annotations (pydantic output-model build) so tool return
annotations were dropped and `_common.run_tool` wraps every result as `{"data": ...}`;
`model_version_id` → `regime_version_id` (pydantic `model_` protected-namespace warning).

### Scope of the security/audit guarantees — known current limitation

The capability-scoping (SQLite authorizer, network/FS guards) and the hash-chained
`agent_call_log` currently apply to **the `todays_call` path specifically**: every
read the orchestrator performs to assemble a decision goes through the broker on a
callee-scoped connection and is recorded.

They do **not** yet cover the whole pipeline. Data ingestion
(`fetch_niftyindices.ingest_index`, `load_india_vix_manual.load_india_vix_csv`,
`calendar_days.load_calendar_days`) and the two heavier validation entry points
(`run_signal_model.compute_oof_predictions`,
`run_pit_evaluation.main` / `regime_reliability_tiers` regeneration) still call
`data_agent.db.get_connection()` directly — an unscoped read/write handle — and are
**not** routed through the broker, so those operations are neither capability-scoped
nor written to `agent_call_log`. This is deliberate and consistent with review
decision 5 (those are consequential, operator-initiated stages run one at a time via
the existing CLIs, not orchestrated). It is a real scope boundary of this phase, not
full end-to-end coverage: wrapping those stages as scoped, brokered MCP tools (and
routing their file writes through `ScopedFS`) is follow-on work.

### Deferred (noted, not blocking the `todays_call` deliverable)

- `data.ingest_prices` / `data.load_vix_csv` (network fetch; `load_india_vix_csv` needs
  a `conn` parameter), `validation.compute_oof_predictions` /
  `run_regime_stratified_eval` (own `get_connection` + file writes) — operator runs
  these via the existing CLIs per review decision 5; wiring them as scoped MCP tools
  is follow-on.
- `ScopedFS` is implemented and tested as a guard, but tools that write files
  (`regenerate_reliability_tiers`) call pandas `to_csv` directly rather than routing
  through it — the `todays_call` path writes no files, so this doesn't bite it.
- Server startup imports lightgbm/sklearn/jumpmodels per subprocess → one
  `todays_call` over MCP takes ~1–2 min. Fine for an "ask for today's call"
  invocation; not a hot path.
