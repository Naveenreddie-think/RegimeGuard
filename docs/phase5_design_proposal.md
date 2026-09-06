# Phase 5 Design Proposal — MCP-Orchestrated Multi-Agent Architecture + Security-by-Design

Status: **finalized after review (2026-09-06). Cleared for implementation.** Covers
proposal §5.2 (multi-agent, MCP-orchestrated architecture) and §5.3 (security-by-design:
tool-permission scoping + signed, tamper-evident audit trail).

### Review decisions locked

1. **Real MCP**, with an in-process broker as the documented fallback. Build order:
   broker + `capabilities.py` + SQLite authorizer + audit chain + deny-path tests
   **first**; MCP server/client layer on top.
2. `agent_call_log` lives in the **main `data/regimeguard.db`** (join-ability with
   `todays_call_log` / `model_versions` outweighs marginal isolation; the authorizer
   blocks non-broker writes regardless of file).
3. **Plain SHA-256 hash chain, no HMAC** — HMAC implies a key-management story this
   project does not have; a plain chain matches the honest "tamper-evident, not
   tamper-proof" scope without overclaiming.
4. **Validation "historical only" is an explicit, checkable rule**: read-only access +
   `todays_call_log` / `agent_call_log` SELECT denied + **the broker rejects any
   Validation tool call whose date arguments fall within `EMBARGO_DAYS` of the last
   available bar**. Read-only alone restricts query *type*, not date *range* — the range
   bound has to be real.
5. **No single-button `refresh_pipeline`.** The operator invokes each consequential stage
   (`data.ingest_prices`, `regime.fit_and_register`, `training.fit_and_register_signal`,
   the validation runs) explicitly — consistent with the human-review-over-automation
   stance already established for recalibration.
6. **Concurrent-writer safety** (raised in review): WAL gives many-readers/one-writer at
   the *database-file* level, not per-table, so partitioning writes by agent role reduces
   but does not remove contention (e.g. a Data write vs. the broker's `agent_call_log`
   write). Every connection sets `PRAGMA busy_timeout = 5000`, and the broker wraps every
   write in a bounded retry on `SQLITE_BUSY` (see §4.3).

**This phase repackages Phases 1–4; it re-derives nothing.** Every tool below is a thin
adapter over an already-built, already-verified function. No research finding, model,
threshold, or evaluation is recomputed or revalidated — only the seam between components
changes.

---

## 0. What exists (the surface being wrapped)

| Layer | Module | Public functions that become tool interfaces |
|---|---|---|
| Data | `data_agent/fetch_niftyindices.py` | `ingest_index(conn, config, start, end)` |
| Data | `data_agent/load_india_vix_manual.py` | `load_india_vix_csv(csv_path)` *(opens its own conn — needs a `conn` param)* |
| Data | `data_agent/calendar_days.py` | `load_calendar_days(conn, start, end)`, `find_unexplained_gaps(conn, instrument_id, start, end)` |
| Data (shared read) | `data_agent/db.py` | `get_connection`, `current_bars` view |
| Regime | `regime_detection/fit_and_register.py` | `fit_and_register(conn, cutoff, k, jump_penalty, notes)` |
| Regime | `regime_detection/quarterly_walk.py` | `run_quarterly_walk(conn, k, jump_penalty)`, `point_in_time_regime_label(conn, model_version, as_of)`, `load_point_in_time_labels(conn, ids)` |
| Regime | `regime_detection/monitoring.py` | `check_recalibration_trigger(conn, model_version_id)` |
| Regime | `regime_detection/regime_db.py` | `get_active_model_version`, `load_model_version` (reads); `save_model_version`, `save_regime_labels` (writes — internal) |
| Regime (shared read) | `regime_detection/features.py` | `build_feature_matrix(conn)`, `FEATURE_COLUMNS` |
| Training | `signal_model/fit_and_register_signal.py` | `fit_and_register_signal(conn, as_of, coverage_td, notes)` |
| Training | `signal_model/registry.py` | `get_active_signal_model_version`, `load_signal_model_version`, `load_signal_prediction` (reads); `save_*` (internal) |
| Validation | `signal_model/run_signal_model.py` | `compute_oof_predictions()` |
| Validation | `signal_model/run_pit_evaluation.py` | `main()` → the PIT regime-stratified eval |
| Validation | `signal_model/evaluate.py` | `regime_stratified_metrics(...)` |
| Validation | `signal_model/significance.py` | `regime_permutation_test`, `block_bootstrap_metric`, `paired_block_bootstrap`, `benjamini_hochberg` |
| Validation | `signal_model/regime_reliability_tiers.py` | `compute_tiers(conn)`, `load_reliability_tiers(path)` |
| Orchestrator | `signal_model/todays_call.py` | `decide_todays_call(conn, as_of, ...)`, `append_to_log`, `RegimeGuardCall` |

Already `conn`-parameterised (broker-ready): `ingest_index`, `load_calendar_days`,
`find_unexplained_gaps`, `fit_and_register`, `run_quarterly_walk`,
`point_in_time_regime_label`, `check_recalibration_trigger`, all `regime_db` /
`registry` funcs, `build_feature_matrix`, `decide_todays_call`. **Needs a signature
change to accept an injected `conn`:** `load_india_vix_csv` (and the several `main()`
CLIs, which become tool wrappers).

---

## 1. Agent → MCP tool boundaries

Five agents. Four are MCP **servers** (stdio); the Orchestrator is an MCP **client** that
sequences them. Each server advertises **only its own** tool set — an agent literally
cannot enumerate or call another agent's tools (MCP tool discovery is the first scoping
layer; §2's broker is the second).

### 1.1 Data Agent — `regimeguard-data`

| Tool | Wraps | Notes |
|---|---|---|
| `data.ingest_prices(index, start, end)` | `fetch_niftyindices.ingest_index` (+ `INDEX_CONFIGS` lookup) | outbound HTTP to niftyindices.com only |
| `data.load_vix_csv(csv_path)` | `load_india_vix_manual.load_india_vix_csv` | `csv_path` must resolve under `data/raw/india_vix/manual/` |
| `data.rebuild_calendar(start, end)` | `calendar_days.load_calendar_days` | REPLACE semantics; logs an `ingestion_runs` row |
| `data.find_gaps(symbol, start, end)` | `calendar_days.find_unexplained_gaps` | read-only |
| `data.get_bars(symbol, start, end)` | new: `SELECT … FROM current_bars` | the read interface other agents *may* use instead of touching the DB (they mostly use `build_feature_matrix` directly on a read-only handle) |

**Internal / never exposed:** `fetch_chunk`, `land_raw`, `parse_row`, `load_bars`,
`chunk_date_range`, `validate_and_parse`, `_match_columns`,
`_verify_market_traded_normally`, `insert_bar`, `start/finish_ingestion_run`.

**Writes:** `daily_bars`, `instruments`, `ingestion_runs`, `calendar_days`. **Denied:**
everything else, all outbound HTTP except niftyindices.com, all filesystem except
`data/raw/**`.

### 1.2 Regime Detection Agent — `regimeguard-regime`

| Tool | Wraps | Notes |
|---|---|---|
| `regime.fit_and_register(cutoff, k, jump_penalty, notes)` | `fit_and_register.fit_and_register` | consequential — creates + activates a `model_versions` row; operator-initiated only (see §1.6 governance) |
| `regime.run_quarterly_walk()` | `quarterly_walk.run_quarterly_walk` | heavy one-shot backfill |
| `regime.point_in_time_label(as_of, model_version_id=None)` | `quarterly_walk.point_in_time_regime_label` (+ `get_active_model_version` when id omitted) | in-memory re-fit, **writes nothing**; returns regime id, run length, OOD inputs |
| `regime.check_drift(model_version_id=None)` | `monitoring.check_recalibration_trigger` | in-memory shadow-fit, **writes nothing**; returns tier1/tier2/recommendation |
| `regime.get_active_version()` / `regime.load_version(id)` | `regime_db.get_active_model_version` / `load_model_version` | read-only |

**Internal:** `features._*`, `compute_nifty_features`, `compute_vix_features`;
`jump_model_fit` / `hmm_fit` grid internals; `monitoring._reconstruct_pipeline`,
`compute_tier1_drift`, `compute_tier2_shadow_fit`; `regime_db.save_model_version`,
`save_regime_labels` (reachable **only** via `fit_and_register` / `run_quarterly_walk`).

**Writes:** `model_versions`, `regime_labels`. **Denied:** `signal_model_versions`,
`signal_predictions`, `todays_call_log`, `daily_bars`/`calendar_days`/`ingestion_runs`
writes (SELECT allowed), all HTTP, filesystem except `regime_detection/results/**`.

### 1.3 Model Training Agent — `regimeguard-training`

| Tool | Wraps | Notes |
|---|---|---|
| `training.fit_and_register_signal(as_of=None, coverage_td=63, notes)` | `fit_and_register_signal.fit_and_register_signal` | creates + activates a `signal_model_versions` row + its `signal_predictions` |
| `training.get_active_signal_version()` / `training.load_signal_version(id)` | `registry.get_active_signal_model_version` / `load_signal_model_version` | read-only |

**Internal:** `lgbm_model.*`, `walk_forward.generate_folds`, `target.compute_*`,
`baselines.momentum_baseline`, `registry.save_signal_model_version`,
`save_signal_predictions` (only via the fit tool).

**Writes:** `signal_model_versions`, `signal_predictions`. **Denied:** `model_versions`,
`regime_labels`, `daily_bars`, `todays_call_log` writes (SELECT allowed for
`model_versions`/`daily_bars`, needed to build features + resolve the active regime
version), all HTTP, filesystem except `signal_model/results/**`.

### 1.4 Validation Agent — `regimeguard-validation`

| Tool | Wraps | Notes |
|---|---|---|
| `validation.compute_oof_predictions()` | `run_signal_model.compute_oof_predictions` | the leakage-safe purged/embargoed walk-forward OOF run; writes `oof_predictions.csv`, `fold_summary.csv` |
| `validation.run_regime_stratified_eval(label_source)` | `run_pit_evaluation` / `run_signal_model` eval path | `label_source ∈ {"pit","hindsight"}`; writes `model_regime_stratified*.csv`, `significance_bh*.csv` |
| `validation.run_significance(test, series_ref, ...)` | `significance.regime_permutation_test` / `block_bootstrap_metric` / `benjamini_hochberg` | on the frozen OOF file |
| `validation.regenerate_reliability_tiers()` | `regime_reliability_tiers.compute_tiers` + write CSV | the `monitored`/`none` tier table `decide_todays_call` consumes |
| `validation.get_reliability_tiers()` | `regime_reliability_tiers.load_reliability_tiers` | read the CSV |

**Internal:** `evaluate.compute_metrics` / `strategy_pnl` / `sharpe_like` /
`max_drawdown`; `significance._block_bootstrap_indices` / `_block_permute`.

**Writes:** *no DB table*. Filesystem: `signal_model/results/**` only. **Denied:** all DB
writes (read-only handle), **SELECT on `todays_call_log` and `agent_call_log`** (it is an
evaluator, not an auditor, and must not see live decisions), all HTTP, and — the concrete
form of "historical only" — **its tools take no `as_of`/"today" argument**; they operate
on the frozen evaluation set (OOF file + historical `regime_labels`). The broker rejects
any Validation tool call carrying a date within `EMBARGO_DAYS` of the last available bar.

### 1.5 Orchestrator — `regimeguard-orchestrator` (MCP client)

| Tool (operator-facing) | Sequence |
|---|---|
| `orchestrator.todays_call(as_of=None)` | `regime.get_active_version` → `regime.point_in_time_label` → `regime.check_drift` → `validation.get_reliability_tiers` → `training.get_active_signal_version` + prediction read → **assemble `RegimeGuardCall`** (shared pure function, §5) → write `todays_call_log` |
| `orchestrator.explain(as_of)` | read the `todays_call_log` row + `SELECT * FROM agent_call_log WHERE trace_id = ? ORDER BY id` (the full call tree) |

There is **no `refresh_pipeline` tool** (review decision 5). The retraining flow is the
operator invoking each consequential stage explicitly, in order, reviewing between them:
`data.ingest_prices` → `data.rebuild_calendar` → `data.find_gaps` (stop on any
unexplained gap) → `regime.fit_and_register` → `training.fit_and_register_signal` →
`validation.compute_oof_predictions` → `validation.run_regime_stratified_eval` →
`validation.regenerate_reliability_tiers`. Each is its own `agent_call_log` entry.

**Writes:** `todays_call_log` only. **Denied:** direct HTTP, model fitting, DB writes to
any other table. It *only* sequences tool calls. No agent may call the Orchestrator
(acyclic call graph).

### 1.6 Governance notes (unchanged from Phase 2/4, restated at the tool layer)

- `regime.fit_and_register`, `training.fit_and_register_signal`, and
  `regime.run_quarterly_walk` are **consequential** (they change the active model). They
  are **operator-initiated only** — invoked one at a time, with review between stages
  (review decision 5); the Orchestrator never chains them autonomously. Every invocation
  is logged at `WARN`-equivalent prominence in `agent_call_log`.
- `orchestrator.todays_call` remains **non-actionable** (`RegimeGuardCall.actionable`
  is still a hard-coded `false`; Phase 4 §1.4).
- **The system has no trade capability at all** — see §2.4.

---

## 2. Tool-permission scoping — enforced, not documented

The Agent Security Testbed lesson, applied: **model/agent judgement is not the control
point; the tool layer is.** Enforcement must hold even if a server's tool code is buggy,
or an LLM is later dropped in as orchestrator and told to misbehave.

### 2.1 One declarative grant table — `agents/capabilities.py`

```python
class AgentName(str, Enum):
    DATA, REGIME, TRAINING, VALIDATION, ORCHESTRATOR = ...

@dataclass(frozen=True)
class Grant:
    db_write:  frozenset[str]         # tables this agent may INSERT/UPDATE/DELETE
    db_read:   frozenset[str] | ALL   # tables it may SELECT (ALL minus an explicit deny set)
    db_read_deny: frozenset[str]      # tables denied even for SELECT
    fs_write_prefixes: tuple[str, ...]
    net_hosts: frozenset[str]         # allowed outbound hosts (usually empty)
    may_call:  frozenset[AgentName]   # which other agents' tools it may invoke

GRANTS: dict[AgentName, Grant] = { ... }   # the whole policy, in one place
```

Changing what an agent can do = editing one dict. Everything below reads from it.

### 2.2 DB enforcement — SQLite's own authorizer (`sqlite3.Connection.set_authorizer`)

Each server, at startup, opens its DB connection and installs an **authorizer callback**
built from its `Grant`. SQLite invokes this callback while *compiling* every statement,
for every table/column/action touched, and a return of `SQLITE_DENY` makes the statement
fail before it executes — regardless of how the SQL was constructed. This is the same
primitive SQLite ships for sandboxing untrusted SQL; it is real, at-the-engine
enforcement, not a string check.

- `SQLITE_INSERT/UPDATE/DELETE` on a table not in `db_write` → `DENY`.
- `SQLITE_READ` on a table in `db_read_deny` (e.g. Validation → `todays_call_log`) → `DENY`.
- `SQLITE_CREATE_*` / `SQLITE_DROP_*` / `SQLITE_ALTER_TABLE` → `DENY` for **every** agent.
  Schema creation is a privileged bootstrap step (`agents/bootstrap.py`, run once with an
  unscoped handle); no agent can alter schema at runtime.
- The scoped connection is built by the broker and is the **only** DB handle a tool
  receives — a tool cannot call `get_connection()` to get an unscoped one (the broker
  passes `conn`; `get_connection` is not importable in the server's tool namespace, and a
  lint/test check enforces that).
- pandas `read_sql_query` / `to_sql` run on the same connection, so they are covered too.

A `PRAGMA journal_mode=WAL` is set once at bootstrap so the four server processes +
orchestrator can hold concurrent read connections with a single writer per table (writes
are already partitioned by agent role, so there is never write contention on one table).

### 2.3 Filesystem + network enforcement

- **`ScopedFS`**: every tool's file writes go through a wrapper that resolves the target
  path and rejects it unless it is under one of the agent's `fs_write_prefixes`. Reads are
  unrestricted (the repo is not secret); writes are the risk.
- **Network**: the Data server is the only one constructed with an HTTP client, and that
  client is a `requests.Session` from a factory that refuses any host not in
  `net_hosts` (a mounted transport adapter that raises on disallowed hosts). Every other
  server installs, at startup, a `socket`-level guard that raises on any outbound
  `connect()` — so even an accidental `import requests; requests.get(...)` in a
  non-Data tool fails hard. A test asserts each non-Data server cannot open a socket.

### 2.4 "Cannot trigger trades" — by construction, not by permission

There is **no trade tool, no broker/exchange client, no `orders` table, and no `trade`
capability in the grant vocabulary** anywhere in the system. "The Data Agent cannot
trigger trades" is therefore not a rule that could be misconfigured — the capability does
not exist to grant. This is the strongest available form of the guarantee and it ties
directly to Phase 4's `actionable: false`: the system is *incapable* of trading, and
that property is testable (assert no module imports an execution/broker SDK; assert the
grant vocabulary has no trade verb).

### 2.5 The deny-path test suite (the actual proof)

A `tests/test_agent_scoping.py` that, for each agent, builds its real scoped handle and
asserts the **denials**, e.g.:

- Validation `SELECT * FROM todays_call_log` → raises `not authorized`.
- Regime `INSERT INTO signal_model_versions …` → raises.
- Training `INSERT INTO model_versions …` → raises.
- Data `INSERT INTO todays_call_log …` → raises.
- Any agent `CREATE TABLE …` / `DROP TABLE …` → raises.
- Non-Data server `socket.create_connection(...)` → raises.
- Training `ScopedFS.open("data/raw/x", "w")` → raises; `open("signal_model/results/x","w")` → ok.
- `orchestrator` not in any agent's `may_call` (no cycles).

These tests are the deliverable that makes §2 a fact rather than a claim.

---

## 3. Audit-trail extension at the tool-call level

Phase 4's `todays_call_log` records the **decision**. With agents calling agents, the
**chain** needs recording so "why did it abstain at moment X" resolves to every input and
the agent call that produced it.

### 3.1 `agent_call_log` — one row per tool invocation, written by the broker

| column | meaning |
|---|---|
| `id` | PK |
| `trace_id` | one per top-level operator request (one `orchestrator.todays_call` = one trace) |
| `parent_call_id` | the call that triggered this one → reconstructs the call tree |
| `seq` | order within the trace |
| `ts_start`, `ts_end` | wall-clock span |
| `caller` | `"operator"` or an `AgentName` |
| `callee_agent`, `tool_name` | what was invoked |
| `args_json` | the **validated** args (post-schema-check) |
| `result_summary_json` | compact result — e.g. `{regime:2, run_length_td:20}` for `point_in_time_label`, **not** the full reconstruction; DataFrame results stored as shape + a digest + the output file path |
| `status` | `ok` / `denied` / `error` |
| `denial_reason` | populated when the broker blocked the call (which grant clause, which table/host) |
| `grant_digest` | hash of the `Grant` in effect, so a later policy change is visible in the trail |
| `code_rev` | `git rev-parse --short HEAD` |
| `prev_hash`, `row_hash` | hash chain — see §3.3 |

Written **by the broker**, not by the tools — a tool cannot forget to log, and a
**denied** call is still logged (`status='denied'`). The enforcement layer produces the
audit record, not the enforced component.

### 3.2 Linking decisions to their trace

Add `trace_id` to `todays_call_log` (small additive migration). Then:

```
orchestrator.explain(as_of):
    row  = SELECT * FROM todays_call_log WHERE as_of_date = ? ORDER BY id DESC LIMIT 1
    tree = SELECT * FROM agent_call_log WHERE trace_id = row.trace_id ORDER BY seq
```

For 2020-03-25 that yields: the `ABSTAIN` record (reason `M1_edge_zone_stress`) **plus**
the ordered calls — `regime.point_in_time_label` → `{regime:2, run:20}`,
`regime.check_drift` → `{tier1:0.30, fires:true}`, `validation.get_reliability_tiers` →
`{2:"monitored"}`, `training.get_active_signal_version` → `…` — every M1–M6 input tied to
the agent call that produced it, chain-verified.

### 3.3 Tamper-evidence — a hash chain (honest scope of "signed, tamper-evident")

Each `agent_call_log` row carries
`row_hash = sha256(prev_hash ‖ canonical_json(row without row_hash))`, `prev_hash` being
the previous row's `row_hash` (genesis = 64 zeros). A `verify_audit_chain(conn)` walks
the table and reports the first broken link. Any retroactive edit, reorder, or deletion
breaks the chain from that point and is detectable.

- This is **tamper-evident**, not tamper-proof — matches the proposal's exact word.
- **Plain SHA-256 chain, no HMAC** (review decision 3). HMAC would need a key-management
  story this project does not have; a plain chain still detects any edit, reorder, or
  deletion by anyone who cannot also rewrite every subsequent `row_hash`, which is the
  honest bound and it is stated as such.
- Same `prev_hash`/`row_hash` pair added to `todays_call_log` (Phase 4 didn't have it).
- No key management, no external service — appropriate for a single-operator local system,
  and the limitations are stated rather than papered over.

---

## 4. MCP implementation — recommendation: **real MCP**

### 4.1 Recommendation

Build it as **real Model Context Protocol servers + a real MCP client**, using the
official `mcp` Python SDK (add `mcp>=1.2` to `requirements.txt`; needs Python ≥3.10, we
have 3.12):

- **Four stdio MCP servers** — `regimeguard-data`, `-regime`, `-training`, `-validation`
  — each an `mcp.server` exposing that agent's tools with JSON-Schema-typed inputs,
  launched as a subprocess.
- **The Orchestrator is a real MCP client** (`mcp.client`) that spawns/connects the four
  servers and calls their tools by name. The `todays_call` sequence is **deterministic
  Python in the client** — no LLM planner. The MCP value here is the typed tool boundary,
  the per-server tool-surface minimisation, and the auditability; none of that needs an
  LLM, and a deterministic client keeps `RegimeGuardCall` records reproducible. (An LLM
  orchestrator can be dropped in later — the tools are already MCP-standard — but that is
  explicitly not this phase.)
- **The broker, `capabilities.py`, and the audit chain live in a shared `agents/` library**
  every server imports. A server builds its `ScopedDB` from `GRANTS[its_identity]` and
  installs its network guard **at startup, before any tool runs**. A server **cannot
  widen its own grant**: the grant is keyed by the server's fixed identity and defined in
  a module the server only reads.

### 4.2 Why real MCP, honestly

- The builder's adjacent work (Agent Security Testbed) is about MCP agent security
  specifically. A working MCP implementation where **tool-surface scoping + engine-level
  resource authorization + a broker-written tamper-evident trail** all compose is a
  materially stronger artifact than an in-process simulation of the same shapes.
- Scope is genuinely bounded because **the tools are thin adapters over verified code**.
  Estimated: 4 server modules (~120–180 lines each, mostly tool schemas), 1 client
  orchestrator (~150), the shared broker/capabilities/audit lib (~250), the deny-path
  test suite (~150). No algorithms, no research, no retraining.
- MCP's own design already gives layer 1 of the security story for free: a server
  advertises only its tools, so an agent cannot even *discover* out-of-role tools. The
  broker is layer 2 (defense in depth, for buggy tools / future LLM orchestrator).

### 4.3 The one real cost, and the fallback

- **Cost:** four server subprocesses share `data/regimeguard.db`.
  - `PRAGMA journal_mode = WAL` (set once at bootstrap) → concurrent readers + one writer.
  - **WAL's write lock is database-file-wide, not per-table**, so partitioning writes by
    agent role reduces contention (no two agents write the same *table*) but a Data write
    can still collide with the broker's `agent_call_log` write. Therefore: **every
    connection sets `PRAGMA busy_timeout = 5000`** (SQLite blocks-and-retries internally
    for up to 5 s), **and the broker's write path is wrapped in an explicit bounded retry**
    — up to 5 attempts with exponential backoff (50 ms → 800 ms) on `sqlite3.OperationalError`
    "database is locked", surfacing a clear error only if all attempts fail. Read paths
    rely on WAL + `busy_timeout` alone.
  - Windows stdio-subprocess lifecycle management is the other fiddly bit.
- **Fallback (if MCP wiring overruns scope):** an **in-process broker** — same
  `capabilities.py` grants, same `ScopedDB` authorizer, same `agent_call_log` hash chain
  — with "agents" as Python objects holding scoped handles instead of subprocesses. This
  keeps 100% of the security + audit story and loses only the literal protocol boundary.
- **Build order that de-risks this:** broker + `capabilities.py` + audit chain + deny-path
  tests **first** (identical either way and independently valuable), then the MCP servers
  + client on top. If the MCP layer blows scope, the in-process path is a complete,
  honest stopping point — but real MCP is the target.

---

## 5. One supporting refactor (small, improves Phase 4 too)

Extract the M1–M6 logic + `RegimeGuardCall` assembly from `decide_todays_call` into a
**pure function** `assemble_regimeguard_call(regime_info, monitoring, tiers, signal_pred,
as_of, thresholds) -> RegimeGuardCall` that takes already-gathered inputs and returns the
record. Then:

- `signal_model/todays_call.decide_todays_call` = "gather via direct imports → assemble"
  (the existing, verified path, unchanged in behaviour).
- `agents/orchestrator.todays_call` = "gather via MCP tool calls → assemble".

Both call the **same assembler**, so the M1–M6 decision logic has exactly one
implementation. A test asserts the two paths produce byte-identical `RegimeGuardCall`
records for a spread of `as_of` dates.

---

## 6. Scope boundary

**In scope:**
1. `agents/capabilities.py` (grants), `agents/broker.py` (`ScopedDB` authorizer,
   `ScopedFS`, network guard, brokered-call logging), `agents/audit.py`
   (`agent_call_log` + hash chain + `verify_audit_chain`), `agents/bootstrap.py`.
2. Four MCP stdio servers wrapping the existing functions per §1.
3. `agents/orchestrator.py` — MCP client + `todays_call` / `explain` (no `refresh_pipeline`).
4. The §5 pure-assembler refactor.
5. `tests/test_agent_scoping.py` — the deny-path suite (§2.5).
6. Additive migrations: `trace_id` on `todays_call_log`; `prev_hash`/`row_hash` on
   `todays_call_log` and `agent_call_log`.
7. `load_india_vix_csv` gains a `conn` parameter.
8. `PRAGMA busy_timeout = 5000` on every connection + a bounded `SQLITE_BUSY` retry in
   the broker's write path (§4.3).

**Explicitly out of scope:**
- Any change to Phase 1–4 research, models, thresholds, or evaluation logic.
- An LLM orchestrator (deterministic client only; noted as a future drop-in).
- Real cryptographic signing / KMS / non-repudiation (plain SHA-256 hash chain is the
  honest "tamper-evident"; limits stated).
- Deployment / hosting / dashboard (proposal §5.4, a later phase).
- Multi-user auth, RBAC beyond the five fixed agent identities.
- Autonomous (non-operator-initiated) retraining; a single-button `refresh_pipeline`.

---

## 7. Review decisions — all resolved

1. **Real MCP**, in-process broker as documented fallback, broker-first build order. ✅
2. `agent_call_log` in the **main DB** (join-ability > isolation; authorizer blocks
   non-broker writes regardless). ✅
3. **Plain SHA-256 chain, no HMAC** (no key-management story to justify HMAC; honest
   "tamper-evident, not tamper-proof"). ✅
4. **Validation "historical only" is an explicit checkable rule** — read-only +
   `todays_call_log`/`agent_call_log` SELECT denied + **broker rejects Validation tool
   calls whose date args fall within `EMBARGO_DAYS` of the last available bar**
   (read-only alone restricts query type, not date range). ✅
5. **No `refresh_pipeline`** — operator invokes each consequential stage explicitly,
   with review between. ✅
6. **Concurrent-writer safety** — `PRAGMA busy_timeout = 5000` everywhere + bounded
   exponential-backoff retry on `SQLITE_BUSY` in the broker write path; WAL's write lock
   is file-wide, so role-partitioned writes reduce but don't remove contention. ✅

## 8. Build order (this phase)

1. **`agents/capabilities.py`** — `AgentName`, `Grant`, `GRANTS`, `CALL_GRAPH`. Pure data.
2. **`agents/audit.py`** — `agent_call_log` schema, SHA-256 chain append, `verify_audit_chain`.
3. **`agents/broker.py`** — `ScopedDB` (SQLite authorizer from a `Grant`), `ScopedFS`,
   network guard, `busy_timeout` + `SQLITE_BUSY` retry, `brokered_call()` (validates args,
   runs the tool with the scoped handle, writes the `agent_call_log` row incl. on denial).
4. **`agents/bootstrap.py`** — one-time privileged schema setup + `PRAGMA journal_mode=WAL`.
5. **`tests/test_agent_scoping.py`** — the deny-path suite (§2.5). **Real output reviewed
   before proceeding.**
6. **§5 refactor** — extract `assemble_regimeguard_call`; assert direct vs. (stub) agent
   path produce identical records.
7. **Four MCP servers** + **`agents/orchestrator.py`** client, `todays_call` / `explain`.
8. End-to-end: `orchestrator.todays_call` for the branch-exercising `as_of` set from
   Phase 4, asserting records match `decide_todays_call` and the `agent_call_log` chain
   verifies.

No commits or pushes at any point. Real output — especially the deny-path suite —
reviewed before any piece is considered done.
