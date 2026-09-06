"""The whole permission policy, in one place - Phase 5 design §2.1.

Every enforcement mechanism (the SQLite authorizer in broker.ScopedDB, the filesystem
guard, the network guard, the call-graph check) reads from `GRANTS` / `CALL_GRAPH`
below. Changing what an agent may do is a one-edit change here; the enforcement
follows automatically.

Design stance: **explicit allowlists**, not allow-all-minus-deny. An agent can read a
table only if it is listed in that agent's `db_read`; it can write only what is in
`db_write`; it can create/drop/alter nothing (schema is a privileged bootstrap step);
it can open outbound sockets only to hosts in `net_hosts` (empty for everyone but
Data); it can write files only under `fs_write_prefixes`; it can call another agent's
tools only if that agent is in `may_call` (only the Orchestrator may call anyone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from signal_model.walk_forward import EMBARGO_DAYS


class AgentName(str, Enum):
    DATA = "data"
    REGIME = "regime"
    TRAINING = "training"
    VALIDATION = "validation"
    ORCHESTRATOR = "orchestrator"
    BROKER = "broker"  # the enforcer itself - writes only agent_call_log


# --- the tables in data/regimeguard.db, grouped by owning domain ---
DATA_TABLES = frozenset({"instruments", "ingestion_runs", "daily_bars", "calendar_days"})
REGIME_TABLES = frozenset({"model_versions", "regime_labels"})
SIGNAL_TABLES = frozenset({"signal_model_versions", "signal_predictions"})
DECISION_TABLE = frozenset({"todays_call_log"})
AUDIT_TABLE = frozenset({"agent_call_log"})
ALL_TABLES = DATA_TABLES | REGIME_TABLES | SIGNAL_TABLES | DECISION_TABLE | AUDIT_TABLE

# PRAGMAs an agent connection may issue. Setting WAL is a bootstrap-only privileged
# step; agents may still *read* journal_mode, and need busy_timeout / foreign_keys
# per-connection. Introspection pragmas are read-only and harmless.
ALLOWED_PRAGMAS = frozenset({
    "busy_timeout", "foreign_keys", "journal_mode",
    "table_info", "table_xinfo", "index_list", "index_info",
})

BUSY_TIMEOUT_MS = 5000  # PRAGMA busy_timeout on every connection (§4.3)


@dataclass(frozen=True)
class Grant:
    db_read: frozenset[str]
    db_write: frozenset[str]
    fs_write_prefixes: tuple[str, ...]
    net_hosts: frozenset[str]
    may_call: frozenset[AgentName] = field(default_factory=frozenset)
    # Validation only: reject any tool call whose date arguments land within
    # EMBARGO_DAYS trading rows of the last available bar. None = no such limit.
    reject_dates_within_embargo_of_last_bar: bool = False

    def __post_init__(self) -> None:
        stray_r = self.db_read - ALL_TABLES
        stray_w = self.db_write - ALL_TABLES
        if stray_r or stray_w:
            raise ValueError(f"Grant references unknown tables: {stray_r | stray_w}")
        if not self.db_write <= self.db_read:
            raise ValueError(f"writable tables not all readable: {self.db_write - self.db_read}")


GRANTS: dict[AgentName, Grant] = {
    AgentName.DATA: Grant(
        db_read=DATA_TABLES,
        db_write=DATA_TABLES,
        fs_write_prefixes=("data/raw/",),
        net_hosts=frozenset({"www.niftyindices.com"}),
    ),
    AgentName.REGIME: Grant(
        db_read=DATA_TABLES | REGIME_TABLES,
        db_write=REGIME_TABLES,
        fs_write_prefixes=("regime_detection/results/",),
        net_hosts=frozenset(),
    ),
    AgentName.TRAINING: Grant(
        # daily_bars + model_versions + regime_labels: build features, resolve the
        # active regime version, derive training labels. signal tables: its own writes.
        db_read=frozenset({"daily_bars"}) | REGIME_TABLES | SIGNAL_TABLES,
        db_write=SIGNAL_TABLES,
        fs_write_prefixes=("signal_model/results/",),
        net_hosts=frozenset(),
    ),
    AgentName.VALIDATION: Grant(
        # Everything it needs to score the frozen evaluation set. Explicitly NOT
        # todays_call_log / agent_call_log - it is an evaluator, not an auditor, and
        # must not see live decisions.
        db_read=frozenset({"daily_bars"}) | REGIME_TABLES | SIGNAL_TABLES,
        db_write=frozenset(),  # read-only
        fs_write_prefixes=("signal_model/results/",),
        net_hosts=frozenset(),
        reject_dates_within_embargo_of_last_bar=True,
    ),
    AgentName.ORCHESTRATOR: Grant(
        db_read=ALL_TABLES,  # assembly needs the model tables; explain needs both logs
        db_write=DECISION_TABLE,
        fs_write_prefixes=(),  # sequences tool calls; writes no files itself
        net_hosts=frozenset(),
        may_call=frozenset({AgentName.DATA, AgentName.REGIME, AgentName.TRAINING, AgentName.VALIDATION}),
    ),
    AgentName.BROKER: Grant(
        db_read=AUDIT_TABLE,   # read prev_hash to extend the chain
        db_write=AUDIT_TABLE,  # write the agent_call_log row (incl. on denial)
        fs_write_prefixes=(),
        net_hosts=frozenset(),
    ),
}

# Who may invoke whose tools. Only the Orchestrator calls anyone; no agent calls
# another; nothing calls the Orchestrator (acyclic, single-rooted).
CALL_GRAPH: dict[AgentName, frozenset[AgentName]] = {
    name: GRANTS[name].may_call for name in AgentName
}


def _assert_call_graph_acyclic_and_rooted() -> None:
    # no self-calls
    for a, callees in CALL_GRAPH.items():
        assert a not in callees, f"{a} may call itself"
    # nothing may call the Orchestrator or the Broker
    for a, callees in CALL_GRAPH.items():
        assert AgentName.ORCHESTRATOR not in callees, f"{a} may call the orchestrator (cycle risk)"
        assert AgentName.BROKER not in callees, f"{a} may call the broker directly"
    # only the Orchestrator has any callees
    for a, callees in CALL_GRAPH.items():
        if a is not AgentName.ORCHESTRATOR:
            assert not callees, f"{a} is not the orchestrator but may call {callees}"


_assert_call_graph_acyclic_and_rooted()

# "no trade capability exists anywhere" (§2.4): the grant vocabulary has no trade verb,
# no orders table, no execution host. This tuple is what tests assert stays empty.
TRADE_VOCABULARY: tuple[str, ...] = ()  # intentionally empty and must stay empty


__all__ = [
    "AgentName", "Grant", "GRANTS", "CALL_GRAPH", "ALL_TABLES", "ALLOWED_PRAGMAS",
    "BUSY_TIMEOUT_MS", "EMBARGO_DAYS", "TRADE_VOCABULARY",
]
