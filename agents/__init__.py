"""RegimeGuard multi-agent layer (Phase 5).

Repackages the Phases 1-4 pipeline as five capability-scoped agents behind MCP tool
boundaries, with tool-layer permission enforcement (SQLite's own statement authorizer)
and a broker-written, hash-chained audit trail. Wraps already-verified code - it
re-derives no research finding.

See docs/phase5_design_proposal.md.
"""
