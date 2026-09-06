"""Phase 6 - deployment. Modal-hosted operational face of the Phase 1-5 system.

Nothing here re-derives a research finding or changes a model, threshold, or the
M1-M6 decision logic. It gives the built system a persistent home (a Modal Volume
holding the one SQLite `regimeguard.db`), a scheduled + on-demand daily decision job
that runs the real MCP orchestrator path, and a minimal read-only Streamlit dashboard.

See docs/phase6_design_proposal.md (approved) and docs/phase6_deployment.md (runbook).
"""
