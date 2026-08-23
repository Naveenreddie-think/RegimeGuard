# CLAUDE_CODE_BRIEF.md — RegimeGuard

## What this document is

This is a planning brief, not a finished blueprint. The attached `Project_Proposal_RegimeGuard.md` defines the problem, the solution shape, and the design decisions already locked. Your job is to take that proposal and produce a **detailed, phase-by-phase technical plan** — including your own research, your own judgment calls, and your own recommendations where the proposal leaves room for engineering choices. This is not a request to scaffold files yet. This is a planning and research pass.

## Your role in this phase

1. Read the full proposal.
2. For each major component (regime detection, leakage-safe validation, signal model, confidence/abstention logic, MCP agent architecture, security scoping, deployment), do real research and propose the **specific technical approach** you'd take — methods, libraries, validation techniques, architecture patterns — with reasoning for each choice, not just a restatement of the proposal's language.
3. Where you see a better approach than what's in the proposal, say so explicitly and argue for it. Genuinely strong alternative ideas are welcome and expected — this is a discussion, not a transcription task.
4. Flag anything in the proposal that's underspecified, risky, or likely to cause problems downstream (e.g., data quality issues, statistical pitfalls, architectural bottlenecks) before it becomes expensive to fix.
5. Produce a phase-by-phase implementation plan (data → regime detection → model → validation → agents/MCP → security → deployment → documentation), with a clear definition of "done" for each phase.

## Locked decisions — do not revisit these without flagging why

- **Data granularity:** daily first (price + India VIX). Intraday is an explicit later phase, not part of this build.
- **Data source:** NSE archives directly.
- **Prediction target:** direction (primary, short-horizon up/down/flat on Nifty/Bank Nifty). Volatility is a secondary, regime-context target feeding the confidence/abstention logic — not a standalone prediction.
- **Core principle:** build something real and document it honestly. No detail should be included anywhere (code comments, README, FINDINGS.md, this plan) that references company names, job descriptions, or "why this is a good portfolio project." The project stands on its own technical merit only.

## Working process — how we'll use your output

1. You produce the detailed plan/research for a phase (or the whole project, your call on how to chunk it).
2. We review it together here, in this chat — not by trusting it blindly. If an idea is genuinely strong, we tell you to implement it. If something's off, we discuss corrections and better approaches before any code is written.
3. Only after a phase's design is validated does implementation begin for that phase.
4. **You do not commit or push to GitHub.** Ever, at any point, regardless of milestone or verification status. All git commits and pushes are done manually. If a phase is complete and verified, say so clearly and stop — do not run `git commit` or `git push` yourself.
5. File-by-file delivery is preferred over scaffolding entire folder structures at once, consistent with how every other project in this portfolio has been built.
6. Same incremental discipline as prior projects: no next phase starts until the current one is reviewed, verified with real output (not assumed), and explicitly approved.

## What "genuinely strong" means when evaluating your proposals

An idea gets implemented if it: (a) is statistically/technically sound and defensible under scrutiny, (b) doesn't overclaim what the data or method can actually support, and (c) keeps the system honest — meaning it should make failures and limitations visible rather than hidden. Ideas that make results look better than they are, or that add complexity without a clear justification, should be flagged as weak even if you generated them yourself — don't talk yourself into a plan just because it's more impressive-sounding.

## Start here

Begin with a research-and-plan pass on **Phase 1: Data acquisition and regime detection foundation** — NSE archive access approach, exact historical depth achievable with clean data, the regime-detection methodology (and alternatives you'd consider beyond what the proposal suggests), and how regime stability will be validated before anything downstream depends on it. Present that first; we validate it together before moving to the next phase.
