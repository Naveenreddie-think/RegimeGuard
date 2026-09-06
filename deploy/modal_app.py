"""Modal app for RegimeGuard - Phase 6 (approved: docs/phase6_design_proposal.md).

One Modal app, one Volume (`regimeguard-data`) holding the single SQLite
`regimeguard.db`, one image. Three functions:

- `decide_scheduled` - `modal.Cron`, weekdays post-close (IST). Runs the real MCP
  orchestrator path so the deployed system actually exercises Phase 5 (four scoped
  stdio subprocesses + the hash-chained agent_call_log). Often re-emits ABSTAIN/stale
  if nobody has refreshed the data - that is the point (a silently frozen dashboard
  would be the failure mode).
- `decide` - the same thing on demand: `modal run deploy/modal_app.py --as-of 2020-03-25`.
- `dashboard` - a scale-to-zero Streamlit web endpoint, read-only.

SQLite stays (the Phase 5 security layer is built on sqlite3's statement authorizer,
which has no Postgres equivalent). The Volume is the source of truth; the in-repo
`data/regimeguard.db` is dropped from git and only used to seed the Volume once - see
docs/phase6_deployment.md.

No commits or pushes are performed by this project's tooling, ever.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = "regimeguard"
VOLUME_NAME = "regimeguard-data"
VOL_MOUNT = "/data"

def _git_sha() -> str:
    """Resolved client-side at deploy time and baked into the image env, so the
    container (built without .git) still records provenance in the audit tables via
    signal_model.todays_call._code_rev(). Returns 'unknown' when git is unavailable
    (e.g. re-import inside the container - the baked value is what actually ships)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(REPO),
        ).stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


# --- image: project deps (pinned) + streamlit; repo copied to /app, data/ excluded --
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")  # OpenMP runtime - lightgbm needs it
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .workdir("/app")
    .env({"PYTHONPATH": "/app", "REGIMEGUARD_CODE_REV": _git_sha()})
    # add_local_* must come last (Modal adds these at container start, not build time)
    .add_local_dir(
        str(REPO),
        remote_path="/app",
        ignore=[
            "**/__pycache__", "**/__pycache__/**", "**/*.py[cod]",
            ".git", ".git/**", ".pytest_cache", ".pytest_cache/**",
            ".venv", ".venv/**", "venv", "venv/**", "env", "env/**",
            ".claude", ".claude/**",
            "data", "data/**",            # all data lives on the Volume, not the image
        ],
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_decide(as_of: str | None) -> dict:
    """Copy the Volume db local, run the MCP orchestrator, copy back, commit."""
    from deploy.db_sync import local_db_session

    with local_db_session():
        # imported inside the session so the whole stack (incl. the four subprocesses
        # the orchestrator spawns, which inherit os.environ) sees REGIMEGUARD_DB.
        from agents.orchestrator import todays_call

        record = todays_call(as_of=as_of, log=True)
        d = record.to_dict()

    vol.commit()
    summary = {
        "as_of_date": d["as_of_date"],
        "generated_at": d["generated_at"],
        "disposition": d["disposition"],
        "reliability_tier": d["reliability_tier"],
        "trace_id": d["audit"].get("trace_id"),
        "code_rev": d["audit"].get("code_rev"),
    }
    print(json.dumps(summary, indent=2))
    return d


@app.function(
    volumes={VOL_MOUNT: vol},
    schedule=modal.Cron("30 13 * * 1-5", timezone="Asia/Kolkata"),  # 19:00 IST, Mon-Fri
    timeout=1200,
)
def decide_scheduled() -> dict:
    return _run_decide(None)


@app.function(volumes={VOL_MOUNT: vol}, timeout=1200)
def decide(as_of: str | None = None) -> dict:
    return _run_decide(as_of)


@app.function(volumes={VOL_MOUNT: vol}, min_containers=0, timeout=1800)
@modal.web_server(8000, startup_timeout=180, label="regimeguard-dashboard")
def dashboard() -> None:
    # Best-effort: pick up the latest daily commit on cold start.
    try:
        vol.reload()
    except Exception as exc:  # noqa: BLE001
        print(f"volume reload skipped: {exc}")
    subprocess.Popen(
        [
            "streamlit", "run", "/app/deploy/dashboard.py",
            "--server.port", "8000",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.gatherUsageStats", "false",
        ]
    )


@app.local_entrypoint()
def main(as_of: str | None = None) -> None:
    """`modal run deploy/modal_app.py [--as-of YYYY-MM-DD]` - one on-demand decision."""
    decide.remote(as_of)
