"""Copy-local / checkpoint / commit-back helper for the Modal-hosted jobs.

Modal Volumes are not a POSIX filesystem - no distributed file locking, FUSE-backed -
so we never run write-active SQLite (WAL needs real `fcntl` locks + a `-shm` mmap)
directly against the mount. The pattern, for a job that is the *only* writer for its
run:

    1. copy the canonical db from the Volume mount to local container disk
    2. point the whole stack at the copy via REGIMEGUARD_DB (broker.db_path() reads it;
       the MCP orchestrator passes os.environ to its four subprocesses, so they inherit
       it too)
    3. run the work
    4. WAL-checkpoint the copy, copy it back over the Volume file
    5. caller calls volume.commit()

A crash before step 4 leaves the Volume file untouched (the run just re-runs). The
decision job only ever appends to the hash-chained logs, so a re-run is safe.

`local_db_session` takes plain paths and imports no `modal` - it is unit-testable and
runs fine locally with overridden paths.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# Defaults match the Modal container layout: the Volume is mounted at /data and the
# repo is copied to /app (whose data_agent/db.py resolves DB_PATH to /app/data/... ,
# i.e. the same working copy - so code paths that ignore REGIMEGUARD_DB still agree).
VOLUME_DB = Path("/data/regimeguard.db")
LOCAL_DB = Path("/app/data/regimeguard.db")


def _checkpoint(db: Path) -> None:
    """Fold the WAL back into the main file so the single file we copy to the Volume
    is self-contained (no -wal / -shm sidecars to carry)."""
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()


@contextmanager
def local_db_session(volume_db: Path | str = VOLUME_DB, local_db: Path | str = LOCAL_DB):
    """Context manager around one write-job's DB access.

    Yields the local working-copy path with REGIMEGUARD_DB set to it. On a clean exit
    it checkpoints and copies the working copy back over `volume_db`; the caller is
    responsible for the subsequent `volume.commit()`. On an exception nothing is
    copied back and REGIMEGUARD_DB is restored.
    """
    volume_db = Path(volume_db)
    local_db = Path(local_db)

    if not volume_db.exists():
        raise FileNotFoundError(
            f"No database on the Volume at {volume_db}. Seed it once with\n"
            f"    modal volume put regimeguard-data <local regimeguard.db> /regimeguard.db\n"
            f"See docs/phase6_deployment.md ('Seeding the Volume')."
        )

    local_db.parent.mkdir(parents=True, exist_ok=True)
    # start from a clean single file; drop any stale sidecars from a previous run
    for p in (local_db, local_db.with_name(local_db.name + "-wal"),
              local_db.with_name(local_db.name + "-shm")):
        p.unlink(missing_ok=True)
    shutil.copy2(volume_db, local_db)

    prev = os.environ.get("REGIMEGUARD_DB")
    os.environ["REGIMEGUARD_DB"] = str(local_db)
    try:
        yield local_db
        _checkpoint(local_db)
        shutil.copy2(local_db, volume_db)
    finally:
        if prev is None:
            os.environ.pop("REGIMEGUARD_DB", None)
        else:
            os.environ["REGIMEGUARD_DB"] = prev
