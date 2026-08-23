"""Exchange-level trading-calendar classification.

Every weekday in the analysis window gets exactly one row: 'trading', 'holiday', or
'outage'. This is what turns a date missing from daily_bars into an explicit,
auditable fact instead of something requiring after-the-fact investigation, per
proposal §2's gap-classification requirement. Weekends are intentionally excluded —
they carry no ambiguity worth recording, and a weekday with no daily_bars row and a
'trading' classification here is exactly the case find_unexplained_gaps() surfaces.

Primary source: pandas_market_calendars' XNSE calendar (community-maintained;
actively corrected against real exchange changes — e.g. it has a tracked fix for the
2024-05-20 India election-day closure).

Cross-checked against NSE's own published holiday circulars for the validation
window (H2 2010) before trusting XNSE for the full pull:
- 2010-09-10 (Ramzan Id), 2010-11-17 (Bakri Id), 2010-12-17 (Moharram) — XNSE marks
  all three as non-trading. Matches NSE's own circular-derived holiday list.
- 2010-11-05 (Diwali / Laxmi Puja) — officially a market holiday for regular trading,
  but NSE runs a special abbreviated "Muhurat trading" session that evening and
  publishes a real official close for it. XNSE correctly marks this date as a
  *trading* day (not a holiday) — consistent with the real OHLC data already loaded
  into daily_bars for this date (see fetch_niftyindices.py's module docstring). No
  special-case handling was needed for Muhurat days: XNSE already gets this right.

Known outage days (system failures on a day that was never a scheduled holiday) are
a small hardcoded exception list below — no calendar library tracks unplanned
outages, since by definition they weren't scheduled off.

Because load_calendar_days() replaces rather than appends (see its docstring), a
rebuild that changes prior classifications would otherwise leave no trace. Every
call logs an ingestion_runs row recording how many dates were newly classified vs.
had their classification changed vs. were unchanged, so a rebuild's effect is
always reconstructable from the audit trail even though the table itself only
holds the current answer.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pandas_market_calendars as mcal

from data_agent.db import finish_ingestion_run, get_connection, start_ingestion_run

XNSE_CALENDAR_NAME = "XNSE"
CALENDAR_SOURCE = f"pandas_market_calendars:{XNSE_CALENDAR_NAME}"

KNOWN_OUTAGE_DAYS: dict[date, str] = {
    date(2021, 2, 24): (
        "NSE system-wide outage: cash and derivatives trading halted for the full "
        "day due to a technical failure. Not a scheduled holiday - the exchange "
        "was expected to trade normally. Note: niftyindices.com still has a real "
        "index close on record for this date - the outage is flagged here as a "
        "known-anomalous trading day, not treated as missing data."
    ),
}

# Ad-hoc, non-recurring trading holidays XNSE's calendar does not track (special
# one-time closures for a specific event, not part of the regular annual holiday
# cycle a generic exchange-calendar library encodes). Found by investigating every
# date find_unexplained_gaps() surfaced after the first full-window build - each
# was a real 'trading' day per XNSE with no corresponding daily_bars row, which
# is exactly the case this table exists to catch rather than paper over. Each
# entry below is corroborated by NSE/BSE closure coverage in financial press,
# cited in the reason.
KNOWN_SPECIAL_HOLIDAYS: dict[date, str] = {
    date(2020, 5, 25): (
        "Eid-ul-Fitr (Ramzan Id) trading holiday. Not in XNSE's calendar for this "
        "year. Confirmed via 2020 Eid al-Fitr observance date (moon sighting placed "
        "Eid on 24-25 May 2020 across India); less directly confirmed than the other "
        "three entries here, which have explicit 'NSE/BSE closed' news coverage."
    ),
    date(2024, 1, 22): (
        "Special one-time trading holiday for the Ayodhya Ram Mandir 'Pran "
        "Pratishtha' consecration ceremony, declared under Section 25 of the "
        "Negotiable Instruments Act, 1881. Not a recurring annual holiday, so "
        "absent from XNSE's calendar."
    ),
    date(2024, 11, 20): (
        "Special trading holiday for Maharashtra Legislative Assembly election "
        "polling day. Ad-hoc election closure, not in XNSE's calendar."
    ),
    date(2026, 1, 15): (
        "Special trading holiday for Maharashtra municipal corporation election "
        "polling day. Ad-hoc election closure, not in XNSE's calendar."
    ),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_days (
    trade_date TEXT PRIMARY KEY,
    day_type TEXT NOT NULL CHECK (
        day_type IN ('trading', 'holiday', 'outage', 'unexplained_gap')
    ),
    reason TEXT,
    source TEXT NOT NULL,
    verified_at TEXT NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def classify_range(start: date, end: date) -> list[dict]:
    """Classify every weekday in [start, end]. See module docstring for sourcing."""
    cal = mcal.get_calendar(XNSE_CALENDAR_NAME)
    schedule = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    trading_days = {d.date() for d in schedule.index}

    verified_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    current = start
    while current <= end:
        if current.weekday() >= 5:  # Saturday / Sunday
            current += timedelta(days=1)
            continue
        if current in KNOWN_OUTAGE_DAYS:
            day_type, reason = "outage", KNOWN_OUTAGE_DAYS[current]
        elif current in KNOWN_SPECIAL_HOLIDAYS:
            day_type, reason = "holiday", KNOWN_SPECIAL_HOLIDAYS[current]
        elif current in trading_days:
            day_type, reason = "trading", None
        else:
            day_type, reason = "holiday", "Not a trading day per XNSE exchange calendar"
        rows.append(
            {
                "trade_date": current.isoformat(),
                "day_type": day_type,
                "reason": reason,
                "source": CALENDAR_SOURCE,
                "verified_at": verified_at,
            }
        )
        current += timedelta(days=1)
    return rows


def load_calendar_days(conn: sqlite3.Connection, start: date, end: date) -> int:
    """(Re)compute calendar_days for [start, end] and write it, replacing any
    existing row for the same date.

    Unlike daily_bars, calendar_days is a recomputable *judgment*, not raw
    observational data - it reflects our current best classification logic
    (XNSE + known outages + known special holidays), which can improve over time
    (as it just did, after find_unexplained_gaps() surfaced four ad-hoc holidays
    XNSE didn't know about). Re-running this after adding a correction is expected
    to update prior rows, not skip them - hence REPLACE rather than daily_bars'
    append-only/supersede model.

    Because a rebuild can silently overwrite a prior classification, every call is
    wrapped in an ingestion_runs record noting exactly how many dates were added,
    how many changed classification (with which dates and old->new values), and
    how many were unchanged - that's the audit trail for what would otherwise be
    an invisible in-place update.
    """
    ensure_schema(conn)
    run_id = start_ingestion_run(conn, "calendar_days", start, end)

    existing = dict(
        conn.execute(
            "SELECT trade_date, day_type FROM calendar_days WHERE trade_date BETWEEN ? AND ?",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    )

    new_rows = classify_range(start, end)
    added, changed, unchanged = [], [], 0
    for row in new_rows:
        prior = existing.get(row["trade_date"])
        if prior is None:
            added.append(row["trade_date"])
        elif prior != row["day_type"]:
            changed.append((row["trade_date"], prior, row["day_type"]))
        else:
            unchanged += 1
        conn.execute(
            "INSERT OR REPLACE INTO calendar_days "
            "(trade_date, day_type, reason, source, verified_at) VALUES (?, ?, ?, ?, ?)",
            (row["trade_date"], row["day_type"], row["reason"], row["source"], row["verified_at"]),
        )
    conn.commit()

    notes = f"{len(added)} added, {len(changed)} changed, {unchanged} unchanged"
    if changed:
        notes += "; changed: " + ", ".join(f"{d} {old}->{new}" for d, old, new in changed)
    finish_ingestion_run(conn, run_id, "success", notes)

    return {"written": len(new_rows), "added": len(added), "changed": changed, "unchanged": unchanged}


def find_unexplained_gaps(conn: sqlite3.Connection, instrument_id: int, start: date, end: date) -> list[str]:
    """Trading days (per calendar_days) with no daily_bars row for this instrument.

    This is the real gap-classification check: calendar_days on its own only says
    what the exchange calendar expects; this join is what catches an actual silent
    data gap (calendar says trading, but our fetch has nothing for that date).
    """
    rows = conn.execute(
        """
        SELECT c.trade_date
        FROM calendar_days c
        LEFT JOIN daily_bars b
            ON b.trade_date = c.trade_date
            AND b.instrument_id = ?
            AND b.superseded_by IS NULL
        WHERE c.day_type = 'trading'
          AND c.trade_date BETWEEN ? AND ?
          AND b.id IS NULL
        ORDER BY c.trade_date
        """,
        (instrument_id, start.isoformat(), end.isoformat()),
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calendar_days classification")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    conn = get_connection()
    try:
        result = load_calendar_days(conn, args.start, args.end)
        print(
            f"calendar_days: {result['written']} rows for {args.start} to {args.end} "
            f"({result['added']} added, {len(result['changed'])} changed, "
            f"{result['unchanged']} unchanged)"
        )
        for trade_date, old, new in result["changed"]:
            print(f"  changed: {trade_date} {old} -> {new}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
