"""Exchange-level trading-calendar classification.

Every calendar day in the analysis window gets exactly one row: 'trading', 'holiday',
'outage', 'weekend_no_session', or 'special_session'. This is what turns a date
missing from daily_bars into an explicit, auditable fact instead of something
requiring after-the-fact investigation, per proposal §2's gap-classification
requirement.

**Weekends are covered, not skipped** (changed after 2024-03-02 was discovered
missing from India VIX while having a real Nifty/Bank Nifty bar — see
KNOWN_SPECIAL_SESSIONS below). Ordinary Saturdays/Sundays get 'weekend_no_session';
the small number of dates NSE actually ran real trading on a weekend get
'special_session'. find_unexplained_gaps() treats 'trading' and 'special_session'
identically (both are dates real data is expected) — only 'weekend_no_session',
'holiday', and 'outage' are exempt from that check.

Primary source: pandas_market_calendars' XNSE calendar (community-maintained;
actively corrected against real exchange changes — e.g. it has a tracked fix for the
2024-05-20 India election-day closure). XNSE only models weekdays, so weekend
classification (below) is entirely our own KNOWN_SPECIAL_SESSIONS list.

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

# Every weekend date NSE actually ran real trading on, 2010-07-19 to present.
# Compiled by querying daily_bars directly for every Sat/Sun with a real
# Nifty/Bank Nifty bar (authoritative - this is literally what ended up in the
# published historical index archive) rather than trying to enumerate NSE's
# circulars from the outside, since SEBI mandates *monthly* mock-trading Saturdays
# that mostly don't produce real archived index data - an external circular search
# would be dominated by irrelevant mock sessions. 18 dates found this way; each is
# categorized below with its confirming evidence, not asserted from the DB query
# alone.
#
# Category A - Diwali Muhurat trading landing on a weekend (5 dates, each date
# confirmed against independently-documented Lakshmi Puja dates for that year):
# Category B - Union Budget Day special session landing on a weekend (4 dates, each
# confirmed against the documented Budget presentation date for that year - the
# last working day of February through 2016, 1 February from 2017 onward):
# Category C - Disaster-Recovery/BCP site switchover drills or holiday-driven
# reschedules (9 dates; 3 individually confirmed via direct news coverage, 6 from
# 2012-2014 inferred by elimination - real weekend trading data exists, doesn't
# match any Diwali or Budget date, consistent with NSE's long-documented DR-testing
# practice, but no individual news citation was found for each - flagged as
# lower-confidence than the other two categories, not asserted as equally certain).
KNOWN_SPECIAL_SESSIONS: dict[date, str] = {
    # --- Category A: Diwali Muhurat trading on a weekend ---
    date(2013, 11, 3): "Diwali Muhurat trading (Diwali/Lakshmi Puja fell on Sun 3-Nov-2013).",
    date(2016, 10, 30): "Diwali Muhurat trading (Diwali/Lakshmi Puja fell on Sun 30-Oct-2016).",
    date(2019, 10, 27): "Diwali Muhurat trading (Diwali/Lakshmi Puja fell on Sun 27-Oct-2019).",
    date(2020, 11, 14): "Diwali Muhurat trading (Diwali/Lakshmi Puja fell on Sat 14-Nov-2020).",
    date(2023, 11, 12): "Diwali Muhurat trading (Diwali/Lakshmi Puja fell on Sun 12-Nov-2023), confirmed via direct news coverage.",
    # --- Category B: Union Budget Day special session on a weekend ---
    date(2015, 2, 28): "Union Budget Day special session (2015 Budget presented Sat 28-Feb-2015, the pre-2017 last-working-day-of-February convention).",
    date(2020, 2, 1): "Union Budget Day special session (2020 Budget presented Sat 1-Feb-2020).",
    date(2025, 2, 1): "Union Budget Day special session (2025 Budget presented Sat 1-Feb-2025), confirmed via direct news coverage.",
    date(2026, 2, 1): "Union Budget Day special session (2026 Budget presented Sun 1-Feb-2026), confirmed via direct news coverage.",
    # --- Category C: DR/BCP drills and holiday-reschedules ---
    date(2024, 1, 20): "Trading day shifted from Mon 22-Jan-2024 (declared a holiday for the Ayodhya Ram Mandir consecration) to Sat 20-Jan-2024 to preserve the week's trading-day count. Confirmed via direct news coverage.",
    date(2024, 3, 2): "SEBI-mandated Disaster Recovery site switchover drill - two live sessions (9:15-10:00, 11:30-12:30), intra-day primary-to-DR switchover in equity and equity derivatives. Confirmed via direct news coverage. This is the date that surfaced this whole gap - VIX was not found loaded for it (see resolution note below).",
    date(2024, 5, 18): "SEBI-mandated Disaster Recovery site switchover drill. Confirmed via direct news coverage.",
    date(2012, 1, 7): "Real Nifty/Bank Nifty trading data exists for this Saturday; doesn't match any Diwali or Budget date for 2012. Inferred as an early DR/BCP live-trading test - not individually news-confirmed.",
    date(2012, 3, 3): "Real Nifty/Bank Nifty trading data exists for this Saturday; doesn't match any Diwali or Budget date for 2012. Inferred as an early DR/BCP live-trading test - not individually news-confirmed.",
    date(2012, 4, 28): "Real Nifty/Bank Nifty trading data exists for this Saturday; doesn't match any Diwali or Budget date for 2012. Inferred as an early DR/BCP live-trading test - not individually news-confirmed.",
    date(2012, 9, 8): "Real Nifty/Bank Nifty trading data exists for this Saturday; doesn't match any Diwali or Budget date for 2012. Inferred as an early DR/BCP live-trading test - not individually news-confirmed.",
    date(2013, 5, 11): "Real Nifty/Bank Nifty trading data exists for this Saturday; doesn't match any Diwali or Budget date for 2013. Inferred as an early DR/BCP live-trading test - not individually news-confirmed.",
    date(2014, 3, 22): "Real Nifty/Bank Nifty trading data exists for this Saturday; doesn't match any Diwali or Budget date for 2014. Inferred as an early DR/BCP live-trading test - not individually news-confirmed.",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_days (
    trade_date TEXT PRIMARY KEY,
    day_type TEXT NOT NULL CHECK (
        day_type IN (
            'trading', 'holiday', 'outage', 'unexplained_gap',
            'weekend_no_session', 'special_session'
        )
    ),
    reason TEXT,
    source TEXT NOT NULL,
    verified_at TEXT NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate_day_type_constraint(conn)


def _migrate_day_type_constraint(conn: sqlite3.Connection) -> None:
    """SQLite can't ALTER a CHECK constraint in place, so a DB created under the
    old constraint (day_type without 'weekend_no_session'/'special_session') needs
    its table rebuilt. Safe here specifically because calendar_days is documented
    as a recomputable judgment, not append-only observational data (see
    load_calendar_days docstring) - preserving existing rows via a rename-copy-drop
    is just a mechanical fix for a constraint that's now too narrow, not a data
    migration with any risk of losing something irreplaceable.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='calendar_days'"
    ).fetchone()
    if row is None or "special_session" in row[0]:
        return  # table doesn't exist yet, or already has the current constraint
    conn.execute("ALTER TABLE calendar_days RENAME TO calendar_days_old")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO calendar_days (trade_date, day_type, reason, source, verified_at) "
        "SELECT trade_date, day_type, reason, source, verified_at FROM calendar_days_old"
    )
    conn.execute("DROP TABLE calendar_days_old")
    conn.commit()


def classify_range(start: date, end: date) -> list[dict]:
    """Classify every calendar day in [start, end]. See module docstring for sourcing."""
    cal = mcal.get_calendar(XNSE_CALENDAR_NAME)
    schedule = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    trading_days = {d.date() for d in schedule.index}

    verified_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    current = start
    while current <= end:
        is_weekend = current.weekday() >= 5
        if current in KNOWN_OUTAGE_DAYS:
            day_type, reason = "outage", KNOWN_OUTAGE_DAYS[current]
        elif current in KNOWN_SPECIAL_SESSIONS:
            day_type, reason = "special_session", KNOWN_SPECIAL_SESSIONS[current]
        elif is_weekend:
            day_type, reason = "weekend_no_session", None
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
    """Trading days *and special sessions* (per calendar_days) with no daily_bars
    row for this instrument.

    This is the real gap-classification check: calendar_days on its own only says
    what the exchange calendar expects; this join is what catches an actual silent
    data gap (calendar says data should exist, but our fetch has nothing for that
    date). 'special_session' is included alongside 'trading' - a Saturday NSE
    genuinely traded on (e.g. a DR drill) is exactly as much a "should have data"
    date as an ordinary weekday, which is how 2024-03-02 was caught in the first
    place. 'weekend_no_session', 'holiday', and 'outage' are excluded - a missing
    bar on an ordinary weekend or a scheduled holiday isn't a gap at all, and an
    outage day is a known, separately-flagged anomaly rather than a plain gap.
    """
    rows = conn.execute(
        """
        SELECT c.trade_date
        FROM calendar_days c
        LEFT JOIN daily_bars b
            ON b.trade_date = c.trade_date
            AND b.instrument_id = ?
            AND b.superseded_by IS NULL
        WHERE c.day_type IN ('trading', 'special_session')
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
