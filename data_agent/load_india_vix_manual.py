"""Load manually-downloaded India VIX historical data into the same point-in-time
schema used for Nifty/Bank Nifty.

Why manual: NSE-direct automated access to VIX history is confirmed hard-blocked -
a plain HTTP client, a headless browser, and the NSEpy library all failed against
Akamai's bot protection (see the compiled Phase 1 plan). VIX is pulled by hand from
NSE's own historical VIX report page
(https://www.nseindia.com/reports-indices-historical-vix) and dropped as a CSV file
for this script to validate and load - VIX access stays manual, not automated.

Expected CSV format - paste NSE's export as-is, don't reformat it by hand:
    Date,Open,High,Low,Close
    19-Jul-2010,24.4525,25.2825,23.9275,24.7000
    20-Jul-2010,...

No Volume column - VIX is a computed index level, not a traded instrument, same as
Nifty/Bank Nifty. Column names are matched case-insensitively, extra columns (e.g.
an "Index Name" column, as NSE's Nifty export has) are ignored rather than
rejected, and a couple of date formats are accepted. NSE's exact VIX export header
hasn't been independently verified the way niftyindices.com's API response was, so
if this script can't confidently identify Date/Open/High/Low/Close it fails loudly
with exactly what it found in the header - it does not guess.

Every load also checks the file's own date range against calendar_days for missed
trading days, same principle as the Nifty/Bank Nifty gap check - reused directly
from calendar_days.py rather than re-implemented.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date, datetime
from pathlib import Path

from data_agent.calendar_days import find_unexplained_gaps
from data_agent.db import (
    finish_ingestion_run,
    get_connection,
    get_or_create_instrument,
    insert_bar,
    start_ingestion_run,
)

SOURCE_NAME = "nseindia.com (manual)"
SYMBOL = "INDIAVIX"
DISPLAY_NAME = "India VIX"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "india_vix" / "manual"

COLUMN_ALIASES: dict[str, set[str]] = {
    "date": {"date"},
    "open": {"open"},
    "high": {"high"},
    "low": {"low"},
    "close": {"close"},
}

DATE_FORMATS = ["%d-%b-%Y", "%d %b %Y", "%d-%m-%Y", "%Y-%m-%d"]

# India VIX has ranged roughly 2.3 (Feb 2016 all-time low) to 86.6 (Mar 2020
# all-time high). A wide band that just catches obvious paste/unit errors, not a
# strict rule.
PLAUSIBLE_RANGE = (1.0, 150.0)


class ValidationError(Exception):
    pass


def _match_columns(header: list[str]) -> dict[str, str]:
    normalized = {h.strip().lower(): h for h in header}
    resolved = {}
    for field, aliases in COLUMN_ALIASES.items():
        found = next((normalized[a] for a in aliases if a in normalized), None)
        if found is None:
            raise ValidationError(
                f"Could not find a '{field}' column. Header row was: {header}. "
                f"Expected one of {sorted(aliases)} (case-insensitive)."
            )
        resolved[field] = found
    return resolved


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValidationError(f"Could not parse date {value!r} with any of {DATE_FORMATS}")


def validate_and_parse(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValidationError("CSV appears to be empty (no header row).")
        columns = _match_columns(reader.fieldnames)

        rows = []
        seen_dates = set()
        for i, raw_row in enumerate(reader, start=2):  # header is row 1
            trade_date = _parse_date(raw_row[columns["date"]])
            if trade_date in seen_dates:
                raise ValidationError(f"Duplicate date {trade_date} at row {i}.")
            seen_dates.add(trade_date)

            try:
                open_ = float(raw_row[columns["open"]])
                high = float(raw_row[columns["high"]])
                low = float(raw_row[columns["low"]])
                close = float(raw_row[columns["close"]])
            except ValueError as exc:
                raise ValidationError(f"Non-numeric OHLC value at row {i}: {raw_row}") from exc

            for label, val in [("open", open_), ("high", high), ("low", low), ("close", close)]:
                if not (PLAUSIBLE_RANGE[0] <= val <= PLAUSIBLE_RANGE[1]):
                    raise ValidationError(
                        f"Row {i}: {label}={val} is outside a plausible India VIX "
                        f"range {PLAUSIBLE_RANGE} - check for a paste/unit error."
                    )
            if not (low <= open_ <= high and low <= close <= high):
                raise ValidationError(
                    f"Row {i}: OHLC values inconsistent (open/close must fall within "
                    f"[low, high]) - open={open_} high={high} low={low} close={close}"
                )

            rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )
    if not rows:
        raise ValidationError("CSV had a header but no data rows.")
    return rows


def load_india_vix_csv(csv_path: Path) -> dict:
    rows = validate_and_parse(csv_path)
    dates = sorted(r["trade_date"] for r in rows)
    start, end = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    landed_path = RAW_DIR / f"india_vix_{start.isoformat()}_{end.isoformat()}.csv"
    shutil.copy2(csv_path, landed_path)

    conn = get_connection()
    try:
        instrument_id = get_or_create_instrument(conn, SYMBOL, DISPLAY_NAME, SOURCE_NAME)
        run_id = start_ingestion_run(conn, SOURCE_NAME, start, end)
        try:
            inserted = 0
            for row in rows:
                if insert_bar(
                    conn, instrument_id, row["trade_date"],
                    row["open"], row["high"], row["low"], row["close"],
                    None, SOURCE_NAME, str(landed_path), run_id,
                ):
                    inserted += 1
            conn.commit()
            finish_ingestion_run(
                conn, run_id, "success", f"{len(rows)} rows validated, {inserted} new"
            )
        except Exception as exc:
            finish_ingestion_run(conn, run_id, "failed", str(exc))
            raise

        gaps = find_unexplained_gaps(conn, instrument_id, start, end)
    finally:
        conn.close()

    return {
        "total_rows": len(rows),
        "inserted": inserted,
        "start": start,
        "end": end,
        "landed_path": landed_path,
        "unexplained_gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and load a manually-downloaded India VIX CSV"
    )
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    try:
        result = load_india_vix_csv(args.csv_path)
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}")
        raise SystemExit(1)

    print(
        f"India VIX: {result['total_rows']} rows validated "
        f"({result['start']} to {result['end']}), {result['inserted']} new rows loaded."
    )
    print(f"Raw file landed at {result['landed_path']}")
    if result["unexplained_gaps"]:
        print(
            f"WARNING: {len(result['unexplained_gaps'])} trading day(s) in this range "
            f"have no VIX bar: {result['unexplained_gaps']}"
        )
    else:
        print("No unexplained gaps in this range.")


if __name__ == "__main__":
    main()
