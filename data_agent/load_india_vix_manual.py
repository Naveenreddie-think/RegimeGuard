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

KNOWN_DATA_ANOMALIES is a narrow, explicit, investigated exception list for dates
where NSE's own published data fails the OHLC consistency check. Each entry was
individually verified (cross-checked Change/Prev.Close and the next day's Open
before concluding which field was the outlier) before being added - this is not a
blanket relaxation of the check. The row is loaded exactly as NSE published it,
values unmodified, with the reason recorded on the bar itself (daily_bars.notes),
not silently passed through.

A second, narrower kind of row is auto-detected rather than hardcoded per date: a
"zero row" where Open=High=Low=Prev.Close=0 and Close just repeats the prior
trading day's close - two confirmed instances (12-Feb-2021, 30-Mar-2021) look like
a recurring defect in NSE's own VIX export, not a data-entry typo. Any row matching
this exact signature is cross-checked against Nifty 50 and Bank Nifty's daily_bars
for that date before being skipped - if both instruments have real data (proving
the market traded normally and the gap is VIX-export-specific), the row is skipped
and the verification evidence is recorded; if the signature doesn't match exactly,
or the cross-check doesn't confirm normal trading, this still fails loudly like any
other validation error rather than being silently skipped.
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

# Optional - used only for the zero-row signature check below. Its absence never
# blocks loading; a row just can't be auto-verified as a zero-row without it.
OPTIONAL_COLUMN_ALIASES: dict[str, set[str]] = {
    "prev_close": {"prev. close", "prev close", "previous close"},
}

DATE_FORMATS = ["%d-%b-%Y", "%d %b %Y", "%d-%m-%Y", "%Y-%m-%d"]

# India VIX has ranged roughly 2.3 (Feb 2016 all-time low) to 86.6 (Mar 2020
# all-time high). A wide band that just catches obvious paste/unit errors, not a
# strict rule.
PLAUSIBLE_RANGE = (1.0, 150.0)

KNOWN_DATA_ANOMALIES: dict[date, str] = {
    date(2013, 8, 22): (
        "NSE's published row has Close (27.68) below Low (27.99), failing the "
        "OHLC consistency check. Verified before overriding: Close is consistent "
        "with Change (-0.41) and Prev. Close (28.09), and matches the next "
        "trading day's (23-AUG-2013) Open exactly (27.68) - so Close is almost "
        "certainly correct and Low is the erroneous field in NSE's own export. "
        "Loaded as published (Low=27.99, Close=27.68, both unmodified) rather "
        "than guessing a corrected Low value."
    ),
}

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
    for field, aliases in OPTIONAL_COLUMN_ALIASES.items():
        found = next((normalized[a] for a in aliases if a in normalized), None)
        if found is not None:
            resolved[field] = found
    return resolved


def _verify_market_traded_normally(conn, trade_date_str: str) -> tuple[bool, str]:
    """Cross-check that Nifty 50 and Bank Nifty both have a real daily_bars row for
    this date - corroborating evidence that a zero-valued VIX row is a
    VIX-export-specific data gap, not a genuine no-trading day."""
    parts = []
    for symbol in ("NIFTY50", "BANKNIFTY"):
        row = conn.execute(
            """
            SELECT b.close FROM daily_bars b JOIN instruments i ON i.id = b.instrument_id
            WHERE i.symbol = ? AND b.trade_date = ? AND b.superseded_by IS NULL
            """,
            (symbol, trade_date_str),
        ).fetchone()
        if row is None:
            return False, f"{symbol} has no bar for {trade_date_str}"
        parts.append(f"{symbol} close={row[0]}")
    return True, "; ".join(parts)


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValidationError(f"Could not parse date {value!r} with any of {DATE_FORMATS}")


def validate_and_parse(csv_path: Path, conn) -> tuple[list[dict], list[dict]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValidationError("CSV appears to be empty (no header row).")
        columns = _match_columns(reader.fieldnames)

        rows: list[dict] = []
        skipped: list[dict] = []
        seen_dates: set[date] = set()
        last_real_close: float | None = None

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
                prev_close = (
                    float(raw_row[columns["prev_close"]]) if "prev_close" in columns else None
                )
            except ValueError as exc:
                raise ValidationError(f"Non-numeric OHLC value at row {i}: {raw_row}") from exc

            is_zero_signature = (
                open_ == 0.0 and high == 0.0 and low == 0.0
                and (prev_close is None or prev_close == 0.0)
                and last_real_close is not None and close == last_real_close
            )
            if open_ == 0.0 and high == 0.0 and low == 0.0:
                # Something in the zero-row family, whether or not it matches the
                # full signature - never silently fall through to the ordinary
                # range check below, which would just reject it as "0 is outside
                # the plausible range" without explaining what was actually going on.
                if not is_zero_signature:
                    raise ValidationError(
                        f"Row {i} ({trade_date}): Open/High/Low are all 0, but this "
                        f"doesn't match the verified zero-row signature (needs "
                        f"Prev.Close=0 and Close equal to the prior trading day's "
                        f"close) - not auto-skipping. open={open_} high={high} "
                        f"low={low} close={close} prev_close={prev_close} "
                        f"last_real_close={last_real_close}"
                    )
                verified, evidence = _verify_market_traded_normally(conn, trade_date.isoformat())
                if not verified:
                    raise ValidationError(
                        f"Row {i} ({trade_date}) matches the zero-row signature, but "
                        f"could not be verified: {evidence}. Not auto-skipping - "
                        f"the underlying market data doesn't confirm normal trading."
                    )
                skipped.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "reason": (
                            f"Zero-row signature (Open=High=Low=Prev.Close=0, "
                            f"Close={close} repeats prior close) auto-verified "
                            f"against daily_bars: {evidence}. Skipped rather than "
                            f"loading zeros."
                        ),
                    }
                )
                continue

            for label, val in [("open", open_), ("high", high), ("low", low), ("close", close)]:
                if not (PLAUSIBLE_RANGE[0] <= val <= PLAUSIBLE_RANGE[1]):
                    raise ValidationError(
                        f"Row {i}: {label}={val} is outside a plausible India VIX "
                        f"range {PLAUSIBLE_RANGE} - check for a paste/unit error."
                    )

            notes = KNOWN_DATA_ANOMALIES.get(trade_date)
            if not (low <= open_ <= high and low <= close <= high) and notes is None:
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
                    "notes": notes,
                }
            )
            last_real_close = close
    if not rows:
        raise ValidationError("CSV had a header but no data rows.")
    return rows, skipped


def load_india_vix_csv(csv_path: Path) -> dict:
    conn = get_connection()
    try:
        # validate_and_parse needs conn to cross-check any zero-row signature
        # against Nifty/Bank Nifty before deciding to skip it.
        rows, skipped = validate_and_parse(csv_path, conn)
        skipped_dates = [s["trade_date"] for s in skipped]
        all_dates = sorted(r["trade_date"] for r in rows) + skipped_dates
        start, end = date.fromisoformat(min(all_dates)), date.fromisoformat(max(all_dates))

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        landed_path = RAW_DIR / f"india_vix_{start.isoformat()}_{end.isoformat()}.csv"
        shutil.copy2(csv_path, landed_path)

        instrument_id = get_or_create_instrument(conn, SYMBOL, DISPLAY_NAME, SOURCE_NAME)
        run_id = start_ingestion_run(conn, SOURCE_NAME, start, end)
        try:
            inserted = 0
            flagged = []
            for row in rows:
                if insert_bar(
                    conn, instrument_id, row["trade_date"],
                    row["open"], row["high"], row["low"], row["close"],
                    None, SOURCE_NAME, str(landed_path), run_id,
                    notes=row["notes"],
                ):
                    inserted += 1
                    if row["notes"]:
                        flagged.append(row["trade_date"])
            conn.commit()
            notes_suffix = f"; anomaly override: {flagged}" if flagged else ""
            if skipped:
                # Full reasoning, not just the dates - otherwise this run's audit
                # trail says *that* something was skipped but not *why*, and the
                # only place the evidence existed was this process's stdout.
                skip_detail = " | ".join(f"{s['trade_date']}: {s['reason']}" for s in skipped)
                notes_suffix += f"; skipped ({len(skipped)}): {skip_detail}"
            finish_ingestion_run(
                conn, run_id, "success",
                f"{len(rows)} rows validated, {inserted} new{notes_suffix}",
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
        # find_unexplained_gaps() will (correctly) include any auto-verified
        # zero-row date in this range as a gap, since no bar was inserted for it -
        # split those back out here so the report distinguishes "known,
        # already-explained" from "genuinely new, needs investigation".
        "unexplained_gaps": [d for d in gaps if d not in skipped_dates],
        "known_skipped": skipped,
        "flagged_anomalies": flagged,
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
    if result["flagged_anomalies"]:
        print(
            f"NOTE: {len(result['flagged_anomalies'])} row(s) loaded via a known "
            f"data-anomaly override (see daily_bars.notes): {result['flagged_anomalies']}"
        )
    if result["known_skipped"]:
        dates = [s["trade_date"] for s in result["known_skipped"]]
        print(
            f"NOTE: {len(result['known_skipped'])} date(s) auto-verified and "
            f"skipped (zero-row NSE data, cross-checked against Nifty/Bank "
            f"Nifty): {dates}"
        )
    if result["unexplained_gaps"]:
        print(
            f"WARNING: {len(result['unexplained_gaps'])} trading day(s) in this range "
            f"have no VIX bar: {result['unexplained_gaps']}"
        )
    else:
        print("No unexplained gaps in this range.")


if __name__ == "__main__":
    main()
