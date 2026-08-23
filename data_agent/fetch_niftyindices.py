"""Fetch and load Nifty 50 / Bank Nifty daily OHLC from niftyindices.com.

Endpoint was found by inspecting niftyindices.com's own client-side JS
(IISLComponet.js, served from liveindexsa.niftyindices.com) and verified directly
with real requests, not assumed:

- The widely-documented `/Backpage.aspx/getHistoricaldatatabletoString` route
  (referenced by several existing scraper tutorials/libraries) is retired — it now
  redirects to a Sitefinity CMS login page, even from a cookie-warmed session.
- The real, currently-working route is `/BackPage/getHistoricaldatatabletoString`
  (no `.aspx`, different capitalization). No auth, no session warm-up, no cookies
  required — confirmed from a cold request.
- The site's own "Download CSV" button does not hit a server URL at all; it
  serializes the already-rendered results table client-side. This endpoint is what
  populates that table, so it returns the same data the manual download produces.
- Cross-checked byte-for-byte against a manually downloaded reference CSV (Nifty 50,
  01-Jul-2010 to 30-Dec-2010): identical 128 rows, including the Diwali Muhurat-
  trading session on 05-Nov-2010, which niftyindices.com reports a real close for
  despite it being an official market holiday for regular trading.
- The UI enforces a 365-day range limit, but only in client-side JS before the
  request is sent — the server accepts multi-year ranges (tested up to 3 years,
  750 rows, no truncation). Chunking below is for resumability and manageable raw
  file sizes, not because the server requires it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from data_agent.db import (
    finish_ingestion_run,
    get_connection,
    get_or_create_instrument,
    insert_bar,
    start_ingestion_run,
)

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
REFERER = "https://www.niftyindices.com/reports/historical-data"
SOURCE_NAME = "niftyindices.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
WINDOW_START = date(2010, 7, 19)  # confirmed India VIX floor; whole system is gated to it


@dataclass(frozen=True)
class IndexConfig:
    symbol: str  # our internal instrument symbol
    trading_index_name: str  # exact name niftyindices.com expects (its "Trading_Index_Name")
    raw_subdir: str


INDEX_CONFIGS: dict[str, IndexConfig] = {
    "nifty50": IndexConfig(
        symbol="NIFTY50", trading_index_name="NIFTY 50", raw_subdir="nifty50"
    ),
    "banknifty": IndexConfig(
        symbol="BANKNIFTY", trading_index_name="Nifty Bank", raw_subdir="banknifty"
    ),
}


def chunk_date_range(
    start: date, end: date, chunk_years: int = 1
) -> list[tuple[date, date]]:
    """Split [start, end] into chunk_years-sized calendar-year-aligned windows."""
    chunks: list[tuple[date, date]] = []
    chunk_start = start
    while chunk_start <= end:
        year_end = date(chunk_start.year + chunk_years - 1, 12, 31)
        chunk_end = min(year_end, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def fetch_chunk(
    trading_index_name: str,
    start: date,
    end: date,
    *,
    max_retries: int = 5,
    timeout: int = 30,
) -> list[dict]:
    """POST one date-range request, retrying with exponential backoff on failure."""
    cinfo = (
        "{'name':'" + trading_index_name.upper() + "',"
        "'startDate':'" + start.strftime("%d-%b-%Y") + "',"
        "'endDate':'" + end.strftime("%d-%b-%Y") + "',"
        "'indexName':'" + trading_index_name + "'}"
    )
    payload = {"cinfo": cinfo}
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": REFERER,
        "User-Agent": USER_AGENT,
    }

    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Some ASP.NET WebMethod responses wrap the payload as {"d": "<json string>"}
            # instead of returning the array directly; handle both, matching the site's
            # own client-side code.
            if isinstance(data, dict) and "d" in data:
                data = json.loads(data["d"]) if isinstance(data["d"], str) else data["d"]
            if not isinstance(data, list):
                raise ValueError(f"Unexpected response shape: {type(data)}")
            return data
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            logger.warning(
                "Fetch attempt %d/%d failed for %s %s-%s: %s",
                attempt, max_retries, trading_index_name, start, end, exc,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(
        f"Failed to fetch {trading_index_name} {start}-{end} after {max_retries} attempts"
    ) from last_exc


def land_raw(raw_subdir: str, start: date, end: date, data: list[dict]) -> Path:
    """Write the exact response payload to the raw landing zone before any parsing."""
    out_dir = RAW_DIR / raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{raw_subdir}_{start.isoformat()}_{end.isoformat()}.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path


def parse_row(row: dict) -> dict:
    trade_date = datetime.strptime(row["HistoricalDate"], "%d %b %Y").date()
    return {
        "trade_date": trade_date.isoformat(),
        "open": float(row["OPEN"]),
        "high": float(row["HIGH"]),
        "low": float(row["LOW"]),
        "close": float(row["CLOSE"]),
    }


def load_bars(
    conn: sqlite3.Connection,
    instrument_id: int,
    rows: list[dict],
    raw_file_ref: Path,
    ingestion_run_id: int,
) -> int:
    """Parse and insert rows via the shared insert_bar(), skipping dates already
    loaded (idempotent re-runs)."""
    inserted = 0
    for row in rows:
        parsed = parse_row(row)
        if insert_bar(
            conn, instrument_id, parsed["trade_date"],
            parsed["open"], parsed["high"], parsed["low"], parsed["close"],
            None, SOURCE_NAME, str(raw_file_ref), ingestion_run_id,
        ):
            inserted += 1
    conn.commit()
    return inserted


def ingest_index(
    conn: sqlite3.Connection,
    config: IndexConfig,
    start: date,
    end: date,
    chunk_years: int = 1,
) -> None:
    instrument_id = get_or_create_instrument(
        conn, config.symbol, config.trading_index_name, SOURCE_NAME
    )
    for chunk_start, chunk_end in chunk_date_range(start, end, chunk_years):
        run_id = start_ingestion_run(conn, SOURCE_NAME, chunk_start, chunk_end)
        try:
            data = fetch_chunk(config.trading_index_name, chunk_start, chunk_end)
            raw_path = land_raw(config.raw_subdir, chunk_start, chunk_end, data)
            inserted = load_bars(conn, instrument_id, data, raw_path, run_id)
            finish_ingestion_run(
                conn, run_id, "success", f"{len(data)} rows fetched, {inserted} new"
            )
            logger.info(
                "%s %s to %s: %d rows fetched, %d new rows loaded",
                config.symbol, chunk_start, chunk_end, len(data), inserted,
            )
        except Exception as exc:
            finish_ingestion_run(conn, run_id, "failed", str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Nifty 50 / Bank Nifty daily OHLC from niftyindices.com"
    )
    parser.add_argument(
        "--index", choices=list(INDEX_CONFIGS), action="append",
        help="Index to fetch (default: both)",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--chunk-years", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    indices = args.index or list(INDEX_CONFIGS)
    conn = get_connection()
    try:
        for key in indices:
            ingest_index(conn, INDEX_CONFIGS[key], args.start, args.end, args.chunk_years)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
