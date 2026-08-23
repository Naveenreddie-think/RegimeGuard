Inbox for manually-downloaded NSE exports (currently: India VIX historical CSVs from
https://www.nseindia.com/reports-indices-historical-vix) before they're run through
`data_agent/load_india_vix_manual.py`.

Separate from `data/raw/india_vix/manual/`, which is that script's own managed,
immutable landing zone — files land there automatically on a successful load. This
folder is just the drop point; nothing here is read by any other part of the pipeline.
