"""incremental_scraper.py — watermark-driven incremental ingestion.

The worker's ``scrape_new_data`` task fetches only records *newer* than the
stored high-water mark, normalises them to the applicants schema, inserts them
idempotently, and advances the watermark — so repeated "Pull Data" clicks keep
adding just the new rows and never create duplicates.

The fetch step reads the LLM-cleaned data set (the same file the base load uses,
bind-mounted read-only at ``/data`` in the container) and returns the next batch
of records whose result id exceeds the watermark.  This stands in for a live
Grad Café scrape while keeping the flow deterministic and container-friendly:
the id is monotonic in posting order, so "id greater than the watermark" is
exactly "posted after what we've already ingested".
"""

import json
import os

from db.load_data import (
    DATA_FILE,
    advance_watermark,
    insert_rows,
    read_watermark,
    to_row,
)

# Number of new records ingested per scrape_new_data task (per button click).
SCRAPE_BATCH = int(os.environ.get("SCRAPE_BATCH", "500"))


def fetch_new_records(since: int, batch: int, filepath: str = DATA_FILE) -> list[dict]:
    """Return up to ``batch`` normalised rows with a result id greater than ``since``.

    :param since: Current watermark; only records with a larger id are returned.
    :type since: int
    :param batch: Maximum number of records to return.
    :type batch: int
    :param filepath: Path to the JSON data set.
    :type filepath: str
    :returns: Normalised row dicts sorted by ascending result id.
    :rtype: list[dict]
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    rows = [to_row(record) for record in records]
    fresh = [row for row in rows if (row["result_id"] or 0) > since]
    fresh.sort(key=lambda row: row["result_id"])
    return fresh[:batch]


def handle_scrape_new_data(conn, payload: dict | None = None) -> dict:
    """Ingest the next batch of new records within the caller's transaction.

    Reads the watermark (or ``payload['since']`` when provided), fetches new
    records, inserts them idempotently, and advances the watermark to the
    maximum id seen.  The caller (the consumer) owns the commit/ack.

    :param conn: An open psycopg connection supplied by the consumer.
    :param payload: Optional task payload. Recognised keys: ``since`` (override
        the watermark), ``batch`` (override the batch size), ``source_file``
        (override the data-set path, used by tests).
    :type payload: dict or None
    :returns: A summary dict — ``inserted``, ``since``, ``watermark``.
    :rtype: dict
    """
    payload = payload or {}
    batch = int(payload.get("batch", SCRAPE_BATCH))
    filepath = payload.get("source_file", DATA_FILE)

    with conn.cursor() as cur:
        raw_since = payload.get("since")
        since = read_watermark(cur) if raw_since is None else int(raw_since)
        rows = fetch_new_records(since, batch, filepath)
        insert_rows(cur, rows)
        ids = [row["result_id"] for row in rows if row["result_id"] is not None]
        watermark = max(ids) if ids else since
        if ids:
            advance_watermark(cur, watermark)

    return {"inserted": len(rows), "since": since, "watermark": watermark}
