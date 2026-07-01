"""load_data.py — schema initialisation + JSON→PostgreSQL loader.

This module owns the database *schema* for the whole stack and the initial
bulk load of ``applicant_data.json``.  It is used two ways:

* At worker start-up (imported) to create the schema and seed the base data.
* Standalone (``python -m db.load_data``) to initialise a database by hand.

Schema
------
* ``applicants``            — one row per Grad Café result (Module 3 schema).
* ``ingestion_watermarks``  — high-water mark per source for incremental,
  idempotent ingestion (see :mod:`etl.incremental_scraper`).
* ``analysis_summary``      — cached scalar metrics rendered by the web UI and
  refreshed by the worker's ``recompute_analytics`` task.

SQL-injection defence
---------------------
Every statement is built with ``psycopg.sql`` composition — identifiers are
quoted with :class:`psycopg.sql.Identifier` and all row values are bound through
``%(name)s`` placeholders, never string-formatted into the SQL text.
"""

import json
import os
import re
from datetime import datetime

from psycopg import sql

from db.db import get_conn

# Natural key used for idempotent inserts and watermarking.
SOURCE = "gradcafe"
TABLE = "applicants"

# Default location of the LLM-cleaned data set.  In the container the data
# directory is bind-mounted read-only at /data (see docker-compose.yml); the
# fallback resolves to src/data/applicant_data.json for local runs.
DATA_FILE = os.environ.get(
    "DATA_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "applicant_data.json"),
)

_RESULT_ID_RE = re.compile(r"(\d+)(?!.*\d)")

_CREATE_APPLICANTS = sql.SQL("""
CREATE TABLE IF NOT EXISTS {tbl} (
    p_id                       SERIAL PRIMARY KEY,
    result_id                  BIGINT,
    program                    TEXT,
    comments                   TEXT,
    date_added                 DATE,
    url                        TEXT UNIQUE,
    status                     TEXT,
    term                       TEXT,
    us_or_international         TEXT,
    gpa                        FLOAT,
    gre                        FLOAT,
    gre_v                      FLOAT,
    gre_aw                     FLOAT,
    degree                     TEXT,
    llm_generated_program      TEXT,
    llm_generated_university   TEXT
)
""").format(tbl=sql.Identifier(TABLE))

_CREATE_WATERMARKS = sql.SQL("""
CREATE TABLE IF NOT EXISTS ingestion_watermarks (
    source        TEXT PRIMARY KEY,
    last_seen     TEXT,
    updated_at    TIMESTAMPTZ DEFAULT now()
)
""")

_CREATE_SUMMARY = sql.SQL("""
CREATE TABLE IF NOT EXISTS analysis_summary (
    metric        TEXT PRIMARY KEY,
    value         DOUBLE PRECISION,
    updated_at    TIMESTAMPTZ DEFAULT now()
)
""")

_INSERT_STMT = sql.SQL("""
INSERT INTO {tbl}
    (result_id, program, comments, date_added, url, status, term,
     us_or_international, gpa, gre, gre_v, gre_aw, degree,
     llm_generated_program, llm_generated_university)
VALUES
    (%(result_id)s, %(program)s, %(comments)s, %(date_added)s, %(url)s,
     %(status)s, %(term)s, %(us_or_international)s, %(gpa)s, %(gre)s,
     %(gre_v)s, %(gre_aw)s, %(degree)s, %(llm_generated_program)s,
     %(llm_generated_university)s)
ON CONFLICT (url) DO NOTHING
""").format(tbl=sql.Identifier(TABLE))

_COUNT_STMT = sql.SQL("SELECT COUNT(*) FROM {tbl}").format(tbl=sql.Identifier(TABLE))

_SEED_WATERMARK = sql.SQL("""
INSERT INTO ingestion_watermarks (source, last_seen, updated_at)
VALUES (%(source)s, %(last_seen)s, now())
ON CONFLICT (source) DO UPDATE
    SET last_seen  = GREATEST(
            ingestion_watermarks.last_seen::bigint,
            EXCLUDED.last_seen::bigint
        )::text,
        updated_at = now()
""")

_READ_WATERMARK = sql.SQL(
    "SELECT last_seen FROM ingestion_watermarks WHERE source = %s"
)


def create_schema(cur) -> None:
    """Create all tables (idempotently) using an open cursor.

    :param cur: An open psycopg cursor.
    :returns: None
    :rtype: None
    """
    cur.execute(_CREATE_APPLICANTS)
    cur.execute(_CREATE_WATERMARKS)
    cur.execute(_CREATE_SUMMARY)


def parse_result_id(url: str | None) -> int | None:
    """Extract the trailing integer id from a Grad Café result URL.

    ``https://www.thegradcafe.com/result/1020294`` → ``1020294``.  This id is
    monotonic in posting order, which makes it a natural watermark key.

    :param url: A result URL, or None.
    :type url: str or None
    :returns: The parsed integer id, or None when no digits are present.
    :rtype: int or None
    """
    if not url:
        return None
    match = _RESULT_ID_RE.search(url)
    return int(match.group(1)) if match else None


def _parse_date(raw: str | None):
    """Parse 'Jun 02, 2026' into a ``datetime.date``, or return None.

    :param raw: Date string in 'Mon DD, YYYY' or 'Month DD, YYYY' format.
    :type raw: str or None
    :returns: Parsed date, or None if unrecognised.
    :rtype: datetime.date or None
    """
    if not raw:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def to_row(entry: dict) -> dict:
    """Map one JSON record to the parameter dict for :data:`_INSERT_STMT`.

    :param entry: A single applicant record from the LLM-cleaned JSON.
    :type entry: dict
    :returns: Dict with keys matching the INSERT statement parameter names.
    :rtype: dict
    """
    semester = entry.get("semester") or ""
    year = entry.get("year") or ""
    term = f"{semester} {year}".strip() or None

    return {
        "result_id":                parse_result_id(entry.get("url")),
        "program":                  entry.get("program") or None,
        "comments":                 entry.get("comments") or None,
        "date_added":               _parse_date(entry.get("date_added")),
        "url":                      entry.get("url") or None,
        "status":                   entry.get("status") or None,
        "term":                     term,
        "us_or_international":       entry.get("student_type") or None,
        "gpa":                      entry.get("gpa"),
        "gre":                      entry.get("gre"),
        "gre_v":                    entry.get("gre_v"),
        "gre_aw":                   entry.get("gre_aw"),
        "degree":                   entry.get("degree") or None,
        "llm_generated_program":    entry.get("llm-generated-program") or None,
        "llm_generated_university": entry.get("llm-generated-university") or None,
    }


def insert_rows(cur, rows: list[dict]) -> None:
    """Idempotently insert normalised applicant rows on an open cursor.

    :param cur: An open psycopg cursor.
    :param rows: Parameter dicts produced by :func:`to_row`.
    :type rows: list[dict]
    :returns: None
    :rtype: None
    """
    if rows:
        cur.executemany(_INSERT_STMT, rows)


def read_watermark(cur, source: str = SOURCE) -> int:
    """Return the current high-water mark for ``source`` (0 if unset).

    :param cur: An open psycopg cursor.
    :param source: Watermark source key.
    :type source: str
    :returns: The stored last-seen id as an int, or 0 when absent.
    :rtype: int
    """
    cur.execute(_READ_WATERMARK, (source,))
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def advance_watermark(cur, last_seen: int, source: str = SOURCE) -> None:
    """Advance the watermark to ``last_seen`` (never moves backwards).

    :param cur: An open psycopg cursor.
    :param last_seen: The maximum id observed in the batch.
    :type last_seen: int
    :param source: Watermark source key.
    :type source: str
    :returns: None
    :rtype: None
    """
    cur.execute(_SEED_WATERMARK, {"source": source, "last_seen": str(last_seen)})


def _resolve_limit(limit: int | None) -> int | None:
    """Resolve the base-load size, falling back to ``INITIAL_LOAD_LIMIT``.

    A base load smaller than the full data set leaves higher-id records for the
    incremental scraper to ingest, so the UI visibly grows on "Pull Data".

    :param limit: Explicit limit, or None to read the environment.
    :type limit: int or None
    :returns: The resolved limit, or None to load every record.
    :rtype: int or None
    """
    if limit is not None:
        return limit
    env = os.environ.get("INITIAL_LOAD_LIMIT")
    return int(env) if env else None


def load_data(filepath: str = DATA_FILE, database_url: str | None = None,
              limit: int | None = None) -> int:
    """Create the schema and bulk-load applicant records. Returns the row count.

    Records are sorted by result id and, when a limit applies, only the lowest
    ids are loaded so the incremental scraper has newer ids to fetch later.
    Inserts are idempotent (``ON CONFLICT (url) DO NOTHING``) and the watermark
    is advanced to the maximum id loaded.

    :param filepath: Path to the JSON file of applicant dicts.
    :type filepath: str
    :param database_url: Optional Postgres URL override.
    :type database_url: str or None
    :param limit: Base-load size; None loads every record (or ``INITIAL_LOAD_LIMIT``).
    :type limit: int or None
    :returns: Total number of rows in the applicants table after loading.
    :rtype: int
    """
    limit = _resolve_limit(limit)
    with open(filepath, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    rows = [to_row(record) for record in records]
    rows.sort(key=lambda row: row["result_id"] or 0)
    if limit is not None:
        rows = rows[:limit]

    max_id = max((row["result_id"] or 0 for row in rows), default=0)

    with get_conn(database_url) as conn:
        with conn.cursor() as cur:
            create_schema(cur)
            insert_rows(cur, rows)
            advance_watermark(cur, max_id)
            cur.execute(_COUNT_STMT)
            total = cur.fetchone()[0]

    return total


if __name__ == "__main__":  # pragma: no cover
    TOTAL = load_data()
    print(f"applicants table now contains {TOTAL:,} rows")
