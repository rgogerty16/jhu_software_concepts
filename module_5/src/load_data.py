"""
load_data.py — Create the applicants table and load Module 2 data into PostgreSQL.

SQL Injection Defence
---------------------
All SQL statements are constructed with ``psycopg.sql.SQL`` objects so that:

* Table and column names are safely quoted via ``sql.Identifier``.
* Row values are bound through ``%s`` / ``%(name)s`` parameter placeholders
  passed to ``cur.execute`` / ``cur.executemany`` — never interpolated into
  the SQL text directly.

Idempotent: uses ``INSERT ... ON CONFLICT (url) DO NOTHING`` so re-running
never creates duplicates (url is the natural unique key for each entry).

Usage::

    python load_data.py
"""

import json
import os
from datetime import datetime

from psycopg import sql

from db import get_conn

DATA_FILE = os.path.join(
    os.path.dirname(__file__), "../module_2/llm_extend_applicant_data.json"
)

# sql.SQL objects are constructed once at module load — they are reused on
# every call to load_data() without re-parsing the SQL text.
_CREATE_TABLE_STMT = sql.SQL("""
CREATE TABLE IF NOT EXISTS {tbl} (
    p_id                  SERIAL PRIMARY KEY,
    program               TEXT,
    comments              TEXT,
    date_added            DATE,
    url                   TEXT UNIQUE,
    status                TEXT,
    term                  TEXT,
    us_or_international   TEXT,
    gpa                   FLOAT,
    gre                   FLOAT,
    gre_v                 FLOAT,
    gre_aw                FLOAT,
    degree                TEXT,
    llm_generated_program      TEXT,
    llm_generated_university   TEXT
)
""").format(tbl=sql.Identifier("applicants"))

_INSERT_STMT = sql.SQL("""
INSERT INTO {tbl}
    (program, comments, date_added, url, status, term,
     us_or_international, gpa, gre, gre_v, gre_aw, degree,
     llm_generated_program, llm_generated_university)
VALUES
    (%(program)s, %(comments)s, %(date_added)s, %(url)s, %(status)s, %(term)s,
     %(us_or_international)s, %(gpa)s, %(gre)s, %(gre_v)s, %(gre_aw)s, %(degree)s,
     %(llm_generated_program)s, %(llm_generated_university)s)
ON CONFLICT (url) DO NOTHING
""").format(tbl=sql.Identifier("applicants"))

_COUNT_STMT = sql.SQL(
    "SELECT COUNT(*) FROM {tbl} LIMIT {lim}"
).format(
    tbl=sql.Identifier("applicants"),
    lim=sql.Literal(1),
)


def _parse_date(raw: str | None):
    """Parse 'Jun 02, 2026' into a datetime.date, or return None.

    :param raw: Date string in 'Mon DD, YYYY' or 'Month DD, YYYY' format.
    :type raw: str or None
    :returns: Parsed date, or None if raw is falsy or unrecognised.
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


def _to_row(entry: dict, idx: int) -> dict:
    """Map one JSON entry to a parameter dict for INSERT_SQL.

    :param entry: A single applicant record from the module_2 JSON output.
    :type entry: dict
    :param idx: Index of the entry in the source list (unused; kept for callers).
    :type idx: int
    :returns: Dict with keys matching the INSERT_SQL parameter names.
    :rtype: dict
    """
    semester = entry.get("semester") or ""
    year = entry.get("year") or ""
    term = f"{semester} {year}".strip() or None

    return {
        "program":               entry.get("program") or None,
        "comments":              entry.get("comments") or None,
        "date_added":            _parse_date(entry.get("date_added")),
        "url":                   entry.get("url") or None,
        "status":                entry.get("status") or None,
        "term":                  term,
        "us_or_international":   entry.get("student_type") or None,
        "gpa":                   entry.get("gpa"),
        "gre":                   entry.get("gre"),
        "gre_v":                 entry.get("gre_v"),
        "gre_aw":                entry.get("gre_aw"),
        "degree":                entry.get("degree") or None,
        "llm_generated_program":      entry.get("llm-generated-program") or None,
        "llm_generated_university":   entry.get("llm-generated-university") or None,
    }


def load_data(filepath: str = DATA_FILE, database_url: str | None = None) -> int:
    """Load JSON applicant data into PostgreSQL. Returns total row count after insert.

    :param filepath: Path to a JSON file containing a list of applicant dicts.
    :type filepath: str
    :param database_url: Optional Postgres connection string. When None, db.py
                         reads DATABASE_URL from the environment.
    :type database_url: str or None
    :returns: Total number of rows in the applicants table after loading.
    :rtype: int
    """
    print(f"Reading {filepath} ...")
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records):,} records read")

    rows = [_to_row(r, i) for i, r in enumerate(records)]

    with get_conn(database_url) as conn:
        with conn.cursor() as cur:
            # Statement construction is done at module level (_CREATE_TABLE_STMT,
            # _INSERT_STMT, _COUNT_STMT).  Here we only execute them with data.
            cur.execute(_CREATE_TABLE_STMT)
            cur.executemany(_INSERT_STMT, rows)
            cur.execute(_COUNT_STMT)
            total = cur.fetchone()[0]

    print(f"  Table now contains {total:,} rows")
    print(f"  (ON CONFLICT DO NOTHING skips any duplicate URLs)")
    return total


if __name__ == "__main__":  # pragma: no cover
    total = load_data()
    print(f"\nDone — {total:,} rows in applicants table.")
