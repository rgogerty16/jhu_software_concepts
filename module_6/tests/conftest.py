"""conftest.py — shared fixtures for the Module 6 suite.

Puts ``src``, ``src/web`` and ``src/worker`` on ``sys.path`` so every service
module imports in one process, exactly as it does inside its container. Provides
a clean test database, a small deterministic data set, and a Flask test client
wired to a fake publisher (no real broker touched).
"""

import json
import os
import pathlib
import sys

import psycopg
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (SRC, SRC / "web", SRC / "worker"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Import after the path is set up so the service packages resolve.
# pylint: disable=wrong-import-position
from db.load_data import create_schema, insert_rows, to_row  # noqa: E402

TEST_DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/gradcafe_m6_test")

_TABLES = "applicants, ingestion_watermarks, analysis_summary"


def _record(result_id, status="Accepted", student_type="International",
            term_year="2026", semester="Fall", gpa=3.6, program="Computer Science"):
    """Build one raw JSON-style applicant record for seeding/scraping tests."""
    return {
        "program": program,
        "raw_program": f"{program} Masters",
        "degree": "Masters",
        "status": status,
        "date_added": "Jun 02, 2026",
        "semester": semester,
        "year": term_year,
        "student_type": student_type,
        "url": f"https://www.thegradcafe.com/result/{result_id}",
        "comments": "seed",
        "gpa": gpa,
        "gre": 320.0,
        "gre_v": 160.0,
        "gre_aw": 4.5,
        "llm-generated-program": program,
        "llm-generated-university": "Johns Hopkins University",
    }


@pytest.fixture(scope="session")
def db_url():
    """Return the test database URL (shared for the whole session)."""
    return TEST_DB_URL


@pytest.fixture()
def clean_db(db_url):
    """Drop and recreate the schema before the test; drop it again after."""
    conn = psycopg.connect(db_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_TABLES} CASCADE")
        create_schema(cur)
    conn.close()

    yield db_url

    conn = psycopg.connect(db_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_TABLES} CASCADE")
    conn.close()


@pytest.fixture()
def sample_records():
    """A deterministic data set: 12 CS rows (mixed status/type) → drives every query."""
    records = []
    for i in range(2000, 2012):
        status = "Accepted" if i % 2 == 0 else ("Rejected" if i % 3 else "Waitlisted")
        student_type = "International" if i % 2 else "American"
        records.append(_record(i, status=status, student_type=student_type,
                               gpa=3.0 + (i % 5) * 0.2))
    return records


@pytest.fixture()
def seeded_db(clean_db, sample_records):
    """A clean DB pre-loaded with ``sample_records`` (committed)."""
    conn = psycopg.connect(clean_db)
    conn.autocommit = True
    with conn.cursor() as cur:
        insert_rows(cur, [to_row(r) for r in sample_records])
    conn.close()
    return clean_db


@pytest.fixture()
def temp_data_file(tmp_path, sample_records):
    """Write ``sample_records`` to a temp JSON file and return its path."""
    path = tmp_path / "applicant_data.json"
    path.write_text(json.dumps(sample_records), encoding="utf-8")
    return str(path)


@pytest.fixture()
def client(seeded_db):
    """Flask test client wired to the seeded DB and a fake (recording) publisher."""
    from app import create_app  # pylint: disable=import-outside-toplevel

    calls = []

    def fake_publish(kind, payload=None, headers=None):
        calls.append({"kind": kind, "payload": payload, "headers": headers})

    flask_app = create_app({
        "TESTING": True,
        "DATABASE_URL": seeded_db,
        "PUBLISH": fake_publish,
    })
    flask_app.config["CALLS"] = calls
    with flask_app.test_client() as test_client:
        yield test_client
