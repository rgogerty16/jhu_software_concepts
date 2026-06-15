"""
test_db_insert.py — Tests for database writes, schema, idempotency, and queries.

What we're testing:
  - After POST /pull-data, new rows exist in the DB with the required schema
  - Required fields (url, status, term) are non-null
  - Pulling the same data twice does NOT create duplicate rows (idempotency)
  - run_queries() returns a dict with all the keys the template expects

Why idempotency matters: the "Pull Data" button can be clicked multiple times.
If each click inserted duplicates, your analysis numbers would be wrong and
the UNIQUE constraint on url would start throwing errors.
"""

import time
import pytest
import psycopg
from conftest import SAMPLE_ROW


# ── Schema / insert tests ────────────────────────────────────────────────────

@pytest.mark.db
def test_db_starts_empty(clean_db):
    """The test database should start with zero rows.

    This verifies our clean_db fixture is working correctly.
    If this test fails, another test is leaking state into the DB — a bug
    in the test suite itself that would make other tests unreliable.
    """
    conn = psycopg.connect(clean_db)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM applicants;")
        count = cur.fetchone()[0]
    conn.close()
    assert count == 0


@pytest.mark.db
def test_pull_data_inserts_rows(app_client, clean_db):
    """After POST /pull-data, new rows should exist in the database.

    We POST to /pull-data (which triggers the fake scraper), wait briefly
    for the background thread to complete, then query the DB directly to
    confirm a row was inserted.

    Note the small sleep: the fake scraper runs in a background thread.
    We need to give it a moment to finish. We use a short poll loop instead
    of a fixed sleep so the test doesn't waste time if the thread finishes early.
    """
    app_client.post("/pull-data")
    # Poll for the row with a timeout instead of a fixed sleep
    deadline = time.time() + 3.0  # 3 second timeout
    count = 0
    while time.time() < deadline:
        conn = psycopg.connect(clean_db)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM applicants;")
            count = cur.fetchone()[0]
        conn.close()
        if count > 0:
            break
        time.sleep(0.05)
    assert count > 0, "No rows inserted after POST /pull-data"


@pytest.mark.db
def test_inserted_row_has_required_non_null_fields(clean_db):
    """Rows inserted into applicants must have the required non-null fields.

    We insert SAMPLE_ROW directly (bypassing the HTTP layer) and then query
    back to verify url, status, and term are non-null.

    We test the DB layer directly here (not through the HTTP layer) because:
    1. It's faster — no HTTP overhead
    2. It isolates the concern: we're testing the schema, not the endpoint
    """
    conn = psycopg.connect(clean_db)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO applicants
                (program, comments, date_added, url, status, term,
                 us_or_international, gpa, gre, gre_v, gre_aw, degree,
                 llm_generated_program, llm_generated_university)
            VALUES
                (%(program)s, %(comments)s, %(date_added)s, %(url)s,
                 %(status)s, %(term)s, %(us_or_international)s,
                 %(gpa)s, %(gre)s, %(gre_v)s, %(gre_aw)s, %(degree)s,
                 %(llm_generated_program)s, %(llm_generated_university)s)
            ON CONFLICT (url) DO NOTHING;
        """, SAMPLE_ROW)
        # Read it back and check required fields
        cur.execute("SELECT url, status, term FROM applicants WHERE url = %(url)s;", SAMPLE_ROW)
        row = cur.fetchone()
    conn.close()

    assert row is not None, "Row was not inserted"
    url, status, term = row
    assert url is not None,    "url must not be NULL"
    assert status is not None, "status must not be NULL"
    assert term is not None,   "term must not be NULL"


@pytest.mark.db
def test_duplicate_pull_does_not_create_duplicates(clean_db):
    """Inserting the same URL twice should result in exactly one row.

    This tests the ON CONFLICT (url) DO NOTHING constraint that prevents
    the "Pull Data" button from creating duplicate entries when clicked
    multiple times.

    We insert SAMPLE_ROW twice and assert the count is still 1.
    """
    conn = psycopg.connect(clean_db)
    conn.autocommit = True
    insert_sql = """
        INSERT INTO applicants
            (program, comments, date_added, url, status, term,
             us_or_international, gpa, gre, gre_v, gre_aw, degree,
             llm_generated_program, llm_generated_university)
        VALUES
            (%(program)s, %(comments)s, %(date_added)s, %(url)s,
             %(status)s, %(term)s, %(us_or_international)s,
             %(gpa)s, %(gre)s, %(gre_v)s, %(gre_aw)s, %(degree)s,
             %(llm_generated_program)s, %(llm_generated_university)s)
        ON CONFLICT (url) DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.execute(insert_sql, SAMPLE_ROW)  # first insert
        cur.execute(insert_sql, SAMPLE_ROW)  # duplicate — should be silently skipped
        cur.execute("SELECT COUNT(*) FROM applicants;")
        count = cur.fetchone()[0]
    conn.close()
    assert count == 1, f"Expected 1 row after duplicate insert, got {count}"


@pytest.mark.db
def test_run_queries_returns_expected_keys(clean_db):
    """run_queries() must return a dict containing all keys the template uses.

    The template (index.html) references specific keys like q1_fall_2026_count,
    q2_pct_international, etc. If run_queries() ever drops one of these keys,
    the page will throw a Jinja TemplateError. This test catches that early.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from query_data import run_queries

    results = run_queries(database_url=clean_db)

    required_keys = [
        "total_entries",
        "q1_fall_2026_count",
        "q2_pct_international",
        "q3_avg_gpa",
        "q3_avg_gre",
        "q3_avg_gre_v",
        "q3_avg_gre_aw",
        "q4_avg_gpa_american_fall2026",
        "q5_pct_accepted_fall2026",
        "q6_avg_gpa_accepted_fall2026",
        "q7_jhu_masters_cs",
        "q8_raw_top4_phd_cs_2026",
        "q9_llm_top4_phd_cs_2026",
        "q10_top_acceptance_programs",
        "q11_gpa_by_status",
    ]
    for key in required_keys:
        assert key in results, f"Key {key!r} missing from run_queries() result"
