"""
conftest.py — Shared pytest fixtures for the module_4 test suite.

pytest automatically discovers and loads this file before any tests run.
Fixtures defined here are available to every test file without importing.

Key concepts:
  - A *fixture* is a function decorated with @pytest.fixture. pytest calls it
    automatically and injects its return value into any test that names it as
    a parameter.
  - *scope* controls how often the fixture runs:
      "function" (default) → re-run for every test (clean slate each time)
      "session"            → run once for the whole test session (expensive setup)
  - conftest.py is the right place for fixtures shared across multiple files.
"""

import os
import pytest
import psycopg

# ── Test database URL ────────────────────────────────────────────────────────
# We create a separate "gradcafe_test" database so tests never touch your
# real data. The CI workflow also sets this env var (see tests.yml).
TEST_DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost/gradcafe_test"
)

# ── Minimal applicant row used by DB tests ───────────────────────────────────
# This matches the required Module-3 schema exactly. We use the same dict
# in multiple test files, so it lives here.
SAMPLE_ROW = {
    "program":                "Computer Science",
    "comments":               "Test entry",
    "date_added":             "2026-01-15",
    "url":                    "https://thegradcafe.com/survey/test-unique-001",
    "status":                 "Accepted",
    "term":                   "Fall 2026",
    "us_or_international":    "American",
    "gpa":                    3.90,
    "gre":                    165.0,
    "gre_v":                  162.0,
    "gre_aw":                 5.0,
    "degree":                 "masters",
    "llm_generated_program":  "Computer Science",
    "llm_generated_university": "Johns Hopkins University",
}


@pytest.fixture(scope="session")
def db_url():
    """Return the test database URL.

    scope="session" means this runs once per pytest session (not once per test).
    All DB tests share the same URL string — no need to re-derive it.

    :returns: Postgres connection string for the test database.
    :rtype: str
    """
    return TEST_DB_URL


@pytest.fixture(scope="function")
def clean_db(db_url):
    """Create the applicants table, yield, then drop it.

    This is the fixture that makes DB tests safe and idempotent:
      1. Before the test: create a fresh empty table
      2. Run the test
      3. After the test: drop the table so the next test starts clean

    scope="function" means this runs for *every* test that requests it.
    The "yield" pattern is pytest's way of doing setup/teardown in one function:
    everything before yield is setup; everything after yield is teardown.

    :param db_url: Injected by the db_url fixture above.
    :type db_url: str
    :yields: The database URL so tests can pass it to the functions under test.
    :rtype: str
    """
    conn = psycopg.connect(db_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        # Drop any leftover table from a previous failed test run
        cur.execute("DROP TABLE IF EXISTS applicants;")
        cur.execute("""
            CREATE TABLE applicants (
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
            );
        """)
    conn.close()

    yield db_url   # ← the test runs here

    # Teardown: drop the table so next test starts clean
    conn = psycopg.connect(db_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS applicants;")
    conn.close()


@pytest.fixture()
def fake_scraper():
    """Return a fake scraper callable that inserts one SAMPLE_ROW into the DB.

    The real scraper hits the Grad Café website and takes ~10 minutes.
    Tests must NEVER do that. Instead, app.config["SCRAPER"] is set to this
    function — it inserts one known row instantly so tests can verify behavior
    without any network access.

    The function signature matches the real scraper: (database_url: str) -> None.

    :returns: A callable that inserts SAMPLE_ROW into the test DB.
    :rtype: callable
    """
    def _fake(database_url: str) -> None:
        # The table may be dropped by teardown before this thread finishes
        # (e.g. in busy-state tests that don't need the insert to succeed).
        # Silently ignore UndefinedTable so the thread exits cleanly.
        try:
            conn = psycopg.connect(database_url)
        except Exception:
            return
        conn.autocommit = True
        try:
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
        except Exception:
            pass
        finally:
            conn.close()

    return _fake


@pytest.fixture()
def app_client(clean_db, fake_scraper):
    """Create a Flask test client wired to the test DB and fake scraper.

    This is the fixture most tests will use for HTTP-level testing.
    Flask's test client lets you make GET/POST requests to your app routes
    from Python code, without starting a real server.

    How it's wired:
      - DATABASE_URL → clean_db (throwaway test DB, not production)
      - SCRAPER → fake_scraper (fast in-memory fake, not real Grad Café)
      - TESTING=True → Flask disables error catching so test failures are clear

    :param clean_db: Injected by the clean_db fixture (provides test DB URL).
    :param fake_scraper: Injected by the fake_scraper fixture.
    :yields: A Flask test client for making requests to the app.
    :rtype: flask.testing.FlaskClient
    """
    import sys, os
    # Ensure src/ is on the path so `from app import create_app` works
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from app import create_app
    flask_app = create_app({
        "TESTING": True,
        "DATABASE_URL": clean_db,
        "SCRAPER": fake_scraper,
    })

    with flask_app.test_client() as client:
        yield client
