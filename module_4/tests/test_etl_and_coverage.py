"""
test_etl_and_coverage.py — Tests for the ETL layer and remaining coverage gaps.

Covers:
  - load_data._parse_date (date parsing helper)
  - load_data._to_row (JSON → DB row mapping)
  - load_data.load_data (end-to-end insert into test DB)
  - pull_and_load.pull_and_load (with injected fakes — no real scraper)
  - query_data.print_results (stdout output)
  - app._default_scraper (mocked subprocess — no real subprocess spawned)

Why we mock subprocess for _default_scraper:
  The real scraper spawns a subprocess that runs pull_and_load.py, which in turn
  would try to import selenium and scrape the web. That's 10 minutes and network
  access — unacceptable in a test suite. unittest.mock.patch replaces
  subprocess.Popen with a fake that records what it was called with, so we can
  verify the code path without any real side effects.
"""

import json
import os
import sys
import tempfile
import io
from datetime import date
from unittest.mock import patch, MagicMock

import pytest
import psycopg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── _parse_date tests ────────────────────────────────────────────────────────

@pytest.mark.db
@pytest.mark.parametrize("raw,expected", [
    ("Jun 02, 2026",      date(2026, 6, 2)),
    ("January 15, 2025",  date(2025, 1, 15)),
    (None,                None),
    ("",                  None),
    ("not a date",        None),
])
def test_parse_date(raw, expected):
    """_parse_date should handle short month, long month, None, empty, and garbage.

    We use @pytest.mark.parametrize so each (raw, expected) pair runs as its
    own test case. That means 5 tests from one function — DRY and explicit.
    """
    from load_data import _parse_date
    assert _parse_date(raw) == expected


# ── _to_row tests ────────────────────────────────────────────────────────────

@pytest.mark.db
def test_to_row_maps_fields_correctly():
    """_to_row should correctly map a JSON entry dict to the DB parameter dict."""
    from load_data import _to_row
    entry = {
        "program":               "Computer Science",
        "comments":              "Test comment",
        "date_added":            "Jun 02, 2026",
        "url":                   "https://example.com/1",
        "status":                "Accepted",
        "semester":              "Fall",
        "year":                  "2026",
        "student_type":          "American",
        "gpa":                   3.9,
        "gre":                   165.0,
        "gre_v":                 160.0,
        "gre_aw":                5.0,
        "degree":                "masters",
        "llm-generated-program": "Computer Science",
        "llm-generated-university": "Johns Hopkins University",
    }
    row = _to_row(entry, 0)
    assert row["program"] == "Computer Science"
    assert row["term"] == "Fall 2026"
    assert row["date_added"] == date(2026, 6, 2)
    assert row["us_or_international"] == "American"
    assert row["llm_generated_program"] == "Computer Science"
    assert row["llm_generated_university"] == "Johns Hopkins University"


@pytest.mark.db
def test_to_row_empty_semester_year_yields_none_term():
    """When semester and year are both missing, term should be None."""
    from load_data import _to_row
    row = _to_row({"url": "https://example.com/x"}, 0)
    assert row["term"] is None


# ── load_data end-to-end ─────────────────────────────────────────────────────

@pytest.mark.db
def test_load_data_inserts_from_json(clean_db):
    """load_data() should read a JSON file and insert rows into the DB."""
    from load_data import load_data

    # Build a minimal JSON file with one record and write it to a temp file
    records = [{
        "program": "CS",
        "comments": "Test",
        "date_added": "Jun 01, 2026",
        "url": "https://gradcafe.com/load-test-001",
        "status": "Accepted",
        "semester": "Fall",
        "year": "2026",
        "student_type": "American",
        "gpa": 3.8,
        "gre": 165.0,
        "gre_v": 160.0,
        "gre_aw": 4.5,
        "degree": "masters",
        "llm-generated-program": "Computer Science",
        "llm-generated-university": "MIT",
    }]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(records, f)
        tmp_path = f.name

    try:
        total = load_data(tmp_path, database_url=clean_db)
        assert total == 1
    finally:
        os.unlink(tmp_path)


@pytest.mark.db
def test_load_data_idempotent(clean_db):
    """load_data() called twice with the same file should still yield 1 row."""
    from load_data import load_data

    records = [{
        "url": "https://gradcafe.com/idempotent-001",
        "status": "Accepted",
        "semester": "Fall",
        "year": "2026",
    }]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(records, f)
        tmp_path = f.name

    try:
        load_data(tmp_path, database_url=clean_db)
        total = load_data(tmp_path, database_url=clean_db)
        assert total == 1
    finally:
        os.unlink(tmp_path)


# ── pull_and_load tests ──────────────────────────────────────────────────────

@pytest.mark.db
def test_pull_and_load_calls_scraper_and_inserts(clean_db):
    """pull_and_load() should call the scrape fn, save fn, and load fn in order."""
    from pull_and_load import pull_and_load

    # Build a minimal record for the fake scraper to return
    fake_record = {
        "url": "https://gradcafe.com/pull-test-001",
        "status": "Accepted",
        "semester": "Fall",
        "year": "2026",
    }

    calls = []

    def fake_scrape(max_entries):
        calls.append("scraped")
        return [fake_record]

    def fake_save(records, filepath):
        calls.append("saved")
        # Write real JSON so load_fn can open it
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(records, fh)

    from load_data import load_data as real_load

    def fake_load(filepath, database_url=None):
        calls.append("loaded")
        return real_load(filepath, database_url=clean_db)

    pull_and_load(scrape_fn=fake_scrape, save_fn=fake_save, load_fn=fake_load,
                  database_url=clean_db)

    assert calls == ["scraped", "saved", "loaded"]

    # Verify row actually landed in the DB
    conn = psycopg.connect(clean_db)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM applicants;")
        count = cur.fetchone()[0]
    conn.close()
    assert count == 1


@pytest.mark.db
def test_pull_and_load_uses_real_load_data_when_load_fn_not_injected(clean_db):
    """When load_fn is None, pull_and_load should lazy-import and use real load_data.

    This covers lines 47-49 (the `if load_fn is None:` branch) which only run
    when the caller doesn't inject a custom loader. We still inject scrape and
    save fakes to avoid touching the network.
    """
    from pull_and_load import pull_and_load

    fake_record = {
        "url": "https://gradcafe.com/lazy-load-test-001",
        "status": "Accepted",
        "semester": "Fall",
        "year": "2026",
    }

    def fake_scrape(max_entries):
        return [fake_record]

    def fake_save(records, filepath):
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(records, fh)

    # load_fn=None forces the lazy import of load_data inside pull_and_load
    pull_and_load(scrape_fn=fake_scrape, save_fn=fake_save, load_fn=None,
                  database_url=clean_db)

    conn = psycopg.connect(clean_db)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM applicants;")
        count = cur.fetchone()[0]
    conn.close()
    assert count == 1


@pytest.mark.db
def test_pull_and_load_no_records_returns_early():
    """pull_and_load() should return early when the scraper returns no records."""
    from pull_and_load import pull_and_load

    calls = []

    def fake_scrape(max_entries):
        return []  # empty — simulates a scraper that found nothing new

    def fake_save(records, filepath):
        calls.append("saved")  # should NOT be called

    def fake_load(filepath, database_url=None):
        calls.append("loaded")  # should NOT be called

    pull_and_load(scrape_fn=fake_scrape, save_fn=fake_save, load_fn=fake_load)
    assert calls == [], "save and load should not be called when scraper returns empty"


# ── query_data.print_results ─────────────────────────────────────────────────

@pytest.mark.analysis
def test_print_results_outputs_all_questions(capsys):
    """print_results() should print a summary of all query results to stdout.

    pytest's built-in `capsys` fixture captures stdout/stderr output so we can
    assert on what print() would display, without actually printing to the terminal.
    """
    from query_data import print_results

    # Build a minimal results dict that matches what run_queries() returns
    sample_results = {
        "q1_fall_2026_count": 100,
        "q2_pct_international": 45.51,
        "q3_avg_gpa": 3.77,
        "q3_avg_gre": 261.45,
        "q3_avg_gre_v": 160.75,
        "q3_avg_gre_aw": 4.36,
        "q4_avg_gpa_american_fall2026": 3.79,
        "q5_pct_accepted_fall2026": 36.94,
        "q6_avg_gpa_accepted_fall2026": 3.76,
        "q7_jhu_masters_cs": 8,
        "q8_raw_top4_phd_cs_2026": 0,
        "q9_llm_top4_phd_cs_2026": 28,
        "q10_top_acceptance_programs": [
            {"program": "Business Analytics", "total": 13, "accepted": 12, "rate_pct": 92.3}
        ],
        "q11_gpa_by_status": [
            {"status": "Accepted", "n": 7389, "avg_gpa": 3.783}
        ],
    }

    print_results(sample_results)
    captured = capsys.readouterr()

    assert "Grad Café SQL Analysis Results" in captured.out
    assert "Q1" in captured.out
    assert "45.51%" in captured.out
    assert "Q10" in captured.out
    assert "Business Analytics" in captured.out


# ── app._default_scraper ─────────────────────────────────────────────────────

@pytest.mark.web
def test_default_scraper_calls_subprocess(tmp_path):
    """_default_scraper should launch pull_and_load.py as a subprocess.

    We mock subprocess.Popen so no real process is created.
    The mock records what arguments Popen was called with so we can verify
    the function built the right command.

    unittest.mock.patch temporarily replaces the real subprocess.Popen
    with a MagicMock object. When the code under test calls Popen(...),
    it calls our mock instead. After the `with` block, the real Popen is restored.
    """
    from app import _default_scraper

    # MagicMock.return_value is what Popen() returns — i.e. the process object.
    # We give it a .wait() method that does nothing so the code doesn't hang.
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0

    with patch("app.subprocess.Popen", return_value=mock_proc) as mock_popen:
        _default_scraper("postgresql://localhost/gradcafe_test")

    # Verify Popen was called once
    mock_popen.assert_called_once()
    # Verify it was passed the python executable and a path ending in pull_and_load.py
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("pull_and_load.py")
    # Verify DATABASE_URL was injected into the subprocess environment
    assert kwargs["env"]["DATABASE_URL"] == "postgresql://localhost/gradcafe_test"
    # Verify .wait() was called (we waited for the subprocess to finish)
    mock_proc.wait.assert_called_once()
