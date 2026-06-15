"""
test_integration_end_to_end.py — End-to-end integration tests.

What we're testing:
  - The full pull → update → render flow works as one connected sequence
  - Multiple pulls with overlapping data stay idempotent (no duplicates)

What makes these "integration" tests vs "unit" tests:
  Unit tests check one thing in isolation (one route, one DB operation).
  Integration tests check that multiple components work *together*.
  Here we're exercising the HTTP layer, the fake scraper, the DB write,
  and the template render — all in one test.

The downside: when an integration test fails, the cause could be in any layer.
That's why we have the unit tests too — they narrow it down.
"""

import re
import time
import pytest
import psycopg
from bs4 import BeautifulSoup
from conftest import SAMPLE_ROW


def _wait_for_rows(db_url: str, expected: int, timeout: float = 3.0) -> int:
    """Poll the DB until row count reaches `expected` or timeout expires.

    :param db_url: Postgres connection string.
    :type db_url: str
    :param expected: Row count to wait for.
    :type expected: int
    :param timeout: Max seconds to wait.
    :type timeout: float
    :returns: Actual row count when polling stopped.
    :rtype: int
    """
    deadline = time.time() + timeout
    count = 0
    while time.time() < deadline:
        conn = psycopg.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM applicants;")
            count = cur.fetchone()[0]
        conn.close()
        if count >= expected:
            break
        time.sleep(0.05)
    return count


@pytest.mark.integration
def test_full_pull_update_render_flow(app_client, clean_db):
    """End-to-end: pull data → update analysis → render shows updated results.

    This test exercises the three main user actions in sequence:
      1. Click "Pull Data" → rows get added to the DB
      2. Click "Update Analysis" → page refreshes
      3. Verify the refreshed page contains analysis output

    If any of these steps breaks the chain, this test catches it.
    """
    # Step 1: Pull data — fake scraper inserts SAMPLE_ROW
    pull_response = app_client.post("/pull-data")
    assert pull_response.status_code == 202
    assert pull_response.get_json()["ok"] is True

    # Wait for the background thread to insert the row
    count = _wait_for_rows(clean_db, expected=1)
    assert count >= 1, "No rows in DB after pull"

    # Step 2: Update analysis — should redirect to /analysis (not 409)
    update_response = app_client.post("/update-analysis", follow_redirects=True)
    assert update_response.status_code == 200

    # Step 3: Render — verify the page shows analysis content with labels
    page_text = BeautifulSoup(update_response.data.decode(), "html.parser").get_text()
    assert "Analysis" in page_text
    assert "Answer:" in page_text


@pytest.mark.integration
def test_multiple_pulls_stay_idempotent(app_client, clean_db):
    """Two pulls with the same data should result in exactly one row (no duplicates).

    This mirrors the real scenario where a user clicks "Pull Data" twice
    by mistake, or the same entries appear in two consecutive scrapes.
    The UNIQUE constraint on `url` plus ON CONFLICT DO NOTHING must hold
    even when triggered through the full HTTP + thread path.
    """
    # First pull
    app_client.post("/pull-data")
    _wait_for_rows(clean_db, expected=1)

    # Second pull with the same data (fake scraper always inserts SAMPLE_ROW)
    app_client.post("/pull-data")
    _wait_for_rows(clean_db, expected=1)  # still expect exactly 1

    conn = psycopg.connect(clean_db)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM applicants;")
        count = cur.fetchone()[0]
    conn.close()
    assert count == 1, f"Duplicate rows created after two pulls: {count} rows"


@pytest.mark.integration
def test_render_shows_correctly_formatted_percentages_after_pull(app_client, clean_db):
    """After a pull, rendered percentages must still have exactly two decimal places.

    This is the integration version of test_percentages_have_two_decimal_places —
    it verifies the formatting holds end-to-end (not just on an empty DB).
    With real data in the DB, the percentage calculation runs for real and the
    two-decimal format is applied by the template.
    """
    # Insert data so percentages are non-zero
    app_client.post("/pull-data")
    _wait_for_rows(clean_db, expected=1)

    response = app_client.get("/analysis")
    page_text = BeautifulSoup(response.data.decode(), "html.parser").get_text()

    percentages = re.findall(r'\d+\.\d+%', page_text)
    two_decimal = re.compile(r'^\d+\.\d{2}%$')
    for pct in percentages:
        assert two_decimal.match(pct), (
            f"After pull, percentage {pct!r} is not formatted to exactly two decimal places"
        )


@pytest.mark.integration
def test_cannot_update_while_pull_running(app_client, clean_db):
    """Update is blocked during a pull and allowed after the pull finishes.

    This is the integration test for the busy-gate: we start a pull,
    confirm update is blocked, wait for the pull to finish, then confirm
    update succeeds.
    """
    # Start a pull — fake scraper is fast but runs in a thread
    app_client.post("/pull-data")

    # Immediately try to update — should be blocked
    block_response = app_client.post("/update-analysis")
    assert block_response.status_code == 409
    assert block_response.get_json()["busy"] is True

    # Wait for the scraper thread to finish (poll pull-status)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = app_client.get("/pull-status").get_json()
        if not status["running"]:
            break
        time.sleep(0.05)

    # Now update should succeed
    allow_response = app_client.post("/update-analysis", follow_redirects=True)
    assert allow_response.status_code == 200
