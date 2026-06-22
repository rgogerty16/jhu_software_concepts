"""
test_buttons.py — Tests for the Pull Data and Update Analysis button endpoints.

What we're testing:
  - POST /pull-data returns 202 {"ok": true} when idle
  - POST /pull-data returns 409 {"busy": true} when already running
  - POST /update-analysis returns a redirect (200 after follow) when idle
  - POST /update-analysis returns 409 {"busy": true} when a pull is running

Why busy-state matters: the assignment requires that if you click "Update Analysis"
while a scrape is running, nothing should happen. Without this guard, the page
could render stale partial data mid-scrape, which is misleading.

Note on testing async behavior: The real scraper runs in a background thread and
takes minutes. Tests must NEVER wait on that. Instead, we *inject* the busy state
directly using app._scrape_running = True. This makes the test instant and
deterministic — no sleep(), no timing races.
"""

import pytest


@pytest.mark.buttons
def test_pull_data_returns_202_when_idle(app_client):
    """POST /pull-data should return 202 with {"ok": true} when not busy.

    202 means "Accepted" — the request was received and processing has started
    (in the background). It's different from 200 ("OK, done") because the scrape
    hasn't finished yet; it's just been kicked off.
    """
    response = app_client.post("/pull-data")
    assert response.status_code == 202
    data = response.get_json()
    assert data["ok"] is True


@pytest.mark.buttons
def test_pull_data_returns_409_when_busy(app_client):
    """POST /pull-data should return 409 with {"busy": true} if already running.

    We inject the busy state by calling /pull-data once to start the scraper,
    then immediately call it again. The fake scraper (from conftest.py) runs in
    a background thread, so the second call arrives while the first is still
    "running" from the app's perspective.

    409 means "Conflict" — the request is valid but can't be fulfilled right now
    because of the current server state.
    """
    # First call starts the scraper
    app_client.post("/pull-data")
    # Second call should be rejected because a scrape is in progress
    response = app_client.post("/pull-data")
    assert response.status_code == 409
    data = response.get_json()
    assert data["busy"] is True


@pytest.mark.buttons
def test_update_analysis_returns_redirect_when_idle(app_client):
    """POST /update-analysis should redirect to /analysis when not busy.

    follow_redirects=True means the test client follows the redirect and returns
    the final page response. We check for 200 (the analysis page loaded) rather
    than checking for the 302 redirect itself. This mirrors the real user
    experience: click the button → page refreshes with latest data.
    """
    response = app_client.post("/update-analysis", follow_redirects=True)
    assert response.status_code == 200


@pytest.mark.buttons
def test_update_analysis_returns_409_when_busy(app_client):
    """POST /update-analysis should return 409 {"busy": true} during a pull.

    We inject busy state by starting a pull first, then immediately trying to
    update. This ensures the guard logic works: you can't update while pulling.
    """
    # Start a scrape to put the app into busy state
    app_client.post("/pull-data")
    # Now try to update — should be blocked
    response = app_client.post("/update-analysis")
    assert response.status_code == 409
    data = response.get_json()
    assert data["busy"] is True


@pytest.mark.buttons
def test_pull_status_idle(app_client):
    """GET /pull-status should report not running when idle.

    This is the polling endpoint that the browser JavaScript calls every few
    seconds to keep the UI in sync with server state.
    """
    response = app_client.get("/pull-status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["running"] is False
    assert data["started_at"] is None


@pytest.mark.buttons
def test_pull_status_busy(app_client):
    """GET /pull-status should report running=True after a pull is started."""
    app_client.post("/pull-data")
    response = app_client.get("/pull-status")
    data = response.get_json()
    assert data["running"] is True
