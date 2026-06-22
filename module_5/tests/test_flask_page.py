"""
test_flask_page.py — Tests for Flask app factory, route existence, and page rendering.

What we're testing:
  - The create_app() factory returns a real Flask app (not None, not broken)
  - All expected routes are registered
  - GET /analysis returns 200 and renders the required page components

Why this matters: if the factory is broken or a route is missing, every other
test in the suite would fail in confusing ways. These tests catch structural
breakage immediately and clearly.
"""

import sys
import os
import pytest
from bs4 import BeautifulSoup

# Add src/ to the path so imports work when pytest runs from module_4/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from app import create_app


# ── App factory tests ────────────────────────────────────────────────────────

@pytest.mark.web
def test_create_app_returns_flask_instance():
    """create_app() should return a Flask object, not None or an exception.

    This is the most basic sanity check: does the factory work at all?
    If this fails, nothing else will work either.
    """
    from flask import Flask
    # We pass TESTING=True so Flask surfaces errors instead of swallowing them,
    # and a fake DATABASE_URL so it doesn't try to connect to Postgres just to
    # create the app object.
    app = create_app({"TESTING": True, "DATABASE_URL": "postgresql://localhost/gradcafe_test"})
    assert app is not None
    assert isinstance(app, Flask)


@pytest.mark.web
@pytest.mark.parametrize("route,methods", [
    ("/analysis", ["GET"]),
    ("/",         ["GET"]),
    ("/pull-data", ["POST"]),
    ("/pull-status", ["GET"]),
    ("/update-analysis", ["POST"]),
])
def test_all_routes_registered(route, methods):
    """Every expected route should be registered on the app.

    We use @pytest.mark.parametrize here — one test function runs once per
    (route, methods) pair. This is DRY: five assertions from one function.

    We check the URL map (app.url_map) rather than making HTTP requests,
    because we just want to confirm the route exists — not what it returns.
    That's what the page-load tests below are for.
    """
    app = create_app({"TESTING": True, "DATABASE_URL": "postgresql://localhost/gradcafe_test"})
    # app.url_map is a Werkzeug Map of all registered routes
    registered = {rule.rule for rule in app.url_map.iter_rules()}
    assert route in registered, f"Route {route!r} not found in {registered}"


# ── Page rendering tests ─────────────────────────────────────────────────────
# These tests use the `app_client` fixture from conftest.py, which gives us
# a Flask test client wired to a clean test DB and a fake scraper.
# We parse the HTML response with BeautifulSoup so we can search it like a
# real browser would, without any brittle string matching.

@pytest.mark.web
def test_analysis_page_returns_200(app_client):
    """GET /analysis should respond with HTTP 200 OK.

    200 means "the server found the page and returned it successfully."
    Any other code (500 = server error, 404 = not found) means something broke.
    """
    response = app_client.get("/analysis")
    assert response.status_code == 200


@pytest.mark.web
def test_analysis_page_contains_analysis_heading(app_client):
    """The /analysis page should contain the word 'Analysis'.

    This confirms the right template rendered (not a blank page, not an error).
    We use BeautifulSoup to parse HTML so we're searching structured content,
    not doing fragile substring matching on raw HTML bytes.
    """
    response = app_client.get("/analysis")
    # response.data is raw bytes; .decode() turns it into a string
    soup = BeautifulSoup(response.data.decode(), "html.parser")
    # get_text() extracts all visible text, stripping tags
    page_text = soup.get_text()
    assert "Analysis" in page_text


@pytest.mark.web
def test_analysis_page_has_pull_data_button(app_client):
    """The page must contain a button with data-testid='pull-data-btn'.

    We use data-testid (not class or text) because that's a stable selector
    that won't break if we restyle or relabel the button.
    """
    response = app_client.get("/analysis")
    soup = BeautifulSoup(response.data.decode(), "html.parser")
    # find() returns the first matching tag, or None if not found
    btn = soup.find(attrs={"data-testid": "pull-data-btn"})
    assert btn is not None, "Pull Data button with data-testid='pull-data-btn' not found"


@pytest.mark.web
def test_analysis_page_has_update_analysis_button(app_client):
    """The page must contain a button with data-testid='update-analysis-btn'."""
    response = app_client.get("/analysis")
    soup = BeautifulSoup(response.data.decode(), "html.parser")
    btn = soup.find(attrs={"data-testid": "update-analysis-btn"})
    assert btn is not None, "Update Analysis button with data-testid='update-analysis-btn' not found"


@pytest.mark.web
def test_analysis_page_contains_answer_labels(app_client):
    """The rendered page should contain at least one 'Answer:' label.

    The assignment requires analysis items to be labeled with 'Answer:'.
    This test checks that the template is rendering those labels — if they're
    missing, formatting tests will also fail and users won't see labeled output.
    """
    response = app_client.get("/analysis")
    page_text = BeautifulSoup(response.data.decode(), "html.parser").get_text()
    assert "Answer:" in page_text, "No 'Answer:' labels found on the analysis page"


@pytest.mark.web
def test_root_redirects_to_analysis(app_client):
    """GET / should redirect to /analysis (backwards compatibility).

    follow_redirects=False lets us inspect the redirect itself, not the final page.
    A 302 response means "temporarily moved here" — the standard redirect code.
    """
    response = app_client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert "/analysis" in response.headers["Location"]
