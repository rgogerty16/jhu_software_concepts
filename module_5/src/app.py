"""
app.py — Flask web application for Grad Café SQL analysis.

Routes:
  GET  /analysis        — main analysis page
  GET  /                — redirects to /analysis (backwards compat)
  POST /pull-data       — start background scrape + DB load
  GET  /pull-status     — JSON: is a scrape currently running?
  POST /update-analysis — refresh analysis (rejects if scrape running)

Design note: ``create_app()`` is a *factory function* — it builds and returns a
Flask app instead of creating one at module level.  This lets tests call
``create_app({"TESTING": True, "DATABASE_URL": "..."})`` to get a clean,
isolated app pointed at a test database, without touching production state.

Database credentials are read from individual environment variables
(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME) via ``db._build_url()``.
The legacy DATABASE_URL single-string env var is still accepted as an
override for local development and CI compatibility.
"""

import os
import subprocess
import sys
import threading
from datetime import datetime

from flask import Flask, jsonify, redirect, render_template, url_for

from db import _build_url
from query_data import run_queries


def create_app(config: dict | None = None):
    """Build and return a configured Flask application instance.

    :param config: Optional dict of Flask config overrides.  Tests use this to
        inject ``TESTING=True``, a throwaway ``DATABASE_URL``, and a fake
        ``SCRAPER`` callable so no real network or database is touched.
    :type config: dict or None
    :returns: A fully configured Flask application instance.
    :rtype: flask.Flask
    """
    app = Flask(__name__)

    # ── Default config ────────────────────────────────────────────────────────
    # DATABASE_URL: prefer the explicit single-URL env var (legacy / CI path);
    # otherwise assemble from the individual DB_* env vars.
    app.config["DATABASE_URL"] = (
        os.environ.get("DATABASE_URL") or _build_url()
    )
    # SCRAPER: callable that pulls new data from Grad Café.
    # Production default spawns a real subprocess; tests swap for a fast fake.
    app.config["SCRAPER"] = _default_scraper

    # Scrape state is stored in app.config so each test app starts clean and
    # there are no underscore-prefixed attributes to trigger protected-access
    # warnings.
    app.config["SCRAPE_RUNNING"] = False
    app.config["SCRAPE_STARTED_AT"] = None
    app.config["SCRAPE_LOCK"] = threading.Lock()

    # Apply any caller-supplied overrides last so they win.
    if config:
        app.config.update(config)

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/analysis")
    def analysis():
        """Render the main analysis page with fresh query results.

        :returns: Rendered HTML page with analysis results.
        :rtype: flask.Response
        """
        results = run_queries(database_url=app.config["DATABASE_URL"])
        return render_template(
            "index.html",
            results=results,
            scrape_running=app.config["SCRAPE_RUNNING"],
            scrape_started=app.config["SCRAPE_STARTED_AT"],
            now=datetime.now(),
        )

    @app.get("/")
    def root():
        """Redirect bare root to /analysis for backwards compatibility.

        :returns: Redirect response to /analysis.
        :rtype: flask.Response
        """
        return redirect(url_for("analysis"))

    @app.post("/pull-data")
    def pull_data():
        """Start a background scrape if one is not already running.

        Returns 202 ``{"ok": true}`` when the scrape is successfully started.
        Returns 409 ``{"busy": true}`` when a scrape is already in progress.

        :returns: JSON status response.
        :rtype: flask.Response
        """
        with app.config["SCRAPE_LOCK"]:
            if app.config["SCRAPE_RUNNING"]:
                return jsonify({"busy": True}), 409
            app.config["SCRAPE_RUNNING"] = True
            app.config["SCRAPE_STARTED_AT"] = datetime.now()

        def _thread():
            try:
                app.config["SCRAPER"](app.config["DATABASE_URL"])
            finally:
                with app.config["SCRAPE_LOCK"]:
                    app.config["SCRAPE_RUNNING"] = False
                    app.config["SCRAPE_STARTED_AT"] = None

        threading.Thread(target=_thread, daemon=True).start()
        return jsonify({"ok": True}), 202

    @app.get("/pull-status")
    def pull_status():
        """Return JSON indicating whether a scrape is currently running.

        :returns: JSON with ``running`` bool and optional ``started_at`` timestamp.
        :rtype: flask.Response
        """
        started = app.config["SCRAPE_STARTED_AT"]
        return jsonify({
            "running": app.config["SCRAPE_RUNNING"],
            "started_at": started.isoformat() if started else None,
        })

    @app.post("/update-analysis")
    def update_analysis():
        """Redirect to /analysis — blocked while a pull is running.

        Returns 409 ``{"busy": true}`` if a pull is currently running.
        Otherwise redirects to GET /analysis.

        :returns: JSON error or redirect response.
        :rtype: flask.Response
        """
        if app.config["SCRAPE_RUNNING"]:
            return jsonify({"busy": True}), 409
        return redirect(url_for("analysis"))

    return app


def _default_scraper(database_url: str) -> None:
    """Run pull_and_load.py as a subprocess — the production scraper.

    Uses a context manager (``with subprocess.Popen(...) as proc``) to ensure
    the process handle is released immediately after ``.wait()`` returns.

    :param database_url: Postgres connection string forwarded to the subprocess
        via the DATABASE_URL environment variable.
    :type database_url: str
    :returns: None
    :rtype: None
    """
    script = os.path.join(os.path.dirname(__file__), "pull_and_load.py")
    env = {**os.environ, "DATABASE_URL": database_url}
    with subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    ) as proc:
        proc.wait()


# ── Dev server entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    # Debug mode is OFF by default (secure). Opt in for local dev only via
    # FLASK_DEBUG=1 — never enable debug on an internet-facing deployment.
    _debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    create_app().run(debug=_debug, port=5000)
