"""
app.py — Flask web application for Grad Café SQL analysis.

Routes:
  GET  /analysis        — main analysis page
  GET  /                — redirects to /analysis (backwards compat)
  POST /pull-data       — start background scrape + DB load
  GET  /pull-status     — JSON: is a scrape currently running?
  POST /update-analysis — refresh analysis (rejects if scrape running)

Design note: create_app() is a *factory function* — it builds and returns a
Flask app instead of creating one at module level. This lets tests call
create_app({"TESTING": True, "DATABASE_URL": "..."}) to get a clean,
isolated app pointed at a test database, without touching production state.

Database credentials are read from individual environment variables
(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME) via db._build_url().
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


def create_app(config: dict | None = None):
    """
    Application factory — build and return a configured Flask app.

    :param config: Optional dict of Flask config overrides. Tests use this to
                   inject TESTING=True, DATABASE_URL, and a fake scraper callable
                   so no real network or DB is touched.
    :type config: dict or None
    :returns: A fully configured Flask application instance.
    :rtype: flask.Flask
    """
    # __name__ tells Flask where to find templates/ and static/ relative to
    # this file. Since everything lives in src/, this still resolves correctly.
    app = Flask(__name__)

    # ── Default config ───────────────────────────────────────────────────────
    # DATABASE_URL: prefer the explicit single-URL env var (legacy / CI path);
    # otherwise assemble the URL from the individual DB_* env vars.
    app.config["DATABASE_URL"] = (
        os.environ.get("DATABASE_URL") or _build_url()
    )
    # SCRAPER: the callable that pulls new data.
    # Production default runs the real pull_and_load script.
    # Tests swap this for a fast fake that returns canned rows instantly.
    app.config["SCRAPER"] = _default_scraper

    # Apply any caller-supplied overrides last so they win.
    if config:
        app.config.update(config)

    # ── Per-app scrape state ─────────────────────────────────────────────────
    # Attached to the app object (not module globals) so each test's
    # app instance starts with a completely clean slate.
    app._scrape_lock = threading.Lock()
    app._scrape_running = False
    app._scrape_started_at = None

    # ── Routes ───────────────────────────────────────────────────────────────

    @app.get("/analysis")
    def analysis():
        """Render the main analysis page with fresh query results.

        :returns: Rendered HTML page with analysis results.
        :rtype: flask.Response
        """
        # Import inside the factory so that when tests override DATABASE_URL
        # the import happens after config is set.
        from query_data import run_queries
        results = run_queries(database_url=app.config["DATABASE_URL"])
        return render_template(
            "index.html",
            results=results,
            scrape_running=app._scrape_running,
            scrape_started=app._scrape_started_at,
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

        Returns 202 {"ok": true} when the scrape is successfully started.
        Returns 409 {"busy": true} when a scrape is already running.

        :returns: JSON status response.
        :rtype: flask.Response
        """
        with app._scrape_lock:
            if app._scrape_running:
                return jsonify({"busy": True}), 409
            app._scrape_running = True
            app._scrape_started_at = datetime.now()

        def _thread():
            try:
                # Call the injectable scraper. In production this hits the web;
                # in tests this is a fake that inserts canned rows instantly.
                app.config["SCRAPER"](app.config["DATABASE_URL"])
            finally:
                with app._scrape_lock:
                    app._scrape_running = False
                    app._scrape_started_at = None

        threading.Thread(target=_thread, daemon=True).start()
        return jsonify({"ok": True}), 202

    @app.get("/pull-status")
    def pull_status():
        """Return JSON indicating whether a scrape is currently running.

        :returns: JSON with 'running' bool and optional 'started_at' timestamp.
        :rtype: flask.Response
        """
        return jsonify({
            "running": app._scrape_running,
            "started_at": (
                app._scrape_started_at.isoformat()
                if app._scrape_started_at else None
            ),
        })

    @app.post("/update-analysis")
    def update_analysis():
        """Redirect to /analysis to refresh results — blocked when busy.

        Returns 409 {"busy": true} if a pull is currently running.
        Otherwise redirects to GET /analysis.

        :returns: JSON error or redirect response.
        :rtype: flask.Response
        """
        if app._scrape_running:
            return jsonify({"busy": True}), 409
        return redirect(url_for("analysis"))

    return app


def _default_scraper(database_url: str) -> None:
    """
    Production scraper: runs pull_and_load.py as a subprocess.

    :param database_url: Postgres connection string passed to the subprocess.
    :type database_url: str
    :returns: None
    :rtype: None
    """
    script = os.path.join(os.path.dirname(__file__), "pull_and_load.py")
    env = {**os.environ, "DATABASE_URL": database_url}
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    proc.wait()


# ── Dev server entry point ────────────────────────────────────────────────────
# Only runs when you do `python app.py` directly.
if __name__ == "__main__":  # pragma: no cover
    create_app().run(debug=True, port=5000)
