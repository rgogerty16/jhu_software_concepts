"""app — Flask web tier for the Grad Café analytics microservice.

``create_app()`` is a factory: it builds and returns an app so tests can inject
config (a throwaway ``DATABASE_URL`` and a fake ``PUBLISH`` callable) without
touching the real database or broker.

Routes
------
* ``GET  /``               — redirect to /analysis.
* ``GET  /analysis``       — render the analysis page (live queries + cache).
* ``POST /pull-data``      — enqueue ``scrape_new_data`` → 202 (503 on failure).
* ``POST /update-analysis``— enqueue ``recompute_analytics`` → 202 (503 on failure).
* ``GET  /healthz``        — liveness probe for Docker Compose.

The write buttons never block on work: they publish a task and return 202
immediately, so the request is fast and the worker does the heavy lifting.
"""

from datetime import datetime

from flask import Flask, jsonify, redirect, render_template, url_for

from app.query_data import run_queries
from publisher import publish_task


def _enqueue(publish, kind: str, message: str):
    """Publish a task and return a JSON 202, or a JSON 503 if the broker fails.

    :param publish: The publisher callable (``publish_task`` or a test fake).
    :param kind: Task kind to enqueue.
    :type kind: str
    :param message: Human-readable queued-status message for the banner.
    :type message: str
    :returns: A ``(response, status_code)`` tuple.
    :rtype: tuple
    """
    try:
        publish(kind)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return jsonify({"status": "error", "kind": kind, "error": str(exc)}), 503
    return jsonify({"status": "queued", "kind": kind, "message": message}), 202


def create_app(config: dict | None = None) -> Flask:
    """Build and return a configured Flask application.

    :param config: Optional Flask config overrides (tests inject ``DATABASE_URL``
        and a fake ``PUBLISH`` callable).
    :type config: dict or None
    :returns: A configured Flask app.
    :rtype: flask.Flask
    """
    app = Flask(__name__)
    app.config["DATABASE_URL"] = None
    app.config["PUBLISH"] = publish_task
    if config:
        app.config.update(config)

    @app.get("/analysis")
    def analysis():
        """Render the analysis page with fresh query results."""
        results = run_queries(database_url=app.config["DATABASE_URL"])
        return render_template("index.html", results=results, now=datetime.now())

    @app.get("/")
    def root():
        """Redirect bare root to /analysis."""
        return redirect(url_for("analysis"))

    @app.post("/pull-data")
    def pull_data():
        """Enqueue an incremental scrape and return 202 immediately."""
        return _enqueue(app.config["PUBLISH"], "scrape_new_data",
                        "Data pull queued — the worker is scraping new entries.")

    @app.post("/update-analysis")
    def update_analysis():
        """Enqueue an analytics recompute and return 202 immediately."""
        return _enqueue(app.config["PUBLISH"], "recompute_analytics",
                        "Analytics recompute queued — the cache will refresh shortly.")

    @app.get("/healthz")
    def healthz():
        """Liveness probe used by the Compose healthcheck."""
        return jsonify({"status": "ok"})

    return app
