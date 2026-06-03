"""
app.py — Flask web application for Grad Café SQL analysis.

Routes:
  GET  /           — main analysis page
  POST /pull-data  — start background scrape + DB load
  GET  /pull-status — JSON: is a scrape currently running?
  POST /update     — refresh analysis (rejects if scrape running)
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime

from flask import Flask, jsonify, redirect, render_template, request, url_for

from query_data import run_queries

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Scraper state (in-process flag; fine for single-worker dev server)
# ---------------------------------------------------------------------------
_scrape_lock = threading.Lock()
_scrape_running = False
_scrape_started_at: datetime | None = None
_scrape_proc: subprocess.Popen | None = None

SCRAPE_SCRIPT = os.path.join(os.path.dirname(__file__), "pull_and_load.py")


def _scrape_thread():
    """Run pull_and_load.py in a subprocess; reset flag when done."""
    global _scrape_running, _scrape_started_at, _scrape_proc
    try:
        _scrape_proc = subprocess.Popen(
            [sys.executable, SCRAPE_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _scrape_proc.wait()
    finally:
        with _scrape_lock:
            _scrape_running = False
            _scrape_started_at = None
            _scrape_proc = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    results = run_queries()
    return render_template(
        "index.html",
        results=results,
        scrape_running=_scrape_running,
        scrape_started=_scrape_started_at,
        now=datetime.now(),
    )


@app.post("/pull-data")
def pull_data():
    """Start a background scrape if one is not already running."""
    global _scrape_running, _scrape_started_at
    with _scrape_lock:
        if _scrape_running:
            return jsonify({"status": "already_running"}), 200
        _scrape_running = True
        _scrape_started_at = datetime.now()

    t = threading.Thread(target=_scrape_thread, daemon=True)
    t.start()
    return jsonify({"status": "started"}), 202


@app.get("/pull-status")
def pull_status():
    """Return JSON status of the background scrape."""
    return jsonify({
        "running": _scrape_running,
        "started_at": _scrape_started_at.isoformat() if _scrape_started_at else None,
    })


@app.post("/update")
def update():
    """Refresh the analysis page — does nothing if a scrape is running."""
    if _scrape_running:
        return jsonify({"status": "scrape_running"}), 409
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
