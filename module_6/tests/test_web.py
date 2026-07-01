"""Tests for the web tier: publisher, Flask routes, rendering, entrypoint."""

import json
from unittest.mock import MagicMock

import pytest

import publisher
from app import create_app
from app.query_data import run_queries
from etl.query_data import handle_recompute_analytics
import psycopg


# ── Publisher ───────────────────────────────────────────────────────────────

@pytest.mark.publisher
def test_open_channel_declares_durable_topology(monkeypatch):
    """_open_channel reads RABBITMQ_URL and declares the durable topology."""
    fake_conn = MagicMock()
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setattr(publisher.pika, "URLParameters", lambda url: ("params", url))
    monkeypatch.setattr(publisher.pika, "BlockingConnection", lambda params: fake_conn)

    conn, channel = publisher._open_channel()
    assert conn is fake_conn
    assert channel is fake_conn.channel.return_value
    channel.exchange_declare.assert_called_once_with(
        exchange="tasks", exchange_type="direct", durable=True)
    channel.queue_declare.assert_called_once_with(queue="tasks_q", durable=True)
    channel.queue_bind.assert_called_once_with(
        exchange="tasks", queue="tasks_q", routing_key="tasks")


@pytest.mark.publisher
def test_publish_task_persistent_message(monkeypatch):
    """publish_task sends a compact, persistent (delivery_mode=2) JSON message."""
    conn, channel = MagicMock(), MagicMock()
    monkeypatch.setattr(publisher, "_open_channel", lambda: (conn, channel))

    publisher.publish_task("scrape_new_data", {"a": 1}, {"h": "v"})

    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["exchange"] == "tasks"
    assert kwargs["routing_key"] == "tasks"
    assert kwargs["properties"].delivery_mode == 2
    assert kwargs["properties"].headers == {"h": "v"}
    body = json.loads(kwargs["body"])
    assert body["kind"] == "scrape_new_data"
    assert body["payload"] == {"a": 1}
    assert "ts" in body
    conn.close.assert_called_once()          # closed in finally


@pytest.mark.publisher
def test_publish_task_closes_connection_on_error(monkeypatch):
    """A publish failure propagates but the connection is still closed."""
    conn, channel = MagicMock(), MagicMock()
    channel.basic_publish.side_effect = RuntimeError("broker down")
    monkeypatch.setattr(publisher, "_open_channel", lambda: (conn, channel))

    with pytest.raises(RuntimeError):
        publisher.publish_task("recompute_analytics")
    conn.close.assert_called_once()


# ── Flask routes ────────────────────────────────────────────────────────────

@pytest.mark.web
def test_root_redirects_to_analysis(client):
    """GET / redirects to /analysis."""
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/analysis" in resp.headers["Location"]


@pytest.mark.web
def test_analysis_page_renders(client):
    """GET /analysis renders the page with the seeded data."""
    resp = client.get("/analysis")
    assert resp.status_code == 200
    assert b"Analysis" in resp.data
    assert b"Cached Summary" in resp.data


@pytest.mark.web
def test_analysis_page_renders_populated_cache(client, seeded_db):
    """After a recompute the cached-summary cards render real numbers."""
    with psycopg.connect(seeded_db) as conn:
        handle_recompute_analytics(conn, {})
    resp = client.get("/analysis")
    assert resp.status_code == 200
    assert b"Last recomputed" in resp.data


@pytest.mark.web
def test_pull_data_enqueues_202(client):
    """POST /pull-data publishes scrape_new_data and returns 202."""
    resp = client.post("/pull-data")
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["status"] == "queued"
    assert body["kind"] == "scrape_new_data"
    assert client.application.config["CALLS"][0]["kind"] == "scrape_new_data"


@pytest.mark.web
def test_update_analysis_enqueues_202(client):
    """POST /update-analysis publishes recompute_analytics and returns 202."""
    resp = client.post("/update-analysis")
    assert resp.status_code == 202
    assert resp.get_json()["kind"] == "recompute_analytics"
    assert client.application.config["CALLS"][0]["kind"] == "recompute_analytics"


@pytest.mark.web
def test_healthz(client):
    """GET /healthz returns ok for the Compose healthcheck."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


@pytest.mark.web
def test_enqueue_returns_503_on_publish_failure(clean_db):
    """When the broker is unreachable the endpoint returns 503, not 500."""
    def failing_publish(kind, payload=None, headers=None):
        raise RuntimeError("broker unreachable")
    app = create_app({"TESTING": True, "DATABASE_URL": clean_db, "PUBLISH": failing_publish})
    with app.test_client() as test_client:
        resp = test_client.post("/pull-data")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "error"


# ── Read-side queries ───────────────────────────────────────────────────────

@pytest.mark.web
def test_run_queries_live_and_cached(seeded_db):
    """run_queries returns live Q-values and reads the worker-maintained cache."""
    before = run_queries(database_url=seeded_db)
    assert before["total_entries"] == 12
    assert before["summary"] == {}                 # cache empty until a recompute
    assert before["summary_updated"] is None
    assert before["q11_gpa_by_status"]             # statuses present in seed
    assert before["q10_top_acceptance_programs"]   # one program with >= 10 rows

    with psycopg.connect(seeded_db) as conn:
        handle_recompute_analytics(conn, {})
    after = run_queries(database_url=seeded_db)
    assert after["summary"]["total_applicants"] == 12.0
    assert after["summary_updated"] is not None


@pytest.mark.web
def test_run_py_builds_app():
    """run.py exposes a ready-to-serve Flask app at module import."""
    import run  # pylint: disable=import-outside-toplevel
    assert run.app.name == "app"
