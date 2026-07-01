"""Tests for the worker: incremental scraper, analytics recompute, consumer."""

import json
import types
from unittest.mock import MagicMock

import psycopg
import pytest

import consumer
from etl import incremental_scraper as scraper
from etl.query_data import handle_recompute_analytics, recompute_summary
from db.load_data import read_watermark


# ── Incremental scraper ─────────────────────────────────────────────────────

@pytest.mark.etl
def test_fetch_new_records_respects_since(temp_data_file):
    """fetch_new_records returns only ids greater than `since`, sorted, capped."""
    everything = scraper.fetch_new_records(0, 100, temp_data_file)
    assert [r["result_id"] for r in everything] == list(range(2000, 2012))
    later = scraper.fetch_new_records(2008, 100, temp_data_file)
    assert [r["result_id"] for r in later] == [2009, 2010, 2011]
    assert len(scraper.fetch_new_records(0, 3, temp_data_file)) == 3


@pytest.mark.etl
def test_handle_scrape_inserts_and_advances(clean_db, temp_data_file):
    """A scrape inserts new rows and advances the watermark to the max id."""
    with psycopg.connect(clean_db) as conn:
        result = scraper.handle_scrape_new_data(conn, {"source_file": temp_data_file, "batch": 4})
    assert result == {"inserted": 4, "since": 0, "watermark": 2003}
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM applicants")
        assert cur.fetchone()[0] == 4
        assert read_watermark(cur) == 2003


@pytest.mark.etl
def test_handle_scrape_since_override_and_exhaustion(clean_db, temp_data_file):
    """A `since` past the data set yields zero inserts and leaves the watermark."""
    with psycopg.connect(clean_db) as conn:
        result = scraper.handle_scrape_new_data(
            conn, {"source_file": temp_data_file, "since": 9999})
    assert result == {"inserted": 0, "since": 9999, "watermark": 9999}


# ── Analytics recompute ─────────────────────────────────────────────────────

@pytest.mark.etl
def test_recompute_summary_populates_cache(seeded_db):
    """recompute_summary computes every metric and upserts it into the cache."""
    with psycopg.connect(seeded_db) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            metrics = recompute_summary(cur)
            cur.execute("SELECT COUNT(*) FROM analysis_summary")
            assert cur.fetchone()[0] == len(metrics) == 5
    assert metrics["total_applicants"] == 12.0
    assert 0.0 <= metrics["pct_accepted"] <= 100.0


@pytest.mark.etl
def test_handle_recompute_accepts_none_payload(seeded_db):
    """The recompute handler tolerates a missing payload."""
    with psycopg.connect(seeded_db) as conn:
        out = handle_recompute_analytics(conn, None)
    assert set(out["metrics"]) == {
        "total_applicants", "distinct_universities", "distinct_programs",
        "avg_gpa", "pct_accepted",
    }


# ── Consumer: topology / dispatch / message handling ────────────────────────

@pytest.mark.consumer
def test_declare_topology_is_durable():
    """declare_topology declares a durable direct exchange, queue, and binding."""
    channel = MagicMock()
    consumer.declare_topology(channel)
    channel.exchange_declare.assert_called_once_with(
        exchange="tasks", exchange_type="direct", durable=True)
    channel.queue_declare.assert_called_once_with(queue="tasks_q", durable=True)
    channel.queue_bind.assert_called_once_with(
        exchange="tasks", queue="tasks_q", routing_key="tasks")


@pytest.mark.consumer
def test_dispatch_routes_and_commits(clean_db, temp_data_file, monkeypatch):
    """dispatch runs the mapped handler in a transaction that commits on success."""
    monkeypatch.setattr(consumer, "get_conn", lambda *a, **k: psycopg.connect(clean_db))
    consumer.dispatch("scrape_new_data", {"source_file": temp_data_file, "batch": 2})
    consumer.dispatch("recompute_analytics", {})
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM applicants")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM analysis_summary")
        assert cur.fetchone()[0] == 5


@pytest.mark.consumer
def test_dispatch_unknown_kind_raises():
    """dispatch rejects an unregistered task kind before opening a connection."""
    with pytest.raises(ValueError):
        consumer.dispatch("nope", {})


@pytest.mark.consumer
def test_on_message_acks_on_success(monkeypatch):
    """A good message is dispatched and acknowledged."""
    monkeypatch.setattr(consumer, "dispatch", lambda kind, payload: {"ok": kind})
    channel = MagicMock()
    method = types.SimpleNamespace(delivery_tag=7)
    body = json.dumps({"kind": "recompute_analytics", "payload": {}}).encode()
    consumer.on_message(channel, method, None, body)
    channel.basic_ack.assert_called_once_with(delivery_tag=7)
    channel.basic_nack.assert_not_called()


@pytest.mark.consumer
def test_on_message_nacks_on_handler_error(monkeypatch):
    """A handler error rolls back and nacks without requeue (no poison loop)."""
    def boom(kind, payload):
        raise RuntimeError("handler failed")
    monkeypatch.setattr(consumer, "dispatch", boom)
    channel = MagicMock()
    method = types.SimpleNamespace(delivery_tag=9)
    body = json.dumps({"kind": "scrape_new_data"}).encode()
    consumer.on_message(channel, method, None, body)
    channel.basic_nack.assert_called_once_with(delivery_tag=9, requeue=False)


@pytest.mark.consumer
def test_on_message_nacks_malformed_json():
    """Malformed JSON is dropped (nack, requeue=false), never crashing the worker."""
    channel = MagicMock()
    method = types.SimpleNamespace(delivery_tag=3)
    consumer.on_message(channel, method, None, b"{not-json")
    channel.basic_nack.assert_called_once_with(delivery_tag=3, requeue=False)


# ── Consumer: start-up load and broker wiring ───────────────────────────────

@pytest.mark.consumer
def test_initial_load_success(clean_db, monkeypatch):
    """initial_load seeds data then recomputes the cache."""
    monkeypatch.setattr(consumer, "load_data", lambda: 10)
    monkeypatch.setattr(consumer, "get_conn", lambda *a, **k: psycopg.connect(clean_db))
    consumer.initial_load()
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM analysis_summary")
        assert cur.fetchone()[0] == 5


@pytest.mark.consumer
def test_initial_load_swallows_errors(monkeypatch):
    """initial_load never crashes the worker if seeding fails."""
    def boom():
        raise FileNotFoundError("no data file")
    monkeypatch.setattr(consumer, "load_data", boom)
    consumer.initial_load()   # must not raise


@pytest.mark.consumer
def test_build_channel(monkeypatch):
    """build_channel opens a blocking connection and returns its channel."""
    fake_conn = MagicMock()
    monkeypatch.setattr(consumer.pika, "BlockingConnection", lambda params: fake_conn)
    conn, channel = consumer.build_channel("amqp://guest:guest@localhost:5672/")
    assert conn is fake_conn
    assert channel is fake_conn.channel.return_value


@pytest.mark.consumer
def test_main_wires_consumer(monkeypatch):
    """main seeds, declares topology, sets prefetch=1, and starts consuming."""
    monkeypatch.setattr(consumer, "initial_load", lambda: None)
    channel = MagicMock()
    monkeypatch.setattr(consumer, "build_channel", lambda url: (MagicMock(), channel))
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    consumer.main()
    channel.basic_qos.assert_called_once_with(prefetch_count=1)
    channel.basic_consume.assert_called_once()
    channel.start_consuming.assert_called_once()
