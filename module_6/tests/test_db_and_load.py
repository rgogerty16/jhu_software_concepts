"""Tests for the shared db package: connection helper, schema, loader, watermark."""

import json

import psycopg
import pytest

from db import db as dbmod
from db.load_data import (
    advance_watermark,
    insert_rows,
    load_data,
    parse_result_id,
    read_watermark,
    to_row,
    _parse_date,
    _resolve_limit,
)

pytestmark = pytest.mark.db


def test_build_url_defaults(monkeypatch):
    """build_url uses DB_* with sensible host/port/name defaults."""
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    assert dbmod.build_url() == "postgresql://u:p@localhost:5432/gradcafe"


def test_build_url_explicit(monkeypatch):
    """build_url honours every DB_* override."""
    for key, val in {
        "DB_USER": "a", "DB_PASSWORD": "b", "DB_HOST": "h",
        "DB_PORT": "6000", "DB_NAME": "n",
    }.items():
        monkeypatch.setenv(key, val)
    assert dbmod.build_url() == "postgresql://a:b@h:6000/n"


def test_resolve_url_precedence(monkeypatch):
    """resolve_url prefers the explicit override, then DATABASE_URL, then DB_*."""
    assert dbmod.resolve_url("postgresql://override") == "postgresql://override"
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")
    assert dbmod.resolve_url() == "postgresql://from-env"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    assert dbmod.resolve_url().startswith("postgresql://u:p@")


def test_get_conn_connects(clean_db):
    """get_conn opens a usable connection against the resolved URL."""
    with dbmod.get_conn(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1


def test_parse_result_id():
    """parse_result_id extracts the trailing integer, else None."""
    assert parse_result_id("https://www.thegradcafe.com/result/1020294") == 1020294
    assert parse_result_id(None) is None
    assert parse_result_id("no-digits-here") is None


def test_parse_date_variants():
    """_parse_date handles both month formats, None, and unparseable input."""
    assert _parse_date(None) is None
    assert _parse_date("Jun 02, 2026").year == 2026        # %b path
    assert _parse_date("June 02, 2026").month == 6         # %B path (after continue)
    assert _parse_date("not a date") is None


def test_to_row_missing_fields():
    """to_row maps empty/missing fields to None and derives term."""
    row = to_row({"url": None, "semester": "", "year": ""})
    assert row["result_id"] is None
    assert row["term"] is None
    assert row["program"] is None


def test_resolve_limit(monkeypatch):
    """_resolve_limit: explicit wins; else INITIAL_LOAD_LIMIT; else None."""
    assert _resolve_limit(7) == 7
    monkeypatch.setenv("INITIAL_LOAD_LIMIT", "42")
    assert _resolve_limit(None) == 42
    monkeypatch.delenv("INITIAL_LOAD_LIMIT", raising=False)
    assert _resolve_limit(None) is None


def test_insert_rows_empty_is_noop(clean_db):
    """insert_rows with no rows does nothing (guard) and never errors."""
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            insert_rows(cur, [])
            cur.execute("SELECT COUNT(*) FROM applicants")
            assert cur.fetchone()[0] == 0


def test_watermark_read_write(clean_db):
    """read_watermark defaults to 0; advance_watermark moves forward only."""
    with psycopg.connect(clean_db) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            assert read_watermark(cur) == 0
            advance_watermark(cur, 500)
            assert read_watermark(cur) == 500
            advance_watermark(cur, 100)          # never regresses
            assert read_watermark(cur) == 500


def test_load_data_with_limit_then_full(temp_data_file, clean_db, monkeypatch):
    """Limited base load seeds the watermark; a later full load is idempotent."""
    monkeypatch.delenv("INITIAL_LOAD_LIMIT", raising=False)
    total = load_data(filepath=temp_data_file, database_url=clean_db, limit=5)
    assert total == 5
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            assert read_watermark(cur) == 2004      # 5th-lowest id (2000..2004)

    total = load_data(filepath=temp_data_file, database_url=clean_db, limit=None)
    assert total == 12                              # remaining 7 added, no dupes


def test_load_data_empty_file(tmp_path, clean_db):
    """Loading an empty data set is a no-op that seeds watermark 0."""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([]), encoding="utf-8")
    assert load_data(filepath=str(empty), database_url=clean_db) == 0
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            assert read_watermark(cur) == 0
