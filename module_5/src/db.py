"""
db.py — shared database connection helper for module_5.

Supports two credential patterns (checked in priority order):

1. Explicit override — callers (tests, CI) pass ``database_url`` directly.
2. Single URL env var — ``DATABASE_URL=postgresql://user:pass@host/db``
   (legacy path, still works for local dev and CI).
3. Individual env vars — ``DB_USER``, ``DB_PASSWORD``, ``DB_HOST``,
   ``DB_PORT``, ``DB_NAME``.  This is the recommended production pattern
   because it limits blast radius: rotating the password does not require
   changing the host or database name, and each var can be managed
   independently in a secrets manager.
"""

import os

import psycopg

# Fallback host/port used when the individual vars are not fully set.
_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = "5432"
_DEFAULT_DB = "gradcafe"


def _build_url() -> str:
    """Construct a PostgreSQL URL from individual environment variables.

    Reads ``DB_USER``, ``DB_PASSWORD``, ``DB_HOST``, ``DB_PORT``, and
    ``DB_NAME`` from the process environment.  ``DB_HOST`` and ``DB_PORT``
    are optional and default to localhost:5432.

    :raises KeyError: If ``DB_USER`` or ``DB_PASSWORD`` are not set.
    :returns: A psycopg-compatible connection URL string.
    :rtype: str
    """
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ.get("DB_HOST", _DEFAULT_HOST)
    port = os.environ.get("DB_PORT", _DEFAULT_PORT)
    name = os.environ.get("DB_NAME", _DEFAULT_DB)
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_conn(database_url: str | None = None):
    """Return an open psycopg connection.

    Connection URL resolution order:
    1. ``database_url`` parameter (used by tests and CI for isolation).
    2. ``DATABASE_URL`` environment variable (legacy single-URL config).
    3. Individual ``DB_*`` environment variables (recommended for production).

    :param database_url: Direct connection URL override.  When provided,
        environment variables are ignored entirely.
    :type database_url: str or None
    :returns: An open psycopg connection.
    :rtype: psycopg.Connection
    """
    url = database_url or os.environ.get("DATABASE_URL") or _build_url()
    return psycopg.connect(url)
