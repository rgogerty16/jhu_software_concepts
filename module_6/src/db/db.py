"""db.py — shared PostgreSQL connection helper for the web and worker services.

Credential resolution order (first match wins):

1. Explicit override — callers pass ``database_url`` directly (tests, CI).
2. ``DATABASE_URL`` env var — the single-URL form used by Docker Compose and
   CI (e.g. ``postgresql://gradcafe:gradcafe@db:5432/gradcafe``).
3. Individual ``DB_*`` env vars — ``DB_USER``, ``DB_PASSWORD``, ``DB_HOST``,
   ``DB_PORT``, ``DB_NAME`` — the recommended production pattern because each
   value can be rotated independently in a secrets manager.

Both services import this one module, so connection handling lives in exactly
one place across the whole stack.
"""

import os

import psycopg

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = "5432"
_DEFAULT_DB = "gradcafe"


def build_url() -> str:
    """Assemble a PostgreSQL URL from the individual ``DB_*`` env vars.

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


def resolve_url(database_url: str | None = None) -> str:
    """Return the connection URL using the documented resolution order.

    :param database_url: Direct override; when provided it is returned as-is.
    :type database_url: str or None
    :returns: The resolved PostgreSQL connection URL.
    :rtype: str
    """
    return database_url or os.environ.get("DATABASE_URL") or build_url()


def get_conn(database_url: str | None = None) -> psycopg.Connection:
    """Open and return a psycopg connection using :func:`resolve_url`.

    :param database_url: Optional direct connection URL override.
    :type database_url: str or None
    :returns: An open psycopg connection (usable as a context manager).
    :rtype: psycopg.Connection
    """
    return psycopg.connect(resolve_url(database_url))
