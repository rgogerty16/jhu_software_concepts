"""db.py: shared database connection helper for the Module 13 website.

Carried over from Module 5, with one behavioural change that matters here.

Module 5's ``_build_url()`` read ``DB_USER`` and ``DB_PASSWORD`` with
``os.environ[...]``, so it raised ``KeyError`` when they were unset, and
``create_app()`` called it eagerly, meaning an unconfigured environment took the
entire site down at startup. In this module the analysis page is no longer the
only page: the "Will You Get In?" predictor does not touch Postgres at all, and
it must keep working on a machine that has no database configured. So URL
resolution never raises. It falls back to a local socket connection to the
``gradcafe`` database, which is the convention Module 3 established, and any
connection failure is handled at query time by the route instead.

Resolution order:

1. An explicit ``database_url`` argument, used by the app config and tests.
2. ``DATABASE_URL`` in the environment.
3. Individual ``DB_USER`` / ``DB_PASSWORD`` / ``DB_HOST`` / ``DB_PORT`` /
   ``DB_NAME`` variables, when a user is set.
4. ``postgresql:///gradcafe``, a local peer-authenticated socket connection.
"""

import os

import psycopg

# Fallback host/port/name used when the individual vars are not fully set.
_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = "5432"
_DEFAULT_DB = "gradcafe"


def build_url() -> str:
    """Assemble a PostgreSQL URL from the individual environment variables.

    Unlike the Module 5 version this never raises: with no ``DB_USER`` set it
    returns a local socket URL rather than a ``KeyError``.

    :returns: A psycopg-compatible connection URL string.
    :rtype: str
    """
    name = os.environ.get("DB_NAME", _DEFAULT_DB)
    user = os.environ.get("DB_USER")
    if not user:
        # An empty host means "local socket", which is how a default Homebrew
        # Postgres install authenticates the current OS user.
        return f"postgresql:///{name}"
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", _DEFAULT_HOST)
    port = os.environ.get("DB_PORT", _DEFAULT_PORT)
    credentials = f"{user}:{password}" if password else user
    return f"postgresql://{credentials}@{host}:{port}/{name}"


def resolve_url(database_url: str | None = None) -> str:
    """Resolve the connection URL to use, without touching the network.

    :param database_url: Explicit override. When provided, everything else is
        ignored.
    :type database_url: str or None
    :returns: The connection URL.
    :rtype: str
    """
    return database_url or os.environ.get("DATABASE_URL") or build_url()


def get_conn(database_url: str | None = None):
    """Return an open psycopg connection.

    :param database_url: Direct connection URL override. When provided,
        environment variables are ignored entirely.
    :type database_url: str or None
    :returns: An open psycopg connection.
    :rtype: psycopg.Connection
    :raises psycopg.Error: If the database cannot be reached. Callers that render
        a page are expected to catch this and degrade gracefully.
    """
    return psycopg.connect(resolve_url(database_url))
