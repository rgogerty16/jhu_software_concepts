"""
db.py — shared database connection helper for module_4.

Reads DATABASE_URL from the environment so that tests and CI can point at
a throwaway database without touching this file.

Expected format: postgresql://user:password@host:port/dbname
Homebrew default (no password, OS user): postgresql://localhost/gradcafe
"""

import os

import psycopg

# Fall back to the same local DB used in module_3 when no env var is set.
_DEFAULT_URL = "postgresql://localhost/gradcafe"


def get_conn(database_url: str | None = None):
    """Return an open psycopg connection.

    :param database_url: Override the DATABASE_URL env var (used by tests).
    :type database_url: str or None
    :returns: An open psycopg connection.
    :rtype: psycopg.Connection
    """
    url = database_url or os.environ.get("DATABASE_URL", _DEFAULT_URL)
    return psycopg.connect(url)
