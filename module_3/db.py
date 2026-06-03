"""
db.py — shared database connection helper for module_3.

All scripts import get_conn() from here so connection parameters live
in one place. Adjust DB_* constants if your local PostgreSQL setup differs.
"""

import psycopg

DB_NAME = "gradcafe"
DB_USER = None          # None → psycopg uses the OS username (default for Homebrew)
DB_HOST = "localhost"
DB_PORT = 5432


def get_conn():
    """Return an open psycopg connection to the gradcafe database."""
    kwargs = dict(dbname=DB_NAME, host=DB_HOST, port=DB_PORT)
    if DB_USER:
        kwargs["user"] = DB_USER
    return psycopg.connect(**kwargs)
