Operational Notes
=================

This page covers runtime behaviour, data management, and common maintenance tasks
for the Grad Café Analytics service.

Busy-State Policy
-----------------

Only one scrape can run at a time. The Flask app uses a threading lock plus two
boolean flags (``_scrape_running``, ``_scrape_started_at``) to enforce this:

* ``POST /pull-data`` while idle → starts the scraper thread, returns ``{"ok": true}`` / 202.
* ``POST /pull-data`` while busy → immediately returns ``{"busy": true}`` / 409.
* ``POST /update-analysis`` while busy → immediately returns ``{"busy": true}`` / 409.
* ``GET /pull-status`` → always returns ``{"running": bool, "started_at": str|null}``.

The browser polls ``/pull-status`` every two seconds to update the progress spinner.
The "Update Analysis" button is disabled while a scrape is running.

Idempotency
-----------

``load_data.py`` inserts rows with::

   INSERT INTO applicants (...) VALUES (...) ON CONFLICT (url) DO NOTHING

The ``url`` column is the natural unique key (the canonical Grad Café post URL).
Running a pull twice or loading the same JSON file twice is safe — duplicate rows
are silently skipped. The row count returned by ``load_data()`` reflects only
newly inserted rows.

This means:

* Re-running ``python load_data.py`` on the same data file is always safe.
* Clicking "Pull Data" multiple times (if the first one finishes before the second
  fires) appends only genuinely new records.

Uniqueness Keys
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Table
     - Unique constraint
     - Reason
   * - ``applicants``
     - ``url`` (TEXT UNIQUE)
     - Each post on The Grad Café has a stable, unique URL that serves as the
       natural surrogate for a single applicant report.

Troubleshooting
---------------

**The spinner never stops / "Pull Data" button stays busy**
    The scraper subprocess may have crashed. Check the Flask server stdout for
    tracebacks. Restart the Flask app; ``_scrape_running`` resets to ``False``
    on startup because it is an in-process flag (not persisted to DB).

**Analysis page shows zeros for all percentages**
    No rows in the ``applicants`` table. Run ``load_data.py`` to seed from the
    Module 2 JSON file, or click "Pull Data" → wait for the pull to finish →
    click "Update Analysis".

**``psycopg.OperationalError: connection refused``**
    PostgreSQL is not running. On macOS::

       brew services start postgresql@17

    In GitHub Actions: the postgres service container must be healthy before
    pytest starts (the ``--health-cmd pg_isready`` health check handles this).

**``relation "applicants" does not exist``**
    The table has not been created yet. The schema is initialised lazily by
    ``load_data.py`` (it runs ``CREATE TABLE IF NOT EXISTS`` on first import).
    Run ``python load_data.py`` once to create the table.

Database Backup and Restore
---------------------------

Dump the production database::

   pg_dump gradcafe > gradcafe_backup.sql

Restore::

   psql gradcafe < gradcafe_backup.sql

To reset and reload from the Module 2 JSON export::

   psql -c "DROP TABLE IF EXISTS applicants;" gradcafe
   cd module_4/src && python load_data.py

Continuous Integration
----------------------

GitHub Actions runs the full test suite on every push and pull request to
``main``. The workflow (``module_4/tests.yml``) spins up a ``postgres:17``
service container, sets ``DATABASE_URL`` to the container's test database, and
enforces ``--cov-fail-under=100``.

A green badge on the ``main`` branch guarantees:

* All 41 tests pass.
* Every line in ``module_4/src/`` is exercised by at least one test.
* No network calls reach The Grad Café during CI.

Adding New Queries
------------------

1. Add the SQL and result key in ``query_data.py::run_queries()``.
2. Reference the new key in ``templates/index.html`` with an ``Answer:`` label
   and ``data-testid`` attribute.
3. Add at least one test in ``test_db_insert.py`` (for the query) and
   ``test_analysis_format.py`` (for the rendered format) to maintain 100% coverage.
4. Document the new query in ``architecture.rst`` under the ETL layer section.
