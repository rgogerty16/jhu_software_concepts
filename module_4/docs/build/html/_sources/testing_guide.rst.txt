Testing Guide
=============

The test suite lives in ``module_4/tests/`` and uses **pytest** with the
**pytest-cov** plugin. All tests must be marked with at least one of the
five registered markers; 100% branch coverage of ``module_4/src/`` is enforced.

Running Tests
-------------

Full suite (requires ``DATABASE_URL`` pointing at a test database)::

   cd module_4
   DATABASE_URL=postgresql://localhost/gradcafe_test pytest tests/

By marker::

   pytest -m web                               # page rendering only
   pytest -m "buttons or analysis"             # endpoints + formatting
   pytest -m "web or buttons or analysis or db or integration"  # everything

With verbose output::

   pytest tests/ -v

Markers
-------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - What it covers
   * - ``@pytest.mark.web``
     - Flask app factory, route registration, page rendering (``test_flask_page.py``).
       Also used for ``_default_scraper`` subprocess test.
   * - ``@pytest.mark.buttons``
     - ``POST /pull-data`` and ``POST /update-analysis`` endpoint behaviour,
       busy-state gating, ``GET /pull-status`` (``test_buttons.py``).
   * - ``@pytest.mark.analysis``
     - ``Answer:`` label presence, two-decimal percentage formatting,
       ``print_results()`` stdout (``test_analysis_format.py``, ``test_etl_and_coverage.py``).
   * - ``@pytest.mark.db``
     - Schema, inserts, idempotency, ``run_queries()`` key contract,
       ``_parse_date``, ``_to_row``, ``load_data``, ``pull_and_load``
       (``test_db_insert.py``, ``test_etl_and_coverage.py``).
   * - ``@pytest.mark.integration``
     - End-to-end pull→update→render flows, multi-pull idempotency, busy-gate
       lifecycle (``test_integration_end_to_end.py``).

Test Fixtures (``conftest.py``)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Fixture
     - Scope
     - Description
   * - ``db_url``
     - session
     - Returns the ``DATABASE_URL`` string. Derived once per pytest session.
   * - ``clean_db``
     - function
     - Creates a fresh ``applicants`` table before each test; drops it after.
       Uses ``yield`` so setup and teardown live in one function.
   * - ``fake_scraper``
     - function
     - Returns a callable ``(database_url: str) -> None`` that inserts one
       known ``SAMPLE_ROW`` instantly. Silently ignores ``UndefinedTable``
       errors caused by teardown racing a background thread.
   * - ``app_client``
     - function
     - Creates a Flask test client wired to ``clean_db`` and ``fake_scraper``.
       Injects ``TESTING=True``. This is what most tests use.

Stable UI Selectors
-------------------

Tests locate buttons by ``data-testid`` attribute, not by CSS class or text:

.. code-block:: python

   btn = soup.find(attrs={"data-testid": "pull-data-btn"})
   btn = soup.find(attrs={"data-testid": "update-analysis-btn"})

These attributes are set in ``templates/base.html`` and are guaranteed stable
across styling or copy changes.

Test Doubles
------------

``fake_scraper``
    Inserts ``SAMPLE_ROW`` (one known applicant record) into the test DB.
    Replaces ``app.config["SCRAPER"]`` so no Grad Café network calls occur.

``fake_scrape`` / ``fake_save`` / ``fake_load`` (inline in ETL tests)
    Injected directly into ``pull_and_load()`` via its function parameters.
    ``fake_save`` writes real JSON so the load step can open the file.

``unittest.mock.patch("app.subprocess.Popen", ...)``
    Used in ``test_default_scraper_calls_subprocess`` to verify the subprocess
    command, environment injection, and ``.wait()`` call without spawning anything.

Coverage
--------

pytest-cov enforces 100% coverage of ``src/``::

   pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=100

The only excluded lines are ``if __name__ == "__main__":`` entry points
(marked ``# pragma: no cover``) and the module_2 scraper import block in
``pull_and_load.py`` (requires selenium, excluded from CI).

Troubleshooting
---------------

**"connection refused" on test startup**
    PostgreSQL is not running. Start it with ``brew services start postgresql@17``
    and create the test DB: ``psql -c "CREATE DATABASE gradcafe_test;" postgres``.

**"UndefinedTable: relation applicants does not exist"**
    A test is running without the ``clean_db`` fixture (which creates the table).
    Make sure any DB-touching test requests ``clean_db`` or ``app_client``
    (which depends on ``clean_db``) as a parameter.

**PytestUnhandledThreadExceptionWarning**
    Benign in busy-state tests where teardown drops the table before a
    background thread finishes. The ``fake_scraper`` silently catches this.
    All tests still pass.

**Coverage below 100%**
    Run ``pytest --cov=src --cov-report=term-missing`` and look at the
    "Missing" column. Add a test that exercises those lines, or add
    ``# pragma: no cover`` if the line is a CLI entry point.
