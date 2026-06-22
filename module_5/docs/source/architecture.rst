Architecture
============

The application follows a classic **three-tier** pattern.

.. code-block:: text

   ┌─────────────────────────────────────────────┐
   │  Presentation tier  (Flask + Jinja2 + CSS)  │
   │  templates/base.html · templates/index.html  │
   └───────────────┬─────────────────────────────┘
                   │ HTTP
   ┌───────────────▼─────────────────────────────┐
   │  Application tier  (Python)                  │
   │  app.py · query_data.py · load_data.py       │
   │  pull_and_load.py                            │
   └───────────────┬─────────────────────────────┘
                   │ psycopg / SQL
   ┌───────────────▼─────────────────────────────┐
   │  Data tier  (PostgreSQL)                     │
   │  gradcafe database · applicants table        │
   └─────────────────────────────────────────────┘

Web Layer (``app.py``)
----------------------

``create_app(config)`` is the **application factory**. It accepts an optional
``config`` dict so tests can inject a test database URL and a fake scraper without
touching module-level globals.

Routes:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Path
     - Method
     - Description
   * - ``/analysis``
     - GET
     - Runs all SQL queries and renders ``index.html`` with results.
   * - ``/``
     - GET
     - Redirects to ``/analysis`` for backwards compatibility.
   * - ``/pull-data``
     - POST
     - Starts the scraper in a background thread. Returns ``{"ok": true}`` / 202
       or ``{"busy": true}`` / 409 if a scrape is already running.
   * - ``/pull-status``
     - GET
     - Returns ``{"running": bool, "started_at": str|null}`` — polled by JS.
   * - ``/update-analysis``
     - POST
     - Redirects to ``/analysis`` if idle; returns ``{"busy": true}`` / 409
       during a pull.

ETL Layer
---------

``load_data.py``
    Reads the Module 2 JSON export and populates the ``applicants`` table.
    Idempotent: ``ON CONFLICT (url) DO NOTHING`` prevents duplicate rows.

``pull_and_load.py``
    Orchestrates a live scrape: calls ``scrape_fn`` → ``save_fn`` →
    ``load_fn``. All three are injectable so tests substitute fast fakes.

``query_data.py``
    Runs 11 SQL queries (9 required + 2 original) and returns a ``dict``
    consumed by the Flask template. Also exposes ``print_results()`` for
    CLI usage.

Data Layer
----------

Single table: **applicants**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Column
     - Type
     - Description
   * - ``p_id``
     - SERIAL PK
     - Auto-generated surrogate key.
   * - ``url``
     - TEXT UNIQUE
     - Natural unique key — the canonical Grad Café post URL.
   * - ``program``
     - TEXT
     - Raw scraped program name (university + department).
   * - ``status``
     - TEXT
     - Accepted / Rejected / Waitlisted / Other.
   * - ``term``
     - TEXT
     - Constructed as ``"{semester} {year}"`` (e.g. "Fall 2026").
   * - ``llm_generated_university``
     - TEXT
     - Normalised university name produced by the Module 2 LLM step.
   * - ``llm_generated_program``
     - TEXT
     - Normalised program name from the same LLM step.
   * - *(+ 8 more)*
     - —
     - ``gpa``, ``gre``, ``gre_v``, ``gre_aw``, ``degree``, ``comments``,
       ``date_added``, ``us_or_international``.

Dependency Injection Pattern
----------------------------

To keep tests fast and deterministic, the application uses **dependency
injection** rather than hard-coded imports for anything that touches the
network or a real database:

* ``app.config["DATABASE_URL"]`` — overridden in tests to point at
  ``gradcafe_test`` instead of the production ``gradcafe`` DB.
* ``app.config["SCRAPER"]`` — replaced in tests with a callable that
  inserts one known row instantly, never hitting Grad Café.
* ``pull_and_load(scrape_fn, save_fn, load_fn)`` — all three functions
  are parameters; tests inject fakes for all of them.
