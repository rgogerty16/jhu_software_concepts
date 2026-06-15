# Module 4: Testing & Documentation — Grad Café Analytics

**Name:** Ryan Gogerty  
**JHED ID:** rgogerty  
**Course:** Modern Concepts in Python  
**Assignment:** Module 4 — pytest, pytest-cov, Sphinx, Read the Docs

---

## Overview

This module adds a complete pytest test suite (41 tests, 100% branch coverage) and
Sphinx developer documentation to the Grad Café Analytics Flask application built
in Modules 2–3. CI runs on every push via GitHub Actions.

---

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 17 (`brew install postgresql@17`)
- Chrome (for the Selenium scraper; not needed to run tests)

### 1. Start PostgreSQL and create databases

```bash
brew services start postgresql@17
psql -c "CREATE DATABASE gradcafe;" postgres
psql -c "CREATE DATABASE gradcafe_test;" postgres
```

### 2. Install dependencies

```bash
pip install -r module_4/requirements.txt
```

### 3. Load Module 2 data

```bash
cd module_4/src
python load_data.py
```

### 4. Run the Flask app

```bash
python app.py
```

Visit `http://127.0.0.1:5000`. The app redirects `/` → `/analysis`.

---

## Running Tests

```bash
cd module_4
DATABASE_URL=postgresql://localhost/gradcafe_test pytest tests/
```

This runs all 41 tests with 100% coverage enforcement. By marker:

```bash
pytest -m web           # Flask app factory + page rendering
pytest -m buttons       # /pull-data, /update-analysis, /pull-status endpoints
pytest -m analysis      # Answer: labels, two-decimal percentages
pytest -m db            # Schema, inserts, idempotency, SQL query contract
pytest -m integration   # End-to-end pull → update → render flows
```

> CI output: `175 stmts, 0 missed, 100%`

---

## Documentation

Full developer docs are published on Read the Docs:

**https://jhu-software-concepts.readthedocs.io/en/latest/**

Covers setup, three-tier architecture, API reference (auto-generated from docstrings),
testing guide, and operational notes.

To build the docs locally:

```bash
cd module_4/docs
make html
open build/html/index.html
```

---

## Project Structure

```
module_4/
├── src/
│   ├── app.py              # Flask app factory (create_app), routes, _default_scraper
│   ├── db.py               # psycopg connection helper
│   ├── query_data.py       # SQL analysis queries → dict
│   ├── load_data.py        # JSON → PostgreSQL loader (idempotent)
│   ├── pull_and_load.py    # ETL orchestrator (injectable scrape/save/load fns)
│   └── templates/
│       ├── base.html       # Buttons (data-testid), JS polling
│       └── index.html      # Analysis results with Answer: labels
├── tests/
│   ├── conftest.py                    # Fixtures: db_url, clean_db, fake_scraper, app_client
│   ├── test_flask_page.py             # 12 web tests
│   ├── test_buttons.py                # 6 button/endpoint tests
│   ├── test_analysis_format.py        # 4 formatting tests
│   ├── test_db_insert.py              # 5 database tests
│   ├── test_integration_end_to_end.py # 4 integration tests
│   └── test_etl_and_coverage.py       # 10 ETL + coverage tests
├── docs/
│   └── source/
│       ├── conf.py              # Sphinx config (autodoc, napoleon, rtd theme)
│       ├── index.rst            # Table of contents
│       ├── overview.rst         # Setup and prerequisites
│       ├── architecture.rst     # Three-tier diagram, routes, schema
│       ├── api_reference.rst    # automodule directives for all 5 src/ modules
│       ├── testing_guide.rst    # Markers, fixtures, test doubles, troubleshooting
│       └── operational_notes.rst # Busy-state policy, idempotency, ops runbook
├── pytest.ini
├── requirements.txt
├── coverage_summary.txt
└── README.md
```

---

## Key Design Decisions

### create_app() factory pattern
The Flask app is wrapped in a function rather than created at module level. This
lets tests inject a throwaway database URL and a fake scraper without ever touching
the network or the production database.

### Dependency injection for ETL
`pull_and_load(scrape_fn, save_fn, load_fn)` accepts all three steps as parameters.
Tests substitute instant in-process fakes, so the entire ETL pipeline is tested
without Selenium or network access.

### 100% coverage strategy
Every `if __name__ == "__main__":` block is marked `# pragma: no cover` (CLI entry
points, not business logic). The module_2 Selenium import block is also excluded
(requires Chrome). All other logic is covered by the test suite.

### data-testid selectors
HTML buttons carry `data-testid` attributes so tests locate them by stable
identifier, not by CSS class or display text that might change.

---

## GitHub Actions (CI)

Workflow: `.github/workflows/tests.yml`  
Trigger: push / pull request to `main`  
Service: `postgres:17` with `--health-cmd pg_isready`  
Command: `pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=100`
