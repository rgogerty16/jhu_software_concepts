# Module 5 — Software Assurance & Secure SQL

**Name:** Ryan Gogerty  
**JHED ID:** rgogerty  
**Course:** Modern Concepts in Python  
**Assignment:** Module 5 — Pylint, SQL Injection Defenses, pydeps, Snyk, uv

---

## What Changed from Module 4

| Topic | Module 4 | Module 5 |
|-------|----------|----------|
| Pylint | not enforced | 10/10 required |
| SQL queries | raw string building | `psycopg sql.SQL` composition |
| DB credentials | single `DATABASE_URL` | individual `DB_*` env vars |
| LIMIT | only on Q10 | every query |
| Packaging | none | `setup.py` + editable install |
| Supply-chain scan | none | Snyk |
| CI | pytest only | pylint + pydeps + Snyk + pytest |

---

## Fresh Install

### pip + venv (standard)

```bash
cd module_5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # editable install — makes src/ importable anywhere
```

### uv (faster, lock-file reproducible)

```bash
cd module_5
pip install uv            # install uv if not already present
uv venv                   # creates .venv using uv's faster resolver
source .venv/bin/activate
uv pip sync requirements.txt   # installs exactly the pinned versions
uv pip install -e .            # editable install
```

`uv pip sync` is stricter than `pip install` — it removes packages that are
NOT in requirements.txt, giving you a perfectly clean, reproducible environment
every time.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values.  **Never commit `.env`.**

```bash
# Option A — individual vars (recommended for production)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gradcafe
DB_USER=gradcafe_app
DB_PASSWORD=changeme

# Option B — single URL (takes precedence if set)
DATABASE_URL=postgresql://gradcafe_app:changeme@localhost:5432/gradcafe
```

### Least-Privilege Database User

The app is read-only after initial data load.  Create a restricted user:

```sql
CREATE USER gradcafe_app WITH PASSWORD 'changeme';
GRANT CONNECT ON DATABASE gradcafe TO gradcafe_app;
GRANT SELECT ON TABLE applicants TO gradcafe_app;
-- No INSERT, UPDATE, DELETE, DROP, or ALTER privileges
```

---

## Running the App

```bash
cd module_5/src
DATABASE_URL=postgresql://localhost/gradcafe python app.py
```

Visit `http://127.0.0.1:5000`.

---

## Running Tests

Create the test database once:

```bash
psql -c "CREATE DATABASE gradcafe_test;" postgres
```

Run the full suite:

```bash
cd module_5
DATABASE_URL=postgresql://localhost/gradcafe_test pytest tests/
```

By marker:

```bash
pytest -m web
pytest -m "buttons or analysis"
pytest -m db
pytest -m integration
```

---

## Pylint

Run from `module_5/` (the `.pylintrc` is here):

```bash
pylint src/ --rcfile=.pylintrc --fail-under=10
```

Expected output: `Your code has been rated at 10.00/10`

No messages are silenced — every point is earned.

---

## Dependency Graph

Generate the SVG (requires Graphviz `dot` on PATH):

```bash
pip install pydeps
pydeps src/app.py --noshow -T svg -o dependency.svg
```

The committed `dependency.svg` shows the full module dependency chain.

---

## Snyk Security Scan

```bash
npm install -g snyk
snyk auth
cd module_5
snyk test
```

See `snyk-analysis.png` for the screenshot of the scan results.

---

## GitHub Actions CI

Workflow: `.github/workflows/ci.yml`  
Triggers on every push or PR that touches `module_5/`.

Four jobs run in parallel:
1. **lint** — `pylint src/ --fail-under=10`
2. **dependency-graph** — regenerates `dependency.svg`, fails if missing
3. **snyk** — `snyk test` (advisory findings reported, does not block merge)
4. **test** — `pytest --cov=src --cov-fail-under=100`

---

## Project Structure

```
module_5/
├── src/
│   ├── app.py              # Flask app factory, routes, _default_scraper
│   ├── db.py               # psycopg connection helper (individual env vars)
│   ├── query_data.py       # 11 SQL queries via psycopg sql.SQL composition
│   ├── load_data.py        # JSON → PostgreSQL loader (idempotent)
│   ├── pull_and_load.py    # ETL orchestrator (injectable scrape/save/load)
│   └── templates/
├── tests/
├── .github/workflows/ci.yml
├── dependency.svg
├── snyk-analysis.png
├── setup.py
├── requirements.txt
├── .pylintrc
├── .env.example
├── pytest.ini
├── module_5_report.pdf
└── README.md
```
