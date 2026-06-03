# Module 3: Relational Databases & SQL Analysis — Grad Café

**Name:** Ryan Gogerty  
**JHED ID:** rgogerty  
**Course:** Modern Concepts in Python  
**Assignment:** Module 3 — SQL, PostgreSQL, Flask  
**Due:** TBD

---

## Overview

This module loads the 30,000-entry Grad Café dataset from Module 2 into a local PostgreSQL database, queries it with SQL to answer analytical questions about graduate school applicants, and displays the results on a dynamic Flask webpage with live data-refresh capabilities.

---

## Prerequisites

- Python 3.10+
- PostgreSQL 17 (installed via Homebrew: `brew install postgresql@17`)
- Chrome (for the scraper's Selenium driver)

---

## Setup

### 1. Start PostgreSQL

```bash
brew services start postgresql@17
```

### 2. Create the database

```bash
psql -c "CREATE DATABASE gradcafe;" postgres
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

The scraper dependencies (selenium, beautifulsoup4, webdriver-manager) are needed for the "Pull Data" button. If you only want the database and analysis without scraping, `flask` and `psycopg[binary]` are sufficient.

---

## Running the Project

### Load Module 2 data into PostgreSQL

```bash
python load_data.py
```

Reads `../module_2/llm_extend_applicant_data.json` and populates the `applicants` table. Safe to re-run — duplicate URLs are silently skipped (`ON CONFLICT DO NOTHING`).

### Run SQL analysis (console output)

```bash
python query_data.py
```

Prints answers to all 11 analysis questions to stdout.

### Launch the Flask web app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in a browser.

---

## File Structure

```
module_3/
├── db.py               # Shared psycopg connection helper
├── load_data.py        # Create table + load module_2 JSON into PostgreSQL
├── query_data.py       # All SQL analysis queries
├── app.py              # Flask application
├── pull_and_load.py    # Background scrape + DB insert (called by Pull Data button)
├── templates/
│   ├── base.html       # Header, buttons, JS polling logic
│   └── index.html      # Query results display
├── static/
│   └── style.css       # Page styling
├── screenshots/
│   ├── webpage.png     # Screenshot of running Flask page
│   └── console_output.png  # Screenshot of query_data.py output
├── limitations.pdf     # Written reflection on self-submitted data limitations
├── requirements.txt
└── README.md
```

---

## Database Schema

Single table: `applicants`

| Column | Type | Source (Module 2 field) |
|---|---|---|
| `p_id` | SERIAL PRIMARY KEY | auto-generated |
| `program` | TEXT | `program` |
| `comments` | TEXT | `comments` |
| `date_added` | DATE | `date_added` (parsed from "Jun 02, 2026") |
| `url` | TEXT UNIQUE | `url` |
| `status` | TEXT | `status` |
| `term` | TEXT | `semester` + `year` → "Fall 2026" |
| `us_or_international` | TEXT | `student_type` |
| `gpa` | FLOAT | `gpa` |
| `gre` | FLOAT | `gre` |
| `gre_v` | FLOAT | `gre_v` |
| `gre_aw` | FLOAT | `gre_aw` |
| `degree` | TEXT | `degree` |
| `llm_generated_program` | TEXT | `llm-generated-program` |
| `llm_generated_university` | TEXT | `llm-generated-university` |

---

## Flask Webpage

The single-page Flask app (`/`) renders all query results dynamically from PostgreSQL on each request.

### Pull Data button
Starts a background subprocess that runs `pull_and_load.py`, which scrapes ~1,000 new entries from Grad Café and inserts them into the database (duplicates skipped). A status bar appears while the scrape runs. The button is disabled during an active scrape.

### Update Analysis button
Re-runs all SQL queries and refreshes the page with the latest data. If a scrape is currently running, the button displays a warning message instead of refreshing, preventing stale partial data from being shown.

Both buttons use JavaScript to poll `/pull-status` (a lightweight JSON endpoint) to keep the UI in sync with the background subprocess state.

### Architecture: Three-Tier
- **Presentation tier:** HTML/CSS/JS templates rendered by Flask
- **Application tier:** Flask + Python query logic (`query_data.py`, `app.py`)
- **Data tier:** PostgreSQL (`gradcafe` database, `applicants` table)

---

## SQL Query Summary

| # | Question | Key SQL clauses |
|---|---|---|
| Q1 | Fall 2026 entry count | `WHERE term = 'Fall 2026'` |
| Q2 | % International | `SUM(CASE WHEN ... = 'International') / COUNT(*)` |
| Q3 | Avg GPA / GRE / GRE V / GRE AW | `AVG` with valid-range filters |
| Q4 | Avg GPA, American + Fall 2026 | `WHERE us_or_international = 'American' AND term = 'Fall 2026'` |
| Q5 | % Accepted, Fall 2026 | `SUM(CASE WHEN status = 'Accepted') / COUNT(*)` |
| Q6 | Avg GPA of Fall 2026 acceptances | `WHERE status = 'Accepted' AND term = 'Fall 2026'` |
| Q7 | JHU Master's CS entries | `LIKE '%johns hopkins%'` on `llm_generated_university` |
| Q8 | Top-4 PhD CS acceptances, raw fields | `LIKE` on `program` — returns 0 (no university in raw field) |
| Q9 | Same with LLM fields | `LIKE` on `llm_generated_university` — returns 28 |
| Q10 | Top programs by acceptance rate | `GROUP BY`, `HAVING COUNT(*) >= 10`, `ORDER BY rate DESC` |
| Q11 | Avg GPA by admission outcome | `GROUP BY status` |

**Note on Q8 vs Q9:** The raw `program` field contains only the program name (e.g., "Computer Science") with no university information, so university-based filtering on raw fields returns 0. The LLM-generated `llm_generated_university` field correctly maps entries to standardized university names, yielding 28 matches — demonstrating the practical value of the Module 2 LLM cleaning step.

---

## Data Quality Notes

- **GRE AW outliers:** Grad Café allows free-text entry, so a small number of entries contain impossible GRE AW values (e.g., 99.99, 73.0). All average calculations filter to the valid range (0–6).
- **GPA outliers:** Similarly filtered to 0–4.0.
- **GRE total vs. section scores:** Some entries report combined GRE (260–340) and some report single-section scores (130–170); both are stored in the `gre` column. Averages are filtered to 130–340.
- **Missing values:** Fields not reported by an applicant are stored as NULL (not empty string), consistent with Module 2 cleaning.

---

## Known Bugs

None.
