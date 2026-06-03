# Module 2: Web Scraping — Grad Cafe

**Name:** Ryan Gogerty  
**JHED ID:** rgogerty  
**Course:** Modern Concepts in Python  
**Assignment:** Module 2 — Web Scraping  
**Due:** TBD

---

## Overview

This module scrapes graduate school application data from [Grad Cafe](https://www.thegradcafe.com/) and cleans it using a self-hosted local LLM. The output is a structured JSON dataset of 30,000+ applicant entries for use in future modules.

---

## Approach

### Scraping (`scrape.py`)

**Workflow: urllib.parse → Selenium → BeautifulSoup/regex → JSON**

1. **URL construction:** Python's standard library `urllib.parse` (`urljoin`, `urlencode`, `urlparse`) constructs, validates, and manages all Grad Cafe URLs. The survey pagination URL pattern is `https://www.thegradcafe.com/survey?page=N`.

2. **Page rendering:** Selenium controls a headless Chrome browser to fully render each page before extraction. Grad Cafe uses dynamic content (JavaScript-rendered tables), so a static `urllib.request` fetch alone would miss the applicant rows.

3. **Explicit waits:** `WebDriverWait` waits for at least one `tw-border-none` row to appear in the DOM before reading `page_source`, avoiding flaky hard-coded `sleep` calls.

4. **Parsing:** BeautifulSoup parses the rendered HTML. Each applicant entry spans 2–3 consecutive `<tr>` rows:
   - Row 1 (main): university, program, degree, date added, status, URL
   - Row 2 (detail badges): semester, year, student type, GPA, GRE, GRE V, GRE AW
   - Row 3 (optional): free-text comment

5. **Polite scraping:** A 2-second delay between every page request. The scraper stops immediately if Selenium cannot load a page (rate-limit / block signal).

6. **Output:** `applicant_data.json` — a JSON array of 30,000 applicant dicts.

### Cleaning (`clean.py`)

- Strips residual HTML tags and decodes HTML entities from all string fields.
- Normalizes whitespace (collapse multiple spaces/newlines to one space).
- Converts empty/whitespace-only strings to `None` for a consistent missing-value representation.
- Parses GPA/GRE fields to `float` where possible; non-numeric values become `None`.
- Adds a derived `notification_date_full` field by combining `notification_date` ("May 27") with `year` ("2026").
- Output: `applicant_data_cleaned.json`

### LLM Standardization (`llm_hosting/app.py`)

The `program` field from Grad Cafe frequently mixes program name and university (e.g., "CS, JHU", "Information Studies, McG") in dozens of abbreviation and spelling variants. Cleaning this manually at 30k rows is not feasible.

**Approach:**
- Self-hosts TinyLlama-1.1B-Chat (GGUF, Q4_K_M quantized) via `llama-cpp-python`.
- Each row's `program` + `university` fields are combined and sent to the model with few-shot examples.
- The model returns `standardized_program` and `standardized_university`.
- A post-processor applies abbreviation expansion (e.g., "UBC" → "University of British Columbia"), common spelling fixes, and fuzzy matching against canonical lists (`canon_universities.txt`, `canon_programs.txt`).
- A result cache avoids redundant LLM calls for repeated program+university combos (very common across 30k rows).
- Parallelized with `ThreadPoolExecutor`; LLM calls serialized internally by a lock since llama.cpp is not thread-safe.
- Output: `llm_extend_applicant_data.json` — original records with two added fields: `llm-generated-program` and `llm-generated-university`.

The original `raw_program` field is preserved unchanged throughout all cleaning passes for traceability.

---

## robots.txt Compliance

**robots.txt was checked programmatically before any scraping began** using `robots_check.py`, which:
1. Fetches and displays the full `robots.txt` from `https://www.thegradcafe.com/robots.txt`
2. Uses `urllib.robotparser.RobotFileParser` to check whether our target paths (`/survey`, `/`) are permitted for a standard browser user agent (`*`)
3. Lists explicitly blocked bots (e.g., ClaudeBot, GPTBot, YandexBot) to confirm our Chrome/Selenium scraper is not among them
4. Reports any crawl-delay directive

**Result:** `/survey` and `/` are both **ALLOWED** for the wildcard user agent. Our Selenium scraper uses Chrome's default user agent, which is not blocked by any directive.

See `screenshot.jpg` for evidence of the robots.txt check output.

**How the scraper complies:**
- Only scrapes `/survey` pages, which are explicitly permitted
- Uses Chrome's standard user agent (not impersonating a blocked bot)
- Enforces a 2-second delay between every page request (polite crawling)
- Stops immediately if the site returns no results or times out (respects rate limits)
- Does not bypass any login, CAPTCHA, access control, or rate limit

---

## Setup & Run Instructions

### Requirements

```bash
# Main scraper/cleaner
pip install -r requirements.txt

# LLM standardizer (separate environment or add to main)
cd llm_hosting
pip install -r requirements.txt
cd ..
```

**Python 3.10+ required.** Uses the walrus operator (`:=`) introduced in 3.8 and match syntax from 3.10.

**Browser:** Chrome must be installed. `webdriver-manager` automatically downloads the matching ChromeDriver — no manual driver setup needed.

### 1. Check robots.txt

```bash
python robots_check.py
```

### 2. Scrape data

```bash
python scrape.py
```

Produces `applicant_data.json` (~30,000 entries). Takes approximately 50 minutes at 2 s/page.

### 3. Clean data

```bash
python clean.py
```

Produces `applicant_data_cleaned.json`.

### 4. Run LLM standardization

```bash
cd llm_hosting
python app.py --file ../applicant_data.json --out ../llm_extend_applicant_data.json
```

The model (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`) is downloaded automatically from Hugging Face on first run into `llm_hosting/models/`. Subsequent runs reuse the local copy. Takes 20–60 minutes depending on CPU.

---

## Selenium Setup

- **Browser:** Google Chrome (system-installed)
- **Driver:** ChromeDriver, managed automatically by `webdriver-manager`
- **Mode:** Headless (`--headless=new`)
- **Waits:** Explicit `WebDriverWait` — waits for `tw-border-none` class elements before reading `page_source`

---

## Data Fields

Each record in `applicant_data.json` contains:

| Field | Description |
|---|---|
| `university` | University name |
| `program` | Program name |
| `raw_program` | Original combined program+degree text (preserved for traceability) |
| `degree` | Degree type (Masters / PhD) |
| `status` | Accepted / Rejected / Waitlisted / Interview |
| `notification_date` | Date of decision (e.g., "May 27") |
| `date_added` | Date entry was added to Grad Cafe |
| `semester` | Program start semester (Fall / Spring / Summer / Winter) |
| `year` | Program start year |
| `student_type` | American / International / Other |
| `url` | Link to individual applicant entry on Grad Cafe |
| `comments` | Applicant-provided free-text comment |
| `gpa` | GPA (float or null) |
| `gre` | GRE total score (float or null) |
| `gre_v` | GRE Verbal score (float or null) |
| `gre_aw` | GRE Analytical Writing score (float or null) |

`llm_extend_applicant_data.json` adds:

| Field | Description |
|---|---|
| `llm-generated-program` | Standardized program name from LLM + post-processor |
| `llm-generated-university` | Standardized university name from LLM + post-processor |

---

## Cleaning Edge Cases & Known Imperfections

- **GRE/GPA availability:** Not all Grad Cafe entries include GRE or GPA data. These fields are `null` when the applicant did not report them. The scraper correctly captures these values when present as detail-row badges (e.g., "GPA 3.90", "GRE 320", "GRE V 165", "GRE AW 4.00").
- **Notification date + year:** Grad Cafe stores notification dates without a year (e.g., "May 27"). The `notification_date_full` derived field pairs this with the program start year, which is a reasonable but imperfect proxy (a rejection in December for a Fall start may actually belong to the prior year).
- **LLM standardization limitations:** TinyLlama (1.1B parameters) occasionally produces inconsistent or partially incorrect standardizations, particularly for less common university names or highly abbreviated inputs. The canonical lists (`canon_universities.txt`, `canon_programs.txt`) and fuzzy post-processor catch most common cases but do not guarantee perfection. The original `raw_program` and `program` fields are preserved for re-processing.
- **Duplicate-like entries:** Grad Cafe allows multiple submissions for the same applicant outcome. The scraper does not deduplicate — it preserves all records as uploaded.

---

## Known Bugs

None.
