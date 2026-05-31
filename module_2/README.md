# Module 2: Web Scraping — Grad Cafe

**Name:** Ryan Gogerty  
**Course:** Modern Concepts in Python  
**Assignment:** Module 2 — Web Scraping  
**Due:** TBD

---

## Overview

This module scrapes graduate school application data from [Grad Cafe](https://www.gradcafe.com/) and cleans it using a self-hosted local LLM. The output is a structured JSON dataset of 30,000+ applicant entries for use in future modules.

---

## Approach

*(To be filled in as the project is built)*

### Scraping

- **URL management:** `urllib3` is used to construct, validate, and inspect Grad Cafe URLs.
- **Page rendering:** Selenium is used to load pages in a Chrome browser, allowing JavaScript-rendered content to fully load before extraction.
- **Parsing:** BeautifulSoup parses the rendered HTML; regex and string methods extract individual fields.
- **Workflow:** urllib3 (URL construction) → Selenium (render page) → `page_source` → BeautifulSoup/regex (extract) → JSON (store)

### Cleaning

- The `program` field mixes program and university names in many formats (e.g., "JHU", "Johns Hopkins", "Johns Hopkins University").
- A self-hosted local LLM (provided by instructor, under `llm_hosting/`) does a first-pass standardization.
- Output adds cleaned `program_clean` and `university_clean` fields alongside the original raw `program` field.

---

## robots.txt Compliance

*(To be filled in — screenshot and explanation added here)*

---

## Setup & Run Instructions

*(To be filled in)*

### Requirements

```bash
pip install -r requirements.txt
```

### Scrape data

```bash
python scrape.py
```

### Clean data

```bash
python clean.py
```

### Run LLM standardization

```bash
cd llm_hosting
pip install -r requirements.txt
python app.py --file "../applicant_data.json" > ../llm_extend_applicant_data.json
```

---

## Known Bugs

*(None yet)*

---

## Edge Cases & Cleaning Notes

*(To be filled in after LLM cleaning pass)*
