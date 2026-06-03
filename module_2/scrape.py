"""
scrape.py — Grad Cafe web scraper
Workflow: urllib3 (URL construction) → Selenium (render page)
          → BeautifulSoup/regex (extract fields) → JSON (save)
"""

import json
import re
import time
from urllib.parse import urljoin, urlparse, urlencode

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.thegradcafe.com"
SURVEY_PATH = "/survey"
OUTPUT_FILE = "applicant_data.json"
REQUEST_DELAY = 2       # seconds to wait between page requests (be polite)
TARGET_ENTRIES = 30000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_data(max_entries=TARGET_ENTRIES):
    """Pull applicant entries from Grad Cafe and return them as a list of dicts.

    Iterates through paginated survey results, parsing each page with
    BeautifulSoup until max_entries is reached, the site stops returning
    results, or the site rejects a request (rate-limit / block).

    Polite behaviour:
      - REQUEST_DELAY seconds between every page load
      - Stops immediately on any None response from _get_page_source,
        which signals the site has blocked or rate-limited the scraper

    Args:
        max_entries: Stop collecting after this many entries.

    Returns:
        List of applicant dicts, each with the keys defined in _parse_entry.
    """
    results = []
    page = 1
    driver = _init_driver()

    try:
        while len(results) < max_entries:
            url = _build_url(page)

            if not _validate_url(url):
                print(f"  [error] Invalid URL on page {page}, stopping.")
                break

            html = _get_page_source(driver, url)

            # None means the site timed out, blocked, or rate-limited us
            if html is None:
                print(f"  [warn] No response on page {page} — stopping scrape.")
                break

            entries = _parse_page(html)

            # An empty page means we've gone past the last page of results
            if not entries:
                print(f"  [info] No entries on page {page} — end of results.")
                break

            results.extend(entries)

            if page % 50 == 0 or len(results) >= max_entries:
                print(f"  Page {page:>5} | collected {len(results):>6,} entries")

            page += 1
            time.sleep(REQUEST_DELAY)

    finally:
        # Always close the browser, even if an exception is raised mid-scrape
        driver.quit()

    # Trim to exactly max_entries in case the last page pushed us over
    return results[:max_entries]


def save_data(data, filepath=OUTPUT_FILE):
    """Serialize applicant data to a JSON file.

    Writes with indent=2 so the file is human-readable and easy to
    inspect without a JSON viewer.

    Args:
        data:     List of applicant dicts returned by scrape_data().
        filepath: Destination filename (default: applicant_data.json).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data):,} entries to {filepath}")


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _build_url(page_number):
    """Construct the Grad Cafe survey URL for a given page number.

    Uses urljoin + urlencode so we never manually concatenate strings.
    Confirmed URL pattern: https://www.thegradcafe.com/survey?page=N

    Args:
        page_number: Integer page index (1-based).

    Returns:
        Full URL string, e.g. "https://www.thegradcafe.com/survey?page=2"
    """
    base = urljoin(BASE_URL, SURVEY_PATH)
    query = urlencode({"page": page_number})
    return f"{base}?{query}"


def _validate_url(url):
    """Parse a URL with urlparse and confirm it has a scheme and netloc.

    Args:
        url: URL string to validate.

    Returns:
        True if valid, False otherwise.
    """
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


# ---------------------------------------------------------------------------
# Selenium helpers
# ---------------------------------------------------------------------------

def _init_driver():
    """Initialize a headless Chrome WebDriver via webdriver-manager.

    webdriver-manager automatically downloads the correct ChromeDriver
    binary for the installed Chrome version, so no manual path is needed.

    Returns:
        A configured selenium.webdriver.Chrome instance.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")       # run without a visible window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,800")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    return driver


def _get_page_source(driver, url):
    """Navigate to url, wait for applicant results to load, return page HTML.

    Uses an explicit Selenium wait (not sleep) so we only proceed once the
    result rows are actually present in the DOM.

    Page structure confirmed: each entry is 2–3 <tr> elements. Detail rows
    always carry class "tw-border-none", so we wait for that class to appear
    as a signal that at least one full entry has rendered.

    Args:
        driver: Active Chrome WebDriver from _init_driver().
        url:    Page URL to load.

    Returns:
        Rendered HTML string (driver.page_source) after results load.
        Returns None if the expected element never appears (timeout) or
        if the site rejects the request.
    """
    try:
        driver.get(url)
        # Wait until at least one detail row is in the DOM
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tw-border-none"))
        )
        return driver.page_source
    except Exception as e:
        print(f"  [warn] Could not load {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_page(html):
    """Parse one rendered survey page and return a list of entry dicts.

    Page structure (confirmed by live inspection):
      Each applicant entry spans 2–3 consecutive <tr> elements:
        Row 1 (main):   university, program, degree, date, status, url
        Row 2 (detail): semester, year, student_type  (class="tw-border-none")
        Row 3 (comment, optional): free-text comment  (class="tw-border-none")

    We identify the start of a new entry by the presence of the university
    div (class contains "tw-font-medium"), then collect the following
    tw-border-none rows as belonging to that entry.

    Args:
        html: Rendered HTML string from _get_page_source().

    Returns:
        List of dicts, one per applicant entry on the page.
    """
    soup = BeautifulSoup(html, "lxml")
    all_rows = soup.find_all("tr")

    entries = []
    current_group = []

    for row in all_rows:
        # The university div has exactly these three classes; mobile status badges
        # also have tw-font-medium but lack tw-text-gray-900 and tw-text-sm,
        # so this triple-class check avoids false positives.
        is_main = row.find(
            "div",
            class_=lambda c: c and all(
                cls in c for cls in ["tw-font-medium", "tw-text-gray-900", "tw-text-sm"]
            )
        )
        if is_main:
            # This row is the start of a new entry; save the previous group first
            if current_group:
                entry = _parse_entry(current_group)
                if entry:
                    entries.append(entry)
            current_group = [row]
        elif current_group:

            # A detail or comment row belonging to the current entry
            current_group.append(row)

    # Don't forget the final group on the page
    if current_group:
        entry = _parse_entry(current_group)
        if entry:
            entries.append(entry)

    return entries


def _parse_entry(rows):
    """Extract all required fields from one applicant entry (a group of <tr> tags).

    Always returns the same set of keys so every record in applicant_data.json
    has a consistent shape. Missing values are None.

    Args:
        rows: List of BeautifulSoup <tr> Tags — [main_row, detail_row, (comment_row)]

    Returns:
        Dict with keys: university, program, raw_program, degree, status,
        notification_date, date_added, semester, year, student_type, url,
        comments, gpa, gre, gre_v, gre_aw.
    """
    main_row = rows[0]
    detail_row = rows[1] if len(rows) > 1 else None
    extra_rows = rows[2:] if len(rows) > 2 else []

    tds = main_row.find_all("td")

    # --- University (td[0]) ---
    univ_div = main_row.find(
        "div",
        class_=lambda c: c and all(
            cls in c for cls in ["tw-font-medium", "tw-text-gray-900", "tw-text-sm"]
        )
    )
    university = univ_div.get_text(strip=True) if univ_div else ""

    # --- Program + degree (td[1]) ---
    # td[1] contains: <div class="tw-text-gray-900"><span>Program</span>...<span>Degree</span></div>
    program_td = tds[1] if len(tds) > 1 else None
    program_div = program_td.find("div") if program_td else None
    spans = program_div.find_all("span") if program_div else []
    program = spans[0].get_text(strip=True) if spans else ""
    degree = spans[1].get_text(strip=True) if len(spans) > 1 else ""
    # raw_program preserves the original combined text for traceability
    raw_program = program_div.get_text(separator=" ", strip=True) if program_div else ""

    # --- Date added (td[2]) ---
    date_td = tds[2] if len(tds) > 2 else None
    date_added = date_td.get_text(strip=True) if date_td else ""

    # --- Status + notification date (td[3]) ---
    # Raw text is e.g. "Rejected on May 27" or "Accepted on May 27"
    status_td = tds[3] if len(tds) > 3 else None
    raw_status = status_td.get_text(strip=True) if status_td else ""
    status = _normalize_status(raw_status)
    date_match = re.search(r"on\s+(.+)$", raw_status, re.IGNORECASE)
    notification_date = date_match.group(1).strip() if date_match else ""

    # --- URL to individual result page (td[4] contains the link icon) ---
    link = main_row.find("a", href=True)
    entry_url = f"{BASE_URL}{link['href']}" if link else ""

    # --- Semester, year, student type, and score badges (detail row) ---
    semester = ""
    year = ""
    student_type = ""
    gpa = None
    gre = None
    gre_v = None
    gre_aw = None
    if detail_row:
        for badge in detail_row.find_all("div", class_="tw-rounded-md"):
            text = badge.get_text(strip=True)
            # Semester badge: "Fall 2026", "Spring 2025", etc.
            if re.match(r"(Fall|Spring|Summer|Winter)\s+\d{4}$", text):
                parts = text.split()
                semester = parts[0]
                year = parts[1]
            # Student type badge
            elif text in ("American", "International", "Other"):
                student_type = text
            # Score badges: "GPA 3.90", "GRE 320", "GRE V 165", "GRE AW 4.00"
            elif m := re.match(r"GPA\s+([\d.]+)$", text):
                gpa = float(m.group(1))
            elif m := re.match(r"GRE V\s+([\d.]+)$", text):
                gre_v = float(m.group(1))
            elif m := re.match(r"GRE AW\s+([\d.]+)$", text):
                gre_aw = float(m.group(1))
            elif m := re.match(r"GRE\s+([\d.]+)$", text):
                gre = float(m.group(1))

    # --- Comments (optional row containing a <p> tag) ---
    comments = ""
    for extra_row in extra_rows:
        p_tag = extra_row.find("p")
        if p_tag:
            comments = p_tag.get_text(strip=True)
            break

    return {
        "university": university,
        "program": program,
        "raw_program": raw_program,
        "degree": degree,
        "status": status,
        "notification_date": notification_date,
        "date_added": date_added,
        "semester": semester,
        "year": year,
        "student_type": student_type,
        "url": entry_url,
        "comments": comments,
        "gpa": gpa,
        "gre": gre,
        "gre_v": gre_v,
        "gre_aw": gre_aw,
    }


def _normalize_status(raw_status):
    """Normalize a raw status string to a canonical form.

    Raw text from the page is e.g. "Rejected on May 27" or "Accepted on May 27".
    We lowercase and keyword-match to handle any variation.

    Args:
        raw_status: Raw status string scraped from the page.

    Returns:
        One of: 'Accepted', 'Rejected', 'Waitlisted', 'Interview', or ''.
    """
    s = raw_status.lower()
    if "accept" in s:
        return "Accepted"
    elif "reject" in s:
        return "Rejected"
    elif "waitlist" in s or "wait list" in s:
        return "Waitlisted"
    elif "interview" in s:
        return "Interview"
    return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting scrape — target: {TARGET_ENTRIES} entries")
    records = scrape_data()
    if records is not None:
        print(f"Scraped {len(records)} entries")
        save_data(records)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("scrape_data() not yet implemented")
