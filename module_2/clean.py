"""
clean.py — Load and clean raw Grad Cafe applicant data.

Functions:
    load_data()   — load applicant_data.json into a list of dicts
    clean_data()  — normalize all fields; return cleaned list
    save_data()   — save cleaned list to a JSON file

Private helpers:
    _clean_entry()      — apply all cleaning rules to one record
    _clean_text()       — strip HTML tags/entities, collapse whitespace
    _normalize_missing()— convert empty / whitespace-only strings to None
    _parse_numeric()    — parse GPA/GRE strings to float, or return None
    _build_full_date()  — combine notification_date ("May 27") with year ("2026")
"""

import html
import json
import re

from bs4 import BeautifulSoup

INPUT_FILE = "applicant_data.json"
OUTPUT_FILE = "applicant_data_cleaned.json"

# Fields that should be strings (empty string → None)
_TEXT_FIELDS = [
    "university", "program", "raw_program", "degree",
    "status", "notification_date", "date_added",
    "semester", "year", "student_type", "url", "comments",
]

# Fields that should be numeric (any non-numeric value → None)
_NUMERIC_FIELDS = ["gpa", "gre", "gre_v", "gre_aw"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath=INPUT_FILE):
    """Load scraped applicant data from a JSON file.

    Args:
        filepath: Path to the JSON file written by scrape.py.

    Returns:
        List of raw applicant dicts, or empty list if file not found.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data):,} entries from {filepath}")
        return data
    except FileNotFoundError:
        print(f"[error] File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"[error] Invalid JSON in {filepath}: {e}")
        return []


def clean_data(data):
    """Normalize all fields across every applicant record.

    Cleaning operations applied to every entry:
      - Strip any residual HTML tags and decode HTML entities
      - Collapse and strip whitespace in all string fields
      - Convert empty / whitespace-only strings to None
      - Parse GPA/GRE fields to float where possible; None otherwise
      - Add notification_date_full by combining notification_date + year
      - Ensure every record has exactly the same set of keys

    The original 'raw_program' field is preserved unchanged so the
    applicant-provided text is always traceable.

    Args:
        data: List of raw dicts from load_data().

    Returns:
        List of cleaned dicts, one per applicant entry.
    """
    cleaned = [_clean_entry(entry) for entry in data]
    print(f"Cleaned {len(cleaned):,} entries")
    return cleaned


def save_data(data, filepath=OUTPUT_FILE):
    """Save cleaned applicant data to a JSON file.

    Args:
        data:     List of cleaned dicts from clean_data().
        filepath: Destination path (default: applicant_data_cleaned.json).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data):,} cleaned entries to {filepath}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clean_entry(entry):
    """Apply all cleaning rules to a single applicant record.

    Args:
        entry: Raw dict from scrape.py.

    Returns:
        Cleaned dict with the same keys plus 'notification_date_full'.
    """
    cleaned = {}

    # Clean all text fields
    for field in _TEXT_FIELDS:
        raw = entry.get(field)
        cleaned[field] = _normalize_missing(_clean_text(raw))

    # Parse numeric fields
    for field in _NUMERIC_FIELDS:
        cleaned[field] = _parse_numeric(entry.get(field))

    # Derived field: full notification date with year attached
    cleaned["notification_date_full"] = _build_full_date(
        cleaned.get("notification_date"),
        cleaned.get("year"),
    )

    return cleaned


def _clean_text(value):
    """Strip HTML tags, decode entities, and normalize whitespace.

    BeautifulSoup's get_text() during scraping handles most of this, but
    some older entries may contain HTML entities (e.g. &amp;, &#39;) or
    stray tags. This function catches any that slipped through.

    Args:
        value: Raw string value, or None.

    Returns:
        Cleaned string, or None if input was None.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)

    # Decode HTML entities first (e.g. &amp; → &, &#39; → ')
    value = html.unescape(value)

    # Strip any residual HTML tags
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "lxml").get_text(separator=" ")

    # Collapse internal whitespace and strip edges
    value = re.sub(r"\s+", " ", value).strip()

    return value


def _normalize_missing(value):
    """Return None for blank strings; otherwise return the value unchanged.

    Ensures every missing field is consistently None rather than a mix of
    "", " ", "N/A", or similar.

    Args:
        value: String or None.

    Returns:
        None if blank/whitespace-only, else the original value.
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _parse_numeric(value):
    """Convert a GPA or GRE string to float, returning None on failure.

    Handles values like "3.85", "320", "N/A", None, or already-numeric types.

    Args:
        value: Raw value from the scraped entry.

    Returns:
        Float if parseable, else None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _build_full_date(notification_date, year):
    """Combine a partial date string with a year to form a full date.

    Grad Cafe stores notification dates without a year (e.g. "May 27").
    The program start year is a reasonable proxy for the notification year
    (most notifications arrive in the same calendar year as the start term).

    Args:
        notification_date: e.g. "May 27" or None.
        year:              e.g. "2026" or None.

    Returns:
        e.g. "May 27, 2026", or None if either input is missing.
    """
    if not notification_date or not year:
        return None
    return f"{notification_date}, {year}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = load_data()
    if raw:
        cleaned = clean_data(raw)
        save_data(cleaned)
