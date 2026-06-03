"""
pull_and_load.py — Scrape new Grad Café entries and add them to the database.

Called by app.py via subprocess when the user clicks "Pull Data".
Reads the last URL in the DB to avoid scraping pages we already have,
scrapes a fresh batch, and inserts only new records (ON CONFLICT DO NOTHING).
"""

import json
import os
import sys
import tempfile

# Allow imports from module_2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../module_2"))

from scrape import scrape_data, save_data
from load_data import load_data

TEMP_FILE = os.path.join(tempfile.gettempdir(), "gradcafe_new.json")
NEW_PAGES = 50  # scrape ~1,000 new entries per pull (polite batch size)


def pull_and_load():
    print("[pull_and_load] Starting fresh scrape batch ...")
    records = scrape_data(max_entries=NEW_PAGES * 20)

    if not records:
        print("[pull_and_load] No records returned.")
        return

    save_data(records, TEMP_FILE)
    print(f"[pull_and_load] Saved {len(records)} records to {TEMP_FILE}")

    total = load_data(TEMP_FILE)
    print(f"[pull_and_load] DB now contains {total:,} rows (duplicates skipped)")


if __name__ == "__main__":
    pull_and_load()
