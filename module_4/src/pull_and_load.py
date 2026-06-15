"""
pull_and_load.py — Scrape new Grad Café entries and add them to the database.

Called by app.py via subprocess when the user clicks "Pull Data".
Scrapes a fresh batch from Grad Café and inserts only new records
(ON CONFLICT DO NOTHING skips duplicates).
"""

import os
import sys
import tempfile

TEMP_FILE = os.path.join(tempfile.gettempdir(), "gradcafe_new.json")
NEW_PAGES = 50  # scrape ~1,000 new entries per pull (polite batch size)


def pull_and_load(scrape_fn=None, save_fn=None, load_fn=None, database_url=None):
    """Scrape new data and load it into the database.

    Parameters are injectable so that tests can pass fake implementations
    without importing selenium or hitting the network.

    :param scrape_fn: Callable matching scrape_data(max_entries=N) signature.
                      Defaults to the real scrape_data from module_2.
    :type scrape_fn: callable or None
    :param save_fn: Callable matching save_data(records, filepath) signature.
                    Defaults to the real save_data from module_2.
    :type save_fn: callable or None
    :param load_fn: Callable matching load_data(filepath, database_url) signature.
                    Defaults to the real load_data from this package.
    :type load_fn: callable or None
    :param database_url: Postgres connection string passed to load_fn.
    :type database_url: str or None
    :returns: None
    :rtype: None
    """
    # Lazy-import the real implementations only when not injected.
    # This keeps the module importable in tests even if module_2 isn't on the path.
    if scrape_fn is None or save_fn is None:  # pragma: no cover
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../module_2"))
        from scrape import scrape_data, save_data
        if scrape_fn is None:
            scrape_fn = scrape_data
        if save_fn is None:
            save_fn = save_data

    if load_fn is None:
        from load_data import load_data
        load_fn = load_data

    print("[pull_and_load] Starting fresh scrape batch ...")
    records = scrape_fn(max_entries=NEW_PAGES * 20)

    if not records:
        print("[pull_and_load] No records returned.")
        return

    save_fn(records, TEMP_FILE)
    print(f"[pull_and_load] Saved {len(records)} records to {TEMP_FILE}")

    total = load_fn(TEMP_FILE, database_url)
    print(f"[pull_and_load] DB now contains {total:,} rows (duplicates skipped)")


if __name__ == "__main__":  # pragma: no cover
    pull_and_load()
