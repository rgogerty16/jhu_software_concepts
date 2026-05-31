"""
robots_check.py — Verify Grad Cafe robots.txt permits scraping before any
data collection begins.  Run this once and screenshot the output for the README.

Usage:
    python robots_check.py
"""

import urllib.robotparser
import urllib3

GRADCAFE_BASE = "https://www.gradcafe.com"
ROBOTS_URL = f"{GRADCAFE_BASE}/robots.txt"
USER_AGENT = "*"

# Paths we intend to visit during scraping
PATHS_TO_CHECK = [
    "/survey/",
    "/",
]


def check_robots():
    """Fetch and parse robots.txt, then report whether our target paths are allowed."""

    print("=" * 60)
    print("robots.txt Compliance Check")
    print(f"Site: {GRADCAFE_BASE}")
    print(f"robots.txt: {ROBOTS_URL}")
    print("=" * 60)

    # --- fetch raw robots.txt so we can display it ---
    http = urllib3.PoolManager()
    response = http.request("GET", ROBOTS_URL)
    raw_text = response.data.decode("utf-8")

    print("\n--- robots.txt content ---")
    print(raw_text.strip())
    print("-" * 26)

    # --- parse with urllib.robotparser ---
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.read()

    # --- check each path we plan to scrape ---
    print(f"\n--- Path permissions for user-agent '{USER_AGENT}' ---")
    all_allowed = True
    for path in PATHS_TO_CHECK:
        full_url = f"{GRADCAFE_BASE}{path}"
        allowed = parser.can_fetch(USER_AGENT, full_url)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  {status}  {path}")
        if not allowed:
            all_allowed = False

    # --- crawl delay (None means no restriction specified) ---
    delay = parser.crawl_delay(USER_AGENT)
    print(f"\nCrawl-delay directive: {delay if delay is not None else 'none specified'}")

    print("\n--- Verdict ---")
    if all_allowed:
        print("All target paths are permitted. Safe to proceed with scraping.")
    else:
        print("WARNING: One or more paths are blocked. Do not scrape blocked paths.")

    print("=" * 60)
    return all_allowed


if __name__ == "__main__":
    permitted = check_robots()
    if not permitted:
        raise SystemExit("Scraping blocked by robots.txt — aborting.")
