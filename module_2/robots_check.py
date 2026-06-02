"""
robots_check.py — Verify Grad Cafe robots.txt permits scraping before any
data collection begins.  Run this once and screenshot the output for the README.

Usage:
    python robots_check.py
"""

import urllib.robotparser
import urllib3

GRADCAFE_BASE = "https://www.thegradcafe.com"
ROBOTS_URL = f"{GRADCAFE_BASE}/robots.txt"

# Our scraper uses Selenium with Chrome's default user agent — represented
# by the wildcard "*" in robots.txt (not "ClaudeBot", which is blocked).
USER_AGENT = "*"

# Paths we intend to visit during scraping
PATHS_TO_CHECK = [
    "/survey",
    "/",
]

# Bots explicitly blocked by thegradcafe.com (for reference — we are none of these)
BLOCKED_BOTS = [
    "ClaudeBot", "GPTBot", "Amazonbot", "CCBot",
    "Bytespider", "Google-Extended", "ia_archiver", "YandexBot",
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

    # --- check each path we plan to scrape (as a generic browser/crawler) ---
    print(f"\n--- Path permissions for user-agent '{USER_AGENT}' (Chrome/Selenium) ---")
    all_allowed = True
    for path in PATHS_TO_CHECK:
        full_url = f"{GRADCAFE_BASE}{path}"
        allowed = parser.can_fetch(USER_AGENT, full_url)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  {status}  {path}")
        if not allowed:
            all_allowed = False

    # --- confirm our scraper is not one of the blocked bots ---
    print("\n--- Blocked bots (our scraper is NOT any of these) ---")
    for bot in BLOCKED_BOTS:
        survey_url = f"{GRADCAFE_BASE}/survey"
        allowed = parser.can_fetch(bot, survey_url)
        status = "allowed" if allowed else "BLOCKED"
        print(f"  {bot}: {status}")

    # --- crawl delay ---
    delay = parser.crawl_delay(USER_AGENT)
    print(f"\nCrawl-delay directive: {delay if delay is not None else 'none — we self-impose 2s between requests'}")

    print("\n--- Verdict ---")
    if all_allowed:
        print("All target paths are permitted for a standard browser/Selenium user agent.")
        print("Scraper uses Chrome default user agent — not blocked by any directive.")
    else:
        print("WARNING: One or more paths are blocked. Do not scrape blocked paths.")

    print("=" * 60)
    return all_allowed


if __name__ == "__main__":
    permitted = check_robots()
    if not permitted:
        raise SystemExit("Scraping blocked by robots.txt — aborting.")
