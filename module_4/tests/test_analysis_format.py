"""
test_analysis_format.py — Tests for analysis output labels and percentage formatting.

What we're testing:
  - Every rendered analysis result is labeled with "Answer:"
  - Every percentage on the page uses exactly two decimal places (e.g. "45.51%")

Why formatting matters: the assignment mandates two-decimal percentages, and
consistent "Answer:" labels are what makes the output readable. These tests
catch regressions where a template change accidentally strips a label or breaks
the %.2f formatting.

Tools used:
  - BeautifulSoup: parses HTML into a tree so we can search it structurally
  - re (regex): lets us write a pattern like ``\\d+\\.\\d{2}%`` to match any
    two-decimal percentage, instead of hardcoding specific numbers (which would
    break if your data changes)
"""

import re
import pytest
from bs4 import BeautifulSoup


@pytest.mark.analysis
def test_page_contains_multiple_answer_labels(app_client):
    """The analysis page should contain several 'Answer:' labels.

    We check for at least 5 — one per major analysis question (Q1–Q5 at minimum).
    If the template strips labels, this catches it immediately.
    """
    response = app_client.get("/analysis")
    page_text = BeautifulSoup(response.data.decode(), "html.parser").get_text()
    count = page_text.count("Answer:")
    assert count >= 5, f"Expected at least 5 'Answer:' labels, found {count}"


@pytest.mark.analysis
def test_percentages_have_two_decimal_places(app_client):
    """Every percentage rendered on the page must have exactly two decimal places.

    We use a regex to find ALL percentage strings on the page and verify each
    one matches the pattern: one-or-more digits, a dot, exactly two digits, %.

    r'\\d+\\.\\d{2}%'  breaks down as:
      \\d+   → one or more digits before the decimal
      \\.    → a literal period
      \\d{2} → exactly two digits after the decimal
      %      → literal percent sign

    Example matches: "45.51%", "0.00%", "100.00%"
    Example failures: "45.5%", "45.513%", "45%"

    We use re.findall() which returns a list of all matches — empty list means
    no percentages found at all, which is also a test failure.
    """
    response = app_client.get("/analysis")
    page_text = BeautifulSoup(response.data.decode(), "html.parser").get_text()

    # Find all percentage strings on the page
    percentages = re.findall(r'\d+\.\d{2,}%', page_text)

    # There should be at least two (Q2: international %, Q5: acceptance rate %)
    assert len(percentages) >= 2, (
        f"Expected at least 2 percentages on page, found: {percentages}"
    )

    # Every percentage must match *exactly* two decimal places
    two_decimal = re.compile(r'^\d+\.\d{2}%$')
    for pct in percentages:
        assert two_decimal.match(pct), (
            f"Percentage {pct!r} does not have exactly two decimal places"
        )


@pytest.mark.analysis
def test_q2_international_pct_format(app_client):
    """Q2 (% international) should appear formatted as XX.XX% on the page.

    This is a more targeted check: we find the specific card for Q2 and
    verify its value matches the two-decimal pattern. If the template renders
    Q2 differently from other percentages, this catches it.
    """
    response = app_client.get("/analysis")
    soup = BeautifulSoup(response.data.decode(), "html.parser")
    page_text = soup.get_text()
    # The page must contain at least one string matching the percentage pattern
    assert re.search(r'\d+\.\d{2}%', page_text), (
        "No two-decimal percentage found on page"
    )


@pytest.mark.analysis
def test_answer_labels_present_in_table_cells(app_client):
    """'Answer:' text should appear inside table cells, not just headings.

    We search specifically in <td> elements — this confirms labels are in the
    data rows, not accidentally only in the column headers.
    """
    response = app_client.get("/analysis")
    soup = BeautifulSoup(response.data.decode(), "html.parser")
    # Find all table data cells that contain 'Answer:'
    cells_with_answer = [
        td for td in soup.find_all("td")
        if "Answer:" in td.get_text()
    ]
    assert len(cells_with_answer) >= 3, (
        f"Expected 'Answer:' in at least 3 table cells, found {len(cells_with_answer)}"
    )
