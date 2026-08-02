"""Render writeup.md into writeup.pdf.

The write-up is authored once, in Markdown, and this script turns it into the
PDF deliverable so the two can never drift apart. Run it after any edit to
``writeup.md``::

    python make_writeup.py

It understands the small subset of Markdown the write-up actually uses:
headings, paragraphs with ``**bold**`` runs, bullet and numbered lists, fenced
code blocks, pipe tables, and an embedded image.
"""

import re
from pathlib import Path

from fpdf import FPDF

WRITEUP_MARKDOWN_PATH = Path(__file__).with_name("writeup.md")
WRITEUP_PDF_PATH = Path(__file__).with_name("writeup.pdf")

# Page geometry, in millimetres.
PAGE_MARGIN = 16.0
BODY_LINE_HEIGHT = 5.0
CODE_LINE_HEIGHT = 3.4

# The built-in PDF fonts are latin-1 only, so a few common Unicode characters
# are folded down to ASCII before anything is written.
ASCII_REPLACEMENTS = [
    ("–", "-"), ("—", "-"), ("‘", "'"), ("’", "'"),
    ("“", '"'), ("”", '"'), ("…", "..."), ("→", "->"),
    (" ", " "),
]

HEADING_SIZES = {1: 15.0, 2: 12.5, 3: 11.0}


def to_ascii(text):
    """Fold Unicode punctuation down to latin-1-safe ASCII.

    Args:
        text: The text to convert.

    Returns:
        The converted text.
    """
    for source, replacement in ASCII_REPLACEMENTS:
        text = text.replace(source, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def strip_inline_markup(text):
    """Remove Markdown link and code markup that the PDF cannot render.

    Args:
        text: A line of Markdown body text.

    Returns:
        The text with backticks removed and links reduced to their labels.
    """
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("`", "")


class WriteupPDF(FPDF):
    """A PDF with a page-number footer."""

    def footer(self):
        """Draw the page number at the bottom of every page."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def write_heading(pdf, level, text):
    """Write a heading line.

    Args:
        pdf: The PDF being built.
        level: Heading level, 1 to 3.
        text: The heading text.
    """
    pdf.ln(3.0 if level > 1 else 1.0)
    pdf.set_font("Helvetica", "B", HEADING_SIZES.get(level, 11.0))
    pdf.multi_cell(0, BODY_LINE_HEIGHT + 1.5, to_ascii(text),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.0)


def write_rich_text(pdf, text, size=9.5):
    """Write a paragraph, honouring ``**bold**`` runs.

    Args:
        pdf: The PDF being built.
        text: The paragraph text, possibly containing bold markers.
        size: Font size in points.
    """
    for index, fragment in enumerate(text.split("**")):
        if not fragment:
            continue
        # Fragments at odd positions sat between a pair of ** markers.
        pdf.set_font("Helvetica", "B" if index % 2 else "", size)
        pdf.write(BODY_LINE_HEIGHT, to_ascii(fragment))
    pdf.ln(BODY_LINE_HEIGHT)


def write_code_block(pdf, lines):
    """Write a monospace block, preserving the alignment of the training log.

    Args:
        pdf: The PDF being built.
        lines: The raw lines inside the fence.
    """
    pdf.ln(1.5)
    pdf.set_font("Courier", "", 7.0)
    for line in lines:
        pdf.multi_cell(0, CODE_LINE_HEIGHT, to_ascii(line) or " ",
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2.0)


def write_table(pdf, rows):
    """Write a pipe table as a bordered PDF table.

    Args:
        pdf: The PDF being built.
        rows: Table rows, each a list of cell strings, header row first.
    """
    pdf.ln(1.5)
    pdf.set_font("Helvetica", "", 8.0)

    # Give each column a share of the width proportional to its widest cell,
    # so a long label column does not wrap while a numeric one sits half empty.
    column_widths = tuple(
        max(len(strip_inline_markup(row[position])) for row in rows if position < len(row))
        for position in range(len(rows[0])))

    with pdf.table(col_widths=column_widths, line_height=4.6,
                   text_align="LEFT") as table:
        for row_index, cells in enumerate(rows):
            pdf.set_font("Helvetica", "B" if row_index == 0 else "", 8.0)
            row = table.row()
            for cell in cells:
                row.cell(to_ascii(strip_inline_markup(cell)))
    pdf.ln(2.0)


def write_image(pdf, image_path):
    """Place an image, centred, at a readable width.

    Args:
        pdf: The PDF being built.
        image_path: Path to the image file.
    """
    if not image_path.exists():
        return
    pdf.ln(2.0)
    width = pdf.w - 2 * PAGE_MARGIN
    pdf.image(str(image_path), x=PAGE_MARGIN, w=width)
    pdf.ln(3.0)


def parse_table_row(line):
    """Split one pipe-table line into its cells.

    Args:
        line: A Markdown table line beginning with ``|``.

    Returns:
        A list of stripped cell strings.
    """
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_divider(line):
    """Report whether a line is a Markdown table's ``---`` divider.

    Args:
        line: The line to test.

    Returns:
        True if the line only separates a header from its body.
    """
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def render_markdown(pdf, markdown_text):
    """Render the supported Markdown subset into the PDF.

    Args:
        pdf: The PDF being built.
        markdown_text: The full contents of ``writeup.md``.
    """
    lines = markdown_text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            write_code_block(pdf, block)

        elif stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                if not is_table_divider(lines[index]):
                    rows.append(parse_table_row(lines[index]))
                index += 1
            write_table(pdf, rows)
            continue

        elif stripped.startswith("!["):
            match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", stripped)
            if match:
                write_image(pdf, Path(__file__).with_name(match.group(1)))

        elif stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            write_heading(pdf, level, stripped.lstrip("# ").strip())

        elif re.match(r"^(\d+\.|-)\s", stripped):
            marker, _, body = stripped.partition(" ")
            bullet = "-" if marker == "-" else marker
            pdf.set_font("Helvetica", "", 9.5)
            pdf.write(BODY_LINE_HEIGHT, f"  {bullet} ")
            write_rich_text(pdf, strip_inline_markup(body))

        elif stripped:
            # Join the physical lines of one wrapped paragraph.
            paragraph = []
            while index < len(lines) and lines[index].strip() \
                    and not lines[index].strip().startswith(("#", "|", "```", "![")) \
                    and not re.match(r"^(\d+\.|-)\s", lines[index].strip()):
                paragraph.append(lines[index].strip())
                index += 1
            write_rich_text(pdf, strip_inline_markup(" ".join(paragraph)))
            pdf.ln(1.5)
            continue

        index += 1


def main():
    """Build ``writeup.pdf`` from ``writeup.md``."""
    pdf = WriteupPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    pdf.add_page()

    render_markdown(pdf, WRITEUP_MARKDOWN_PATH.read_text(encoding="utf-8"))
    pdf.output(str(WRITEUP_PDF_PATH))

    print(f"Wrote {WRITEUP_PDF_PATH.name} ({pdf.page_no()} pages) "
          f"from {WRITEUP_MARKDOWN_PATH.name}")


if __name__ == "__main__":
    main()
