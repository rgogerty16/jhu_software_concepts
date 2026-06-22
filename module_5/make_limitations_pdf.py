"""Generate limitations.pdf - run once, then delete this script."""
from fpdf import FPDF


def _a(text):
    """Replace common Unicode punctuation with ASCII equivalents for latin-1 fonts."""
    for src, dst in [("–", "-"), ("—", "-"), ("‘", "'"), ("’", "'"),
                     ("“", '"'), ("”", '"'), ("…", "...")]:
        text = text.replace(src, dst)
    return text

TITLE = "Limitations of Self-Submitted Anonymous Graduate Admissions Data"
AUTHOR = "Ryan Gogerty - JHU Modern Concepts in Python, Module 3"

PARA1 = (
    "The most fundamental limitation of Grad Café data is self-selection bias: only applicants "
    "who actively choose to submit their outcomes are represented in the dataset. This creates "
    "a sample that is systematically unrepresentative of the broader applicant population. "
    "High-achieving applicants - those accepted to elite programs or those with strong scores "
    "to brag about - are disproportionately motivated to share their results. The effect is "
    "visible directly in the data: the average GRE Verbal score among reporters in our dataset "
    "is 160.75, compared to the ETS-reported national mean of approximately 150–151 for all "
    "test takers. Similarly, the average GRE Analytical Writing score of 4.36 exceeds national "
    "norms, and the average GPA of 3.77 across all reporters is well above the undergraduate "
    "population average. Applicants with unremarkable or disappointing credentials are less "
    "likely to submit - and those who are rejected without any standout metric may not report "
    "at all - leaving the dataset with an upward-skewed portrait of who actually applies to "
    "graduate programs. This gap between the reporting sample and the true applicant pool means "
    "that any averages derived from Grad Café data should be interpreted as approximate upper "
    "bounds rather than true population means."
)

PARA2 = (
    "Beyond selection bias, self-submitted data introduces a separate layer of reliability "
    "concerns: there is no verification of the information entered. A user who enters a GRE "
    "Analytical Writing score of 99.99 or a GPA of 5.5 faces no correction mechanism; these "
    "values entered the raw dataset and had to be filtered out manually. More subtly, applicants "
    "may misremember scores, round figures, or conflate different score scales (for instance, "
    "reporting a single GRE section score where the field intends a combined score). The "
    "university and program name fields are even more susceptible: \"Johns Hopkins,\" \"JHU,\" "
    "\"John Hopkins\" (misspelled), and \"Johns Hopkins University\" all refer to the same "
    "institution, yet appear as distinct entries without normalization. This is precisely why "
    "the LLM standardization step from Module 2 was necessary - and its impact was measurable: "
    "querying for PhD Computer Science acceptances at top-four universities using raw fields "
    "returned zero results, while the same query against LLM-normalized fields returned 28. "
    "Taken together, these limitations mean that Grad Café analytics are best used for "
    "directional insights and hypothesis generation - identifying rough trends in acceptance "
    "rates, comparing programs, or flagging anomalies - rather than as authoritative statistics. "
    "Any conclusions drawn should be accompanied by the caveat that the underlying data is "
    "voluntary, unverified, and shaped by who chooses to participate."
)


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(26, 39, 68)
        self.cell(0, 10, TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, AUTHOR, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(200, 210, 230)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


pdf = PDF()
pdf.set_margins(25, 25, 25)
pdf.add_page()

pdf.set_font("Helvetica", "B", 10.5)
pdf.set_text_color(26, 39, 68)
pdf.cell(0, 8, "Paragraph 1 - Selection Bias and Score Inflation",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.set_font("Helvetica", "", 10.5)
pdf.set_text_color(30, 30, 30)
pdf.multi_cell(0, 6.5, _a(PARA1))
pdf.ln(8)

pdf.set_font("Helvetica", "B", 10.5)
pdf.set_text_color(26, 39, 68)
pdf.cell(0, 8, "Paragraph 2 - Data Reliability and Verification",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.set_font("Helvetica", "", 10.5)
pdf.set_text_color(30, 30, 30)
pdf.multi_cell(0, 6.5, _a(PARA2))

pdf.output("limitations.pdf")
print("limitations.pdf written.")
