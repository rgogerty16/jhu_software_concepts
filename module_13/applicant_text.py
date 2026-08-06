"""applicant_text.py — the unified applicant text representation.

This module is deliberately the *only* place where an applicant record is turned
into the string a language model reads. ``train_model.py`` imports it to build
training examples and the Flask ``/will-you-get-in`` route imports it to build
inference inputs, so the two can never drift apart. If the template changes, it
changes in one file and both sides move together.

Two design decisions are worth calling out, because both are deliberate:

1. ``Comments`` is the **last** field in the template. Comments are the only
   unbounded field in the dataset (median 9 words, but the longest is 337), so
   putting them last means tokenizer truncation clips free text rather than
   discarding the structured GPA/GRE/degree signal that appears above it.

2. The target label **never** appears in the text. The assignment's illustrative
   example ends with a ``Prediction target: Accepted`` line, but including that
   at training time would teach the model to read its own answer off the input
   and would be impossible to reproduce at inference time. Training and serving
   both use exactly the template below, with no label line.

The two LLM-standardized fields available in the dataset
(``llm-generated-program`` and ``llm-generated-university``) are intentionally
**excluded** — see ``EXCLUDED_FIELDS``. A deployed form cannot reproduce them, so
training on them would create a train/serve skew where the model learned from a
cleaned field that is only ever a duplicate of the user's raw typing in
production.
"""

from __future__ import annotations

import math

# Bumped whenever the template changes, and written into the saved model's
# metadata so a reloaded model can be checked against the code serving it.
TEMPLATE_VERSION = 1

#: Rendered in place of any field that is missing, blank, or out of range. One
#: placeholder is used for every field type so the model sees a single
#: consistent "no data" token rather than a mix of None/NA/empty.
MISSING_PLACEHOLDER = "Unknown"

#: Free-text fields written by the applicant. Three of them, against the
#: assignment's floor of two.
TEXT_FIELDS = ("program", "university", "comments")

#: Structured categorical fields.
CATEGORICAL_FIELDS = ("term", "degree", "us_or_international")

#: Structured scalar fields.
NUMERIC_FIELDS = ("gpa", "gre", "gre_v", "gre_aw")

#: Every field the model actually reads: 3 text + 7 non-text.
MODEL_FIELDS = TEXT_FIELDS + CATEGORICAL_FIELDS + NUMERIC_FIELDS

#: Fields present in the dataset but kept out of the model input, with the
#: reason. Reported in the training log so the omission is a documented choice
#: rather than an oversight.
EXCLUDED_FIELDS = {
    "llm-generated-program": "standardized duplicate of program; a fallback, never a second input",
    "llm-generated-university": "standardized duplicate of university; fallback only, same reason",
    "raw_program": "unnormalized concatenation of program and degree, both already used",
    "status": "this is the prediction target - including it would leak the label",
    "notification_date": "only known after a decision arrives, so unavailable at prediction time",
    "date_added": "posting date leaks the outcome timeline rather than the applicant profile",
    "url": "row identity only; used for deduplication, carries no admissions signal",
}

#: ``(label, field)`` pairs in template order. Comments last, on purpose.
TEMPLATE_LINES = (
    ("Program", "program"),
    ("University", "university"),
    ("Term", "term"),
    ("Degree", "degree"),
    ("Citizenship", "us_or_international"),
    ("GPA", "gpa"),
    ("GRE Quant", "gre"),
    ("GRE Verbal", "gre_v"),
    ("GRE AW", "gre_aw"),
    ("Comments", "comments"),
)

#: The label used for the comments line, needed by :func:`without_comments`.
COMMENTS_LABEL = "Comments"

#: How each scalar is rendered, so "3.5" and "3.50" never tokenize differently.
NUMERIC_FORMATS = {
    "gpa": "{:.2f}",
    "gre": "{:.0f}",
    "gre_v": "{:.0f}",
    "gre_aw": "{:.1f}",
}

#: Inclusive plausible ranges. A value outside its range is treated as missing
#: rather than trusted: the dataset is self-reported, so it contains GPAs on
#: 10-point scales and GRE scores on the retired 200-800 scale, and both would
#: otherwise be read by the model as absurd values on the current scale.
SANITY_RANGES = {
    "gpa": (0.0, 5.0),
    "gre": (130.0, 170.0),
    "gre_v": (130.0, 170.0),
    "gre_aw": (0.0, 6.0),
}

#: Keys accepted for each canonical field. The raw scraped JSON, the Postgres
#: ``applicants`` table, and the web form each name a few fields differently;
#: normalizing here means no caller has to care which source it holds.
#: The LLM-standardized names appear here only as *fallbacks*, never as extra
#: inputs: the scraped JSON carries a raw ``university`` but the Postgres
#: ``applicants`` table only ever stored ``llm_generated_university``, so the
#: database path would otherwise render every university as "Unknown".
FIELD_ALIASES = {
    "program": ("program", "llm_generated_program", "llm-generated-program"),
    "university": ("university", "llm_generated_university", "llm-generated-university"),
    "comments": ("comments",),
    "degree": ("degree",),
    "us_or_international": ("us_or_international", "student_type", "citizenship"),
    "gpa": ("gpa",),
    "gre": ("gre",),
    "gre_v": ("gre_v",),
    "gre_aw": ("gre_aw",),
}

#: Lowercased strings that mean "no value". Note that ``unknown`` is included,
#: so a user who picks the form's "Unknown" option is treated as having left the
#: field blank and renders as the same placeholder.
MISSING_TOKENS = frozenset(
    {"", "-", "--", "n/a", "na", "nan", "none", "null", "unknown", "not reported", "?"}
)


def is_missing(value: object) -> bool:
    """Report whether a raw value means "no data".

    Covers ``None`` and the NaN floats pandas produces for empty cells. The check
    uses ``math.isnan`` rather than pandas' or numpy's null helpers, because this
    module is imported by the Flask app, which has no reason to pull either in.

    Args:
        value: Any raw value.

    Returns:
        True when the value carries no information.
    """
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _clean_text(value: object) -> str | None:
    """Collapse a raw value to a trimmed single-line string, or ``None``.

    Newlines and runs of whitespace are collapsed to single spaces so that one
    template field always occupies exactly one line, no matter what a user
    pasted into the comments box.

    Args:
        value: Any raw value from JSON, a database row, or an HTML form field.

    Returns:
        The cleaned string, or ``None`` when the value means "missing".
    """
    if is_missing(value):
        return None
    text = " ".join(str(value).split())
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def _clean_number(value: object, field: str) -> float | None:
    """Parse a scalar field and reject implausible values.

    Args:
        value: Raw value, which may be a float, an int, or a form string.
        field: Canonical field name, used to look up its plausible range.

    Returns:
        The parsed float, or ``None`` when the value is missing, unparseable, or
        outside the range in :data:`SANITY_RANGES`.
    """
    text = _clean_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    # float() accepts "inf" and "nan" as literal strings, so a user typing either
    # into the form would otherwise reach the range check as a non-finite value.
    if not math.isfinite(number):
        return None
    low, high = SANITY_RANGES[field]
    if not low <= number <= high:
        return None
    return number


def _first_present(raw: dict, keys: tuple[str, ...]) -> object:
    """Return the first value among ``keys`` that is present and non-missing.

    Args:
        raw: Source mapping.
        keys: Candidate key names, in priority order.

    Returns:
        The first usable raw value, or ``None`` if none of the keys carry one.
    """
    for key in keys:
        if key in raw and _clean_text(raw[key]) is not None:
            return raw[key]
    return None


def _resolve_term(raw: dict) -> str | None:
    """Build the application term, from ``term`` directly or ``semester``+``year``.

    The scraped JSON stores the term as two fields (``semester`` and ``year``),
    the Postgres table stores it as one (``term``), and the web form collects it
    as two dropdown/text inputs. All three end up as ``"Fall 2026"``.

    Args:
        raw: Source mapping.

    Returns:
        The term string, or ``None`` when neither form is available.
    """
    direct = _clean_text(raw.get("term"))
    if direct is not None:
        return direct
    semester = _clean_text(raw.get("semester"))
    year = _clean_text(raw.get("year"))
    # Year arrives as "2026" from JSON but may arrive as 2026.0 from a float
    # column, so strip a trailing ".0" before joining.
    if year is not None and year.endswith(".0"):
        year = year[:-2]
    parts = [part for part in (semester, year) if part is not None]
    if not parts:
        return None
    return " ".join(parts)


def normalize_record(raw: dict) -> dict:
    """Reduce any applicant mapping to the ten canonical model fields.

    Missing values become ``None``, scalars become plausibility-checked floats,
    and the term is assembled from whichever representation the source uses.
    The function is idempotent: normalizing an already-normalized record returns
    an equal record.

    Args:
        raw: Applicant mapping from the scraped JSON, the Postgres
            ``applicants`` table, or a submitted HTML form.

    Returns:
        A dict with exactly the keys in :data:`MODEL_FIELDS`, whose values are
        ``str``, ``float``, or ``None``.
    """
    record: dict = {}
    for field in TEXT_FIELDS + ("degree", "us_or_international"):
        record[field] = _clean_text(_first_present(raw, FIELD_ALIASES[field]))
    record["term"] = _resolve_term(raw)
    for field in NUMERIC_FIELDS:
        record[field] = _clean_number(_first_present(raw, FIELD_ALIASES[field]), field)
    return record


def format_field(field: str, value: object) -> str:
    """Render one field's value exactly as the template writes it.

    Args:
        field: Canonical field name.
        value: Normalized value, possibly ``None``.

    Returns:
        The rendered string, or :data:`MISSING_PLACEHOLDER` when the value is
        absent.
    """
    # The NaN check is load-bearing, not defensive: values arriving straight off a
    # pandas DataFrame row are NaN rather than None, and without this a missing
    # GPA would render as "nan" during training while the web form rendered
    # "Unknown" for the same applicant.
    if is_missing(value):
        return MISSING_PLACEHOLDER
    if field in NUMERIC_FORMATS:
        return NUMERIC_FORMATS[field].format(float(value))
    return str(value)


def build_applicant_text(record: dict) -> str:
    """Render a normalized record as the model's input string.

    Args:
        record: Output of :func:`normalize_record`.

    Returns:
        A newline-joined block of ``Label: value`` lines in
        :data:`TEMPLATE_LINES` order, with no trailing newline.
    """
    return "\n".join(
        f"{label}: {format_field(field, record.get(field))}" for label, field in TEMPLATE_LINES
    )


def applicant_text_from_raw(raw: dict) -> str:
    """Normalize a raw mapping and render it in one step.

    Args:
        raw: Applicant mapping in any of the accepted source shapes.

    Returns:
        The model input string.
    """
    return build_applicant_text(normalize_record(raw))


def without_comments(text: str) -> str:
    """Blank the comments line of an already-rendered model input.

    Used for the comments ablation in the evaluation section: re-scoring the
    test set with this transformation applied measures how much the free-text
    field contributes, without retraining anything. It relies on ``Comments``
    being the final template line.

    Args:
        text: A string produced by :func:`build_applicant_text`.

    Returns:
        The same text with the comments value replaced by the missing
        placeholder.
    """
    prefix = f"\n{COMMENTS_LABEL}: "
    head, separator, _ = text.rpartition(prefix)
    if not separator:
        return text
    return f"{head}{prefix}{MISSING_PLACEHOLDER}"


def template_skeleton() -> str:
    """Return the blank template, for printing in the training log and write-up.

    Returns:
        The template with ``{field}`` placeholders instead of values.
    """
    return "\n".join(f"{label}: {{{field}}}" for label, field in TEMPLATE_LINES)


def field_inventory() -> list[tuple[str, str, str]]:
    """Describe every model field, for the training log's field listing.

    Returns:
        A list of ``(field, kind, why it is used)`` tuples in template order.
    """
    reasons = {
        "program": "the applicant's own words for what they applied to; the strongest text signal",
        "university": "institution selectivity differs enormously and the name encodes it",
        "comments": "free text where applicants volunteer research, funding, and fit details",
        "term": "admissions cycles differ in competitiveness, so the cycle is a real covariate",
        "degree": "Masters and PhD acceptance rates differ by more than 40 points",
        "us_or_international": "domestic and international pools are evaluated against each other",
        "gpa": "the most widely reported academic scalar in the dataset",
        "gre": "quantitative score, sparse but highly informative where present",
        "gre_v": "verbal score, matters more for humanities programs",
        "gre_aw": "analytical writing score, the weakest of the three but still reported",
    }
    kinds = {field: "text" for field in TEXT_FIELDS}
    kinds.update({field: "categorical" for field in CATEGORICAL_FIELDS})
    kinds.update({field: "numeric" for field in NUMERIC_FIELDS})
    return [(field, kinds[field], reasons[field]) for _, field in TEMPLATE_LINES]
