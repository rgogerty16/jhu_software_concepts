"""app package: the Grad Café Flask site, extended with an admissions predictor.

This is the Module 5 analysis website with a second page added: **Will You Get
In?**, which collects an applicant profile and scores it with the DistilBERT
model fine-tuned in ``train_model.py``.

Routes:

===========================  ==================================================
``GET  /``                   redirects to ``/analysis``
``GET  /analysis``           the Module 5 SQL analysis page
``GET  /will-you-get-in``    the blank prediction form
``POST /will-you-get-in``    validate, render the model input, and predict
``GET  /healthz``            JSON liveness probe, reports model readiness
===========================  ==================================================

Three things about how this is wired are deliberate:

* **The model loads once, at startup.** ``create_app()`` warms
  ``inference.load_bundle()``, which is memoized, so no request ever reads
  weights from disk and nothing is ever retrained on a page view.
* **Nothing about the model is trusted to be present.** A missing or broken
  ``model/`` directory leaves the site up: the form still renders and explains
  what to run. The same is true of Postgres for the analysis page.
* **The applicant text is built by the training code.** The route calls the same
  ``applicant_text`` helpers ``train_model.py`` used, so a form submission is
  encoded byte-for-byte the way a training row was.

The Module 5 scraper machinery (``/pull-data``, ``/pull-status``,
``/update-analysis``, and the Selenium subprocess behind them) is deliberately
not carried over, because this assignment never exercises it, and dropping it keeps
Selenium, BeautifulSoup, and webdriver-manager out of ``requirements.txt``.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

# The project modules live one level up, beside run.py, and are imported by the
# same names train_model.py uses so there is exactly one copy of the template
# logic in the repository.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import applicant_text  # pylint: disable=wrong-import-position
import inference  # pylint: disable=wrong-import-position

from .query_data import run_queries  # pylint: disable=wrong-import-position

#: Shown on the prediction page, and required by the assignment.
DISCLAIMER = (
    "This is a course project for JHU EN.605.256, not an admissions tool. The model "
    "was fine-tuned on roughly 19,000 self-reported Grad Café posts, which are "
    "voluntarily submitted, unverified, and not a representative sample of "
    "applicants. Its output is a pattern match against that data and is not an "
    "admissions decision, a prediction of one, or advice about where to apply. No "
    "admissions office has any connection to this page."
)

#: Numeric form fields, with the range each is validated against and the label
#: used in error messages. The ranges are the same ones ``applicant_text`` applies
#: to the training data, so the form cannot accept a value training would reject.
NUMERIC_FORM_FIELDS = (
    ("gpa", "GPA", 0.0, 5.0),
    ("gre", "GRE Quant", 130.0, 170.0),
    ("gre_v", "GRE Verbal", 130.0, 170.0),
    ("gre_aw", "GRE Analytical Writing", 0.0, 6.0),
)

#: Free-text form fields.
TEXT_FORM_FIELDS = ("program", "university", "comments")

#: Dropdown choices. "Unknown" is a first-class option, because the model was
#: trained on rows where these were genuinely missing and renders them with the
#: same placeholder.
DEGREE_CHOICES = ("Unknown", "Masters", "PhD", "MFA", "PsyD", "EdD", "JD", "MBA", "Other")
CITIZENSHIP_CHOICES = ("Unknown", "American", "International", "Other")
SEMESTER_CHOICES = ("Unknown", "Fall", "Spring", "Summer", "Winter")

#: Every form field name, used to echo the user's input back on a validation error.
FORM_FIELD_NAMES = (
    TEXT_FORM_FIELDS + ("semester", "year", "degree", "us_or_international")
    + tuple(name for name, _, _, _ in NUMERIC_FORM_FIELDS)
)


def parse_year(raw: str, errors: list[str]) -> str | None:
    """Validate the application year.

    :param raw: Raw submitted value.
    :param errors: List that a message is appended to when validation fails.
    :returns: The year as a string, or None when blank or invalid.
    """
    text = raw.strip()
    if not text:
        return None
    if not text.isdigit() or not 1990 <= int(text) <= 2100:
        errors.append("Year should be a four-digit year such as 2026, or left blank.")
        return None
    return text


def parse_numbers(form, errors: list[str]) -> dict[str, float | None]:
    """Validate the four scalar fields.

    Blank is always acceptable and means "not reported", and the model was trained on
    a dataset where most rows were missing at least one of these. Anything that is
    neither blank nor a plausible number produces a message naming the field and
    its range, and the form is re-rendered rather than the request failing.

    :param form: The submitted form mapping.
    :param errors: List that messages are appended to.
    :returns: A mapping of field name to parsed value or None.
    """
    values: dict[str, float | None] = {}
    for field, label, low, high in NUMERIC_FORM_FIELDS:
        text = form.get(field, "").strip()
        if not text:
            values[field] = None
            continue
        try:
            number = float(text)
        except ValueError:
            errors.append(f"{label} should be a number such as {low + (high - low) / 2:.1f}, "
                          f"or left blank.")
            values[field] = None
            continue
        if not low <= number <= high:
            errors.append(f"{label} should be between {low:g} and {high:g}. "
                          f"You entered {number:g}.")
            values[field] = None
            continue
        values[field] = number
    return values


def parse_form(form) -> tuple[dict, list[str], dict[str, str]]:
    """Turn a submitted form into an applicant record, collecting any problems.

    :param form: The submitted form mapping.
    :returns: A tuple of the applicant record, the list of validation messages,
        and the raw values echoed back so the user does not have to retype.
    """
    errors: list[str] = []
    echo = {name: form.get(name, "").strip() for name in FORM_FIELD_NAMES}

    record: dict = {field: form.get(field, "").strip() for field in TEXT_FORM_FIELDS}
    record["degree"] = form.get("degree", "").strip()
    record["us_or_international"] = form.get("us_or_international", "").strip()
    record["semester"] = form.get("semester", "").strip()
    record["year"] = parse_year(form.get("year", ""), errors)
    record.update(parse_numbers(form, errors))
    return record, errors, echo


def create_app(config: dict | None = None) -> Flask:
    """Build and return the configured Flask application.

    :param config: Optional config overrides, applied last so they win. Tests use
        this to point at a throwaway database or a different model directory.
    :type config: dict or None
    :returns: A configured Flask application.
    :rtype: flask.Flask
    """
    app = Flask(__name__)
    app.config["DATABASE_URL"] = None
    app.config["MODEL_DIR"] = inference.DEFAULT_MODEL_DIR
    if config:
        app.config.update(config)

    # Warm the model at startup rather than on first request. load_bundle is
    # memoized, so this is what guarantees the page never reloads weights or
    # retrains, and it surfaces a missing model as a banner instead of a 500.
    try:
        inference.load_bundle(app.config["MODEL_DIR"])
        app.config["MODEL_READY"] = True
        app.config["MODEL_ERROR"] = None
    except inference.ModelNotAvailableError as error:
        app.config["MODEL_READY"] = False
        app.config["MODEL_ERROR"] = str(error)

    def render_form(**context):
        """Render the prediction page with sensible defaults for every variable.

        :returns: The rendered prediction page.
        :rtype: str
        """
        defaults = {
            "active_page": "predict",
            "disclaimer": DISCLAIMER,
            "degree_choices": DEGREE_CHOICES,
            "citizenship_choices": CITIZENSHIP_CHOICES,
            "semester_choices": SEMESTER_CHOICES,
            "numeric_fields": NUMERIC_FORM_FIELDS,
            "model_ready": app.config["MODEL_READY"],
            "model_error": app.config["MODEL_ERROR"],
            "errors": [],
            "form": {name: "" for name in FORM_FIELD_NAMES},
            "prediction": None,
        }
        defaults.update(context)
        return render_template("predict.html", **defaults)

    @app.get("/analysis")
    def analysis():
        """Render the SQL analysis page, degrading gracefully without Postgres.

        :returns: The rendered analysis page.
        :rtype: str
        """
        results, db_error = None, None
        try:
            results = run_queries(database_url=app.config["DATABASE_URL"])
        except Exception as error:  # pylint: disable=broad-except
            # Any database problem, whether the server is down, the table is
            # missing, or the credentials are wrong, must leave the site usable
            # and must not put a traceback in front of a user. The message names
            # the cause without echoing connection details back to the browser.
            db_error = type(error).__name__
            app.logger.warning("Analysis queries failed: %s: %s", db_error, error)
        return render_template(
            "index.html",
            results=results,
            db_error=db_error,
            active_page="analysis",
            now=datetime.now(),
        )

    @app.get("/")
    def root():
        """Redirect the bare root to the analysis page.

        :returns: A redirect response.
        :rtype: flask.Response
        """
        return redirect(url_for("analysis"))

    @app.get("/will-you-get-in")
    def will_you_get_in():
        """Render the blank prediction form.

        :returns: The rendered prediction page.
        :rtype: str
        """
        return render_form()

    @app.post("/will-you-get-in")
    def submit_prediction():
        """Validate the submitted profile and score it with the fine-tuned model.

        :returns: The prediction page, showing either the result or the problems
            that stopped it being produced.
        :rtype: str
        """
        record, errors, echo = parse_form(request.form)
        if errors:
            return render_form(errors=errors, form=echo)
        if not app.config["MODEL_READY"]:
            return render_form(form=echo)

        try:
            bundle = inference.load_bundle(app.config["MODEL_DIR"])
            prediction = inference.predict_applicant(record, bundle)
        except (inference.ModelNotAvailableError, RuntimeError, ValueError) as error:
            app.logger.exception("Prediction failed")
            return render_form(
                form=echo,
                errors=[f"The model could not score this profile ({type(error).__name__}). "
                        f"Please try again."],
            )

        return render_form(
            form=echo,
            prediction={
                "label": prediction.label,
                "score": prediction.score,
                "accepted_probability": prediction.accepted_probability,
                "accepted_percent": round(prediction.accepted_probability * 100, 1),
                "probabilities": prediction.probabilities,
                "model_input_text": prediction.model_input_text,
                "template_version": applicant_text.TEMPLATE_VERSION,
            },
        )

    @app.get("/healthz")
    def healthz():
        """Report liveness and whether the model is loaded.

        :returns: A small JSON status document.
        :rtype: flask.Response
        """
        return jsonify({"ok": True, "model_ready": app.config["MODEL_READY"]})

    return app
