"""leakage_analysis.py: how much of the model's skill is reading the answer?

Section 5 of ``train_model.py`` shows that keeping the comments field is worth
about 8 points of accuracy. This script asks a harder question about *why*, and it
is the most important number in the project.

Grad Café comments are written **after** a decision arrives, so many of them
describe the outcome rather than the applicant: "Rejected on May 26", "Accepted off
of waitlist", "Just got the rejection email". A model given that text is not
predicting an admissions decision, it is reading one. But a prospective applicant
typing into the "Will You Get In?" form cannot describe an outcome they do not have
yet, so any skill that comes from outcome-revealing language will not transfer to
the deployed page.

This measures the size of that effect without retraining anything. It rebuilds the
same seeded split, splits the test fold by whether the comment contains
outcome-revealing language, and scores the saved model on each part. The gap between
those two numbers is the honest correction to apply to the headline accuracy.

Nothing here changes the model. It is a diagnostic, run after training::

    python leakage_analysis.py
"""

from __future__ import annotations

import re

import numpy as np
import transformers

import train_model as tm
from inference import load_bundle

# Keep the shard-loading progress bar out of the saved transcript.
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

#: Language that describes an admissions *outcome* rather than an applicant's
#: qualifications. Deliberately broad: the goal is to isolate a subset of test rows
#: that is confidently free of outcome language, so over-matching is the safe
#: direction to err in.
OUTCOME_PATTERNS = {
    "rejection": r"\breject|\bdenie|\bdeclin|\bturned down",
    "acceptance": r"\baccept|\badmit|\badmiss|\bgot in\b|\bi'm in\b|\bim in\b",
    "offer": r"\boffer|\bfunded|\bfunding\b|\bstipend|\bfellowship|\bwaive|\bassistantship",
    "waitlist": r"\bwait\s?list",
    "interview": r"\binterview",
    "notification": r"\bportal\b|\bemail\b|\bletter\b|\bheard back\b|\bnotified\b|\bdecision",
}

#: Combined pattern used to classify a single comment.
ANY_OUTCOME_PATTERN = re.compile("|".join(OUTCOME_PATTERNS.values()), re.IGNORECASE)


def mentions_outcome(comment: object) -> bool:
    """Report whether a comment contains outcome-revealing language.

    Args:
        comment: A comment string, or a missing value.

    Returns:
        True when the comment appears to describe an admissions outcome.
    """
    if not isinstance(comment, str) or not comment.strip():
        return False
    return bool(ANY_OUTCOME_PATTERN.search(comment))


def report_corpus_prevalence(frame) -> None:
    """Print how often each kind of outcome language appears, and its skew.

    A pattern whose accepted rate differs sharply from the base rate is carrying
    outcome information rather than applicant information.

    Args:
        frame: The full modelling DataFrame.
    """
    commented = frame[frame["has_comment"]]
    comments = commented["comments"].str.lower()
    base = commented["label"].mean()
    tm.print_banner("OUTCOME LANGUAGE IN THE COMMENTS CORPUS")
    print(f"Rows with a comment: {len(commented):,} of {len(frame):,}")
    print(f"Accepted rate among commented rows: {base:.4f}")
    print()
    print(f"{'pattern':<16}{'rows':>8}{'share':>9}{'accepted rate':>16}{'skew':>8}")
    print("-" * 57)
    for name, pattern in OUTCOME_PATTERNS.items():
        hit = comments.str.contains(pattern, regex=True, na=False)
        rate = commented.loc[hit, "label"].mean() if hit.any() else float("nan")
        print(f"{name:<16}{hit.sum():>8,}{hit.mean():>9.1%}{rate:>16.4f}{rate - base:>+8.3f}")
    any_hit = commented["comments"].map(mentions_outcome)
    print("-" * 57)
    print(f"{'ANY':<16}{any_hit.sum():>8,}{any_hit.mean():>9.1%}")


def report_deployment_estimate(split, bundle, config) -> None:
    """Score the saved model separately on leaking and non-leaking test rows.

    Args:
        split: The reproduced train/test split.
        bundle: The loaded inference bundle.
        config: Resolved run configuration.
    """
    _, eval_loader = tm.build_loaders(split, bundle.tokenizer, config)
    evaluation = tm.evaluate_model(bundle.model, eval_loader, bundle.device)

    leaks = split.test["comments"].map(mentions_outcome).to_numpy()
    no_comment = ~split.test["has_comment"].to_numpy()
    groups = (
        ("all test rows", np.ones(len(leaks), dtype=bool)),
        ("comment reveals an outcome", leaks),
        ("comment, no outcome language", split.test["has_comment"].to_numpy() & ~leaks),
        ("no comment at all", no_comment),
        ("no outcome language available", ~leaks),
    )

    tm.print_banner("MODEL ACCURACY WITH AND WITHOUT OUTCOME LANGUAGE")
    print(f"{'test subset':<32}{'n':>7}{'accuracy':>11}{'F1':>9}{'baseline':>11}")
    print("-" * 70)
    results = {}
    for name, mask in groups:
        metrics = tm.evaluate_subset(evaluation, mask)
        if metrics is None:
            continue
        results[name] = metrics
        print(f"{name:<32}{metrics['count']:>7,}{metrics['accuracy']:>11.4f}"
              f"{metrics['f1']:>9.4f}{metrics['majority_baseline']:>11.4f}")

    headline = results["all test rows"]["accuracy"]
    clean = results["no outcome language available"]["accuracy"]
    leaking = results["comment reveals an outcome"]["accuracy"]
    print()
    tm.print_paragraph(
        f"The headline accuracy of {headline:.4f} is measured on a test fold where "
        f"{leaks.mean():.1%} of rows carry a comment describing the outcome. On those "
        f"rows the model scores {leaking:.4f}. On rows with no outcome language "
        f"available at all - the situation every user of the web form is in, since "
        f"nobody can describe a decision they have not received - it scores "
        f"{clean:.4f}, which is {clean - headline:+.4f} against the headline."
    )
    print()
    tm.print_paragraph(
        f"That second number, {clean:.4f}, is the honest estimate of deployed "
        f"performance, and it is the one the write-up leads with when discussing what "
        f"this system would actually do for a real applicant. It is still well above "
        f"the {results['no outcome language available']['majority_baseline']:.4f} "
        f"majority-class baseline, so the model has learned something real about "
        f"programs, degrees, and scores. But the gap is a reminder that a held-out "
        f"test set only protects against memorizing rows, not against a feature that "
        f"encodes the answer."
    )


def main() -> None:
    """Reproduce the split, load the saved model, and report both analyses."""
    config = tm.parse_arguments([])
    frame, _ = tm.build_dataframe(tm.load_records(config), config)
    frame = tm.add_model_text(frame)
    split = tm.split_dataset(frame)
    bundle = load_bundle(config.paths.model_dir)

    report_corpus_prevalence(frame)
    report_deployment_estimate(split, bundle, config)
    print()
    print("This script changes nothing. It only measures the saved model.")


if __name__ == "__main__":
    main()
