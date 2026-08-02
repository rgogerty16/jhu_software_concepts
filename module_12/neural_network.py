"""Module 12 - a two-layer neural network for graduate-admissions prediction.

This script builds an end-to-end binary classifier that predicts whether a
graduate-school applicant was *Accepted* or *Rejected* from six features:
GPA, GRE Quantitative, GRE Verbal, GRE Analytical Writing, Masters-vs-PhD, and
International-vs-Local.

The network itself is written with NumPy only - no PyTorch, TensorFlow, Keras,
JAX, or scikit-learn neural-network utilities. scikit-learn is used for exactly
one thing: the train/test split.

Run it with::

    python neural_network.py                     # uses ./applicant_data.json
    python neural_network.py --data other.jsonl  # any JSON Lines / JSON array file
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Fixed configuration required by the assignment specification.
# --------------------------------------------------------------------------- #
DEFAULT_DATA_PATH = Path(__file__).with_name("applicant_data.json")

# The six model inputs, in the exact order the network expects them.
FEATURE_COLUMNS = [
    "gpa",
    "gre",
    "gre_v",
    "gre_aw",
    "ms_vs_phd",
    "international_vs_local",
]

# Columns that arrive as strings (or nulls) and must become floats.
NUMERIC_COLUMNS = ["gpa", "gre", "gre_v", "gre_aw"]

# The Grad Cafe corpus carried forward from Module 2 names three fields
# differently than the assignment does. Mapping them here keeps the rest of the
# pipeline written against the assignment's vocabulary, and lets the same script
# read either file without edits.
COLUMN_ALIASES = {
    "status": "applicant_status",
    "degree": "masters_or_phd",
    "student_type": "citizenship",
}

# Only these outcome / degree values survive filtering.
KEPT_STATUSES = ("Accepted", "Rejected")
KEPT_DEGREES = ("Masters", "PhD")


def print_banner(title):
    """Print a section header so the training log is easy to navigate.

    Args:
        title: Text of the section heading.
    """
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Section 1 - load and prepare the applicant dataset
# --------------------------------------------------------------------------- #
def load_applicant_records(data_path):
    """Load applicant records from a JSON Lines file into a list of dicts.

    The assignment supplies the applicants as JSON Lines, where each line is a
    separate JSON object. The Grad Cafe dataset produced in Module 2 is instead
    a single pretty-printed JSON array, so a file that begins with ``[`` is
    parsed as one array. Both shapes yield the same list of records.

    Args:
        data_path: Path to the JSON Lines (or JSON array) applicant file.

    Returns:
        A list of dictionaries, one per applicant record.
    """
    raw_text = data_path.read_text(encoding="utf-8")

    if raw_text.lstrip().startswith("["):
        return json.loads(raw_text)

    return [json.loads(line) for line in raw_text.splitlines() if line.strip()]


def build_dataframe(records):
    """Turn raw applicant records into a filtered, fully numeric DataFrame.

    Preprocessing happens in the order the assignment prescribes: filter on
    outcome, filter on degree, convert the string-valued numeric columns to
    floats, then build the two binary features and the target.

    Args:
        records: List of applicant dictionaries from :func:`load_applicant_records`.

    Returns:
        A DataFrame containing the six feature columns plus ``target``, along
        with the original text columns for reference.
    """
    frame = pd.DataFrame(records)

    # Rename Grad Cafe column names to the assignment's names, but never
    # clobber a column the file already provides under the expected name.
    renames = {old: new for old, new in COLUMN_ALIASES.items()
               if old in frame.columns and new not in frame.columns}
    frame = frame.rename(columns=renames)

    # Keep only decided outcomes (drop Waitlisted / Interview) and only the two
    # degree types the model is meant to distinguish.
    frame = frame[frame["applicant_status"].isin(KEPT_STATUSES)]
    frame = frame[frame["masters_or_phd"].isin(KEPT_DEGREES)].copy()

    # Convert the numeric columns from strings to floats. Values that cannot be
    # parsed become NaN and are filled later with training-set medians.
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    # Binary features: PhD = 1 / Masters = 0, International = 1 / Local = 0.
    frame["ms_vs_phd"] = (frame["masters_or_phd"] == "PhD").astype(float)
    frame["international_vs_local"] = (frame["citizenship"] == "International").astype(float)

    # Target: Accepted = 1, Rejected = 0.
    frame["target"] = (frame["applicant_status"] == "Accepted").astype(float)

    return frame


def report_dataset(frame, original_row_count):
    """Print the Section 1 summary of the cleaned dataset.

    Args:
        frame: The cleaned DataFrame from :func:`build_dataframe`.
        original_row_count: Number of records before any filtering.
    """
    print_banner("SECTION 1 - DATASET LOADING, FILTERING, AND FEATURE CONSTRUCTION")
    print(f"Rows in the original dataset        : {original_row_count:,}")
    print(f"Rows remaining after filtering      : {len(frame):,}")
    print(f"Accepted rows                       : {int((frame['target'] == 1).sum()):,}")
    print(f"Rejected rows                       : {int((frame['target'] == 0).sum()):,}")
    feature_list = ", ".join(FEATURE_COLUMNS)
    print(f"Final input features ({len(FEATURE_COLUMNS)})            : {feature_list}")
    print()
    print("First five rows of the cleaned dataframe:")
    print(frame[FEATURE_COLUMNS + ["target"]].head().to_string(index=False))


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the applicant dataset (JSON Lines or JSON array).",
    )
    return parser.parse_args()


def main():
    """Run the full assignment pipeline end to end."""
    arguments = parse_arguments()

    # Section 1 - load, filter, and engineer features.
    records = load_applicant_records(arguments.data)
    frame = build_dataframe(records)
    report_dataset(frame, len(records))


if __name__ == "__main__":
    main()
