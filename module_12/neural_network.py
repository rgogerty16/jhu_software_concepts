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
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Fixed configuration required by the assignment specification.
# --------------------------------------------------------------------------- #
DEFAULT_DATA_PATH = Path(__file__).with_name("applicant_data.json")

RANDOM_SEED = 42
TEST_SIZE = 0.2

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


# --------------------------------------------------------------------------- #
# Section 2 - split the data and preprocess it without leaking test information
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainingStatistics:
    """Per-feature statistics learned from the training set only.

    The same three vectors are reused to preprocess the test set and, later, the
    artificial applicants, so that every input the model ever sees is measured
    on one consistent scale.

    Attributes:
        medians: Median of each feature, used to fill missing values.
        means: Mean of each feature after missing values were filled.
        standard_deviations: Standard deviation of each feature, with any zero
            replaced by 1 so that scaling never divides by zero.
    """

    medians: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray


def split_dataset(frame):
    """Split the cleaned data into 80% training and 20% testing matrices.

    This is the only place scikit-learn is used; the network itself is pure
    NumPy.

    Args:
        frame: The cleaned DataFrame from :func:`build_dataframe`.

    Returns:
        Tuple ``(x_train, x_test, y_train, y_test)`` of NumPy arrays. The target
        arrays are column vectors of shape ``(n_samples, 1)`` so they line up
        with the network's single output unit.
    """
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = frame["target"].to_numpy(dtype=float).reshape(-1, 1)

    return train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )


def fill_missing_values(matrix, medians):
    """Replace NaNs with the supplied per-column medians.

    Args:
        matrix: Feature matrix of shape ``(n_samples, n_features)``.
        medians: Median of each feature, taken from the training set.

    Returns:
        A new matrix with no missing values.
    """
    return np.where(np.isnan(matrix), medians, matrix)


def fit_training_statistics(x_train):
    """Learn medians, means, and standard deviations from the training set.

    Args:
        x_train: Raw (unfilled, unscaled) training feature matrix.

    Returns:
        A :class:`TrainingStatistics` instance.
    """
    medians = np.nanmedian(x_train, axis=0)

    # Means and standard deviations are measured after filling, so that the
    # imputed rows are described by the same statistics used to scale them.
    filled_train = fill_missing_values(x_train, medians)
    means = filled_train.mean(axis=0)
    standard_deviations = filled_train.std(axis=0)

    # A constant feature would otherwise divide by zero; the assignment asks for
    # a standard deviation of 1 in that case, which leaves the column unscaled.
    standard_deviations[standard_deviations == 0.0] = 1.0

    return TrainingStatistics(medians, means, standard_deviations)


def preprocess_features(matrix, statistics):
    """Fill missing values and standardize a feature matrix.

    Args:
        matrix: Raw feature matrix, possibly containing NaNs.
        statistics: Training-set statistics from :func:`fit_training_statistics`.

    Returns:
        The filled and standardized matrix.
    """
    filled = fill_missing_values(matrix, statistics.medians)
    return (filled - statistics.means) / statistics.standard_deviations


def report_split(x_train, x_test, y_train, y_test, statistics):
    """Print the Section 2 summary of the split and its preprocessing.

    Args:
        x_train: Training feature matrix.
        x_test: Test feature matrix.
        y_train: Training targets.
        y_test: Test targets.
        statistics: Training-set statistics used to preprocess both matrices.
    """
    print_banner("SECTION 2 - TRAIN/TEST SPLIT AND LEAKAGE-SAFE PREPROCESSING")
    print(f"Training set size : {len(x_train):,} rows ({1 - TEST_SIZE:.0%})")
    print(f"Test set size     : {len(x_test):,} rows ({TEST_SIZE:.0%})")
    print(f"Split settings    : test_size={TEST_SIZE}, "
          f"random_state={RANDOM_SEED}, shuffle=True")
    print(f"Accepted share    : {y_train.mean():.4f} in train, "
          f"{y_test.mean():.4f} in test")
    print()
    print("Training-set statistics (computed from the 80% training split only):")
    print(f"{'feature':<24}{'median':>12}{'mean':>12}{'std':>12}")
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        print(f"{feature_name:<24}"
              f"{statistics.medians[index]:>12.4f}"
              f"{statistics.means[index]:>12.4f}"
              f"{statistics.standard_deviations[index]:>12.4f}")
    print()
    print("Why these statistics come from the training set only:")
    print("  The test set stands in for applicants the model has never seen. If")
    print("  the medians, means, and standard deviations were computed over the")
    print("  full dataset, every test row would have contributed to the numbers")
    print("  used to fill and scale the training rows - that is data leakage.")
    print("  The reported test score would then be optimistic, because part of")
    print("  the test set's information reached the model during training.")
    print("  Fitting on the training split alone keeps the test set a genuinely")
    print("  held-out estimate of performance on future applicants.")


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

    # Section 2 - split first, then learn the preprocessing on the training half.
    x_train_raw, x_test_raw, y_train, y_test = split_dataset(frame)
    statistics = fit_training_statistics(x_train_raw)
    x_train = preprocess_features(x_train_raw, statistics)
    x_test = preprocess_features(x_test_raw, statistics)
    report_split(x_train, x_test, y_train, y_test, statistics)


if __name__ == "__main__":
    main()
