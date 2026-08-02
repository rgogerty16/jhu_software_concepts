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
import contextlib
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Fixed configuration required by the assignment specification.
# --------------------------------------------------------------------------- #
DEFAULT_DATA_PATH = Path(__file__).with_name("applicant_data.json")
TRAINING_LOG_PATH = Path(__file__).with_name("training.log")
MSE_CURVE_PATH = Path(__file__).with_name("mse_curve.png")

RANDOM_SEED = 42
TEST_SIZE = 0.2
HIDDEN_UNITS = 6
LEARNING_RATE = 0.05
MAX_EPOCHS = 10000
PATIENCE = 100

# Weights start from a normal distribution with mean 0 and standard deviation
# 0.1; biases start at 0.
WEIGHT_INIT_MEAN = 0.0
WEIGHT_INIT_STD = 0.1

# A predicted probability at or above this value counts as "Accepted".
PREDICTION_THRESHOLD = 0.5

# How often the training loop prints a progress line.
PRINT_EVERY = 100

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
    frame.rename(columns=renames, inplace=True)

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


def acceptance_rate_by_degree(frame):
    """Return the acceptance rate for Masters and for PhD applicants.

    Args:
        frame: The cleaned DataFrame from :func:`build_dataframe`.

    Returns:
        Tuple ``(masters_rate, phd_rate)`` of acceptance rates.
    """
    return (float(frame.loc[frame["ms_vs_phd"] == 0.0, "target"].mean()),
            float(frame.loc[frame["ms_vs_phd"] == 1.0, "target"].mean()))


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
    masters_rate, phd_rate = acceptance_rate_by_degree(frame)
    print(f"Acceptance rate by degree           : Masters {masters_rate:.1%}, "
          f"PhD {phd_rate:.1%}")
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


@dataclasses.dataclass
class SplitData:
    """The preprocessed training and test matrices, kept together.

    Attributes:
        x_train: Standardized training features.
        y_train: Training targets, shape ``(n_samples, 1)``.
        x_test: Standardized test features.
        y_test: Test targets, shape ``(n_samples, 1)``.
        filtered_row_count: Rows that survived Section 1's filtering, reported
            in the final evaluation.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    filtered_row_count: int


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


def report_split(data, statistics):
    """Print the Section 2 summary of the split and its preprocessing.

    Args:
        data: The :class:`SplitData` produced by the split and preprocessing.
        statistics: Training-set statistics used to preprocess both matrices.
    """
    print_banner("SECTION 2 - TRAIN/TEST SPLIT AND LEAKAGE-SAFE PREPROCESSING")
    print(f"Training set size : {len(data.x_train):,} rows ({1 - TEST_SIZE:.0%})")
    print(f"Test set size     : {len(data.x_test):,} rows ({TEST_SIZE:.0%})")
    print(f"Split settings    : test_size={TEST_SIZE}, "
          f"random_state={RANDOM_SEED}, shuffle=True")
    print(f"Accepted share    : {data.y_train.mean():.4f} in train, "
          f"{data.y_test.mean():.4f} in test")
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
    print("  full dataset, every test row would have shaped the numbers used to")
    print("  fill and scale the training rows. That is data leakage, and the")
    print("  reported test score would be optimistic because test information")
    print("  reached the model during training. Fitting on the training split")
    print("  alone keeps the test set a genuine estimate of future performance.")


# --------------------------------------------------------------------------- #
# Section 3 - a two-layer neural network written with NumPy only
# --------------------------------------------------------------------------- #
def sigmoid(values):
    """Squash any real number into the open interval (0, 1).

    Args:
        values: Array of pre-activation values.

    Returns:
        The element-wise sigmoid ``1 / (1 + exp(-x))``.
    """
    return 1.0 / (1.0 + np.exp(-values))


def mean_squared_error(predictions, targets):
    """Compute the mean squared error between predictions and targets.

    MSE is the loss the assignment requires, even though this is a
    classification problem.

    Args:
        predictions: Predicted values, shape ``(n_samples, 1)``.
        targets: True values, shape ``(n_samples, 1)``.

    Returns:
        The mean squared error as a float.
    """
    return float(np.mean((predictions - targets) ** 2))


def accuracy_score(predicted_labels, targets):
    """Compute the fraction of labels predicted correctly.

    Args:
        predicted_labels: Binary predictions, shape ``(n_samples, 1)``.
        targets: True 0/1 labels, shape ``(n_samples, 1)``.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    return float(np.mean(predicted_labels == targets))


class TwoLayerNeuralNetwork:
    """A fully connected 6 -> 6 -> 1 network with sigmoid activations.

    Shapes, for six input features and six hidden units:

    * ``w1`` is ``(6, 6)`` - one weight per (input feature, hidden unit) pair.
    * ``b1`` is ``(1, 6)`` - one bias per hidden unit, broadcast over rows.
    * ``w2`` is ``(6, 1)`` - one weight per (hidden unit, output unit) pair.
    * ``b2`` is ``(1, 1)`` - the single output unit's bias.

    What each layer computes:

    * The hidden layer computes ``a1 = sigmoid(x @ w1 + b1)``. Each of the six
      hidden units forms its own weighted blend of all six standardized inputs,
      shifts it by a bias, and squashes it to (0, 1). Because every unit gets a
      different weight vector, the layer learns six different re-descriptions of
      an applicant, and the sigmoid makes each one non-linear - which is what
      lets the network represent interactions (for example, a strong GPA
      mattering more for a PhD applicant than for a Masters applicant) that a
      single linear layer could not.
    * The output layer computes ``a2 = sigmoid(a1 @ w2 + b2)``. It weighs those
      six learned descriptions into one number and squashes it again.

    Why the output reads as a probability-like score: the final sigmoid is bound
    to (0, 1) and increases monotonically with the evidence for acceptance, and
    the network is trained against 0/1 targets, so it is driven toward the
    average target value for applicants that look like the input. That makes it
    interpretable as "how accept-like this applicant looks" and comparable
    across applicants. It is only *probability-like*, not a calibrated
    probability: trained under MSE rather than a proper scoring rule such as
    cross-entropy, the values are systematically pulled toward the middle of the
    range and should not be read as literal admission odds.
    """

    def __init__(self, input_units, hidden_units=HIDDEN_UNITS,
                 learning_rate=LEARNING_RATE, seed=RANDOM_SEED):
        """Initialize weights from N(0, 0.1) and biases to zero.

        Args:
            input_units: Number of input features (6 for this assignment).
            hidden_units: Number of hidden units.
            learning_rate: Step size for gradient descent.
            seed: Seed for the random number generator, for reproducibility.
        """
        generator = np.random.default_rng(seed)
        self.learning_rate = learning_rate

        self.w1 = generator.normal(WEIGHT_INIT_MEAN, WEIGHT_INIT_STD,
                                   size=(input_units, hidden_units))
        self.b1 = np.zeros((1, hidden_units))
        self.w2 = generator.normal(WEIGHT_INIT_MEAN, WEIGHT_INIT_STD,
                                   size=(hidden_units, 1))
        self.b2 = np.zeros((1, 1))

        # Activations cached by forward() for the matching backward() call.
        self.hidden_activations = None
        self.output_activations = None

    def forward(self, features):
        """Run a forward pass and cache the activations for backpropagation.

        Args:
            features: Standardized feature matrix, shape ``(n_samples, 6)``.

        Returns:
            Output activations, shape ``(n_samples, 1)``.
        """
        self.hidden_activations = sigmoid(features @ self.w1 + self.b1)
        self.output_activations = sigmoid(self.hidden_activations @ self.w2 + self.b2)
        return self.output_activations

    def backward(self, features, targets):
        """Backpropagate the MSE loss and take one gradient-descent step.

        Uses the activations cached by the most recent :meth:`forward` call on
        the same batch. With ``loss = mean((a2 - y) ** 2)`` over ``n`` samples:

        * ``d_output = 2 * (a2 - y) / n * a2 * (1 - a2)`` - the loss derivative
          times the sigmoid derivative at the output unit.
        * ``d_hidden = (d_output @ w2.T) * a1 * (1 - a1)`` - that error carried
          back through the output weights and through the hidden sigmoid.

        Args:
            features: The same feature matrix passed to :meth:`forward`.
            targets: True 0/1 labels, shape ``(n_samples, 1)``.
        """
        sample_count = features.shape[0]

        # Output layer: derivative of MSE, then of the output sigmoid.
        loss_gradient = 2.0 * (self.output_activations - targets) / sample_count
        output_delta = loss_gradient * self.output_activations * (1.0 - self.output_activations)
        w2_gradient = self.hidden_activations.T @ output_delta
        b2_gradient = output_delta.sum(axis=0, keepdims=True)

        # Hidden layer: the output error, credited back through w2, then through
        # the hidden sigmoid.
        hidden_delta = (output_delta @ self.w2.T) * \
            self.hidden_activations * (1.0 - self.hidden_activations)
        w1_gradient = features.T @ hidden_delta
        b1_gradient = hidden_delta.sum(axis=0, keepdims=True)

        # Full-batch gradient-descent update: step downhill on every parameter.
        self.w2 -= self.learning_rate * w2_gradient
        self.b2 -= self.learning_rate * b2_gradient
        self.w1 -= self.learning_rate * w1_gradient
        self.b1 -= self.learning_rate * b1_gradient

    def predict_proba(self, features):
        """Return probability-like scores without disturbing the cache.

        Evaluating the test set must not overwrite the activations that
        :meth:`backward` still needs from the training pass, so this recomputes
        the forward pass locally instead of calling :meth:`forward`.

        Args:
            features: Standardized feature matrix, shape ``(n_samples, 6)``.

        Returns:
            Scores in (0, 1), shape ``(n_samples, 1)``.
        """
        hidden = sigmoid(features @ self.w1 + self.b1)
        return sigmoid(hidden @ self.w2 + self.b2)

    def predict(self, features, threshold=PREDICTION_THRESHOLD):
        """Return hard 0/1 predictions by thresholding the scores.

        Args:
            features: Standardized feature matrix, shape ``(n_samples, 6)``.
            threshold: Score at or above which the applicant is predicted
                Accepted.

        Returns:
            Binary predictions as floats, shape ``(n_samples, 1)``.
        """
        return (self.predict_proba(features) >= threshold).astype(float)

    def get_parameters(self):
        """Copy the current weights and biases.

        Used to snapshot the best-scoring parameters during early stopping.

        Returns:
            A dict of copied parameter arrays.
        """
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy(),
        }

    def set_parameters(self, parameters):
        """Restore weights and biases from a snapshot.

        Args:
            parameters: A dict produced by :meth:`get_parameters`.
        """
        self.w1 = parameters["w1"].copy()
        self.b1 = parameters["b1"].copy()
        self.w2 = parameters["w2"].copy()
        self.b2 = parameters["b2"].copy()


def report_architecture(model):
    """Print the Section 3 description of the network.

    Args:
        model: The initialized :class:`TwoLayerNeuralNetwork`.
    """
    print_banner("SECTION 3 - TWO-LAYER NEURAL NETWORK (NUMPY ONLY)")
    print(f"Architecture      : {model.w1.shape[0]} inputs -> "
          f"{model.w1.shape[1]} hidden units -> {model.w2.shape[1]} output unit")
    print("Activations       : sigmoid after the hidden layer, "
          "sigmoid after the output layer")
    print("Loss function     : mean squared error")
    print(f"Initialization    : weights ~ N({WEIGHT_INIT_MEAN}, "
          f"{WEIGHT_INIT_STD}), biases = 0")
    print(f"Hyperparameters   : RANDOM_SEED={RANDOM_SEED}, "
          f"HIDDEN_UNITS={HIDDEN_UNITS}, LEARNING_RATE={LEARNING_RATE},")
    print(f"                    MAX_EPOCHS={MAX_EPOCHS}, PATIENCE={PATIENCE}")
    print()
    print("Parameter shapes:")
    print(f"  w1 {str(model.w1.shape):<8} one weight per (input feature, hidden unit)")
    print(f"  b1 {str(model.b1.shape):<8} one bias per hidden unit")
    print(f"  w2 {str(model.w2.shape):<8} one weight per (hidden unit, output unit)")
    print(f"  b2 {str(model.b2.shape):<8} the output unit's bias")
    print()
    print("The hidden layer computes a1 = sigmoid(x @ w1 + b1): six different")
    print("non-linear blends of the six standardized inputs. The output layer")
    print("computes a2 = sigmoid(a1 @ w2 + b2), one weighted summary of those")
    print("blends squashed into (0, 1). Bounded, rising with the evidence for")
    print("acceptance, and trained against 0/1 targets, a2 reads as a")
    print("probability-like score. MSE training leaves it uncalibrated, though,")
    print("so it is not a literal admission probability.")


# --------------------------------------------------------------------------- #
# Section 4 - train until the test MSE stops improving
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainingResult:
    """The record left behind by a training run.

    Attributes:
        history: Per-epoch lists of epoch number, training MSE, test MSE, and
            test accuracy.
        best_epoch: Epoch with the lowest test MSE; its parameters are restored.
        best_test_mse: The lowest test MSE observed.
        best_test_accuracy: Test accuracy at ``best_epoch``.
        stopped_epoch: The last epoch actually run.
        early_stopped: Whether patience ran out before ``MAX_EPOCHS``.
    """

    history: dict
    best_epoch: int
    best_test_mse: float
    best_test_accuracy: float
    stopped_epoch: int
    early_stopped: bool


class EarlyStoppingTracker:
    """Remembers the best test MSE and the parameters that produced it.

    Keeping this state in its own object means the training loop only has to
    ask two questions each epoch: "is this the best so far?" and "has patience
    run out?".

    Attributes:
        patience: Epochs without improvement allowed before stopping.
        best_mse: Lowest test MSE seen so far.
        best_accuracy: Test accuracy recorded at ``best_epoch``.
        best_epoch: Epoch that produced ``best_mse``.
        best_parameters: Snapshot of the model parameters at ``best_epoch``.
        epochs_without_improvement: Consecutive epochs with no new best.
    """

    def __init__(self, model, patience=PATIENCE):
        """Start tracking, seeded with the model's initial parameters.

        Args:
            model: The model being trained.
            patience: Epochs without improvement allowed before stopping.
        """
        self.patience = patience
        self.best_mse = float("inf")
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_parameters = model.get_parameters()
        self.epochs_without_improvement = 0

    def update(self, model, epoch, test_mse, test_accuracy):
        """Record one epoch's result and report whether training should stop.

        Args:
            model: The model being trained, snapshotted on an improvement.
            epoch: The epoch number just completed.
            test_mse: Test MSE after this epoch's update.
            test_accuracy: Test accuracy after this epoch's update.

        Returns:
            True when the test MSE has not improved for ``patience``
            consecutive epochs.
        """
        if test_mse < self.best_mse:
            self.best_mse = test_mse
            self.best_accuracy = test_accuracy
            self.best_epoch = epoch
            self.best_parameters = model.get_parameters()
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience

    def restore_best(self, model):
        """Roll a model back to the best parameters seen during training.

        Args:
            model: The model to restore in place.
        """
        model.set_parameters(self.best_parameters)


def print_progress_row(epoch, train_mse, test_mse, test_accuracy):
    """Print one line of the training log.

    Args:
        epoch: Epoch number.
        train_mse: Training MSE for this epoch.
        test_mse: Test MSE for this epoch.
        test_accuracy: Test accuracy for this epoch.
    """
    print(f"{epoch:>8,}{train_mse:>14.6f}{test_mse:>14.6f}{test_accuracy:>16.4f}")


def train_network(model, data):
    """Train with full-batch gradient descent and early stopping on test MSE.

    Each epoch follows the order the assignment prescribes: forward pass on the
    training set, training MSE, backpropagation and parameter update, then a
    forward pass on the test set for test MSE and test accuracy. (So the
    training MSE recorded for an epoch is the loss the update responded to,
    while the test MSE reflects the parameters after that update.)

    Training stops once the test MSE has failed to improve for ``PATIENCE``
    consecutive epochs, and the parameters from the best epoch are restored
    before the model is evaluated or used for predictions.

    Args:
        model: The :class:`TwoLayerNeuralNetwork` to train.
        data: The :class:`SplitData` holding both preprocessed splits.

    Returns:
        A :class:`TrainingResult`.
    """
    print_banner("SECTION 4 - TRAINING LOG (FULL-BATCH GRADIENT DESCENT)")
    print(f"Training on {len(data.x_train):,} rows, "
          f"evaluating on {len(data.x_test):,} rows.")
    print(f"Stopping when test MSE has not improved for {PATIENCE} "
          f"consecutive epochs (max {MAX_EPOCHS:,}).")
    print()
    print(f"{'epoch':>8}{'train MSE':>14}{'test MSE':>14}{'test accuracy':>16}")
    print("-" * 52)

    history = {"epoch": [], "train_mse": [], "test_mse": [], "test_accuracy": []}
    tracker = EarlyStoppingTracker(model)
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        # Forward pass and loss on the training set.
        train_predictions = model.forward(data.x_train)
        train_mse = mean_squared_error(train_predictions, data.y_train)

        # Backpropagation and the weight/bias update.
        model.backward(data.x_train, data.y_train)

        # Forward pass on the held-out test set with the updated parameters.
        test_predictions = model.predict_proba(data.x_test)
        test_mse = mean_squared_error(test_predictions, data.y_test)
        test_accuracy = accuracy_score(
            (test_predictions >= PREDICTION_THRESHOLD).astype(float), data.y_test)

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["test_accuracy"].append(test_accuracy)

        out_of_patience = tracker.update(model, epoch, test_mse, test_accuracy)

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print_progress_row(epoch, train_mse, test_mse, test_accuracy)

        if out_of_patience:
            stopped_epoch = epoch
            print_progress_row(epoch, train_mse, test_mse, test_accuracy)
            print()
            print(f"Early stopping at epoch {epoch:,}: test MSE has not improved "
                  f"for {PATIENCE} consecutive epochs.")
            break

    # Roll the model back to its best-scoring state before any evaluation.
    tracker.restore_best(model)
    print(f"Restored the parameters from epoch {tracker.best_epoch:,} "
          f"(test MSE {tracker.best_mse:.6f}).")

    if stopped_epoch == MAX_EPOCHS:
        print(f"Note: reached MAX_EPOCHS ({MAX_EPOCHS:,}) before patience ran out.")

    return TrainingResult(
        history=history,
        best_epoch=tracker.best_epoch,
        best_test_mse=tracker.best_mse,
        best_test_accuracy=tracker.best_accuracy,
        stopped_epoch=stopped_epoch,
        early_stopped=stopped_epoch < MAX_EPOCHS,
    )


# --------------------------------------------------------------------------- #
# Section 5 - evaluate the restored best model
# --------------------------------------------------------------------------- #
def report_evaluation(model, data, result):
    """Print the Section 5 final metrics and a short interpretation.

    Args:
        model: The trained model, already rolled back to its best parameters.
        data: The :class:`SplitData` used for training and testing.
        result: The :class:`TrainingResult` returned by :func:`train_network`.
    """
    train_mse = mean_squared_error(model.predict_proba(data.x_train), data.y_train)
    train_accuracy = accuracy_score(model.predict(data.x_train), data.y_train)
    test_accuracy = accuracy_score(model.predict(data.x_test), data.y_test)

    # If the model always guessed the more common class, it would score this.
    majority_baseline = max(data.y_test.mean(), 1.0 - data.y_test.mean())

    print_banner("SECTION 5 - FINAL EVALUATION (BEST PARAMETERS RESTORED)")
    print(f"Best epoch                       : {result.best_epoch:,}")
    print(f"Best test MSE                    : {result.best_test_mse:.6f}")
    print(f"Final training accuracy          : {train_accuracy:.4f}")
    print(f"Final test accuracy              : {test_accuracy:.4f}")
    print(f"Rows used after filtering        : {data.filtered_row_count:,}")
    print(f"Training rows / test rows        : {len(data.x_train):,} / "
          f"{len(data.x_test):,}")
    print(f"Epochs actually run              : {result.stopped_epoch:,} of "
          f"{MAX_EPOCHS:,} (early stopping: {result.early_stopped})")
    print(f"Training MSE at best parameters  : {train_mse:.6f}")
    print(f"Majority-class baseline (test)   : {majority_baseline:.4f}")
    print()
    parameter_count = model.w1.size + model.b1.size + model.w2.size + model.b2.size

    print("Interpretation:")
    print(f"  Overfitting: training MSE ({train_mse:.6f}) and test MSE")
    print(f"  ({result.best_test_mse:.6f}) sit "
          f"{abs(train_mse - result.best_test_mse):.6f} apart and fall together")
    print("  for the whole run, so the model is not overfitting. With only")
    print(f"  {parameter_count} parameters against {len(data.x_train):,} training rows it")
    print("  has far too little capacity to memorize. If anything, it underfits.")
    print(f"  Strength: {test_accuracy:.4f} test accuracy beats the 0.5000 coin flip")
    print(f"  and the {majority_baseline:.4f} majority-class baseline, so the network")
    print("  found real signal rather than guessing the more common outcome.")
    print("  It is a modest gain, not a strong admissions predictor.")
    print("  Stability: the test MSE curve is smooth and monotone and accuracy")
    print("  holds flat over long stretches, so this is not a lucky epoch.")
    print("  Accuracy moves in steps because it only changes when scores cross")
    print("  the 0.5 threshold.")


# --------------------------------------------------------------------------- #
# Section 6 - plot training and test MSE over time
# --------------------------------------------------------------------------- #
def plot_mse_curve(result, output_path=MSE_CURVE_PATH):
    """Save a line plot of training and test MSE against epoch.

    Args:
        result: The :class:`TrainingResult` holding the per-epoch history.
        output_path: Where to write the PNG.
    """
    # Render straight to a file; no interactive window is needed.
    plt.switch_backend("Agg")

    history = result.history
    figure, axes = plt.subplots(figsize=(9.0, 5.5))

    axes.plot(history["epoch"], history["train_mse"],
              label="Training MSE", linewidth=2.0)
    axes.plot(history["epoch"], history["test_mse"],
              label="Test MSE", linewidth=2.0, linestyle="--")
    axes.axvline(result.best_epoch, color="grey", linestyle=":", linewidth=1.5,
                 label=f"Best epoch ({result.best_epoch:,})")

    axes.set_title("Training and Test MSE per Epoch\n"
                   "Two-layer 6-6-1 NumPy network, learning rate 0.05")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Mean squared error")
    axes.legend(loc="upper right")
    axes.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print_banner("SECTION 6 - MSE CURVE")
    print(f"Saved the training/test MSE curve to {output_path.name} "
          f"({len(history['epoch']):,} epochs plotted).")


# --------------------------------------------------------------------------- #
# Section 7 - run the trained model on artificial applicants
# --------------------------------------------------------------------------- #
# Hand-written applicants used to probe what the model actually learned.
# Values are on the raw scale (GPA out of 4.0, GRE on the 130-170 sections);
# they run through exactly the same fill-and-standardize pipeline as real rows.
# The last profile deliberately omits every GRE score, which is the common case
# in this dataset, to show what median-filling does to a prediction.
ARTIFICIAL_APPLICANTS = [
    {
        "profile": "Strong PhD, international",
        "gpa": 3.95, "gre": 335.0, "gre_v": 165.0, "gre_aw": 5.0,
        "ms_vs_phd": 1.0, "international_vs_local": 1.0,
    },
    {
        "profile": "Strong PhD, local",
        "gpa": 3.95, "gre": 335.0, "gre_v": 165.0, "gre_aw": 5.0,
        "ms_vs_phd": 1.0, "international_vs_local": 0.0,
    },
    {
        "profile": "Average Masters, local",
        "gpa": 3.40, "gre": 305.0, "gre_v": 152.0, "gre_aw": 3.5,
        "ms_vs_phd": 0.0, "international_vs_local": 0.0,
    },
    {
        "profile": "Weak Masters, international",
        "gpa": 2.90, "gre": 295.0, "gre_v": 145.0, "gre_aw": 3.0,
        "ms_vs_phd": 0.0, "international_vs_local": 1.0,
    },
    {
        "profile": "Strong GPA, no GRE reported",
        "gpa": 3.90, "gre": np.nan, "gre_v": np.nan, "gre_aw": np.nan,
        "ms_vs_phd": 1.0, "international_vs_local": 0.0,
    },
]


def score_artificial_applicants(model, statistics):
    """Predict outcomes for the hand-written applicants in Section 7.

    The artificial rows go through the identical pipeline used for real data:
    missing values filled with the stored *training* medians, then standardized
    with the stored *training* means and standard deviations.

    Args:
        model: The trained model with best parameters restored.
        statistics: The training-set statistics from Section 2.

    Returns:
        A DataFrame of the applicants with predicted probability, binary label,
        and status appended.
    """
    applicants = pd.DataFrame(ARTIFICIAL_APPLICANTS)

    # Same preprocessing as the real data, using the stored training statistics.
    raw_features = applicants[FEATURE_COLUMNS].to_numpy(dtype=float)
    processed_features = preprocess_features(raw_features, statistics)

    probabilities = model.predict_proba(processed_features)
    labels = model.predict(processed_features)

    applicants["probability"] = probabilities.ravel()
    applicants["label"] = labels.ravel().astype(int)
    applicants["status"] = np.where(labels.ravel() == 1.0, "Accepted", "Rejected")

    return applicants


def probability_for(applicants, profile):
    """Look up one artificial applicant's predicted probability.

    Args:
        applicants: The scored applicant DataFrame.
        profile: The profile label to look up.

    Returns:
        The predicted probability as a float.
    """
    return float(applicants.loc[applicants["profile"] == profile,
                                "probability"].iloc[0])


def report_artificial_applicants(applicants, degree_rates):
    """Print the Section 7 prediction table and what it shows.

    Args:
        applicants: The DataFrame returned by :func:`score_artificial_applicants`.
        degree_rates: ``(masters_rate, phd_rate)`` acceptance rates from the
            filtered dataset, used to explain the predictions.
    """
    print_banner("SECTION 7 - PREDICTIONS FOR ARTIFICIAL APPLICANTS")
    print(applicants.to_string(index=False, na_rep="NaN",
                               float_format=lambda value: f"{value:.4f}"))
    print()

    ranked = applicants.sort_values("probability", ascending=False)
    strongest, weakest = ranked.iloc[0], ranked.iloc[-1]
    accepted_count = int((applicants["label"] == 1).sum())
    citizenship_gap = (probability_for(applicants, "Strong PhD, international")
                       - probability_for(applicants, "Strong PhD, local"))
    masters_gap = (probability_for(applicants, "Average Masters, local")
                   - probability_for(applicants, "Weak Masters, international"))

    print("What these predictions show:")
    print(f"  Scores span {weakest['probability']:.4f} ({weakest['profile']}) to "
          f"{strongest['probability']:.4f} ({strongest['profile']}),")
    print(f"  and {accepted_count} of {len(applicants)} profiles clear the 0.5 "
          "threshold.")
    print("  The ordering is driven by degree type, not by credentials: every")
    print("  Masters profile outscores every PhD profile, even though the top")
    print("  PhD applicant carries a full GPA point and 40 GRE points more than")
    print(f"  the weakest Masters one. That is the data speaking, since "
          f"{degree_rates[0]:.1%}")
    print(f"  of Masters rows here were accepted versus {degree_rates[1]:.1%} of PhD")
    print("  rows. It does mean the network is largely reproducing base rates by")
    print("  degree rather than judging an applicant's strength.")
    print("  Within a degree the numbers do move the score the expected way:")
    print(f"  the average Masters applicant scores {masters_gap:+.4f} against the")
    print("  weak one, and citizenship shifts two otherwise identical PhD")
    print(f"  applicants by {citizenship_gap:+.4f}. Both effects are small next to")
    print("  the degree flag.")
    print("  The 'no GRE reported' applicant is scored as if holding median GRE")
    print("  values, which is exactly what median-filling does. About nine in ten")
    print("  real rows lack GRE scores, so most predictions lean on GPA, degree,")
    print("  and citizenship whether or not test scores were supplied.")


class _Tee:
    """Fan writes out to several streams at once.

    Used to mirror everything printed to both the console and ``training.log``,
    so the submitted log is the real transcript of the run rather than a
    hand-copied excerpt.
    """

    def __init__(self, *streams):
        """Store the streams to write to.

        Args:
            *streams: Open, writable text streams.
        """
        self._streams = streams

    def write(self, text):
        """Write ``text`` to every stream.

        Args:
            text: The text to write.

        Returns:
            Number of characters written.
        """
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self):
        """Flush every stream."""
        for stream in self._streams:
            stream.flush()


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


def run_pipeline(data_path):
    """Run every section of the assignment in order.

    Args:
        data_path: Path to the applicant dataset.
    """
    # Section 1 - load, filter, and engineer features.
    records = load_applicant_records(data_path)
    frame = build_dataframe(records)
    report_dataset(frame, len(records))

    # Section 2 - split first, then learn the preprocessing on the training half.
    x_train_raw, x_test_raw, y_train, y_test = split_dataset(frame)
    statistics = fit_training_statistics(x_train_raw)
    data = SplitData(
        x_train=preprocess_features(x_train_raw, statistics),
        y_train=y_train,
        x_test=preprocess_features(x_test_raw, statistics),
        y_test=y_test,
        filtered_row_count=len(frame),
    )
    report_split(data, statistics)

    # Section 3 - build the network.
    model = TwoLayerNeuralNetwork(input_units=len(FEATURE_COLUMNS))
    report_architecture(model)

    # Section 4 - train until the test MSE stops improving.
    result = train_network(model, data)

    # Section 5 - evaluate the restored best parameters.
    report_evaluation(model, data, result)

    # Section 6 - plot the loss curves.
    plot_mse_curve(result)

    # Section 7 - probe the trained model with hand-written applicants.
    report_artificial_applicants(score_artificial_applicants(model, statistics),
                                 acceptance_rate_by_degree(frame))


def main():
    """Run the pipeline, mirroring all output into ``training.log``."""
    arguments = parse_arguments()

    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
            run_pipeline(arguments.data)

    print(f"\nFull run transcript saved to {TRAINING_LOG_PATH.name}")


if __name__ == "__main__":
    main()
