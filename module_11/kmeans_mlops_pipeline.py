"""Module 11 - MLOps: MLflow tracking for the Grad Cafe KMeans model.

Builds on the ported Module 9 clustering pipeline (TF-IDF -> PCA -> KMeans) by
logging each training run to a local MLflow tracking server: the required
clustering parameters and the model's ``inertia_`` metric.  Model logging /
registry and an optional wandb backend are added in later commits.

Pipeline
--------
1. Load and clean the Grad Cafe program names.
2. TF-IDF vectorise, then reduce with PCA.
3. Fit KMeans with the required parameters.
4. Log the parameters and the ``inertia_`` metric to MLflow.

Start the tracking server first, e.g.::

    mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "applicant_data.json"

TRACKING_URI = "http://127.0.0.1:8080"   # local MLflow server (localhost:8080)
EXPERIMENT_NAME = "gradcafe-kmeans"
RUN_NAME = "kmeans-gradcafe"

MAX_FEATURES = 1000        # TF-IDF vocabulary cap (matches Module 9)
PCA_COMPONENTS = 80        # dense PCA components (matches Module 9's final run)

# Required clustering parameters for this assignment.  Module 9 chose ~85
# clusters from the elbow analysis; here the four values below are fixed by
# the spec, so they live in one dict that feeds the model directly.
PARAMS = {"max_iter": 500, "n_clusters": 25, "n_init": 5, "random_state": 42}


def load_programs(path):
    """Load Grad Cafe records and return cleaned program names (Module 9 logic)."""
    frame = pd.read_json(path).dropna(subset=["program"])
    frame = frame[frame["program"].astype(str).str.strip() != ""]
    # Rebuild the "program, university" label and split on the FIRST comma, the
    # same normalisation used in Module 9's clean_programs step.
    combined = (frame["program"].astype(str).str.strip() + ", "
                + frame["university"].astype(str).str.strip())
    program = combined.str.split(",", n=1, expand=True)[0]
    return program.str.strip().str.replace(r"\s+", " ", regex=True)


def vectorize(programs):
    """Return a TF-IDF sparse matrix for the program-name Series."""
    return TfidfVectorizer(max_features=MAX_FEATURES).fit_transform(programs)


def reduce_dimensions(matrix, n_components):
    """Densify the TF-IDF *matrix* and reduce it to *n_components* with PCA."""
    # PCA cannot consume a sparse matrix, so densify first; MAX_FEATURES keeps
    # that dense array a manageable size.
    pca = PCA(n_components=n_components, random_state=PARAMS["random_state"])
    return pca.fit_transform(matrix.toarray())


def train_kmeans(features):
    """Fit KMeans with the required parameters and return the fitted model."""
    # Unpack PARAMS straight into the estimator so the values we later track and
    # the values actually trained on can never drift apart.
    return KMeans(**PARAMS).fit(features)


def build_features():
    """Run the shared load -> vectorize -> PCA steps and return the features."""
    programs = load_programs(DATA_FILE)
    matrix = vectorize(programs)
    features = reduce_dimensions(matrix, PCA_COMPONENTS)
    print(f"Programs: {len(programs):,} | TF-IDF: {matrix.shape} | "
          f"PCA features: {features.shape}")
    return features


def track_mlflow(model):
    """Log the run, required parameters and the inertia metric to MLflow."""
    # set_tracking_uri points the client at the server started with, e.g.:
    #   mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params(PARAMS)
        # inertia_ is a model OUTPUT, so it is logged as a metric, not a param.
        mlflow.log_metric("inertia", float(model.inertia_))
    print(f"MLflow: logged run to {TRACKING_URI} (experiment '{EXPERIMENT_NAME}')")


def main():
    """Train KMeans on the Grad Cafe programs and log the run to MLflow."""
    features = build_features()
    model = train_kmeans(features)
    print(f"KMeans inertia: {model.inertia_:,.2f}")
    track_mlflow(model)


if __name__ == "__main__":
    main()
