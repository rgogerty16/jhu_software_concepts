"""Module 11 - MLOps: KMeans clustering pipeline for Grad Cafe program names.

Ported from the Module 9 clustering workflow (TF-IDF -> PCA -> KMeans).  This
first step establishes the plain modelling pipeline; MLflow (and later wandb)
experiment tracking is layered on top in subsequent commits.

Pipeline
--------
1. Load the Grad Cafe applicant records and clean the program names.
2. Vectorise the program names with scikit-learn's ``TfidfVectorizer``.
3. Reduce the features to ``PCA_COMPONENTS`` dense components with PCA.
4. Fit ``KMeans`` with the required clustering parameters and report ``inertia_``.
"""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "applicant_data.json"

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


def main():
    """Train KMeans on the Grad Cafe programs and report inertia."""
    features = build_features()
    model = train_kmeans(features)
    print(f"KMeans inertia: {model.inertia_:,.2f}")


if __name__ == "__main__":
    main()
