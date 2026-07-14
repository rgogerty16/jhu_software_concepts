"""Module 9 - K-Means clustering of Grad Cafe program names.

This script groups similar master's-degree program names together so the
dataset can be analysed at the level of a "program area" rather than the many
slightly different spellings that were scraped from The Grad Cafe.

Pipeline
--------
1. Load the raw applicant records into a Pandas DataFrame.
2. Clean the data: drop rows with no program, rebuild the combined
   ``"program, university"`` label and split it on the first comma into
   separate ``Program`` and ``University`` columns.
3. Vectorise the program names with scikit-learn's ``TfidfVectorizer``.
4. Reduce the TF-IDF features with ``PCA`` (2 components for the scatter plot,
   80 components for the elbow analysis and the final clustering).
5. Cluster the features with ``KMeans`` and attach the labels to the DataFrame.
6. Produce five PNG deliverables and print a short console summary.

All randomness is seeded with ``RANDOM_STATE`` so every run is reproducible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "applicant_data.json"

RANDOM_STATE = 42          # seed shared by every estimator for reproducibility
MAX_FEATURES = 1000        # cap the TF-IDF vocabulary so dense PCA fits in RAM
INITIAL_CLUSTERS = 50      # section 1: the initial 50-cluster experiment
HIGH_DIM_COMPONENTS = 80   # section 2/3: PCA size for the elbow + final run
FINAL_CLUSTERS = 85        # section 3: chosen from the elbow analysis below
ELBOW_MAX_K = 100          # section 2: sweep k over 1..100 (never 0)
SAMPLE_ROWS = 100          # rows shown in the clustered-DataFrame image


def load_data(path):
    """Load the Grad Cafe applicant records from *path* into a DataFrame."""
    return pd.read_json(path)


def _collapse_ws(series):
    """Strip and collapse repeated whitespace in a string Series."""
    return series.str.strip().str.replace(r"\s+", " ", regex=True)


def clean_programs(df):
    """Clean program text and split it into Program and University columns.

    Rows whose ``program`` is missing or blank are removed. The scraped data
    stores ``program`` and ``university`` separately, so the combined
    ``"program, university"`` label is rebuilt and then split on the FIRST
    comma (``n=1``) to reproduce the assignment's cleaning step.
    """
    df = df.dropna(subset=["program"]).copy()
    df = df[df["program"].astype(str).str.strip() != ""]

    combined = (
        df["program"].astype(str).str.strip()
        + ", "
        + df["university"].astype(str).str.strip()
    )
    parts = combined.str.split(",", n=1, expand=True)
    df["Program"] = _collapse_ws(parts[0])
    df["University"] = _collapse_ws(parts[1])
    return df.reset_index(drop=True)


def vectorize_programs(programs):
    """Return a TF-IDF sparse matrix for the given program-name Series."""
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    return vectorizer.fit_transform(programs)


def reduce_dimensions(matrix, n_components):
    """Reduce a sparse TF-IDF *matrix* to *n_components* dense PCA features.

    PCA cannot consume a sparse matrix, so it is densified first; the TF-IDF
    vocabulary is capped (see ``MAX_FEATURES``) to keep that array small.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    features = pca.fit_transform(matrix.toarray())
    return features, pca


def run_kmeans(features, n_clusters):
    """Fit K-Means with the assignment's parameters and return the model."""
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=100,
        n_init=5,
        random_state=RANDOM_STATE,
    )
    model.fit(features)
    return model


def plot_initial_clusters(features2d, labels, path):
    """Scatter the two PCA components, coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        features2d[:, 0], features2d[:, 1], c=labels, cmap="tab20", s=10
    )
    ax.set_title("KMeans Clustering of Programs")
    ax.set_xlabel("KMeans Distance Direction 1")
    ax.set_ylabel("KMeans Distance Direction 2")
    # 50 clusters is too many for a full legend, so show a representative set.
    handles, texts = scatter.legend_elements(num=10)
    ax.legend(handles, texts, title="Cluster", loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_clustered_dataframe(df, labels, path):
    """Attach cluster labels and render a 100-row sample as a PNG table."""
    labeled = df.assign(cluster=labels)
    sample = labeled.loc[:, ["cluster", "Program", "University"]].head(SAMPLE_ROWS)

    fig, ax = plt.subplots(figsize=(10, 24))
    ax.axis("off")
    fig.suptitle("Clustered Programs (100-row sample)", fontsize=12, y=0.995)
    table = ax.table(
        cellText=sample.values,
        colLabels=sample.columns,
        cellLoc="left",
        bbox=[0, 0, 1, 0.97],  # leave headroom so the title does not overlap
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return labeled


def elbow_method(features, path):
    """Sweep k from 1..100 and plot K-Means inertia to locate the elbow.

    The curve declines gradually with no sharp "elbow jut", so roughly 85
    clusters (see ``FINAL_CLUSTERS``) is a reasonable operating point: it is
    far enough down the curve to separate distinct program areas without
    over-fragmenting near-identical names.
    """
    ks = list(range(1, ELBOW_MAX_K + 1))
    inertias = [run_kmeans(features, k).inertia_ for k in ks]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, inertias, "bx-", label="Inertia")
    ax.set_title("The Elbow Method using Inertia")
    ax.set_xlabel("Values of K")
    ax.set_ylabel("Inertia")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return inertias


def identify_cluster(df, keyword):
    """Return the cluster id whose Program values most often contain *keyword*."""
    mask = df["Program"].str.contains(keyword, case=False, na=False)
    if not mask.any():
        return None
    return int(df.loc[mask, "cluster"].value_counts().idxmax())


def plot_gre_box(df, cluster_id, title, path):
    """Box-plot the GRE and GRE Verbal scores for a single cluster."""
    subset = df[df["cluster"] == cluster_id]
    gre = pd.to_numeric(subset["gre"], errors="coerce").dropna()
    gre_v = pd.to_numeric(subset["gre_v"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(
        [gre, gre_v], tick_labels=["GRE", "GRE V"], patch_artist=True
    )
    for patch, color in zip(box["boxes"], ["#1f77b4", "#ff7f0e"]):
        patch.set_facecolor(color)
    ax.set_title(title)
    ax.set_xlabel("GRE Component")
    ax.set_ylabel("Score (points)")
    ax.legend(box["boxes"], ["GRE", "GRE V"], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run the full clustering pipeline and write every deliverable."""
    plt.switch_backend("Agg")  # render to files without a display

    df = clean_programs(load_data(DATA_FILE))
    print(f"Number of Entries: {len(df):,}")
    print(f"Number of Program Input Names: {df['Program'].nunique():,}")

    matrix = vectorize_programs(df["Program"])
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"TF-IDF matrix type: {type(matrix)}")

    features_2d, pca_2d = reduce_dimensions(matrix, 2)
    print(f"PCA feature shape: {features_2d.shape}")
    print(f"PCA configuration: {pca_2d}")

    initial = run_kmeans(features_2d, INITIAL_CLUSTERS)
    plot_initial_clusters(features_2d, initial.labels_, HERE / "initial_cluster.png")
    build_clustered_dataframe(df, initial.labels_, HERE / "clustered_dataFrame.png")

    features_hd, _ = reduce_dimensions(matrix, HIGH_DIM_COMPONENTS)
    elbow_method(features_hd, HERE / "elbow.png")

    labeled = df.assign(cluster=run_kmeans(features_hd, FINAL_CLUSTERS).labels_)
    plot_gre_box(
        labeled,
        identify_cluster(labeled, "philosoph"),
        "GRE and GRE Verbal Scores for Philosophy Majors",
        HERE / "philosophy.png",
    )
    plot_gre_box(
        labeled,
        identify_cluster(labeled, "computer science"),
        "GRE and GRE Verbal Scores for CS Majors",
        HERE / "computer_science.png",
    )
    print(f"All visualizations written to {HERE}")


if __name__ == "__main__":
    main()
