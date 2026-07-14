# Module 9 — K-Means Clustering of Grad Café Programs

- **Name:** Ryan Gogerty
- **JHED:** rgogerty
- **Course:** EN.605.256 — Modern Software Concepts in Python
- **Assignment:** Module 9 — K-Means Clustering

## Overview

`kmeans.py` groups similar master's-degree **program names** from the scraped
Grad Café dataset into a smaller set of "program areas" using K-Means. Many
programs are the same but were entered differently (e.g. *Business and
Economics* vs *Business Economics*); clustering reduces that noise so students
can later be compared at the program level.

The script runs the full pipeline end to end:

1. **Load** the raw applicant records into a Pandas DataFrame.
2. **Clean** — drop rows with no program, rebuild the combined
   `"program, university"` label and split it on the **first comma** into
   separate `Program` and `University` columns, collapsing stray whitespace.
3. **Vectorize** the program names with scikit-learn's `TfidfVectorizer`
   (vocabulary capped at 1,000 features so the dense PCA input stays in memory).
4. **PCA** — reduce the TF-IDF features to **2 components** for the scatter plot
   and to **80 components** for the elbow analysis and final clustering.
5. **K-Means** — an initial 50-cluster experiment, then a final 85-cluster run
   whose labels are returned to the DataFrame.
6. **Analyse** — box-plot the GRE / GRE V score ranges for the Computer-Science-
   like and Philosophy-like clusters.

Every estimator uses `random_state=42`, so results are reproducible.

## How to run

```bash
cd module_9
pip install -r requirements.txt
python kmeans.py
```

The data file (`applicant_data.json`) ships inside this folder, so the script
runs standalone. The elbow step fits K-Means for k = 1..100 and takes a few
minutes.

## Dependencies

See `requirements.txt`: `scikit-learn`, `pandas`, `numpy`, `matplotlib`
(Python 3.10+).

## Outputs

Console summary (entries, unique program names, TF-IDF shape/type, PCA
shape/config) plus five PNG figures:

| File | Contents |
| --- | --- |
| `initial_cluster.png` | Scatter of the 2-D PCA features coloured by the 50 initial clusters. |
| `clustered_dataFrame.png` | 100-row sample of the clustered DataFrame (cluster / Program / University). |
| `elbow.png` | Inertia vs. number of clusters (k = 1..100) — the Elbow Method. |
| `philosophy.png` | GRE & GRE V box plot for the Philosophy-like cluster. |
| `computer_science.png` | GRE & GRE V box plot for the Computer-Science-like cluster. |

For the bundled dataset the cleaned frame has **29,992 entries** across
**2,909 unique program input names**, and the TF-IDF matrix is a
`(29992, 1000)` SciPy sparse (CSR) matrix.

## Choosing the number of clusters

The elbow curve declines gradually with no sharp "elbow jut", so there is no
single obvious k. About **85 clusters** is a reasonable operating point: it is
far enough down the inertia curve to separate genuinely distinct program areas
without over-fragmenting near-identical names. This is documented in the
`elbow_method` docstring in `kmeans.py`.

## Conclusion — does something look amiss?

Comparing the two box plots, the **Computer Science** cluster shows GRE / GRE V
values that are implausible for the score's intended range and are more spread
out than the Philosophy cluster. GRE fields are also missing for the large
majority of applicants, so the surviving values are sparse and clearly include
mis-entered or mixed-scale numbers. This indicates the GRE columns need
**further data cleaning** (range validation and unit normalisation) before they
can be trusted for cross-program comparison.
