# Module 8 — Data Preparation & Statistics (Grad Café on SageMaker)

Extends the Module 7 cloud pipeline into a full data-cleaning and exploratory-
analysis workflow. `module_8.ipynb` **loads the Grad Café JSON dataset from
Amazon S3** into a SageMaker notebook (reusing the Module 7 `boto3` workflow),
**cleans / validates / feature-engineers** it with Pandas + NumPy, runs
**SciPy** statistics, produces **Matplotlib** figures, and **writes the cleaned
dataset back to S3**.

> **AWS reminder:** the SageMaker notebook instance was **STOPPED** after
> completing the assignment (Notebook instances → Stop — stopped, not
> terminated, so it can be reused). No AWS keys, SSH keys, or `.env` files are
> committed.

---

## Folder structure

```
module_8/
├── module_8.ipynb              # main notebook — runs top-to-bottom
├── s3_fetch.py                 # Module 7 boto3 download + new upload_dataset()
├── gradcafe.py                 # helpers: PNG-table renderer, 6 plots, analytics PDF
├── requirements.txt            # boto3, pandas, numpy, scipy, matplotlib
├── .env.example                # AWS_REGION / bucket / keys — no secrets
├── .gitignore
├── README.md                   # this file
│
├── analytics.pdf               # written analytical summary (Q1–Q7)
├── cleaned_gradcafe.json       # cleaned dataset (also uploaded to S3)
├── missingness_summary.csv     # per-column missing count + %
├── summary_statistics.csv      # descriptive stats for GPA/GRE/GRE V/GRE AW
│
└── PNG deliverables:
    initial_dataframe · missingness_summary · low_program_count · date_based ·
    float_columns · outlier_summary · GRE-vs-GRE-V · GPA-vs-GRE ·
    Degree-vs-International · Acceptances-over-Time · GPA-by-Outcome ·
    Numeric-Correlation-Heatmap
```

The raw file the notebook downloads from S3 (`applicant_data_SM.json`) is
git-ignored — it is regenerated on every run and is not a required deliverable.

---

## Run it on SageMaker

1. Log into AWS as `dailyWork-<initials>` and **start** the SageMaker notebook
   instance (the one reused from Module 7, IAM role scoped to your bucket).
2. Upload `module_8.ipynb`, `s3_fetch.py`, `gradcafe.py`, and
   `requirements.txt` into the same Jupyter working directory.
3. In a cell or terminal: `pip install -r requirements.txt`.
4. **Kernel → Restart & Run All.** The notebook downloads the dataset from S3,
   produces every CSV/PNG/PDF deliverable in the working directory, writes
   `cleaned_gradcafe.json`, and uploads it back to the bucket (prints an
   `Upload succeeded -> s3://…` confirmation).
5. Download the executed notebook (with outputs) and the generated files back
   into `module_8/` for submission.
6. **Stop the SageMaker notebook instance.**

**Credentials:** none are hard-coded. `s3_fetch.make_s3_client()` builds the S3
client from boto3's default provider chain — on SageMaker that is the instance's
IAM execution role. Bucket / key / region are overridable via the env vars in
`.env.example` (`S3_BUCKET`, `S3_KEY`, `OUTPUT_FILE`, `S3_CLEAN_KEY`,
`AWS_REGION`).

### Running locally (authoring / verification)

The same notebook runs off-cloud for development: set `LOCAL_DATASET` to a local
copy of `applicant_data.json` and `s3_fetch` reads/writes locally instead of
calling S3. Leave it unset on SageMaker so the real S3 download/upload runs.

```bash
export LOCAL_DATASET=/path/to/applicant_data.json
jupyter nbconvert --to notebook --execute --inplace module_8.ipynb
```

---

## What the cleaning does (key decisions)

* **raw_df** is copied before any changes; rows with a `None` program are dropped.
* **Program / University** — split from the combined program field when a comma
  is present; otherwise the existing `university` column is used. Text columns
  are stripped and internal whitespace collapsed, preserving capitalization.
* **Dates** — `date_added` → datetime. `Status` is split into `outcome`
  (restricted to Accepted / Rejected / Waitlisted / Interviewed) and
  `decision_date`, whose **year is inferred from `term`** (e.g. `Fall 2026`).
* **Degree** — kept only for `Master's` / `PhD` / `PsyD`; other values dropped.
  **US/International** — kept for `International` / `American` / `Other`; blanks
  and unexpected values mapped to `Other`.
* **Numeric** — `GPA`, `GRE`, `GRE V`, `GRE AW` coerced to floats.
* **Derived** — `has_valid_gpa/gre/gre_v`, `days_to_decision`, `decision_speed`
  (`np.select`: 0-30 / 31-60 / 61+ / Unknown), `application_season`
  (Early/Mid/Late Cycle by `date_added` month), and z-score outlier flags
  (`gpa_outlier`, `gre_outlier`, |z| > 3, kept-not-dropped).

All transforms are vectorized Pandas/NumPy; a validation cell reports rows
removed (overall and per rule), the final shape, and raw-vs-cleaned counts. The
written interpretations and answer values live in `analytics.pdf`, every number
computed from the cleaned DataFrame (nothing hard-coded).
