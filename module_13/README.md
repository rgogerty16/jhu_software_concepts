# Module 13 — Fine-Tuning DistilBERT and Deploying "Will You Get In?"

- **Name:** Ryan Gogerty
- **JHED:** rgogerty1
- **Course:** EN.605.256 — Modern Software Concepts in Python
- **Assignment:** Module 13 — Scale & LM Deployment

A pretrained DistilBERT language model, fine-tuned on the Grad Café admissions
dataset collected earlier in the semester, and served from a new page on the
Module 5 Flask website.

Where Module 12 hand-built a two-layer network in NumPy over six structured
features, this module fine-tunes 67 million pretrained parameters over the **full
applicant record** — free text and structured fields together — and deploys the
result as an interactive web page.

> **This is coursework, not an admissions tool.** The model is fitted to roughly
> 19,000 voluntarily posted, unverified, self-reported Grad Café entries. Its
> output is a pattern match against that data. It is not an admissions decision,
> not a prediction of one, and not advice about where to apply.

## What the program does

1. **Loads and filters the dataset.** Reads all 30,000 scraped applicant records
   into a Pandas DataFrame, keeps only the Accepted and Rejected rows, drops repeat
   postings of the same entry URL, normalizes missing values to a single
   representation, converts the scalars to numeric types, and creates the target
   (`label = 1` for Accepted, `0` for Rejected). Every filtering rule reports how
   many rows it cost.
2. **Builds one unified text input per applicant.** Ten labelled lines covering
   three text fields and seven non-text fields, from a single template shared by
   the training script and the web app.
3. **Splits 80/20 and tokenizes.** A stratified `train_test_split` with
   `test_size=0.2`, `random_state=42`, `shuffle=True`, then DistilBERT's own
   WordPiece tokenizer with truncation at 256 tokens and dynamic per-batch padding.
4. **Fine-tunes DistilBERT.** A hand-written PyTorch training loop — custom
   `Dataset`, `DataLoader`, AdamW, a linear warmup-and-decay schedule, gradient
   clipping — rather than the Hugging Face `Trainer`, so every part of the loop is
   visible. The best epoch's parameters are kept, not the last.
5. **Evaluates on the held-out test set.** Accuracy, precision, recall, F1, and a
   confusion matrix, plus class distributions, worked probability examples,
   correctly and incorrectly classified cases, a like-for-like slice against
   Module 12, and an ablation that measures what the comments field actually
   contributes.
6. **Saves and reloads the model.** Weights, tokenizer, and the preprocessing
   metadata needed to reconstruct the input format, then proves the round trip by
   reloading from disk and scoring two fresh applicants.
7. **Serves it from the website.** A new **Will You Get In?** page collects an
   applicant profile, renders it through the same template used in training, and
   shows the predicted class with a confidence score.

## Files included

| File | What it is |
| --- | --- |
| `train_model.py` | The whole training pipeline, printed as the six numbered assignment sections |
| `inference.py` | Loads the saved model once and scores applicants; also a standalone two-example demo |
| `leakage_analysis.py` | Measures how much of the model's skill comes from comments that describe the outcome |
| `leakage_analysis.txt` | Output of that analysis for the committed model |
| `applicant_text.py` | The unified text template — the single source shared by training and serving |
| `run.py` | Starts the Flask website |
| `app/__init__.py` | Flask app factory, routes, and form validation |
| `app/db.py`, `app/query_data.py` | Postgres helpers carried over from Module 5, used by `/analysis` |
| `app/templates/`, `app/static/` | `base.html`, `index.html`, the new `predict.html`, and the stylesheet |
| `model/` | The saved fine-tuned model: sharded weights, tokenizer, and `inference_metadata.json` |
| `applicant_data.json` | The 30,000-record scraped dataset, carried forward from Module 2 |
| `training.log` | Full transcript of the committed training run |
| `metrics.json` | Machine-readable final metrics, including every reported slice |
| `confusion_matrix.png`, `training_curve.png` | Generated evaluation plots |
| `writeup.md`, `writeup.pdf`, `make_writeup.py` | The write-up and the renderer that builds the PDF |
| `screenshots/` | Training output, the blank prediction page, and a completed prediction |
| `requirements.txt` | Dependencies |
| `.pylintrc` | Line-length and module-length limits; no message is disabled |

## How to run

### 1. Install

```bash
cd module_13
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is required; the committed run used 3.12.13.

### 2. Train the model

```bash
python train_model.py
```

Reads `./applicant_data.json`, prints all six sections to the terminal, and
mirrors everything into `training.log`. It writes `model/`, `metrics.json`,
`confusion_matrix.png`, and `training_curve.png`. Expect roughly 20 minutes on a
CPU; the log records the exact wall clock.

Useful flags:

```bash
python train_model.py --smoke            # 2,000 rows, 1 epoch, artifacts under *_smoke names
python train_model.py --epochs 5         # override any hyperparameter
python train_model.py --device mps       # opt in to Apple GPU acceleration (see the note below)
python train_model.py --source postgres  # read from the gradcafe.applicants table instead
```

`--smoke` writes to `model_smoke/`, `training_smoke.log`, and so on, so a pipeline
check can never overwrite the committed run.

> **A note on `--device mps`.** The default is CPU on purpose. A full-length run on
> Apple's Metal backend deadlocked inside the Metal shader compiler
> (`MPSGraphExecutable specializeWithDevice:` → `optimizeOriginalModule`), sleeping
> indefinitely while accumulating no CPU time. Dynamic padding gives each batch a
> distinct tensor shape, every new shape triggers a fresh Metal graph compilation,
> and a long run produces far more shapes than a short one — which is why the
> 2,000-row check passed and the 15,000-row run hung. CPU is slower per step but
> finishes every time. The write-up discusses this at more length.

### 3. Run inference from the saved model

```bash
python inference.py
```

Reloads `model/` in a fresh process — no retraining — and scores two contrasting
applicants, printing the exact text the model read, the predicted class, and the
probabilities.

### 4. Start the website

```bash
python run.py
```

Then open <http://127.0.0.1:5000>:

- **`/will-you-get-in`** — the prediction page. Needs no database.
- **`/analysis`** — the Module 5 SQL analysis page. Needs PostgreSQL with the
  `gradcafe` database and its `applicants` table. If Postgres is not running the
  page explains that and stays navigable; the predictor is unaffected.

`PORT=8080 python run.py` changes the port, and
`DATABASE_URL=postgresql:///gradcafe python run.py` sets the database explicitly.

The model is loaded **once**, when the app starts, so no request ever reads weights
from disk and nothing is ever retrained on a page view. Starting the app before
training succeeds: the prediction page says the model is missing and tells you to
run `train_model.py`.

### 5. Rebuild the write-up PDF

```bash
python make_writeup.py
```

## Outputs produced

- `training.log` — the complete transcript, rewritten on each run
- `metrics.json` — accuracy, precision, recall, F1, and the confusion matrix, for
  the whole test fold and for each reported slice
- `confusion_matrix.png` — the test confusion matrix
- `training_curve.png` — training loss per logged step, and test accuracy and loss
  per epoch
- `model/` — the reloadable fine-tuned model

## Results from the committed run

| Metric | Value |
| --- | --- |
| Modelling rows (after filtering) | 19,324 — 8,799 Accepted / 10,525 Rejected |
| Train / test | 15,459 / 3,865, stratified |
| Epochs / best epoch | 3 / 3 |
| Training wall clock | 29.2 minutes on CPU |
| **Test accuracy** | **0.8111** |
| Precision / Recall / F1 (Accepted) | 0.8156 / 0.7562 / 0.7848 |
| F1 (macro) | 0.8083 |
| Majority-class baseline | 0.5446 (lift **+0.2665**) |
| Module 12 for comparison | 0.7043 on its 24,326 rows |
| Same slice as Module 12 (Masters/PhD) | 0.8102, i.e. **+0.106** |
| Accuracy without comments (ablation) | 0.7265, so the text is worth **+0.085** |
| **Accuracy on inputs a real user can supply** | **0.7617** |

That last row is the number worth taking seriously, and it is the most important
result in the project. Grad Café comments are written *after* a decision arrives, so
64% of them contain language describing an outcome — "Rejected on May 26", "Accepted
off of waitlist". The model learned to read it: it scores 0.8885 on test rows with
such language and 0.7617 on rows without. Anyone using the prediction page is
necessarily in the second group, since they are asking precisely because they do not
know the outcome. `leakage_analysis.py` measures this against the saved model without
retraining, and `writeup.pdf` discusses what follows from it.

Full numbers are in `metrics.json`, `training.log`, and `leakage_analysis.txt`.

## Code quality

```bash
pylint applicant_text.py inference.py run.py train_model.py app/ --rcfile=.pylintrc
```

Scores 10.00/10 with no messages disabled. `.pylintrc` only moves the line-length
limit to 100 and the module-length limit, so that `train_model.py` can hold all six
assignment sections in the order a grader reads them.
