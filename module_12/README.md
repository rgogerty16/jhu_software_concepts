# Module 12 — A Two-Layer Neural Network for Graduate Admissions Prediction

- **Name:** Ryan Gogerty
- **JHED:** rgogerty1
- **Course:** EN.605.256 — Modern Software Concepts in Python
- **Assignment:** Module 12 — Neural Networks

## What the program does

`neural_network.py` trains a fully connected **6 → 6 → 1** neural network,
implemented with **NumPy only**, to predict whether a graduate-school applicant
was *Accepted* or *Rejected*. It runs the whole assignment end to end in one
pass:

1. **Load and prepare** the applicant records — keep only Accepted/Rejected
   outcomes and Masters/PhD degrees, convert `gpa`, `gre`, `gre_v`, and `gre_aw`
   to floats, and build `ms_vs_phd` (PhD = 1), `international_vs_local`
   (International = 1), and `target` (Accepted = 1).
2. **Split and preprocess** — an 80/20 split, then medians, means, and standard
   deviations computed on the **training split only** and applied to both
   halves (a zero standard deviation becomes 1 before scaling).
3. **Build the network** — weights from N(0, 0.1), biases at 0, sigmoid after
   both layers, `forward()`, `backward()`, `predict_proba()`, and `predict()`.
4. **Train** with full-batch gradient descent, printing progress every 100
   epochs and stopping once test MSE has not improved for 100 consecutive
   epochs, then restoring the best epoch's parameters.
5. **Evaluate** the restored model against the majority-class baseline.
6. **Plot** training and test MSE against epoch.
7. **Predict** outcomes for five hand-written artificial applicants pushed
   through the identical preprocessing pipeline.

scikit-learn is used for exactly one thing — `train_test_split`. No PyTorch,
TensorFlow, Keras, JAX, or scikit-learn neural-network utilities appear
anywhere; the forward pass, backpropagation, loss, and training loop are all
hand written.

## Files included

| File | What it is |
| --- | --- |
| `neural_network.py` | The complete solution: sections 1–7 in one file |
| `README.md` | This file |
| `writeup.md` | Write-up source: printouts, results, graph, findings, reflection |
| `writeup.pdf` | The write-up deliverable, rendered from `writeup.md` |
| `make_writeup.py` | Renders `writeup.md` into `writeup.pdf` |
| `training.log` | Full transcript of the training run (generated) |
| `mse_curve.png` | Training/test MSE against epoch (generated) |
| `applicant_data.json` | The Grad Café dataset (30,000 records, from Module 2) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
cd module_12
python3.12 -m venv .venv          # Python 3.10+ required; developed on 3.12
source .venv/bin/activate
pip install -r requirements.txt

python neural_network.py          # reads ./applicant_data.json
```

The run takes about **9 seconds**. To point it at a different dataset:

```bash
python neural_network.py --data path/to/applicants.jsonl
```

The loader accepts either format: **JSON Lines** (one JSON object per line, as
the assignment describes) or a single pretty-printed **JSON array**, which is
how the Grad Café corpus has been stored since Module 2. It also maps that
corpus's `status`, `degree`, and `student_type` fields onto the assignment's
`applicant_status`, `masters_or_phd`, and `citizenship` names, so either file
works without a code change.

To regenerate the PDF write-up after editing `writeup.md`:

```bash
python make_writeup.py
```

## Outputs produced

Running `neural_network.py` prints all seven sections to the console and
produces two files:

- **`training.log`** — the complete transcript, including the dataset summary,
  the training-set statistics, the full epoch-by-epoch progress table, the final
  evaluation, and the artificial-applicant predictions.
- **`mse_curve.png`** — training MSE and test MSE against epoch, with the
  restored best epoch marked.

## Results from the committed run

| Metric | Value |
| --- | --- |
| Rows after filtering | 24,326 of 30,000 (10,687 Accepted / 13,639 Rejected) |
| Train / test split | 19,460 / 4,866 |
| Early stopping | epoch 6,039 (best epoch 5,939) |
| Best test MSE | 0.207908 |
| Final training accuracy | 0.7053 |
| Final test accuracy | **0.7043** |
| Majority-class baseline | 0.5697 |

Test accuracy of 70.4% clears the assignment's 53% bar and the 57.0%
majority-class baseline. The write-up explains where that accuracy comes from —
mostly the Masters/PhD flag, since 75.9% of Masters rows in this dataset were
accepted versus 31.6% of PhD rows — and why that makes the model a weaker
admissions predictor than the headline number suggests.
