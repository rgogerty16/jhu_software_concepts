"""inference.py — load the fine-tuned admissions model and score applicants.

This is the serving side of Module 13. It reloads the model saved by
``train_model.py`` and turns applicant records into predictions, and it is the
only module the Flask app talks to for scoring.

Two properties matter for the deployed page:

* **The model is loaded once.** :func:`load_bundle` is memoized, so the weights
  are read from disk on the first call and reused for every request afterwards.
  Nothing is retrained or re-read when a user visits the page.
* **The text is built by the training code.** The applicant string comes from
  ``applicant_text.build_applicant_text``, the same function ``train_model.py``
  used, so a form submission is encoded exactly the way a training row was.

Run it directly for a self-contained demonstration on two examples::

    python inference.py
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from applicant_text import applicant_text_from_raw, normalize_record

#: Default location of the saved model, beside this file.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"

#: Fallback label mapping, used only if the saved metadata is somehow missing it.
FALLBACK_LABELS = {0: "Rejected", 1: "Accepted"}

#: Fallback truncation length, used only if the saved metadata omits it.
FALLBACK_MAX_LENGTH = 256

#: Batch size used when scoring several applicants at once.
BATCH_SIZE = 16


class ModelNotAvailableError(RuntimeError):
    """Raised when the saved model cannot be found or loaded.

    The Flask app catches this specifically so it can show an explanatory banner
    rather than letting a stack trace reach the browser.
    """


@dataclasses.dataclass
class InferenceBundle:
    """A loaded model and everything needed to use it.

    Attributes:
        model: The fine-tuned sequence classification model, in eval mode.
        tokenizer: The tokenizer saved alongside it.
        metadata: Contents of ``inference_metadata.json``.
        device: Device the model is on.
        labels: Integer class index to human-readable label.
        max_length: Tokenizer truncation length used during training.
    """

    model: torch.nn.Module
    tokenizer: object
    metadata: dict
    device: torch.device
    labels: dict[int, str]
    max_length: int


@dataclasses.dataclass
class Prediction:
    """One applicant's prediction.

    Attributes:
        label: Predicted class name, ``"Accepted"`` or ``"Rejected"``.
        score: Confidence in the predicted class, in ``[0.5, 1.0]``.
        accepted_probability: Model probability of the Accepted class.
        probabilities: Probability for every class, keyed by label name.
        model_input_text: The exact string the model read, for transparency.
    """

    label: str
    score: float
    accepted_probability: float
    probabilities: dict[str, float]
    model_input_text: str


def _resolve_device() -> torch.device:
    """Choose the inference device.

    CPU is the default on purpose. Serving a handful of short sequences per
    request is not compute-bound, and CPU avoids both the accelerator warm-up cost
    on the first request and any device-specific numerical differences between the
    machine that trained the model and the machine serving it.

    Returns:
        The device to run inference on.
    """
    requested = os.environ.get("INFERENCE_DEVICE", "cpu").strip().lower()
    if requested in ("", "cpu"):
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_metadata(model_dir: Path) -> dict:
    """Read the saved preprocessing metadata.

    Args:
        model_dir: Directory holding the saved model.

    Returns:
        The metadata dict, or an empty dict if the file is absent.
    """
    path = model_dir / "inference_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=2)
def load_bundle(model_dir: Path | str = DEFAULT_MODEL_DIR) -> InferenceBundle:
    """Load the fine-tuned model, tokenizer, and metadata from disk.

    Memoized on ``model_dir``, so repeated calls in one process return the same
    already-loaded bundle. This is what keeps the web app from touching disk on
    every request.

    Args:
        model_dir: Directory holding the saved model.

    Returns:
        The loaded :class:`InferenceBundle`.

    Raises:
        ModelNotAvailableError: If the directory is missing, incomplete, or the
            weights fail to load.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise ModelNotAvailableError(
            f"No fine-tuned model at {model_dir}. Run 'python train_model.py' first "
            f"to train and save it."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except (OSError, ValueError, KeyError) as error:
        raise ModelNotAvailableError(
            f"The model at {model_dir} could not be loaded: {error}"
        ) from error

    metadata = _read_metadata(model_dir)
    device = _resolve_device()
    model.to(device)
    model.eval()

    # Prefer the label map saved with the model, then the one on its config, then
    # the hard-coded fallback. Getting this backwards would silently invert every
    # prediction, so it is worth the belt and braces.
    raw_labels = metadata.get("id2label") or model.config.id2label or FALLBACK_LABELS
    labels = {int(key): str(value) for key, value in raw_labels.items()}

    return InferenceBundle(
        model=model,
        tokenizer=tokenizer,
        metadata=metadata,
        device=device,
        labels=labels,
        max_length=int(metadata.get("max_length", FALLBACK_MAX_LENGTH)),
    )


def _score_texts(texts: list[str], bundle: InferenceBundle) -> list[dict[str, float]]:
    """Run the model over already-rendered input strings.

    Args:
        texts: Model input strings.
        bundle: The loaded model bundle.

    Returns:
        One mapping of label name to probability per input.
    """
    results: list[dict[str, float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start : start + BATCH_SIZE]
        encoded = bundle.tokenizer(
            chunk,
            truncation=True,
            max_length=bundle.max_length,
            padding=True,
            return_tensors="pt",
        )
        # DistilBERT's forward does not accept token_type_ids, which the tokenizer
        # emits anyway, so the batch is narrowed to the keys the model wants.
        inputs = {
            key: value.to(bundle.device)
            for key, value in encoded.items()
            if key in ("input_ids", "attention_mask")
        }
        with torch.inference_mode():
            logits = bundle.model(**inputs).logits.float()
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        for row in probabilities:
            results.append({bundle.labels[index]: float(row[index]) for index in range(len(row))})
    return results


def predict_applicants(
    records: list[dict], bundle: InferenceBundle | None = None
) -> list[Prediction]:
    """Score a batch of applicant records.

    Args:
        records: Applicant mappings, in any shape ``normalize_record`` accepts —
            raw scraped JSON, a database row, or submitted form values.
        bundle: A preloaded bundle, or None to use the default cached one.

    Returns:
        One :class:`Prediction` per record, in the same order.

    Raises:
        ModelNotAvailableError: If no saved model can be loaded.
    """
    bundle = bundle or load_bundle()
    texts = [applicant_text_from_raw(record) for record in records]
    scored = _score_texts(texts, bundle)
    predictions = []
    for text, probabilities in zip(texts, scored):
        label = max(probabilities, key=probabilities.__getitem__)
        predictions.append(
            Prediction(
                label=label,
                score=probabilities[label],
                accepted_probability=probabilities.get("Accepted", 0.0),
                probabilities=probabilities,
                model_input_text=text,
            )
        )
    return predictions


def predict_applicant(record: dict, bundle: InferenceBundle | None = None) -> Prediction:
    """Score a single applicant record.

    Args:
        record: Applicant mapping.
        bundle: A preloaded bundle, or None to use the default cached one.

    Returns:
        The prediction for that applicant.

    Raises:
        ModelNotAvailableError: If no saved model can be loaded.
    """
    return predict_applicants([record], bundle)[0]


#: The two applicants scored by the command-line demo: profiles that should sit at
#: opposite ends of the model's range, so a broken reload is obvious at a glance.
DEMO_APPLICANTS = (
    {
        "program": "Computer Science",
        "university": "Johns Hopkins University",
        "semester": "Fall",
        "year": "2026",
        "degree": "Masters",
        "student_type": "American",
        "gpa": 3.92,
        "gre": 169,
        "gre_v": 163,
        "gre_aw": 5.0,
        "comments": "Strong systems background, two internships, and a named scholarship offer.",
    },
    {
        "program": "Physics",
        "university": "Stanford University",
        "semester": "Fall",
        "year": "2026",
        "degree": "PhD",
        "student_type": "International",
        "gpa": 2.75,
        "gre": None,
        "gre_v": None,
        "gre_aw": None,
        "comments": "",
    },
)


def main() -> None:
    """Reload the saved model and score the two demonstration applicants."""
    try:
        bundle = load_bundle()
    except ModelNotAvailableError as error:
        print(f"Cannot run inference: {error}")
        raise SystemExit(1) from error

    print(f"Loaded {bundle.metadata.get('base_model', 'model')} from "
          f"{DEFAULT_MODEL_DIR.name}/ on {bundle.device.type}.")
    print(f"Trained at      : {bundle.metadata.get('trained_at', 'unknown')}")
    print(f"Label mapping   : {bundle.labels}")
    print(f"Max length      : {bundle.max_length}")
    print(f"Template version: {bundle.metadata.get('template_version', 'unknown')}")

    for index, (record, prediction) in enumerate(
        zip(DEMO_APPLICANTS, predict_applicants(list(DEMO_APPLICANTS), bundle)), start=1
    ):
        normalized = normalize_record(record)
        print()
        print(f"--- Example {index}: {normalized['degree']} {normalized['program']}, "
              f"GPA {normalized['gpa']} ---")
        print("Model input:")
        print("\n".join(f"    {line}" for line in prediction.model_input_text.splitlines()))
        print(f"Prediction : {prediction.label}")
        print(f"Model score: {prediction.score:.4f}")
        print(f"P(Accepted): {prediction.accepted_probability:.4f}")

    print()
    print("Both predictions came from weights read off disk, with no retraining.")


if __name__ == "__main__":
    main()
