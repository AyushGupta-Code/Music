# para_to_emo.py
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---- Configuration ----
# Prefer a local folder if present; otherwise use the Hub repo ID. Override with EMOTION_MODEL_DIR.
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOCAL_MODEL_DIR = _REPO_ROOT / "hf_models" / "twitter-roberta-base-emotion"
ENV_MODEL_DIR = "EMOTION_MODEL_DIR"
HUB_MODEL_ID = "cardiffnlp/twitter-roberta-base-emotion"

# Torch device (CPU is fine; keep it explicit and deterministic)
DEVICE = torch.device("cpu")


# ---- Lazy-loaded globals (so we only load once per process) ----
_TOKENIZER = None
_MODEL = None
_LABELS = None


def _resolve_model_src() -> Tuple[str, bool]:
    """
    Returns (model_source, local_only_flag).
    model_source: path or hub id
    local_only_flag: True iff we should force local-only loading
    """
    env_override = os.environ.get(ENV_MODEL_DIR)
    if env_override:
        local_dir = Path(env_override).expanduser().resolve()
    else:
        local_dir = _DEFAULT_LOCAL_MODEL_DIR

    if local_dir.is_dir():
        return str(local_dir), True
    return HUB_MODEL_ID, False


def _load_model_and_tokenizer():
    """Load tokenizer, model, and labels once; cache in globals."""
    global _TOKENIZER, _MODEL, _LABELS
    if _TOKENIZER is not None and _MODEL is not None and _LABELS is not None:
        return

    model_src, local_only = _resolve_model_src()

    # Load tokenizer/model
    _TOKENIZER = AutoTokenizer.from_pretrained(model_src, local_files_only=local_only)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_src, local_files_only=local_only)
    _MODEL.to(DEVICE)
    _MODEL.eval()

    # Derive labels from config (robust to any checkpoint label order)
    id2label = getattr(_MODEL.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        # Sort by id to preserve correct order
        _LABELS = [id2label[i] for i in sorted(id2label.keys(), key=int)]
    else:
        # Fallback for known checkpoint
        _LABELS = ["anger", "joy", "optimism", "sadness"]


@torch.no_grad()
def detect_emotion(paragraph: str) -> Dict[str, float]:
    """
    Returns a dict of emotion -> score for the given paragraph.
    Scores are a probability distribution (sum to 1.0).
    """
    _load_model_and_tokenizer()

    text = (paragraph or "").strip()
    if not text:
        # Return a neutral distribution if empty input
        return {label: (1.0 / len(_LABELS)) for label in _LABELS}

    enc = _TOKENIZER(text, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = _MODEL(**enc).logits  # [1, num_labels]
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

    return {label: float(score) for label, score in zip(_LABELS, probs)}


# Optional: quick CLI test
if __name__ == "__main__":
    sample = "I can't believe this happened. I'm so frustrated right now."
    print(detect_emotion(sample))
