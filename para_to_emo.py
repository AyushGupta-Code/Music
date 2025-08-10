# para_to_emo.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

def detect_emotion(paragraph: str) -> dict:
    """
    Offline emotion detection using your local HF cache.
    Returns scores dict for ['anger', 'joy', 'optimism', 'sadness'].
    """
    model_dir = "/mnt/c/Users/ayush/Desktop/Music/hf_models/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    inputs = tokenizer(paragraph, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])

    labels = ['anger', 'joy', 'optimism', 'sadness']
    return dict(zip(labels, scores))
