from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load pretrained emotion model
def detect_emotion(paragraph):
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(paragraph, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])

    labels = ['anger', 'joy', 'optimism', 'sadness']
    return dict(zip(labels, scores))

