from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL = "cardiffnlp/twitter-roberta-base-emotion"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# model returns logits for these labels:
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def get_emotion(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None, None

    # tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # get model output
    with torch.no_grad():
        logits = model(**inputs).logits

    # softmax for probabilities
    probs = torch.softmax(logits, dim=1).numpy()[0]

    # get top emotion
    top_idx = np.argmax(probs)
    top_emotion = EMOTION_LABELS[top_idx]

    return top_emotion, dict(zip(EMOTION_LABELS, probs))