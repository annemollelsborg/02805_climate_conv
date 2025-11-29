import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

df = pd.read_csv("/Users/ame/02805_climate_conv/data/cleaned_twitter_sentiment_data.csv",
                 encoding="latin1",
                 on_bad_lines="skip")

df["message"] = df["message"].astype(str)

df = df[df["message"].str.strip().replace("nan", np.nan).notna()]

def clean_text(msg):
    msg = msg.lower()
    msg = re.sub(r"http\S+", "", msg)       # remove URLs
    msg = re.sub(r"rt\s*@\w+:", "", msg)    # remove RT prefix
    msg = re.sub(r"@\w+", "", msg)          # remove mentions
    msg = re.sub(r"#", "", msg)             # remove hashtags
    msg = re.sub(r"[^a-z0-9\s]", " ", msg)  # keep letters/numbers
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg

df["clean_text"] = df["message"].apply(clean_text)

df = df[df["clean_text"].str.len() > 0]

def is_valid_unicode(x):
    try:
        x.encode("utf-8")
        return True
    except:
        return False

df = df[df["clean_text"].apply(is_valid_unicode)]

df = df[df["clean_text"].str.len() <= 8000]

df = df.reset_index(drop=True)

load_dotenv()  # loads .env variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_batch(text_batch, client):
    """Embeds a batch of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_batch
    )
    return [item.embedding for item in response.data]

BATCH_SIZE = 500

all_embeddings = []
texts = df["clean_text"].tolist()

print("Embedding", len(texts), "tweets...")

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    print(f"Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} items)")
    batch_embeddings = embed_batch(batch, client)
    all_embeddings.extend(batch_embeddings)

embeddings = np.array(all_embeddings)
print("Embeddings shape:", embeddings.shape)

SAVE_PATH = "/Users/ame/02805_climate_conv/data/tweet_embeddings.npy"

np.save(SAVE_PATH, embeddings)
print(f"Saved embeddings to {SAVE_PATH} with shape {embeddings.shape}")