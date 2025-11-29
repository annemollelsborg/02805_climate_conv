from datasets import load_dataset
import re
import pandas as pd
import numpy as np
import json

# Load the full dataset from Hugging Face
ds = load_dataset("AndresR2909/climate_twitter_text_embeddings")

# Inspect available splits (usually 'train')
print(ds)

# Access the main split
data = ds["train"]

df = data.to_pandas()

# Split metadata dictionary into separate columns
def extract_metadata_field(row, key):
    try:
        return row.get(key)
    except:
        return None

# Ensure metadata stays intact as original
df["date"] = df["metadata"].apply(lambda x: extract_metadata_field(x, "date"))
df["hashtags"] = df["metadata"].apply(lambda x: extract_metadata_field(x, "hashtags"))
df["location"] = df["metadata"].apply(lambda x: extract_metadata_field(x, "location"))
df["sentiment"] = df["metadata"].apply(lambda x: extract_metadata_field(x, "sentiment1"))

# Rename columns to match previous dataset
df = df.rename(columns={"id": "tweetid", "text": "message"})

# Ensure message is string
df["message"] = df["message"].astype(str)
df = df[df["message"].str.strip().replace("nan", np.nan).notna()]

def clean_text(msg):
    msg = msg.lower()
    msg = re.sub(r"http\S+", "", msg)
    msg = re.sub(r"rt\s*@\w+:", "", msg)
    msg = re.sub(r"@\w+", "", msg)
    msg = re.sub(r"#", "", msg)
    msg = re.sub(r"[^a-z0-9\s]", " ", msg)
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
df = df.drop_duplicates(subset="clean_text", keep="first")
df = df.reset_index(drop=True)

print(df.head())

embeddings = np.vstack(df["embeddings"].to_numpy())  # convert list-of-lists to array
print("Embeddings shape:", embeddings.shape)

# Print column names
print("Columns in cleaned DataFrame:", df.columns.tolist())

# Print first 10 rows
print(df.head(10))

# Convert embeddings to a JSON string so they are stored in a clean, parseable format
# (instead of a truncated NumPy-style string) when written to CSV.
df["embeddings"] = df["embeddings"].apply(lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x))

output_path = "/Users/ame/02805_climate_conv/data/cleaned_twitter_embedded_data.csv"
df.to_csv(output_path, index=False)
print(f"Saved cleaned data to {output_path}")