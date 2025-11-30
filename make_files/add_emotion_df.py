import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import help_files.get_emotions as get_emotions  


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_twitter_embedded_data_hashtags_fixed.csv")
print(f"Loading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Data shape before adding emotion:", df.shape)

# Add emotion column based on clean_text
def extract_emotion(text: str):
    top_emotion, _ = get_emotions.get_emotion(text)
    return top_emotion

print("Computing emotions for each tweet (this may take a bit)...")
df["emotion"] = df["clean_text"].apply(extract_emotion)

print("Data shape after adding emotion:", df.shape)

# Save updated dataframe
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_twitter_embedded_data_hashtags_emotion.csv")
df.to_csv(OUT_PATH, index=False)
print(f"Saved updated data with emotion column to:\n{OUT_PATH}")