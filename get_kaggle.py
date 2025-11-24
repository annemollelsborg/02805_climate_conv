import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import re
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

load_dotenv()
print("KAGGLE_USERNAME:", os.getenv("KAGGLE_USERNAME"))  # quick sanity check
print("KAGGLE_KEY exists:", os.getenv("KAGGLE_KEY") is not None)

api = KaggleApi()
api.authenticate()   # uses KAGGLE_USERNAME and KAGGLE_KEY from env vars

download_dir = "/Users/ame/02805_climate_conv/data"

# Create folder if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

api.dataset_download_files(
    "edqian/twitter-climate-change-sentiment-dataset",
    path=download_dir,
    unzip=True  # unzip so you get the actual files
)

print("Files in download_dir:", os.listdir(download_dir))

csv_filename = "twitter_sentiment_data.csv"
csv_path = os.path.join(download_dir, csv_filename)

df = pd.read_csv(csv_path, encoding='latin1')
print(df.head())
print("Shape:", df.shape)

print("Column names:", df.columns.tolist())
print(df.head())        # show first rows
print(df.info())        # show column types
print(df.sample(3))     # show some random tweets