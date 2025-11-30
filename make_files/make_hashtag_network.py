import pandas as pd
import numpy as np
import ast
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

# ----------------------------------------------------
# 1) Load cleaned tweet data
# ----------------------------------------------------
DATA_PATH = "/Users/ame/02805_climate_conv/data/cleaned_twitter_embedded_data_hashtags_fixed.csv"
OUTPUT_PATH = "/Users/ame/02805_climate_conv/networks/hashtag_cooccurrence_network.gexf"

print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df))


# ----------------------------------------------------
# 2) Parse hashtag column into Python lists
# ----------------------------------------------------
def parse_hashtags(cell):
    """Always return a list of lowercase hashtags."""
    if isinstance(cell, list):
        return [str(x).lower().lstrip("#") for x in cell if str(x).strip()]

    if isinstance(cell, str):
        s = cell.strip()
        if s == "" or s.lower() == "nan":
            return []

        # Try list format
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x).lower().lstrip("#") for x in parsed if str(x).strip()]
            except:
                pass

        # Fallback: comma-split
        return [
            p.strip().lower().lstrip("#")
            for p in s.split(",")
            if p.strip()
        ]

    return []


print("Parsing hashtags...")
df["hashtags"] = df["hashtags"].apply(parse_hashtags)

all_tags = df["hashtags"].explode()
print("Unique hashtags in raw data:", all_tags.nunique())


# ----------------------------------------------------
# 3) Keep only the 700 mid-frequency hashtags (20–100)
# ----------------------------------------------------
MIN_FREQ = 20
MAX_FREQ = 100

tag_counts = all_tags.value_counts()
keep_tags = set(tag_counts[(tag_counts >= MIN_FREQ) & (tag_counts <= MAX_FREQ)].index)

print(f"Hashtags kept (freq {MIN_FREQ}-{MAX_FREQ}): {len(keep_tags)}")

# Filter each tweet’s tag list
df["hashtags_filtered"] = df["hashtags"].apply(
    lambda tags: [t for t in tags if t in keep_tags]
)

filtered_tags = df["hashtags_filtered"].explode()
print("Unique hashtags after filtering:", filtered_tags.nunique())


# ----------------------------------------------------
# 4) Build hashtag → list of tweetids
# ----------------------------------------------------
print("Building hashtag → tweetlist map...")
hashtag_map = defaultdict(list)

for row in df.itertuples(index=False):
    tid = str(row.tweetid)
    for tag in row.hashtags_filtered:
        hashtag_map[tag].append(tid)

print("Distinct hashtags in mapping:", len(hashtag_map))


# ----------------------------------------------------
# 5) Build the hashtag–hashtag co-occurrence network
# ----------------------------------------------------
print("Building hashtag–hashtag network...")

G = nx.Graph()

# Add nodes first, with count attribute
for tag in keep_tags:
    G.add_node(tag, count=int(tag_counts[tag]))

# Add edges: co-occurrence inside a tweet
for tags in df["hashtags_filtered"]:
    unique_tags = sorted(set(tags))
    if len(unique_tags) < 2:
        continue

    for a, b in combinations(unique_tags, 2):
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)

print("Hashtag nodes:", G.number_of_nodes())
print("Hashtag edges:", G.number_of_edges())


# ----------------------------------------------------
# 6) Save to GEXF
# ----------------------------------------------------
print(f"Saving network to {OUTPUT_PATH} ...")
nx.write_gexf(G, OUTPUT_PATH)
print("Done!")