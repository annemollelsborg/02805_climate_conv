import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import pandas as pd

# Load cleaned tweet data
df = pd.read_csv("/Users/ame/02805_climate_conv/data/cleaned_twitter_sentiment_data.csv")

# Build nearest-neighbor search over embeddings

EMBEDDINGS_PATH = "/Users/ame/02805_climate_conv/data/tweet_embeddings.npy"

embeddings = np.load(EMBEDDINGS_PATH)

print("Building nearest-neighbor index...")

k = 50  # number of neighbors

nn = NearestNeighbors(n_neighbors=k, metric="cosine")
nn.fit(embeddings)

distances, indices = nn.kneighbors(embeddings)

# Cosine similarity = 1 - cosine distance
similarities = 1 - distances

THRESHOLD = 0.60
G = nx.Graph()

# Make sure tweetid matches node IDs (usually strings)
df["tweetid"] = df["tweetid"].astype(str)

# ---- 1) ADD NODES FIRST ----
for _, row in df.iterrows():
    tid = row["tweetid"]
    G.add_node(
        tid,
        tweetid=tid,
        sentiment=row["sentiment"],   # numeric
        message=row["message"],
        clean_text=row["clean_text"],
    )

# ---- 2) MAP NUMERIC SENTIMENT → LABEL ----
sentiment_map = {
    2: "news",
    1: "pro",
    0: "neutral",
    -1: "anti",
}

df["sentiment_label"] = df["sentiment"].map(sentiment_map)

# Build dict: tweetid → label
label_attr = df.set_index("tweetid")["sentiment_label"].to_dict()

# ---- 3) NOW ADD LABELS TO EXISTING NODES ----
nx.set_node_attributes(G, label_attr, "sentiment_label")

# ---- 4) ADD EDGES ----
edges_added = 0
print("Building network with similarity threshold =", THRESHOLD)

for i in range(len(df)):
    for j_idx, sim in zip(indices[i], similarities[i]):
        if sim >= THRESHOLD and i != j_idx:
            tid1 = df.loc[i, "tweetid"]
            tid2 = df.loc[j_idx, "tweetid"]
            G.add_edge(tid1, tid2, weight=float(sim))
            edges_added += 1


print("Network created.")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

nx.write_gexf(G, "/Users/ame/02805_climate_conv/networks/climate_tweet_network.gexf")
print("Graph saved to networks")