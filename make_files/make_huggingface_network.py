import numpy as np
import pandas as pd
import networkx as nx
try:
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        USE_GPU = True
        print("Using PyTorch with MPS (Metal) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        USE_GPU = True
        print("Using PyTorch with CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        USE_GPU = False
        print("No GPU available, using CPU")
except ImportError:
    USE_GPU = False
    device = None
    print("PyTorch not available, falling back to CPU")

# Load embedded + cleaned dataset
df = pd.read_csv("/Users/ame/02805_climate_conv/data/cleaned_twitter_embedded_data.csv")

# Ensure tweetid is string
df["tweetid"] = df["tweetid"].astype(str)

# Convert embeddings column (string repr) to real vectors if needed
if isinstance(df["embeddings"].iloc[0], str):
    import ast
    df["embeddings"] = df["embeddings"].apply(ast.literal_eval)

embeddings = np.vstack(df["embeddings"].to_numpy())
print("Embeddings shape:", embeddings.shape)

# Build nearest neighbor structure
k = 50                    # neighbors per node
THRESHOLD = 0.8          # cosine similarity threshold

print("Computing pairwise cosine similarities...")

if USE_GPU:
    # Transfer to GPU
    embeddings_tensor = torch.from_numpy(embeddings).float().to(device)

    # Normalize embeddings for cosine similarity
    embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

    # Compute cosine similarity matrix in chunks to avoid memory issues
    n_samples = embeddings_tensor.shape[0]
    chunk_size = min(1000, n_samples)  # Process in chunks

    indices_list = []
    similarities_list = []

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        # Compute similarity for this chunk against all points
        sim_chunk = torch.mm(embeddings_normalized[i:end_i], embeddings_normalized.T)

        # Get top-k for each row
        top_k_similarities, top_k_indices = torch.topk(sim_chunk, k=k, dim=1, largest=True)

        indices_list.append(top_k_indices.cpu())
        similarities_list.append(top_k_similarities.cpu())

        if (i // chunk_size) % 10 == 0:
            print(f"Processed {end_i}/{n_samples} samples...")

    # Concatenate and convert to numpy
    indices = torch.cat(indices_list, dim=0).numpy()
    similarities = torch.cat(similarities_list, dim=0).numpy()
    print("GPU computation complete!")
else:
    # CPU fallback
    from sklearn.neighbors import NearestNeighbors
    print("Building KNN index on CPU...")
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    similarities = 1 - distances

# Create Graph and add all node attributes
G = nx.Graph()

print("Adding nodes with full attributes...")

for idx, row in df.iterrows():
    attributes = row.to_dict()        # add ALL columns as attributes
    tid = row["tweetid"]
    G.add_node(tid, **attributes)

print("Nodes added:", G.number_of_nodes())

# Add edges based on similarity threshold
print("Adding edges based on similarity threshold =", THRESHOLD)
edges_added = 0

for i in range(len(df)):
    tid1 = df.loc[i, "tweetid"]

    for j_idx, sim in zip(indices[i], similarities[i]):
        if i == j_idx:
            continue
        if sim < THRESHOLD:
            continue

        tid2 = df.loc[j_idx, "tweetid"]
        G.add_edge(tid1, tid2, weight=float(sim))
        edges_added += 1

print("Edges added:", edges_added)

# Keep only lightweight node attributes for GEXF export
cols_to_keep = ["tweetid", "message", "clean_text", "date", "location", "sentiment"]


for n, data in G.nodes(data=True):
    keys = list(data.keys())
    for k in keys:
        if k not in cols_to_keep:
            del data[k]

# --- Sanitize remaining attributes for GEXF (must be simple types) ---
for n, data in G.nodes(data=True):
    for k, v in list(data.items()):
        # Convert None or NaN to empty string
        if v is None or (isinstance(v, float) and np.isnan(v)):
            data[k] = ""
        # Convert everything else to string to avoid GEXF dynamic-attribute parsing
        else:
            data[k] = str(v)

print("Final graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Save graph
output_path = "/Users/ame/02805_climate_conv/networks/climate_tweet_network_embedded.gexf"
nx.write_gexf(G, output_path)

print("Graph saved to:", output_path)