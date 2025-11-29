import networkx as nx
import pandas as pd
import numpy as np
import help_files.get_emotions as get_emotions


# Load graph
G = nx.read_gexf("/Users/ame/02805_climate_conv/networks/climate_tweet_network.gexf")
print("Graph loaded from file")

# Load cleaned tweet data
df = pd.read_csv("/Users/ame/02805_climate_conv/data/cleaned_twitter_sentiment_data.csv")
print("Cleaned tweet data loaded")

# Work with ALL nodes; only filter by character limit later
all_node_ids = list(G.nodes())
df_subset = df[df["tweetid"].astype(str).isin(all_node_ids)].copy()

# Filter out tweets with too short text
df_subset = df_subset[df_subset["clean_text"].str.len() >= 100]

print("Selected tweets:", len(df_subset))

emotions = []
emotion_probs = []

for row in df_subset.itertuples(index=False):
    emotion, prob_dict = get_emotions.get_emotion(row.clean_text)
    emotions.append(emotion)
    emotion_probs.append(prob_dict)

df_subset["emotion"] = emotions
df_subset["emotion_probs"] = emotion_probs

# Make sure tweetid matches node IDs (string)
df_subset["tweetid"] = df_subset["tweetid"].astype(str)

# Build mapping: tweetid -> emotion
emotion_attr = df_subset.set_index("tweetid")["emotion"].to_dict()

# Build mapping: tweetid -> emotion_probs (dict of 6 probs)
emotion_probs_attr = df_subset.set_index("tweetid")["emotion_probs"].to_dict()

# Attach as node attributes on G
nx.set_node_attributes(G, emotion_attr, "emotion")
nx.set_node_attributes(G, emotion_probs_attr, "emotion_probs")

# Save the new graph with emotion attributes to file
nx.write_gexf(G, "/Users/ame/02805_climate_conv/networks/climate_tweet_network_with_emotions.gexf")
print("Graph with emotions saved to file")