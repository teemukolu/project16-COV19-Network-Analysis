import pandas as pd
import csv
import networkx as nx
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Restrict dataset to Oct–Nov 2019
# Load TweetsCOV19.tsv file
try:
    data = pd.read_csv(
        "TweetsCOV19.tsv",
        sep="\t",
        low_memory=False,
        on_bad_lines="skip",  # Skip lines with extra fields
        quoting=csv.QUOTE_NONE,  # Avoid quote issues
        names=[
            "Tweet Id", "Username", "Timestamp", "#Followers", "#Friends",
            "#Retweets", "#Favorites", "Entities", "Sentiment", "Mentions",
            "Hashtags", "URLs"
        ],  # Column names from description
        header=None  # No header row
    )
except FileNotFoundError:
    print("Error: TweetsCOV19.tsv not found")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Help print few rows to verify
print("Sample data (first 5 rows):")
print(data[["Tweet Id", "Timestamp"]].head(5))

# Check date range
timestamp_col = "Timestamp"
print("\nDataset date range:")
print(f"Min Timestamp: {data[timestamp_col].min()}")
print(f"Max Timestamp: {data[timestamp_col].max()}")

# Convert timestamp to datetime with specific format
data[timestamp_col] = pd.to_datetime(
    data[timestamp_col],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce"
)

# Filter to Oct–Nov 2019
start_date = "2019-10-01"
end_date = "2019-11-30"
filtered_data = data[(data[timestamp_col] >= start_date) & (data[timestamp_col] <= end_date)]

# Drop rows with invalid timestamps
filtered_data = filtered_data.dropna(subset=[timestamp_col])

# Save filtered dataset
filtered_data.to_csv("filtered_tweets_oct_nov_2019.csv", index=False)
print(f"Task 1 Complete: Saved {len(filtered_data)} tweets to filtered_tweets_oct_nov_2019.csv")

# Task 2: Construct hashtag-based network and compute metrics
# Load filtered dataset from Task 1
try:
    data = pd.read_csv(
        "filtered_tweets_oct_nov_2019.csv",
        low_memory=False
    )
except FileNotFoundError:
    print("Error: filtered_tweets_oct_nov_2019.csv not found")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Helper print to check hashtag prevalence
print("Hashtag counts (top 20):")
print(data["Hashtags"].value_counts().head(20))
print("\nSample non-null Hashtags (first 5):")
print(data[data["Hashtags"] != "null;"]["Hashtags"].head(5))

# Sample data to speed up (adjust sample_size as needed). Full dataset too big to process? Need to change something?
sample_size = 50000  # Change to 50000, 100000, or None (full dataset)
if sample_size is not None:
    data = data.sample(n=sample_size, random_state=42)
    print(f"Sampled {sample_size} tweets")

# Print sample Hashtags to verify format
print("Sample Hashtags (first 5):")
print(data["Hashtags"].head(5))

# Initialize graph
G = nx.Graph()

# Extract Tweet Id and Hashtags
tweet_hashtags = data[["Tweet Id", "Hashtags"]].dropna(subset=["Hashtags"])
tweet_hashtags = tweet_hashtags[tweet_hashtags["Hashtags"] != "null;"]

# Map hashtags to tweet_ids
hashtag_to_tweets = defaultdict(list)
for tweet_id, hashtags in tweet_hashtags.itertuples(index=False):
    # Try space-separated hashtags
    hashtag_list = [h.strip().lower() for h in str(hashtags).split() if h and h != "null"]
    # Fallback: colon/semicolon format
    if not hashtag_list:
        hashtag_list = [h.split(":")[0].strip().lower() for h in str(hashtags).split(";") if h and h != "null"]
    for hashtag in hashtag_list:
        hashtag_to_tweets[hashtag].append(tweet_id)

# Add edges
edge_list = []
for hashtag, tweet_ids in hashtag_to_tweets.items():
    # Generate edges for pairs of tweet_ids
    for i in range(len(tweet_ids)):
        for j in range(i + 1, len(tweet_ids)):
            edge_list.append((tweet_ids[i], tweet_ids[j]))
    if len(edge_list) % 10000 == 0:
        print(f"Generated {len(edge_list)} edges...")

# Add edges to graph
G.add_edges_from(edge_list)
print(f"Added {G.number_of_edges()} edges to graph")

# Add all tweet_ids as nodes
G.add_nodes_from(tweet_hashtags["Tweet Id"])

# Network metrics
metrics = {}

# Number of nodes and edges
metrics["Nodes"] = G.number_of_nodes()
metrics["Edges"] = G.number_of_edges()

# Average path length (for largest component)
largest_component = max(nx.connected_components(G), key=len, default=set())
largest_subgraph = G.subgraph(largest_component)
if len(largest_subgraph) > 1:
    metrics["Avg Path Length"] = nx.average_shortest_path_length(largest_subgraph)
else:
    metrics["Avg Path Length"] = np.nan

# Degree centrality
degree_centrality = nx.degree_centrality(G)
metrics["Max Degree Centrality"] = max(degree_centrality.values()) if degree_centrality else 0
metrics["Min Degree Centrality"] = min(degree_centrality.values()) if degree_centrality else 0
metrics["Avg Degree Centrality"] = np.mean(list(degree_centrality.values())) if degree_centrality else 0

# Number of components
metrics["Components"] = nx.number_connected_components(G)

# Display metrics in a table
metrics_df = pd.DataFrame([metrics], index=["Network"])
print("\nNetwork Metrics:")
print(metrics_df.to_string())

# Save adjacency matrix as CSV
edge_list_df = nx.to_pandas_edgelist(G)
edge_list_df.to_csv("adjacency_matrix.csv", index=False)
print("\nSaved adjacency matrix to adjacency_matrix.csv")

# Save graph
nx.write_gml(G, "hashtag_network.gml")
print("Saved network to hashtag_network.gml")

# Task 3: Identify largest, second largest, and third largest components
components = sorted(nx.connected_components(G), key=len, reverse=True)
component_sizes = [len(c) for c in components]
top_components = component_sizes[:3] if len(component_sizes) >= 3 else component_sizes

print("\nTask 3: Top Connected Components")
for i, size in enumerate(top_components, 1):
    print(f"{i}th Largest Component: {size} nodes")

# Save component sizes
with open("component_sizes.txt", "w") as f:
    for i, size in enumerate(top_components, 1):
        f.write(f"{i}th Largest Component: {size} nodes\n")
print("Task 3: Saved component sizes to component_sizes.txt")

# Task 4: Plot degree centrality distribution and cumulative distribution
degree_centrality_values = list(degree_centrality.values())

# Degree centrality distribution
plt.figure(figsize=(10, 6))
plt.hist(degree_centrality_values, bins=50, log=True, color='blue', alpha=0.7)
plt.title("Degree Centrality Distribution")
plt.xlabel("Degree Centrality")
plt.ylabel("Frequency (log scale)")
plt.grid(True, alpha=0.3)
plt.savefig("degree_centrality_distribution.png")
plt.close()
print("Task 4: Saved degree centrality distribution to degree_centrality_distribution.png")

# Cumulative degree centrality distribution
plt.figure(figsize=(10, 6))
plt.hist(degree_centrality_values, bins=50, cumulative=True, density=True, histtype='step', color='red')
plt.title("Cumulative Degree Centrality Distribution")
plt.xlabel("Degree Centrality")
plt.ylabel("Cumulative Probability")
plt.grid(True, alpha=0.3)
plt.savefig("cumulative_degree_centrality_distribution.png")
plt.close()
print("Task 4: Saved cumulative degree centrality distribution to cumulative_degree_centrality_distribution.png")

