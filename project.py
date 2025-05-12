import pandas as pd
import csv
import networkx as nx
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
from ndlib.models.epidemics.SISModel import SISModel
from itertools import combinations
import powerlaw
from scipy.stats import linregress

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

# Sample data to speed up (adjust sample_size as needed). Full dataset too big to process (takes too long)? Need to change something?
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
for tweet_ids in hashtag_to_tweets.values():
    edge_list.extend(combinations(tweet_ids, 2))

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

# Task 6: Time slicing the network
# Create 10 time slices
data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors='coerce')

# Drop invalid
data = data.dropna(subset=["Timestamp"])

# Sort data chronologically
data = data.sort_values(by="Timestamp")

# Start and end times
start_time = data["Timestamp"].min()
end_time = data["Timestamp"].max()

# Create 10 time bins
time_bins = pd.date_range(start=start_time, end=end_time, periods=11)

# Slice data by time ranges
time_slices = []
for i in range(10):
    slice_df = data[(data["Timestamp"] >= time_bins[i]) & (data["Timestamp"] < time_bins[i+1])]
    time_slices.append(slice_df)

# List for metrics 
subgraph_metrics = []

# Plot
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.flatten()

for i, slice_df in enumerate(time_slices):
    # ID s from slice
    slice_ids = set(slice_df['Tweet Id'])

    # Subgraph
    subgraph = G.subgraph(slice_ids).copy()

    # Metrics calculation
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()
    diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) and num_nodes > 1 else np.nan
    avg_path_lenght = nx.average_shortest_path_length(subgraph) if nx.is_connected(subgraph) and num_nodes > 1 else np.nan

    subgraph_metrics.append({
        "Time slice": i + 1,
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Diameter": diameter,
        "Avg path length": avg_path_lenght
    })

    # Draw subgraph
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_size=10, ax=axes[i])
    axes[i].set_title(f"Slice {i + 1} ({num_nodes} nodes)")

plt.tight_layout()
plt.savefig("subgraphs_over_time.png")
plt.close()
print("Task 6: Saved subgraphs vizualization to subgraphs_over_time.png")

# Save global metrics table 
metrics_df = pd.DataFrame(subgraph_metrics)
metrics_df.to_csv("time_slice_metrics.cvs", index=False)
print("Task 6: Saved time slice metrics to time_slice_metrics.csv")

# Task 7: Triangle scores and evolution of triangles overtime 
triangle_counts = []

for i, slice_df in enumerate(time_slices):
    # Tweet ids
    tweet_ids = set(slice_df['Tweet Id'])

    # Subgraph
    subgraph = G.subgraph(tweet_ids).copy()

    # Count triangles 
    triangle_dic = nx.triangles(subgraph)
    total_triangles = sum(triangle_dic.values()) // 3

    triangle_counts.append({
        "Time slice": i + 1,
        "Triangles": total_triangles
    })

    print(f"Slice {i + 1}: {len(tweet_ids)} tweet IDs → {subgraph.number_of_nodes()} nodes → {total_triangles} triangles")

# Save triangle data
triangle_df = pd.DataFrame(triangle_counts)
triangle_df.to_csv("triangle_evolution.csv", index=False )
print("Task 7: Saved triangle counts to triangle_evolution.csv")

# Plot triangle evolution 
plt.figure(figsize=(10, 6))
plt.plot(triangle_df["Time slice"], triangle_df["Triangles"], marker='o', linestyle='-')
plt.title("Triangle count over time")
plt.xlabel("Time slice")
plt.ylabel("Number of triangles")
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 11))
plt.savefig("triangle_evolution_over_time.png")
plt.close()
print("Task 7: Saves plot to triangle_evolution_over_time.png")

# Task 8: Sentiment evolution over time
sentiment_scores = []

for i, slice_df in enumerate(time_slices):
    pos_values = []
    neg_values = []

    for raw in slice_df["Sentiment"].dropna():
        try:
            parts = raw.strip().split()
            if len(parts) == 2:
                pos = float(parts[0])
                neg = float(parts[1])
                pos_values.append(pos)
                neg_values.append(neg) 
        except:
            continue

    avg_pos = np.mean(pos_values) if pos_values else 0.0
    avg_neg = np.mean(neg_values) if neg_values else 0.0

    sentiment_scores.append({
        "Time slice": i + 1,
        "Avg positive": avg_pos,
        "Avg negative": avg_neg
    })        

    print(f"Slice {i + 1}: Avg positive = {avg_pos:.4f}, Avg negative = {avg_neg:.4f}")

# Save sentiment data
sentiment_df = pd.DataFrame(sentiment_scores)
sentiment_df.to_csv("sentiment_evolution.cvs", index=False)
print("Task 8: Saved sentiment ecolution data to sentiment_evolution.cvs")

# Plot sentiment evolution
plt.figure(figsize=(10, 6))
plt.plot(sentiment_df["Time slice"], sentiment_df["Avg positive"], label='Positive', marker='o', color='green')
plt.plot(sentiment_df["Time slice"], sentiment_df["Avg negative"], label='Negative', marker='o', color='red')
plt.title("Sentiment evolution over time")
plt.xlabel("Time slice")
plt.ylabel("Average sentiment score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 11))
plt.ylim(-2, 2)
plt.savefig("sentiment_evolution_over_time.png")
plt.close()
print("Task 8: Saved sentiment plot to sentiment_evolution_over_time.png")

# Task 9: Negative sentiment spread
tweet_ids = set(time_slices[0]["Tweet Id"])
subgraph = G.subgraph(tweet_ids).copy()

# Initialize SIS model
model = SISModel(subgraph)

# Configuration
# Selected simulation 5: beta = 0.03  lambda = 0.02 
# because this simulation best matches findings in Task 8
config = mc.Configuration()
config.add_model_parameter("beta", 0.03) # Infection chance percent
config.add_model_parameter("lambda", 0.01) # Recovery chance
config.add_model_parameter("fraction_infected", 0.05) # Start with 5 percent of infected
model.set_initial_status(config)

# Stimulate spread of infection
iterations = model.iteration_bunch(10)
infected_counts = [it["node_count"][1] for it in iterations]

# print and plot the results
print("Stimulated infected users per time slice:")
print(infected_counts)

plt.plot(range(1, 11), infected_counts, marker='o')
plt.title("Stimulated infected users per time slice")
plt.xlabel("Time slice")
plt.ylabel("Simulated negative users (infected nodes)")
plt.grid(True, alpha=0.3)
plt.savefig("task9_sis_simulation.png")
plt.close()



# Task 5: Test power-law fit for degree distribution
try:
    G = nx.read_gml("hashtag_network.gml")
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
except FileNotFoundError:
    print("Error: hashtag_network.gml not found. Run Task 2 first.")
    exit()

print("\nTask 5: Testing power-law fit for degree distribution")

# Use raw degrees, exclude degree 0
degrees = [degree for _, degree in G.degree() if degree > 0]
data = np.array(degrees)

if len(data) == 0:
    print("Error: No nodes with degree > 0. Cannot fit power-law.")
    exit()

# Fit power-law distribution
fit = powerlaw.Fit(data, discrete=True)
alpha = fit.power_law.alpha
xmin = fit.power_law.xmin
print(f"Power-law fit: alpha = {alpha:.3f}, xmin = {xmin:.3f}")

# Compute R-squared for degrees >= xmin
data_above_xmin = data[data >= xmin]
if len(data_above_xmin) < 10:
    print("Warning: Too few data points above xmin for reliable R-squared.")
    r_squared = 0.0
else:
    # Empirical CDF
    sorted_data = np.sort(data_above_xmin)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n + 1) / n

    # Fitted power-law CDF
    fitted_cdf = fit.power_law.cdf(sorted_data)

    # R-squared
    ss_tot = np.sum((empirical_cdf - np.mean(empirical_cdf))**2)
    ss_res = np.sum((empirical_cdf - fitted_cdf)**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

print(f"R-squared value: {r_squared:.3f}")

# Assess fit strength
fit_strength = "Weak fit" if r_squared < 0.5 else "Moderate fit" if r_squared < 0.8 else "Strong fit"
print(f"Fit strength: {fit_strength}")

# Save results
with open("powerlaw_fit_results.txt", "w") as f:
    f.write(f"Power-law Fit Results\n")
    f.write(f"Alpha: {alpha:.3f}\n")
    f.write(f"Xmin: {xmin:.3f}\n")
    f.write(f"R-squared: {r_squared:.3f}\n")
    f.write(f"Data points above xmin: {len(data_above_xmin)}\n")
    f.write(f"Fit strength: {fit_strength}\n")
print("Task 5: Saved power-law fit results to powerlaw_fit_results.txt")
