import pandas as pd
import csv

# Task 1: Restrict dataset to Oct–Nov 2019
# Load TweetsCOV19.tsv
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
    print("Error: TweetsCOV19.tsv not found in c:\\Users\\tkolu\\project16-covid-network\\")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Print first few rows to verify
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
