import os
import time
import csv
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import glob
def load_data(path, chunk_size=None):
    """
    Load and clean dataset from a CSV file.

    Args:
        path (str): Path to the dataset CSV file.
        chunk_size (int, optional): For large datasets, read in chunks.

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print(f"[INFO] Loading dataset from: {path}")

    # Load CSV (with optional chunking)
    if chunk_size:
        chunks = pd.read_csv(path, chunksize=chunk_size)
        df = pd.concat(chunks)
    else:
        df = pd.read_csv(path)

    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Check data types and convert

    df['facilityType'] = df['facilityType'].astype(int)
    df['managerVehicle'] = df['managerVehicle'].astype(int)
    print("Sample feature data types:")

    # Convert to numeric and drop rows with key missing values
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df = df.dropna(subset=['distance', 'chargeTimeHrs', 'kwhTotal'])

    # Drop irrelevant columns if they exist
    drop_cols = [
        'created', 'ended', 'startTime', 'endTime', 'weekday', 'platform',
        'userId', 'reportedZip'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    return df

def partition_data(X, y, num_clients):
    """
    Partition dataset by grouping all samples of the same locationId together,
    and assign groups to clients to balance data size approximately.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Targets
        subject_ids (pd.Series): locationId values per sample
        num_clients (int): number of clients to split into

    Returns:
        List of tuples: [(X_client, y_client, subject_ids_client), ...]
    """

    # Combine into one DataFrame for easier handling
    df = X.copy()
    df['target'] = y

    # Group by locationId
    grouped = df.groupby('locationId')

    # Shuffle locationIds
    location_ids = sorted(grouped.groups.keys())
    # Initialize client allocations
    client_data = defaultdict(list)
    client_sizes = [0] * num_clients

    # Assign each locationId group to client with smallest total size so far
    for loc_id in location_ids:
        group_df = grouped.get_group(loc_id)
        # Find client with smallest dataset size
        client_idx = np.argmin(client_sizes)
        client_data[client_idx].append(group_df)
        client_sizes[client_idx] += len(group_df)

    # Combine groups per client and separate features, target, locationId
    clients = []
    for i in range(num_clients):
        client_df = pd.concat(client_data[i])
        X_client = client_df.drop(columns=['target'])
        y_client = client_df['target']
        subject_ids_client = client_df['locationId']
        print(f"Client {i+1}: {len(subject_ids_client.unique())} unique locations, {len(client_df)} samples")
        clients.append((X_client, y_client, subject_ids_client))

    return clients




def load_full_logs(pattern):
    logs = {}
    for filepath in glob.glob(os.path.join("logs/clients", pattern)):
        # Extract client ID from filename like 'latency_client_0.json'
        cid = os.path.basename(filepath).split("_")[-1].split(".")[0]
        with open(filepath, "r") as f:
            logs[cid] = json.load(f)
    return logs
