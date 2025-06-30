from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
import numpy as np
import yaml
import json

app = Flask(__name__)

# ---------------------- Config Loader ---------------------- #
def load_config(path="conf/base.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
NUM_CLIENTS = config["fl_config"]["num_clients"]
NUM_ROUNDS = config["server_config"]["num_rounds"]

# ---------------------- State ---------------------- #
client_weights_store = []
current_round = 0
anomaly_buffer = {}        # client_id -> anomaly count
ban_round_record = {}      # client_id -> round when banned
banned_clients = set()

ANOMALY_THRESHOLD = 5
BAN_LIFETIME = 10          # Unban after 10 global rounds

anomaly_flag_matrix = np.zeros((NUM_CLIENTS, NUM_ROUNDS), dtype=int)

# ---------------------- Logic ---------------------- #
def detect_outliers_kmeans(client_weights):
    flat_weights = {
        c['client_id']: np.concatenate([w.flatten() for w in c['weights']])
        for c in client_weights
    }

    client_ids = list(flat_weights.keys())
    weight_matrix = np.stack(list(flat_weights.values()))

    if len(client_ids) <= 2:
        return []

    kmeans = KMeans(n_clusters=2, random_state=42).fit(weight_matrix)
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    outlier_cluster = unique[np.argmin(counts)]

    outliers = [cid for cid, label in zip(client_ids, labels) if label == outlier_cluster]
    return outliers

def unban_expired_clients():
    to_unban = [cid for cid in banned_clients if current_round - ban_round_record.get(cid, 0) >= BAN_LIFETIME]
    for cid in to_unban:
        print(f"[Unban] Client {cid} auto-unbanned after {BAN_LIFETIME} rounds.")
        banned_clients.remove(cid)
        ban_round_record.pop(cid, None)
        anomaly_buffer[cid] = 0

def save_anomaly_matrix():
    data = anomaly_flag_matrix.tolist()
    with open("anomaly_matrix.json", "w") as f:
        json.dump(data, f)
    print("[Save] Anomaly matrix saved to anomaly_matrix.json")

def aggregate_weights():
    global current_round
    print(f"\n[Aggregation] Round {current_round}")

    unban_expired_clients()

    if not client_weights_store:
        print("[Aggregation] No weights to process.")
        save_anomaly_matrix()
        current_round += 1
        return []

    valid_clients = [c for c in client_weights_store if c['client_id'] not in banned_clients]
    if not valid_clients:
        print("[Aggregation] All clients are banned.")
        save_anomaly_matrix()
        current_round += 1
        return []

    outliers = detect_outliers_kmeans(valid_clients)
    print(f"[Outliers Detected] {outliers}")

    for cid in outliers:
        anomaly_buffer[cid] = anomaly_buffer.get(cid, 0) + 1
        if anomaly_buffer[cid] >= ANOMALY_THRESHOLD and cid not in banned_clients:
            banned_clients.add(cid)
            ban_round_record[cid] = current_round
            print(f"[Banned] Client {cid} reached anomaly threshold at round {current_round}.")

    # Mark outliers in anomaly matrix
    for cid in outliers:
        if cid < NUM_CLIENTS and current_round < NUM_ROUNDS:
            anomaly_flag_matrix[cid, current_round] = 1

    # Mark ALL currently banned clients in anomaly matrix
    for cid in banned_clients:
        if cid < NUM_CLIENTS and current_round < NUM_ROUNDS:
            anomaly_flag_matrix[cid, current_round] = 1

    filtered_clients = [c for c in valid_clients if c['client_id'] not in outliers]
    if not filtered_clients:
        print("[Aggregation] No clients left after filtering.")
        save_anomaly_matrix()
        current_round += 1
        return []

    # MEDIAN aggregation across filtered clients
    all_weights = [c['weights'] for c in filtered_clients]
    num_layers = len(all_weights[0])
    aggregated_weights = []

    for layer_idx in range(num_layers):
        layer_stack = np.stack([client[layer_idx] for client in all_weights])
        median_layer = np.median(layer_stack, axis=0)
        aggregated_weights.append(median_layer.tolist())

    print(f"[Aggregation] Aggregated using median from {len(filtered_clients)} clients.")
    save_anomaly_matrix()
    current_round += 1
    return aggregated_weights

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    data = request.get_json()
    if "weights" not in data or "client_id" not in data:
        return jsonify({"error": "Missing client_id or weights"}), 400

    client_id = data["client_id"]
    if client_id in banned_clients:
        print(f"[Reject] Banned client {client_id} attempted to submit.")
        if client_id < NUM_CLIENTS and current_round < NUM_ROUNDS:
            anomaly_flag_matrix[client_id, current_round] = 1
        return jsonify({"error": "Client is banned"}), 403

    weights = [np.array(w) for w in data["weights"]]

    client_weights_store.append({
        "client_id": client_id,
        "weights": weights
    })

    print(f"[Submit] Received weights from Client {client_id}, Round {current_round + 1}")
    return jsonify({"message": "Weights received"}), 200

@app.route('/get_aggregated_weights', methods=['GET'])
def get_aggregated_weights():
    aggregated_weights = aggregate_weights()
    client_weights_store.clear()
    return jsonify({"aggregated_weights": aggregated_weights}), 200

@app.route('/banned_clients', methods=['GET'])
def get_banned_clients():
    return jsonify({
        "banned_clients": list(banned_clients),
        "ban_round_record": ban_round_record,
        "current_round": current_round
    }), 200

@app.route('/anomaly_matrix', methods=['GET'])
def get_anomaly_matrix():
    return jsonify(anomaly_flag_matrix.tolist()), 200

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    app.run(port=8080)
