from flask import Flask, request, jsonify
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
anomaly_buffer = {}
ban_round_record = {}
banned_clients = set()
temporal_scores = {}

ANOMALY_THRESHOLD = 5
BAN_LIFETIME = 10

anomaly_flag_matrix = np.zeros((NUM_CLIENTS, NUM_ROUNDS), dtype=int)

# ---------------------- Detection Logic ---------------------- #
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def detect_outliers_layerwise(valid_clients, threshold=2.0):
    num_layers = len(valid_clients[0]['weights'])
    medians = [
        np.median(np.stack([c['weights'][i] for c in valid_clients]), axis=0)
        for i in range(num_layers)
    ]

    layer_dists_all = []
    client_scores = []

    for client in valid_clients:
        layer_dists = []
        layer_cosines = []

        for i in range(num_layers):
            w = client['weights'][i]
            median_w = medians[i]
            layer_dists.append(np.linalg.norm(w - median_w))
            layer_cosines.append(cosine_similarity(w.flatten(), median_w.flatten()))

        avg_dist = np.mean(layer_dists)
        avg_cosine = np.mean(layer_cosines)
        layer_dists_all.append(avg_dist)
        client_scores.append((client['client_id'], avg_dist, avg_cosine))

    dist_mean = np.mean(layer_dists_all)
    dist_std = np.std(layer_dists_all)

    outliers = []
    for cid, avg_dist, avg_cosine in client_scores:
        score = temporal_scores.get(cid, 0)
        if avg_dist > dist_mean + threshold * dist_std or avg_cosine < 0.7:
            score = 0.9 * score + 1
        else:
            score = 0.9 * score
        temporal_scores[cid] = score

        # The condition check factors mentioned below is subject to change (given as example)
        if avg_dist > dist_mean + threshold * dist_std or avg_cosine < 0.7 or temporal_scores[cid] > 1:
            outliers.append(cid)
            print(f"[Outlier] Client {cid}: dist={avg_dist:.4f}, cosine={avg_cosine:.4f}, score={score:.2f}")
        else:
            print(f"[OK!] Client {cid}: dist={avg_dist:.4f}, cosine={avg_cosine:.4f}, score={score:.2f}")
    return outliers

def unban_expired_clients():
    to_unban = [cid for cid in banned_clients if current_round - ban_round_record.get(cid, 0) >= BAN_LIFETIME]
    for cid in to_unban:
        print(f"[Unban] Client {cid} auto-unbanned after {BAN_LIFETIME} rounds.")
        banned_clients.remove(cid)
        ban_round_record.pop(cid, None)
        anomaly_buffer[cid] = 0

def save_anomaly_matrix():
    with open("anomaly_matrix.json", "w") as f:
        json.dump(anomaly_flag_matrix.tolist(), f)
    print("[Save] Anomaly matrix saved.")

def aggregate_weights():
    global current_round
    print(f"\n[Aggregation] Round {current_round}")

    unban_expired_clients()

    if not client_weights_store:
        print("[Aggregation] No weights submitted.")
        save_anomaly_matrix()
        current_round += 1
        return []

    valid_clients = [c for c in client_weights_store if c['client_id'] not in banned_clients]
    if not valid_clients:
        print("[Aggregation] All clients are banned.")
        save_anomaly_matrix()
        current_round += 1
        return []

    outliers = detect_outliers_layerwise(valid_clients)

    for cid in outliers:
        anomaly_buffer[cid] = anomaly_buffer.get(cid, 0) + 1
        if anomaly_buffer[cid] >= ANOMALY_THRESHOLD and cid not in banned_clients:
            banned_clients.add(cid)
            ban_round_record[cid] = current_round
            print(f"[Ban] Client {cid} banned at round {current_round}.")
        if cid < NUM_CLIENTS and current_round < NUM_ROUNDS:
            anomaly_flag_matrix[cid, current_round] = 1

    # Mark ALL currently banned clients in the matrix
    for cid in banned_clients:
        if cid < NUM_CLIENTS and current_round < NUM_ROUNDS:
            anomaly_flag_matrix[cid, current_round] = 1

    filtered_clients = [c for c in valid_clients if c['client_id'] not in outliers]

    if not filtered_clients:
        print("[Aggregation] No clients left after filtering.")
        save_anomaly_matrix()
        current_round += 1
        return []

    aggregated = []
    for layers in zip(*[c['weights'] for c in filtered_clients]):
        aggregated.append(np.median(np.stack(layers), axis=0).tolist())

    save_anomaly_matrix()
    current_round += 1
    return aggregated

# ---------------------- Routes ---------------------- #
@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    data = request.get_json()
    if "weights" not in data or "client_id" not in data:
        return jsonify({"error": "Missing client_id or weights"}), 400

    client_id = data["client_id"]
    if client_id in banned_clients:
        print(f"[Reject] Banned client {client_id} tried to submit.")
        if client_id < NUM_CLIENTS and current_round < NUM_ROUNDS:
            anomaly_flag_matrix[client_id, current_round] = 1
        return jsonify({"error": "Client is banned"}), 403

    weights = [np.array(w) for w in data["weights"]]
    client_weights_store.append({"client_id": client_id, "weights": weights})

    print(f"[Submit] Client {client_id} weights received for round {current_round + 1}.")
    return jsonify({"message": "Weights received"}), 200

@app.route('/get_aggregated_weights', methods=['GET'])
def get_aggregated_weights():
    aggregated = aggregate_weights()
    client_weights_store.clear()
    return jsonify({"aggregated_weights": aggregated}), 200

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
