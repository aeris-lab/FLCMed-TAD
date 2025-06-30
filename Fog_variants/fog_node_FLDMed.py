from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

client_weights_store = []

def is_anomalous(distances, index, threshold=2.0):
    """Check if the distance at index is an outlier (beyond threshold * std deviation)."""
    mean = np.mean(distances)
    std = np.std(distances)
    return abs(distances[index] - mean) > threshold * std

def detect_anomalies(client_weights):
    """Detect and remove anomalous client weights."""
    # Flatten weights for distance computation
    flat_weights = [np.concatenate([w.flatten() for w in weights]) for weights in client_weights]
    mean_weights = np.mean(flat_weights, axis=0)

    # Compute L2 distances from mean
    distances = [np.linalg.norm(w - mean_weights) for w in flat_weights]

    # Filter out anomalous clients
    filtered_clients = []
    for i, weights in enumerate(client_weights):
        if not is_anomalous(distances, i):
            filtered_clients.append(weights)
        else:
            print(f"🚨 Client {i} flagged as anomaly and excluded from aggregation.")

    return filtered_clients

def aggregate_weights():
    if not client_weights_store:
        return []

    # Filter out anomalous clients before aggregation
    filtered_weights = detect_anomalies(client_weights_store)
    if not filtered_weights:
        print("All clients were filtered out as anomalies. Returning empty weights.")
        return []

    # Median aggregation from remaining (non-anomalous) clients
    aggregated_weights = []
    for layer_weights in zip(*filtered_weights):
        layer_stack = np.stack(layer_weights, axis=0)
        layer_median = np.median(layer_stack, axis=0)
        aggregated_weights.append(layer_median.tolist())

    return aggregated_weights

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    data = request.get_json()
    if "weights" not in data:
        return jsonify({"error": "Missing weights"}), 400
    client_weights_store.append([np.array(w) for w in data["weights"]])
    return jsonify({"message": "Weights received"}), 200

@app.route('/get_aggregated_weights', methods=['GET'])
def get_aggregated_weights():
    aggregated_weights = aggregate_weights()
    client_weights_store.clear()  # Clear for next round
    return jsonify({"aggregated_weights": aggregated_weights}), 200

@app.route('/receive_weights', methods=['POST'])
def receive_weights():
    data = request.get_json()
    if not data or "weights" not in data:
        return jsonify({"error": "No weights received"}), 400
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    app.run(port=8080)
