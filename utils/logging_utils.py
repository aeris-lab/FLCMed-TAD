import json
import os
import sys
import pickle

def create_directory_if_not_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def log_client_weights(weights, client_id, round_num, log_dir="logs/clients"):
    path = os.path.join(log_dir, f"weights_client_{client_id}.jsonl")
    create_directory_if_not_exists(path)
    with open(path, "a") as f:
        json.dump({"round": round_num, "weights": weights}, f)
        f.write("\n")

def log_latency(cid, duration, round_number):
    filepath = os.path.join("logs/clients", f"latency_client_{cid}.json")
    os.makedirs("logs/clients", exist_ok=True)
    entry = {"round": round_number, "latency_sec": duration}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)

def log_comm_overhead(cid, weights, round_number):
    serialized = pickle.dumps(weights)
    size_bytes = sys.getsizeof(serialized)
    filepath = os.path.join("logs/clients", f"comm_client_{cid}.json")
    os.makedirs("logs/clients", exist_ok=True)
    entry = {"round": round_number, "size_bytes": size_bytes}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)
