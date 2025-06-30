import os
import json
import numpy as np
import tensorflow as tf
from models.model import ev_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.fl_utils import set_model_parameters
import requests
# Optional global history
METRICS_HISTORY = {
    "rounds": [],
    "mse": [],
    "mae": [],
    "r2": []
}
OUTPUT_DIR = "server_metrics"

# ---------------------- Config Loader ---------------------- #
def load_config(path="conf/base.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# ------------------ Fog Node Fetch ------------------------- #
def fetch_fog_aggregated_weights():
    try:
        print("Contacting fog node for aggregated weights...")
        response = requests.get("http://localhost:8080/get_aggregated_weights")
        if response.status_code == 200:
            print("Fog responded with weights.")
            raw_weights = response.json().get("aggregated_weights", [])
            return [np.array(w, dtype=np.float32) for w in raw_weights]
        else:
            print(f"Fog responded with status {response.status_code}")
    except Exception as e:
        print(f"Could not reach fog node: {e}")
    return None
# -------------------Evaluate Function ----------------#
def evaluate_fn(server_round, parameters, config, dataset):
    print(f"\n--- Evaluating Round {server_round} ---")

    df = dataset.copy()

    feature_cols = [
        'chargeTimeHrs','distance','managerVehicle','facilityType',
        'dollars', 'locationId', 'stationId'
    ]
    target_col = 'kwhTotal'

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    # Build and load model
    model = ev_model(input_dim=X.shape[1])
    try:
        fog_weights = fetch_fog_aggregated_weights()
        if fog_weights:
            set_model_parameters(model, fog_weights)
            print("✅ Applied fog node weights.")
        else:
            set_model_parameters(model, parameters)
            print("✅ Applied federated weights.")
    except Exception as e:
        print(f"Error applying weights: {e}")
        set_model_parameters(model, parameters)

    # Compile model for evaluation
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Evaluate
    eval_loss, eval_mae = model.evaluate(val_dataset, verbose=0)
    y_pred = model.predict(X_val, verbose=0).flatten()
    r2 = r2_score(y_val, y_pred)

    # Save metrics
    METRICS_HISTORY["rounds"].append(server_round)
    METRICS_HISTORY["mse"].append(eval_loss)
    METRICS_HISTORY["mae"].append(eval_mae)
    METRICS_HISTORY["r2"].append(r2)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "ev_metrics_history.json"), "w") as f:
        json.dump(METRICS_HISTORY, f, indent=4)

    print(f"Round {server_round} | MSE: {eval_loss:.4f}, MAE: {eval_mae:.4f}, R²: {r2:.4f}")

    return eval_loss, {"mae": eval_mae, "r2": r2}


def fit_config(server_round: int):
    return {"server_round": server_round}
