import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import flwr as fl
import yaml
import time
from tensorflow import keras
from models.model import ev_model
from tensorflow.keras import optimizers
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data, partition_data
from utils.logging_utils import log_latency, log_comm_overhead
from utils.fl_utils import get_model_parameters, set_model_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
FOG_NODE_URL = "http://localhost:8080/submit_weights"
LOG_DIR = "logs/clients"
os.makedirs(LOG_DIR, exist_ok=True)
TRAINING_METRICS_DIR = "training_metrics"
EVALUATION_METRICS_DIR = "evaluation_metrics"
os.makedirs(TRAINING_METRICS_DIR, exist_ok=True)
os.makedirs(EVALUATION_METRICS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Separate global histories for training and evaluation
TRAINING_METRICS_HISTORY = {
    "rounds": [],
    "mse": [],
    "rmse": [],
    "mae": [],
    "r2": []
}

EVALUATION_METRICS_HISTORY = {
    "rounds": [],
    "mse": [],
    "rmse": [],
    "mae": [],
    "r2": []
}

round_number = 0

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X, y, config):
        self.model = model
        self.X_raw = X
        self.y_raw = y
        self.config = config
        self.cid = config.get("cid", "unknown")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare data (scale and split)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values.astype(np.float32))
        y_scaled = y.values.astype(np.float32).reshape(-1, 1)

        X_train, self.X_val, y_train, self.y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        batch_size = 32
        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(batch_size)

    def get_parameters(self, config=None):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    def log_client_weights(self, round_number):
        weights = self.get_parameters()
        weights_serializable = [w.tolist() for w in weights]
        filename = f"weights_client_{self.cid}.jsonl"
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, "a") as f:
            json.dump({
                "round": round_number,
                "weights": weights_serializable
            }, f)
            f.write("\n")

    def fit(self, parameters, config):
        global round_number
        round_number += 1
        self.set_parameters(parameters)
        start_time = time.time()
        self.train_function()
        self.save_model_weights(round_number)
        duration = time.time() - start_time
        log_latency(self.cid, duration, round_number)
        return self.get_parameters(), len(self.X_raw), {}

    def train_function(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        print(f"[Client {self.cid}] Starting training for round {round_number}")
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=30,
            callbacks=[early_stop],
            verbose=0
        )

        y_val_pred = self.model.predict(self.X_val, verbose=0)
        mse_val = mean_squared_error(self.y_val, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        mae_val = mean_absolute_error(self.y_val, y_val_pred)
        r2_val = r2_score(self.y_val, y_val_pred)

        # Log training metrics globally
        TRAINING_METRICS_HISTORY["rounds"].append(round_number)
        TRAINING_METRICS_HISTORY["mse"].append(mse_val)
        TRAINING_METRICS_HISTORY["rmse"].append(rmse_val)
        TRAINING_METRICS_HISTORY["mae"].append(mae_val)
        TRAINING_METRICS_HISTORY["r2"].append(r2_val)

        # Save training metrics JSON
        training_metrics_file = os.path.join(TRAINING_METRICS_DIR, f"metrics_client_{self.cid}.json")
        with open(training_metrics_file, "w") as f:
            json.dump(TRAINING_METRICS_HISTORY, f, indent=4)

        print(f"[Client {self.cid}] Training complete. Training metrics saved.")

    def evaluate(self, parameters, config):
        global round_number
        print(f"[Client {self.cid}] Evaluating round {round_number}")
        self.set_parameters(parameters)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        start_time = time.time()
        loss, mae = self.model.evaluate(self.val_dataset, verbose=0)
        preds = self.model.predict(self.X_val, verbose=0).flatten()
        mse = mean_squared_error(self.y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_val, preds)
        duration = time.time() - start_time

        print(f"[Client {self.cid}] Round {round_number} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Duration: {duration:.2f}s")

        # Log evaluation metrics globally
        EVALUATION_METRICS_HISTORY["rounds"].append(round_number)
        EVALUATION_METRICS_HISTORY["mse"].append(mse)
        EVALUATION_METRICS_HISTORY["rmse"].append(rmse)
        EVALUATION_METRICS_HISTORY["mae"].append(mae)
        EVALUATION_METRICS_HISTORY["r2"].append(r2)

        # Save evaluation metrics as JSONL (append) and JSON (full)
        eval_jsonl_path = os.path.join(EVALUATION_METRICS_DIR, f"metrics_client_{self.cid}.jsonl")
        with open(eval_jsonl_path, "a") as f:
            json.dump({
                "client_id": self.cid,
                "round": round_number,
                "loss": loss,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "duration_sec": duration
            }, f)
            f.write("\n")

        eval_json_path = os.path.join(EVALUATION_METRICS_DIR, f"metrics_client_{self.cid}.json")
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append({
            "client_id": self.cid,
            "round": round_number,
            "loss": loss,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "duration_sec": duration
        })
        with open(eval_json_path, "w") as f:
            json.dump(data, f, indent=4)

        return loss, len(self.X_raw), {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "duration": duration
        }
    def save_model_weights(self, round_number):
        model_save_path = f"client_{self.cid}_model.h5"
        self.model.save(model_save_path)
        logger.info(f" Model weights saved to {model_save_path}.")
        self.send_weights_to_fog(round_number)

    def send_weights_to_fog(self, round_number):
        logger.info(f"Sending weights to fog node...")

        weights = self.get_parameters()
        weights_serializable = [w.tolist() for w in weights]

        # Get weight names from model (if using Keras or similar)
        try:
            weight_names = [w.name if hasattr(w, "name") else f"weight_{i}" for i, w in enumerate(self.model.weights)]
        except Exception as e:
            logger.warning(f"Could not extract weight names: {e}")
            weight_names = [f"weight_{i}" for i in range(len(weights))]

        os.makedirs("logs/clients", exist_ok=True)
        filename = f"weights_client_{self.cid}.jsonl"
        filepath = os.path.join("logs/clients", filename)

        log_comm_overhead(self.cid, weights_serializable, round_number)

        record = {
            "client_id": self.cid,
            "round": round_number,
            "weights": weights_serializable
        }

        if round_number == 0:
            record["weight_names"] = weight_names

        with open(filepath, "a") as f:
            json.dump(record, f)
            f.write("\n")
        try:
            response = requests.post(FOG_NODE_URL, json={"client_id": self.cid, "weights": weights_serializable})
            if response.status_code == 200:
                logger.info(f" Client {self.cid} sent weights to fog node.")
            else:
                logger.error(f"L Failed to send weights to fog node. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"L Error sending weights to fog node: {e}")

def generate_client_fn(config_path: str, dataset):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    df = dataset.copy()
    batch_size = config["fl_config"]["training"]["batch_size"]
    num_clients = config["fl_config"]["num_clients"]

    feature_cols = [
        'chargeTimeHrs','distance','managerVehicle','facilityType',
        'dollars', 'locationId', 'stationId'
    ]
    target_col = 'kwhTotal'

    # Extract data
    X = df[feature_cols]
    y = df[target_col]

    # Partition data
    client_subsets = partition_data(X, y, num_clients=num_clients)

    def client_fn(cid: str):  # Flower expects this signature
        client_id = int(cid)

        # Unpack the correct number of items (3 in your case)
        X_sub, y_sub, subject_ids_sub = client_subsets[client_id]
        model = ev_model(input_dim = 7) #default

        initial_parameters = get_model_parameters(model)
        client_config = {**config["fl_config"]["training"], "cid": client_id}

        # Choose client class (malicious or normal) #Client_Poisoing
        if (client_id == 22):                                                    #configure the client_id you want to poison (eg - client 2)
            client = DataPoisoningClient(model, X_sub, y_sub, client_config)
        elif (client_id == 21):                                                  #configure the client_id you want to poison (eg - client 7)
            client = ModelPoisoningClient(model, X_sub, y_sub, client_config)
        else:
            client = FLClient(model, X_sub, y_sub, client_config)

        client.set_parameters(initial_parameters)
        return client

    return client_fn



class ModelPoisoningClient(FLClient):
    def fit(self, parameters, config):
        start_time = time.time()
        logger.warning(f"[Client {self.cid}] Performing MODEL POISONING")
        trained_weights = self.get_parameters()
        poisoned_weights = [w + 0.3 * np.sign(w) for w in trained_weights]      #configure the factor you want to poison by (eg - 0.3)
        # This can also be simulated after the model training is done successfuly ,
        # by poisong before sending the trained weights to the fog node > the effect is same.
        # That scenario simulates a situation where clients as locally performs correctly,
        # but the sent weight is affecting the model.
        # poisoned_weights = [w.copy() * poison_factor for w in trained_weights]
        self.set_parameters(poisoned_weights)
        global round_number
        round_number += 1
        self.train_function()
        self.save_model_weights(round_number)
        duration = time.time() - start_time
        log_latency(self.cid, duration, round_number)

        return poisoned_weights, len(self.X_raw), {}

class DataPoisoningClient(FLClient):
    def __init__(self, model, X, y, config, poison_mode=2, poison_fraction=0.9):
        """
        poison_mode:
        0 = Scale y by random factor per sample (mild poisoning)
        1 = Add large random noise (strong poisoning)
        2 = Replace y with random values in y's range (extreme poisoning)

        poison_fraction:
        Fraction of data points to poison (e.g. 0.5 for 50%)
        """
        self.datapoison = poison_mode

        # Make a copy of y to avoid modifying original
        y_poisoned = np.copy(y)

        # Select indices to poison
        num_to_poison = int(len(y) * poison_fraction)
        poison_indices = np.random.choice(len(y), num_to_poison, replace=False)

        # Apply poisoning only on selected indices
        if self.datapoison == 0:
            factors = np.random.uniform(0.5, 10.5, size=num_to_poison)          # factor can be adjusted accordingly
            y_poisoned[poison_indices] *= factors

        elif self.datapoison == 1:
            noise = np.random.normal(loc=0.0, scale=np.std(y) * 10, size=num_to_poison) # factor can be adjusted accordingly
            y_poisoned[poison_indices] += noise

        elif self.datapoison == 2:
            extreme_val = y.max() + 50 * np.std(y)                              # factor can be adjusted accordingly
            y_poisoned[poison_indices] = extreme_val

        # Otherwise: no poisoning (data stays clean)
        y_poisoned = pd.Series(y_poisoned, index=y.index)
        super().__init__(model, X, y_poisoned, config)

    def fit(self, parameters, config):
        logger.warning(f"[Client {self.cid}] Data poisoning active (mode {self.datapoison}, partial fraction)")
        return super().fit(parameters, config)
