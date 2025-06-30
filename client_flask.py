import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.model import ev_model
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from utils.data_loader import load_data

app = Flask(__name__)

# --------- Utility: Load model for a specific client --------- #
def load_client_model(client_id: str, input_dim: int):
    model_path = f"client_{client_id}_model.h5"
    if not os.path.exists(model_path):
        return None

    model = ev_model(input_dim=7)
    model.load_weights(model_path)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# --------- Utility: Preprocess the uploaded CSV data --------- #
def preprocess_data(df: pd.DataFrame):
    required_columns = [
        'chargeTimeHrs', 'distance', 'managerVehicle', 'facilityType',
        'dollars', 'locationId', 'stationId'
    ]

    if not all(col in df.columns for col in required_columns):
        missing = list(set(required_columns) - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_columns].copy()
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df.astype(np.float32))
    return normalized

# --------- Route: Predict using uploaded CSV and client_id --------- #
@app.route("/predict", methods=["POST"])
def predict():
    if "client_id" not in request.form:
        return jsonify({"error": "Missing client_id"}), 400
    if "file" not in request.files:
        return jsonify({"error": "Missing CSV file"}), 400

    client_id = request.form["client_id"]
    file = request.files["file"]

    try:
        df = load_data(file, 250000)
        if df.empty:
            return jsonify({"error": "Empty CSV file"}), 400
        processed_data = preprocess_data(df)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    input_dim = processed_data.shape[1]
    model = load_client_model(client_id, input_dim)
    if model is None:
        return jsonify({"error": f"Model for client {client_id} not found"}), 404

    with tf.device('/CPU:0'):
        predictions = model.predict(processed_data).squeeze()
        if predictions.ndim == 0:
            predictions = np.array([predictions])
        predictions = predictions.tolist()

    return jsonify({
        "client_id": client_id,
        "predictions": predictions
    })

# --------- Run the Flask app --------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
