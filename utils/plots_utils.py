import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
import seaborn as sns
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
LOG_DIR = "logs/clients"
LOG_DIR_SERVER = "server_metrics"

OUTPUT_PATH_LATENCY = os.path.join(LOG_DIR, "latency_comm_plot.png")
OUTPUT_PATH_SERVER = os.path.join(LOG_DIR_SERVER, "server_metrics.png")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOG_DIR_SERVER, exist_ok=True)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_client_weight_trends(log_dir="logs/clients", save_data_file="logs/clients/all_clients_weight_means.json"):
    client_files = [f for f in os.listdir(log_dir) if f.startswith("weights_client_") and f.endswith(".jsonl")]

    if not client_files:
        print("No client weight logs found.")
        return

    all_clients_data = {}

    for file in sorted(client_files):
        client_id = file.replace("weights_client_", "").replace(".jsonl", "")
        path = os.path.join(log_dir, file)

        per_layer_means = {}
        rounds = []

        with open(path, "r") as f:
            for line in f:
                record = json.loads(line)
                round_num = record["round"]
                rounds.append(round_num)

                for idx, layer in enumerate(record["weights"]):
                    mean = float(np.mean(np.array(layer)))
                    if idx not in per_layer_means:
                        per_layer_means[idx] = []
                    per_layer_means[idx].append(mean)

        # Store data for exporting
        all_clients_data[client_id] = {
            "rounds": rounds,
            "layer_means": per_layer_means
        }

        # Plot per-layer weight trends
        plt.figure(figsize=(10, 6))
        for layer_idx, values in per_layer_means.items():
            plt.plot(rounds, values, marker='o', label=f"Layer {layer_idx}")

        plt.title(f"Client {client_id} - Layer-wise Weight Means")
        plt.xlabel("Round")
        plt.ylabel("Mean Weight Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{log_dir}/client_{client_id}_layer_trends.png")
        plt.close()

    # Save aggregated data to JSON
    with open(save_data_file, "w") as f:
        json.dump(all_clients_data, f, indent=2)

    print(f"Saved all client weight trends to {save_data_file}")


def plot_server_aggregated_weights(log_file="logs/server/aggregated_weights.jsonl"):
    if not os.path.exists(log_file):
        print("Server aggregated weights log not found.")
        return

    per_layer_means = {}
    rounds = []

    with open(log_file, "r") as f:
        for line in f:
            record = json.loads(line)
            rounds.append(record["round"])
            for idx, layer in enumerate(record["weights"]):
                mean = np.mean(np.array(layer))
                if idx not in per_layer_means:
                    per_layer_means[idx] = []
                per_layer_means[idx].append(mean)

    # Plot per-layer server weight trends
    plt.figure(figsize=(10, 6))
    for layer_idx, values in per_layer_means.items():
        plt.plot(rounds, values, marker='x', label=f"Layer {layer_idx}")

    plt.title("Server - Layer-wise Aggregated Weight Means")
    plt.xlabel("Round")
    plt.ylabel("Mean Weight Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/server/server_layerwise_aggregated_weights.png")
    plt.close()

def compare_client_vs_aggregated(log_dir="logs/clients", server_log="logs/server/aggregated_weights.jsonl"):
    client_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("weights_client_")]
    if not client_files or not os.path.exists(server_log):
        print("Missing client or server weight logs.")
        return

    # Load server data
    server_data = {}
    with open(server_log, "r") as f:
        for line in f:
            record = json.loads(line)
            flat_weights = np.concatenate([np.array(w).flatten() for w in record["weights"]])
            server_data[record["round"]] = np.mean(flat_weights)

    # Load and average client data per round
    round_means = {}
    for file in client_files:
        with open(file, "r") as f:
            for line in f:
                record = json.loads(line)
                round_ = record["round"]
                flat_weights = np.concatenate([np.array(w).flatten() for w in record["weights"]])
                mean = np.mean(flat_weights)
                round_means.setdefault(round_, []).append(mean)

    rounds = sorted(set(server_data.keys()).intersection(round_means.keys()))
    client_avgs = [np.mean(round_means[r]) for r in rounds]
    server_avgs = [server_data[r] for r in rounds]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, client_avgs, label="Client Mean Weights", marker='o')
    plt.plot(rounds, server_avgs, label="Server Aggregated Weights", marker='x')
    plt.xlabel("Round")
    plt.ylabel("Avg Weight Value")
    plt.title("Avg Client vs Server Aggregated Weights per Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/weights_comparison.png")
    plt.close()


def plot_combined(latency_data, comm_data):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Latency Plot ---
    for cid, entries in latency_data.items():
        rounds = [entry["round"] for entry in entries]
        latencies = [entry["latency_sec"] for entry in entries]
        axs[0].plot(rounds, latencies, marker='o', label=f"Client {cid}")
    axs[0].set_title("Latency per Round (per Client)")
    axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Latency (sec)")
    axs[0].legend()
    axs[0].grid(True)

    # --- Communication Overhead Plot ---
    for cid, entries in comm_data.items():
        rounds = [entry["round"] for entry in entries]
        comm_kb = [entry["size_bytes"] / 1024 for entry in entries]
        axs[1].plot(rounds, comm_kb, marker='s', label=f"Client {cid}")
    axs[1].set_title("Communication Overhead per Round (per Client)")
    axs[1].set_xlabel("Round")
    axs[1].set_ylabel("Size (KB)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_LATENCY)
    print(f"Combined plot saved at {OUTPUT_PATH_LATENCY}")

def server_plot():

    # Read data from JSON file
    # Adjust the path
    with open(f'/home/FLCMed-TAD/server_metrics/ev_metrics_history.json', 'r') as f:
        data = json.load(f)

    rounds = data['rounds']
    mse = data['mse']
    mae = data['mae']
    r2 = data['r2']

    # Create plots
    plt.figure(figsize=(12, 8))

    # MSE plot
    plt.subplot(3, 1, 1)
    plt.plot(rounds, mse, marker='o', linestyle='-')
    plt.title('Model Metrics Over Rounds')
    plt.ylim(0,5)
    plt.ylabel('MSE')
    plt.grid(True)

    # MAE plot
    plt.subplot(3, 1, 2)
    plt.plot(rounds, mae, marker='o', linestyle='-', color='orange')
    plt.ylim(0,5)
    plt.ylabel('MAE')
    plt.grid(True)

    # R² plot
    plt.subplot(3, 1, 3)
    plt.plot(rounds, r2, marker='o', linestyle='-', color='green')
    plt.ylim(0,1)
    plt.xlabel('Round')
    plt.ylabel('R²')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT_PATH_SERVER)
    print(f"Server plot saved at {OUTPUT_PATH_SERVER}")


def client_data_plot():
    # Adjust the path
    metrics_dir = '/home/FLCMed-TAD/federated_learning_updated_model_EV/training_metrics'  #adjust the path
    json_files = glob(os.path.join(metrics_dir, 'metrics_client_*.json'))

    if not json_files:
        print("No metrics files found!")
        return

    for filepath in json_files:
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)

            rounds = data['rounds']
            mse = data['mse']
            mae = data['mae']
            r2 = data['r2']
            rmse = data.get('rmse', [])  # In case rmse missing in some files

            # Extract client ID from filename
            clientid = os.path.basename(filepath).split('_')[-1].split('.')[0]

            # Plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            # MSE
            axes[0].plot(rounds, mse, marker='o', color='blue')
            axes[0].set_title(f'Client {clientid} - MSE Over Rounds')
            axes[0].set_xlabel('Round')
            axes[0].set_ylim(0, 5)
            axes[0].set_ylabel('MSE')
            axes[0].grid(True)

            # MAE
            axes[1].plot(rounds, mae, marker='o', color='green')
            axes[1].set_title('MAE Over Rounds')
            axes[1].set_xlabel('Round')
            axes[1].set_ylim(0, 5)
            axes[1].set_ylabel('MAE')
            axes[1].grid(True)

            # R²
            axes[2].plot(rounds, r2, marker='o', color='orange')
            axes[2].set_title('R² Over Rounds')
            axes[2].set_xlabel('Round')
            axes[2].set_ylim(0, 2)
            axes[2].set_ylabel('R²')
            axes[2].grid(True)

            plt.tight_layout()

            # Save plot
            save_path = os.path.join(metrics_dir, f'metrics_client_{clientid}.png')
            plt.savefig(save_path)
            plt.close()  # Close figure to free memory

            print(f"Saved plot for client {clientid} at {save_path}")

        except Exception as e:
            print(f"Failed to process {filepath}: {e}")


def combined_client_server_plot():
    # Adjust the path
    metrics_dir = '/home/FLCMed-TAD/federated_learning_updated_model_EV/training_metrics'
    server_file = '/home/FLCMed-TAD/federated_learning_updated_model_EV/server_metrics/ev_metrics_history.json'

    # Load server data
    with open(server_file, 'r') as f:
        server_data = json.load(f)

    s_rounds = server_data['rounds']
    s_mse = server_data['mse']
    s_mae = server_data['mae']
    s_r2 = server_data['r2']

    # Get client files
    client_files = glob(os.path.join(metrics_dir, 'metrics_client_*.json'))

    if not client_files:
        print("No client metrics files found!")
        return

    # Prepare figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Colors for clients
    client_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot each client
    for idx, filepath in enumerate(client_files):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)

            rounds = data['rounds']
            mse = data['mse']
            mae = data['mae']
            r2 = data['r2']

            client_id = os.path.basename(filepath).split('_')[-1].split('.')[0]
            color = client_colors[idx % len(client_colors)]

            # MSE
            axes[0].plot(rounds, mse, marker='o', linestyle='--', label=f'Client {client_id}', color=color)
            # MAE
            axes[1].plot(rounds, mae, marker='o', linestyle='--', label=f'Client {client_id}', color=color)
            # R2
            axes[2].plot(rounds, r2, marker='o', linestyle='--', label=f'Client {client_id}', color=color)

        except Exception as e:
            print(f"Failed to process {filepath}: {e}")

    # Plot server data
    axes[0].plot(s_rounds, s_mse, marker='x', linestyle='-', color='black', linewidth=2, label='Server')
    axes[1].plot(s_rounds, s_mae, marker='x', linestyle='-', color='black', linewidth=2, label='Server')
    axes[2].plot(s_rounds, s_r2, marker='x', linestyle='-', color='black', linewidth=2, label='Server')

    # Set titles, labels, grid
    axes[0].set_title('MSE Over Rounds')
    #axes[0].set_ylim(0, 5)
    axes[0].autoscale(enable=True, axis='both', tight=True)
    axes[0].set_ylabel('MSE')
    axes[0].grid(True)

    axes[1].set_title('MAE Over Rounds')
    axes[1].autoscale(enable=True, axis='both', tight=True)
    #axes[1].set_ylim(0, 5)
    axes[1].set_ylabel('MAE')
    axes[1].grid(True)

    axes[2].set_title('R² Over Rounds')
    axes[2].autoscale(enable=True, axis='both', tight=True)
    #axes[2].set_ylim(0, 1)
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('R²')
    axes[2].grid(True)

    # Add legends
    for ax in axes:
        ax.legend()

    plt.tight_layout()

    # Save combined plot
    save_path = os.path.join(metrics_dir, 'combined_client_server_metrics.png')
    plt.savefig(save_path)
    plt.show()

    print(f"Combined plot saved at {save_path}")

def moving_average(data, window_size=3):
    """Compute simple moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def fit_trend_line(x, y):
    """Fit and return a linear trend line."""
    coeffs = np.polyfit(x, y, 1)
    return np.poly1d(coeffs)

def client_data_plot_new():
    # Adjust the path
    metrics_dir = '/home/FLCMed-TAD/federated_learning_updated_model_EV/evaluation_metrics'
    json_files = glob(os.path.join(metrics_dir, 'metrics_client_*.json'))

    if not json_files:
        print("No metrics files found!")
        return

    window_size = 3  # Moving average window

    for filepath in json_files:
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)

            rounds = np.array(data['rounds'])
            mse = np.array(data['mse'])
            mae = np.array(data['mae'])
            r2 = np.array(data['r2'])
            rmse = np.array(data.get('rmse', []))

            clientid = os.path.basename(filepath).split('_')[-1].split('.')[0]

            # Compute moving averages
            rounds_ma = rounds[window_size - 1:]
            mse_ma = moving_average(mse, window_size)
            mae_ma = moving_average(mae, window_size)
            r2_ma = moving_average(r2, window_size)
            rmse_ma = moving_average(rmse, window_size) if len(rmse) else None

            # Compute trend lines
            mse_trend = fit_trend_line(rounds, mse)
            mae_trend = fit_trend_line(rounds, mae)
            r2_trend = fit_trend_line(rounds, r2)
            rmse_trend = fit_trend_line(rounds, rmse) if len(rmse) else None

            # Plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            # MSE
            axes[0].plot(rounds_ma, mse_ma, marker='o', color='blue', label=f'Moving Avg (window={window_size})')
            axes[0].plot(rounds, mse_trend(rounds), linestyle='--', color='black', label='Trend Line')
            axes[0].set_title(f'Client {clientid} - MSE Over Rounds')
            axes[0].set_ylabel('MSE')
            axes[0].autoscale(enable=True, axis='both', tight=True)
            #axes[0].set_ylim(0, 5)
            axes[0].grid(True)
            axes[0].legend()

            # MAE
            axes[1].plot(rounds_ma, mae_ma, marker='o', color='green', label=f'Moving Avg (window={window_size})')
            axes[1].plot(rounds, mae_trend(rounds), linestyle='--', color='black', label='Trend Line')
            axes[1].set_title('MAE Over Rounds')
            axes[1].set_ylabel('MAE')
            axes[1].autoscale(enable=True, axis='both', tight=True)
            #axes[1].set_ylim(0, 5)
            axes[1].grid(True)
            axes[1].legend()

            # R²
            axes[2].plot(rounds_ma, r2_ma, marker='o', color='orange', label=f'Moving Avg (window={window_size})')
            axes[2].plot(rounds, r2_trend(rounds), linestyle='--', color='black', label='Trend Line')
            axes[2].set_title('R² Over Rounds')
            axes[2].set_xlabel('Round')
            axes[2].set_ylabel('R²')
            axes[2].autoscale(enable=True, axis='both', tight=True)
            #axes[2].set_ylim(0, 2)
            axes[2].grid(True)
            axes[2].legend()

            plt.tight_layout()

            # Save main metrics plot
            save_path = os.path.join(metrics_dir, f'metrics_client_{clientid}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot for client {clientid} at {save_path}")

        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
