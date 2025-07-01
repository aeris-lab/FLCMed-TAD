import os
import yaml
import flwr as fl
import ray
import subprocess
import time
import glob
import shutil
from clients.client import generate_client_fn
from server.server import evaluate_fn  # Ensure evaluate_fn is using Keras
from utils.data_loader import load_data, load_full_logs
from utils.plots_utils import (
    plot_client_weight_trends,
    plot_combined, server_plot, client_data_plot, combined_client_server_plot, client_data_plot_new
)

FIXED_SET = 250000
# ---------------------- Config Loader ---------------------- #
def load_config(path="conf/base.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# ---------------------- Start Fog Node ---------------------- #
def start_fog_node():
    print("Starting fog node server...")
    return subprocess.Popen(["python3", "fog_node.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def clear_directories(directories):
    """
    Deletes all files and subdirectories inside each given directory.
    Args:
        directories (list of str): List of directory paths to clean.
    """
    for dir_path in directories:
        if os.path.exists(dir_path):
            files = glob.glob(os.path.join(dir_path, '*'))
            for f in files:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
        else:
            print(f"Directory not found: {dir_path}")



# ---------------------- Main Simulation ---------------------- #
def main():
    clear_directories(["logs/clients", "server_metrics","evaluation_metrics","training_metrics"])
    config = load_config()

    # Start fog node server
    fog_process = start_fog_node()
    time.sleep(3)  # Wait for fog node to start

    try:
        # Load dataset or your dataset of choice
        dataset_path = config["fl_config"]["dataset_path_1"]
        dataset = load_data(dataset_path, chunk_size=FIXED_SET)

        # Generate client function (this will be passed to Flower simulation)
        client_fn = generate_client_fn(
            config_path="conf/base.yaml",
            dataset=dataset
        )
        # Define custom evaluation function
        def server_evaluate_fn(server_round, parameters, config_=None):
            # Using Keras-based evaluate function
            return evaluate_fn(server_round, parameters, config, dataset)

        # Define Federated Averaging strategy with evaluation > default (currently acting as a framework > single input - customizable depending on FOG output)
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config["server_config"]["min_available_clients"],
            min_evaluate_clients=config["server_config"]["min_available_clients"],
            min_available_clients=config["server_config"]["min_available_clients"],
            on_fit_config_fn=lambda rnd: {
                "round": rnd,
                "batch_size": config["fl_config"]["training"]["batch_size"],
                "epochs": config["fl_config"]["training"]["epochs"],
                "learning_rate": config["fl_config"]["training"]["lr"]
            },
            evaluate_fn=lambda r, p, c: server_evaluate_fn(r, p, config)  # Use the Keras-based evaluation function here
        )

        # --------------------- Start Ray and Federated Learning --------------------- #
        ray.init(ignore_reinit_error=True)

        print("\n Starting federated learning simulation...\n")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config["fl_config"]["num_clients"],
            config=fl.server.ServerConfig(num_rounds=config["server_config"]["num_rounds"]),
            strategy=strategy,
        )

        print("\n Simulation Complete.")
        print(f" Final Metrics:\n{history.metrics_centralized}")

        # ---------------------- Plotting Results ---------------------- #
        print("\n Showing consolidated server-side metrics plot...")
        server_plot()

        print("\n Showing per-client metrics plots...")
        plot_client_weight_trends()
        client_data_plot()

        # Uncomment the lines below to plot weight trends and aggregated weights if needed
        latency_logs = load_full_logs("latency_client_*.json")
        comm_logs = load_full_logs("comm_client_*.json")
        plot_combined(latency_logs, comm_logs)

    finally:
        # Ensure fog node is shut down gracefully
        print("\nShutting down fog node...")
        fog_process.terminate()
        try:
            fog_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fog_process.kill()
        ray.shutdown()  # Gracefully shut down Ray

if __name__ == "__main__":
    main()
