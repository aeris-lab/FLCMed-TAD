# FLCMed-TAD
Federated Learning Framework for EV Stations Charging 

This repository implements a **Federated Learning** pipeline to detect anomalies in Electric Vehicle (EV) charging data using multiple clients. It supports customizable CPU/GPU settings and is designed to run on **Ubuntu 18.04** with **Python 3.9**.

---

## ⚙️ Setup & Environment

### 1. Clone and Setup Virtual Environment

```bash
git clone https://github.com/aeris-lab/FLCMed-TAD.git
cd FLCMed-TAD
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is failing, install manually (listing some of them below)
also be careful for the version of flower and tensorflow are in sync:
```bash
pip install flask pandas numpy scikit-learn matplotlib tensorflow ray
```

---

##  Important Note on Paths

Search for the keyword `# Adjust the path` throughout the codebase and **update paths** according to your local directory structure to ensure proper file access.

---

##  Federated Learning Flow

### Sequence of Execution:

1. **Activate the virtual environment**

```bash
source venv/bin/activate
```

2. **Run Fog Node** (Intuitive terminal-based status for each round)

```bash
python fog_node.py
```

3. **In a new terminal window**, run:

```bash
python main.py
```

> Shows continuous processing, including training and output loss per round.

---

##  Model & Output Artifacts

- **Client Weights (per layer, per round)**: `logs/clients/`
- **Global Server Metrics**: `server_metrics/`
- **Evaluation Plots per Client**: `evaluation_metrics/`
- **Training Plots per Client**: `training_metrics/`
- **Trained Client Models**:  
  Saved as `client_{client_id}_model.h5` in the root directory
- **Anomaly Logs**:  
  `anomaly_matrix.json` (for FLCMed-TAD and FLKMed-BU modes)

---

##  Dataset

- Dataset: `station_data_dataverse.csv`
- Source: `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NFPQLW`
- Total Columns: **14**
- **7 features** used for training, selected based on correlation analysis

---

##  Configuration
- The primary configuration is located at: conf/base.yaml
- num_clients, num_rounds, dataset_path_1 parameters can be configured directly from here.

---
##  Optional Live Prediction via Flask Client

### 1. Start Flask Client

```bash
python client_flask.py
```

### 2. Send Live Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -F "client_id=8" \
  -F "file=@FLCMed-TAD/dataset/station_data_dataverse_subset.csv"
```

---

##  Cleaning Up

Always terminate Flask and Fog Node servers properly after execution:

1. **Kill Flask Client**
```bash
lsof -i :5000
kill -9 <PID>
```

2. **Kill Fog Node**
```bash
lsof -i :8080
kill -9 <PID>
```

---

##  Switching Fog Node Algorithms

You can easily switch between different **fog node algorithms**:

- All fog node variants are available in the `Fog_variants/` directory.
- Simply open the desired variant and **copy its content** into `fog_node.py`.
- That’s it — you’re ready to rerun with a different algorithm.

---

##  Client Poisoning Setup

To simulate adversarial poisoning behavior:
- Search for #Client_Poisoning in client.py
- You can:
    - Define which clients to poison (poisoned_clients)
    - Select the type of attack (e.g., weight poison, data poison)
---

##  Configuration

- **Python Version**: 3.9  
- **OS**: Ubuntu 18.04  
- CPU/GPU usage can be configured in `main.py` under the TensorFlow session setup.

---

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
