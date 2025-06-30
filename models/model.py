import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ---------------------- Config Loader ---------------------- #
def load_config(path="conf/base.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# ---------------------- Model -------------------------------#
def ev_model(input_dim):
    config = load_config()
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=l2(config["fl_config"]["training"]["lr"])), #set to 64 accordingly
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(config["fl_config"]["training"]["lr"])), #set to 32 accordingly
        Dropout(0.3),
        Dense(1)  # single neuron, linear activation (default)
    ])

    return model