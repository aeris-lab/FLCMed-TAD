from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def ev_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)), #set to 128 accordingly
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)), #set to 64 accordingly
        Dropout(0.3),
        Dense(1)  # single neuron, linear activation (default)
    ])

    return model
