import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# utils/fl_utils.py

def get_model_parameters(model):
    """Get model weights for Flower using Keras."""
    return model.get_weights()

def set_model_parameters(model, parameters):
    """Set model weights for Flower using Keras."""
    model.set_weights(parameters)
