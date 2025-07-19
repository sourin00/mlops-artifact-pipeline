import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_data():
    """Load and return digits dataset."""
    digits = load_digits()
    return digits.data, digits.target

def train_model(X, y, config):
    """Train LogisticRegression model with given configuration."""
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=config.get('random_state', 42),
        multi_class='ovr'
    )
    model.fit(X, y)
    return model

def save_model(model, filepath):
    """Save model using joblib."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load model using joblib."""
    return joblib.load(filepath)

def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    return accuracy, f1