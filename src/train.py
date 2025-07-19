import os
import sys
from utils import load_config, load_data, train_model, save_model, evaluate_model


def main():
    """Main training pipeline."""
    # Load configuration
    config = load_config('../config/config.json')
    print("Configuration loaded:", config)

    # Load data
    X, y = load_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Train model
    print("Training model...")
    model = train_model(X, y, config)

    # Evaluate model
    accuracy, f1 = evaluate_model(model, X, y)
    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Training F1-Score: {f1:.4f}")

    # Save model
    save_model(model, 'model_train.pkl')
    print("Training completed successfully!")


if __name__ == "__main__":
    main()