import os
import sys
import numpy as np
from utils import load_data, load_model, evaluate_model


def main():
    """Main inference pipeline."""
    print("Starting inference pipeline...")

    # Check if model exists
    model_path = 'model_train.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        sys.exit(1)

    # Load the trained model
    print("Loading trained model...")
    model = load_model(model_path)
    print("Model loaded successfully!")

    # Load the dataset (same as training)
    print("Loading dataset...")
    X, y = load_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Evaluate performance
    accuracy, f1 = evaluate_model(model, X, y)

    # Print results
    print("\n=== INFERENCE RESULTS ===")
    print(f"Total samples processed: {len(predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    # Show sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    for i in range(min(10, len(predictions))):
        confidence = np.max(probabilities[i])
        print(f"Sample {i + 1}: Predicted={predictions[i]}, Actual={y[i]}, Confidence={confidence:.3f}")

    # Class-wise prediction counts
    print("\n=== PREDICTION DISTRIBUTION ===")
    unique, counts = np.unique(predictions, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"Digit {digit}: {count} predictions")

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()