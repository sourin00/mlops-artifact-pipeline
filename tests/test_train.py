import pytest
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import load_config, load_data, train_model, save_model, load_model, evaluate_model


class TestTrainingPipeline:
    """Test suite for training pipeline components."""

    def test_config_file_loading(self):
        """Test that configuration file loads successfully."""
        config = load_config('config/config.json')

        # Check that config is loaded
        assert config is not None
        assert isinstance(config, dict)

        # Check required hyperparameters exist
        required_params = ['C', 'solver', 'max_iter']
        for param in required_params:
            assert param in config, f"Missing required parameter: {param}"

        # Check data types
        assert isinstance(config['C'], (int, float)), "C should be numeric"
        assert isinstance(config['solver'], str), "solver should be string"
        assert isinstance(config['max_iter'], int), "max_iter should be integer"

        # Check reasonable values
        assert config['C'] > 0, "C should be positive"
        assert config['max_iter'] > 0, "max_iter should be positive"

    def test_model_creation(self):
        """Test that model creation returns LogisticRegression object."""
        # Load test data
        X, y = load_data()
        config = load_config('config/config.json')

        # Train model
        model = train_model(X, y, config)

        # Verify model type
        assert isinstance(model, LogisticRegression), "Model should be LogisticRegression instance"

        # Check that model is fitted (has required attributes)
        assert hasattr(model, 'coef_'), "Model should have coef_ attribute after fitting"
        assert hasattr(model, 'classes_'), "Model should have classes_ attribute after fitting"

        # Check model parameters match config
        assert model.C == config['C'], "Model C parameter should match config"
        assert model.solver == config['solver'], "Model solver should match config"
        assert model.max_iter == config['max_iter'], "Model max_iter should match config"

    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy."""
        # Load data
        X, y = load_data()
        config = load_config('config/config.json')

        # Train model
        model = train_model(X, y, config)

        # Evaluate model
        accuracy, f1 = evaluate_model(model, X, y)

        # Check accuracy threshold (digits dataset should achieve high accuracy)
        assert accuracy > 0.85, f"Model accuracy {accuracy:.4f} is below threshold 0.85"
        assert f1 > 0.85, f"Model F1-score {f1:.4f} is below threshold 0.85"

        # Check that predictions have correct shape
        predictions = model.predict(X)
        assert predictions.shape[0] == X.shape[0], "Predictions should match input sample count"
        assert set(predictions).issubset(set(y)), "Predictions should only contain valid class labels"

    def test_data_loading(self):
        """Test data loading functionality."""
        X, y = load_data()

        # Check data shapes
        assert X.shape[0] > 0, "Should have samples"
        assert X.shape[1] == 64, "Should have 64 features (8x8 flattened)"
        assert len(y) == X.shape[0], "Labels should match sample count"

        # Check data types
        assert isinstance(X, np.ndarray), "X should be numpy array"
        assert isinstance(y, np.ndarray), "y should be numpy array"

        # Check class labels (digits 0-9)
        unique_labels = set(y)
        expected_labels = set(range(10))
        assert unique_labels == expected_labels, "Should have digits 0-9"

    def test_model_save_load(self):
        """Test model saving and loading functionality."""
        # Train a model
        X, y = load_data()
        config = load_config('config/config.json')
        model = train_model(X, y, config)

        # Save model
        test_model_path = 'test_model.pkl'
        save_model(model, test_model_path)

        # Check file exists
        assert os.path.exists(test_model_path), "Model file should be created"

        # Load model
        loaded_model = load_model(test_model_path)

        # Test loaded model
        assert isinstance(loaded_model, LogisticRegression), "Loaded model should be LogisticRegression"

        # Test predictions match
        original_pred = model.predict(X[:10])  # Test on subset
        loaded_pred = loaded_model.predict(X[:10])
        np.testing.assert_array_equal(original_pred, loaded_pred,
                                      "Loaded model should produce same predictions")

        # Cleanup
        if os.path.exists(test_model_path):
            os.remove(test_model_path)