"""
Test suite for MLP-MNIST Framework
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MLPConfig, DatasetConfig, NoiseConfig
from src.mlp_model import MLPClassifier, ActivationFunction
from src.data_loader import MNISTLoader, NoiseGenerator


class TestActivationFunctions:
    """Test activation functions"""
    
    def test_sigmoid(self):
        x = np.array([-1, 0, 1])
        result = ActivationFunction.sigmoid(x)
        assert result.shape == x.shape
        assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_tanh(self):
        x = np.array([-1, 0, 1])
        result = ActivationFunction.tanh(x)
        assert result.shape == x.shape
        assert np.all(result >= -1) and np.all(result <= 1)
    
    def test_relu(self):
        x = np.array([-1, 0, 1])
        result = ActivationFunction.relu(x)
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_softmax(self):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        result = ActivationFunction.softmax(x)
        assert result.shape == x.shape
        np.testing.assert_almost_equal(result.sum(axis=1), [1, 1])


class TestNoiseGenerator:
    """Test noise generation"""
    
    def test_gaussian_noise(self):
        X = np.random.rand(10, 784)
        X_noisy = NoiseGenerator.add_gaussian_noise(X, noise_level=0.1, seed=42)
        
        assert X_noisy.shape == X.shape
        assert not np.array_equal(X, X_noisy)
        assert np.all(X_noisy >= 0) and np.all(X_noisy <= 1)
    
    def test_salt_pepper_noise(self):
        X = np.random.rand(10, 784) * 0.5  # Mid-range values
        X_noisy = NoiseGenerator.add_salt_pepper_noise(X, probability=0.1, seed=42)
        
        assert X_noisy.shape == X.shape
        # Should have some 0s and 1s
        assert np.any(X_noisy == 0) or np.any(X_noisy == 1)
    
    def test_speckle_noise(self):
        X = np.random.rand(10, 784)
        X_noisy = NoiseGenerator.add_speckle_noise(X, noise_level=0.1, seed=42)
        
        assert X_noisy.shape == X.shape
        assert not np.array_equal(X, X_noisy)
    
    def test_uniform_noise(self):
        X = np.random.rand(10, 784)
        X_noisy = NoiseGenerator.add_uniform_noise(X, noise_level=0.1, seed=42)
        
        assert X_noisy.shape == X.shape
        assert not np.array_equal(X, X_noisy)


class TestMLPClassifier:
    """Test MLP classifier"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.rand(20, 20)
        y_test = np.random.randint(0, 3, 20)
        return X_train, y_train, X_test, y_test
    
    def test_initialization(self):
        config = MLPConfig(hidden_layers=[10, 5], learning_rate=0.01)
        model = MLPClassifier(config)
        
        assert model.config == config
        assert len(model.weights) == 0  # Not initialized yet
    
    def test_fit_predict(self, sample_data):
        X_train, y_train, X_test, y_test = sample_data
        
        config = MLPConfig(
            hidden_layers=[10, 5],
            learning_rate=0.1,
            max_epochs=10,
            batch_size=10
        )
        
        model = MLPClassifier(config)
        model.fit(X_train, y_train, X_test, y_test, verbose=False)
        
        # Check weights initialized
        assert len(model.weights) == 3  # input->hidden1, hidden1->hidden2, hidden2->output
        
        # Check predictions
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert np.all(predictions >= 0) and np.all(predictions < 3)
    
    def test_predict_proba(self, sample_data):
        X_train, y_train, X_test, y_test = sample_data
        
        config = MLPConfig(hidden_layers=[10], max_epochs=5)
        model = MLPClassifier(config)
        model.fit(X_train, y_train, verbose=False)
        
        probas = model.predict_proba(X_test)
        
        # Check shape
        assert probas.shape == (len(X_test), 3)
        
        # Check probabilities sum to 1
        np.testing.assert_almost_equal(probas.sum(axis=1), np.ones(len(X_test)))
        
        # Check probabilities in [0, 1]
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    def test_score(self, sample_data):
        X_train, y_train, X_test, y_test = sample_data
        
        config = MLPConfig(hidden_layers=[10], max_epochs=10)
        model = MLPClassifier(config)
        model.fit(X_train, y_train, verbose=False)
        
        score = model.score(X_test, y_test)
        
        assert 0 <= score <= 1
    
    def test_training_history(self, sample_data):
        X_train, y_train, X_test, y_test = sample_data
        
        config = MLPConfig(hidden_layers=[10], max_epochs=5)
        model = MLPClassifier(config)
        model.fit(X_train, y_train, X_test, y_test, verbose=False)
        
        # Check history is populated
        assert len(model.history.epochs) == 5
        assert len(model.history.train_losses) == 5
        assert len(model.history.train_accuracies) == 5
        assert len(model.history.val_losses) == 5
        assert len(model.history.val_accuracies) == 5
    
    def test_different_activations(self, sample_data):
        X_train, y_train, X_test, y_test = sample_data
        
        for activation in ["sigmoid", "tanh", "relu"]:
            config = MLPConfig(
                hidden_layers=[10],
                activation=activation,
                max_epochs=5
            )
            
            model = MLPClassifier(config)
            model.fit(X_train, y_train, verbose=False)
            
            predictions = model.predict(X_test)
            assert predictions.shape == y_test.shape


class TestConfigurations:
    """Test configuration classes"""
    
    def test_mlp_config_defaults(self):
        config = MLPConfig()
        
        assert isinstance(config.hidden_layers, list)
        assert config.learning_rate > 0
        assert config.max_epochs > 0
        assert config.activation in ["sigmoid", "tanh", "relu"]
    
    def test_dataset_config_defaults(self):
        config = DatasetConfig()
        
        assert config.n_samples > 0
        assert 0 < config.test_size < 1
        assert config.normalize in [True, False]
    
    def test_noise_config_defaults(self):
        config = NoiseConfig()
        
        assert config.noise_type in ["gaussian", "salt_pepper", "speckle", "uniform"]
        assert 0 <= config.noise_level <= 1
        assert 0 <= config.noise_probability <= 1


def test_integration():
    """Integration test with real training"""
    # Create small dataset
    np.random.seed(42)
    X_train = np.random.rand(200, 50)
    y_train = np.random.randint(0, 5, 200)
    X_test = np.random.rand(50, 50)
    y_test = np.random.randint(0, 5, 50)
    
    # Configure
    config = MLPConfig(
        hidden_layers=[20, 10],
        learning_rate=0.1,
        max_epochs=20,
        batch_size=20
    )
    
    # Train
    model = MLPClassifier(config)
    model.fit(X_train, y_train, X_test, y_test, verbose=False)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Basic sanity checks
    assert train_score > 0.1  # Should be better than random (0.2 for 5 classes)
    assert test_score > 0.1
    
    # Check architecture info
    info = model.get_architecture_info()
    assert info['input_size'] == 50
    assert info['hidden_layers'] == [20, 10]
    assert info['output_size'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
