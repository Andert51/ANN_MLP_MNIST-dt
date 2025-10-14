"""
Multi-Layer Perceptron implementation with detailed tracking
"""
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
import time

from .config import MLPConfig


@dataclass
class TrainingHistory:
    """Store training history"""
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    epochs: List[int]
    learning_rates: List[float]
    weights_history: List[List[np.ndarray]]
    biases_history: List[List[np.ndarray]]
    training_times: List[float]
    
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []
        self.learning_rates = []
        self.weights_history = []
        self.biases_history = []
        self.training_times = []


class ActivationFunction:
    """Activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid"""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @classmethod
    def get_activation(cls, name: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative"""
        if name == "sigmoid":
            return cls.sigmoid, cls.sigmoid_derivative
        elif name == "tanh":
            return cls.tanh, cls.tanh_derivative
        elif name == "relu":
            return cls.relu, cls.relu_derivative
        else:
            raise ValueError(f"Unknown activation function: {name}")


class MLPClassifier:
    """Multi-Layer Perceptron Classifier"""
    
    def __init__(self, config: MLPConfig):
        self.config = config
        self.weights = []
        self.biases = []
        self.history = TrainingHistory()
        self.activation_fn, self.activation_derivative = ActivationFunction.get_activation(
            config.activation
        )
        np.random.seed(config.random_seed)
        
    def _initialize_weights(self, input_size: int, output_size: int):
        """Initialize weights using He initialization"""
        layers = [input_size] + self.config.hidden_layers + [output_size]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # He initialization
            limit = np.sqrt(2.0 / layers[i])
            w = np.random.randn(layers[i], layers[i + 1]) * limit
            b = np.zeros((1, layers[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation (softmax for output layer, configured activation for hidden)
            if i == len(self.weights) - 1:
                a = ActivationFunction.softmax(z)
            else:
                a = self.activation_fn(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_pass(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                      z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass - compute gradients"""
        m = X.shape[0]
        
        # One-hot encode y
        y_onehot = np.zeros((m, self.weights[-1].shape[1]))
        y_onehot[np.arange(m), y] = 1
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = activations[-1] - y_onehot
        dW[-1] = np.dot(activations[-2].T, delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Hidden layers gradients
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(z_values[i])
            dW[i] = np.dot(activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, db
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        m = y_true.shape[0]
        y_onehot = np.zeros((m, y_pred.shape[1]))
        y_onehot[np.arange(m), y_true] = 1
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_onehot * np.log(y_pred_clipped)) / m
        
        return loss
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy"""
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'MLPClassifier':
        """Train the MLP"""
        # Initialize weights
        input_size = X_train.shape[1]
        output_size = len(np.unique(y_train))
        self._initialize_weights(input_size, output_size)
        
        # Training loop
        n_batches = len(X_train) // self.config.batch_size
        
        for epoch in range(self.config.max_epochs):
            epoch_start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations, z_values = self._forward_pass(X_batch)
                
                # Backward pass
                dW, db = self._backward_pass(X_batch, y_batch, activations, z_values)
                
                # Update weights
                for i in range(len(self.weights)):
                    self.weights[i] -= self.config.learning_rate * dW[i]
                    self.biases[i] -= self.config.learning_rate * db[i]
            
            # Compute metrics
            train_activations, _ = self._forward_pass(X_train)
            train_loss = self._compute_loss(y_train, train_activations[-1])
            train_acc = self._compute_accuracy(y_train, train_activations[-1])
            
            epoch_time = time.time() - epoch_start_time
            
            # Store history
            self.history.epochs.append(epoch)
            self.history.train_losses.append(train_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.learning_rates.append(self.config.learning_rate)
            self.history.training_times.append(epoch_time)
            
            # Store weights snapshot (every 10 epochs to save memory)
            if epoch % 10 == 0:
                self.history.weights_history.append([w.copy() for w in self.weights])
                self.history.biases_history.append([b.copy() for b in self.biases])
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_activations, _ = self._forward_pass(X_val)
                val_loss = self._compute_loss(y_val, val_activations[-1])
                val_acc = self._compute_accuracy(y_val, val_activations[-1])
                
                self.history.val_losses.append(val_loss)
                self.history.val_accuracies.append(val_acc)
            
            # Early stopping check
            if len(self.history.train_losses) > 10:
                recent_losses = self.history.train_losses[-10:]
                if max(recent_losses) - min(recent_losses) < self.config.tolerance:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        activations, _ = self._forward_pass(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        activations, _ = self._forward_pass(X)
        return activations[-1]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_architecture_info(self) -> Dict:
        """Get information about the network architecture"""
        return {
            'input_size': self.weights[0].shape[0],
            'hidden_layers': self.config.hidden_layers,
            'output_size': self.weights[-1].shape[1],
            'total_layers': len(self.weights),
            'total_parameters': sum(w.size + b.size for w, b in zip(self.weights, self.biases)),
            'activation': self.config.activation
        }
