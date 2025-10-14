"""
Data loading and preprocessing module
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import pickle
from pathlib import Path

from .config import DatasetConfig, NoiseConfig

console = Console()


class MNISTLoader:
    """Load and preprocess MNIST dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_original = None
        self.X_test_original = None
        
    def load_data(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST dataset"""
        cache_path = Path("output/data/mnist_cache.pkl")
        
        if use_cache and cache_path.exists():
            console.print("[cyan]📦 Loading MNIST from cache...[/cyan]")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            self.X_train_original = data['X_train_original']
            self.X_test_original = data['X_test_original']
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Loading MNIST dataset...", total=None)
                
                # Load MNIST
                mnist = fetch_openml('mnist_784', version=1, parser='auto')
                X, y = mnist.data.values, mnist.target.values.astype(int)
                
                progress.update(task, description="[cyan]Sampling data...")
                # Sample if needed
                if self.config.n_samples < len(X):
                    indices = np.random.RandomState(self.config.random_seed).choice(
                        len(X), self.config.n_samples, replace=False
                    )
                    X, y = X[indices], y[indices]
                
                progress.update(task, description="[cyan]Splitting dataset...")
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=self.config.test_size,
                    random_state=self.config.random_seed,
                    stratify=y
                )
                
                # Store original data
                self.X_train_original = X_train.copy()
                self.X_test_original = X_test.copy()
                
                progress.update(task, description="[cyan]Normalizing data...")
                # Normalize
                if self.config.normalize:
                    X_train = X_train / 255.0
                    X_test = X_test / 255.0
                
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                
                # Cache data
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'X_train': self.X_train,
                        'X_test': self.X_test,
                        'y_train': self.y_train,
                        'y_test': self.y_test,
                        'X_train_original': self.X_train_original,
                        'X_test_original': self.X_test_original
                    }, f)
                
                progress.update(task, description="[green]✓ Dataset loaded successfully!")
        
        console.print(f"[green]✓ Training samples: {len(self.X_train)}[/green]")
        console.print(f"[green]✓ Test samples: {len(self.X_test)}[/green]")
        console.print(f"[green]✓ Features: {self.X_train.shape[1]}[/green]")
        console.print(f"[green]✓ Classes: {len(np.unique(self.y_train))}[/green]")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_subset(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a subset of the data"""
        indices_train = np.random.choice(len(self.X_train), min(n_samples, len(self.X_train)), replace=False)
        indices_test = np.random.choice(len(self.X_test), min(n_samples // 4, len(self.X_test)), replace=False)
        
        return (
            self.X_train[indices_train],
            self.X_test[indices_test],
            self.y_train[indices_train],
            self.y_test[indices_test]
        )


class NoiseGenerator:
    """Generate different types of noise for data augmentation"""
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, noise_level: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
        """Add Gaussian noise to data"""
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, noise_level, X.shape)
        return np.clip(X + noise, 0, 1)
    
    @staticmethod
    def add_salt_pepper_noise(X: np.ndarray, probability: float = 0.05, seed: Optional[int] = None) -> np.ndarray:
        """Add salt and pepper noise"""
        if seed is not None:
            np.random.seed(seed)
        X_noisy = X.copy()
        
        # Salt
        salt_mask = np.random.random(X.shape) < probability / 2
        X_noisy[salt_mask] = 1
        
        # Pepper
        pepper_mask = np.random.random(X.shape) < probability / 2
        X_noisy[pepper_mask] = 0
        
        return X_noisy
    
    @staticmethod
    def add_speckle_noise(X: np.ndarray, noise_level: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
        """Add speckle (multiplicative) noise"""
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(1, noise_level, X.shape)
        return np.clip(X * noise, 0, 1)
    
    @staticmethod
    def add_uniform_noise(X: np.ndarray, noise_level: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
        """Add uniform noise"""
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.uniform(-noise_level, noise_level, X.shape)
        return np.clip(X + noise, 0, 1)
    
    @classmethod
    def apply_noise(cls, X: np.ndarray, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
        """Apply noise based on configuration"""
        if config.noise_type == "gaussian":
            return cls.add_gaussian_noise(X, config.noise_level, seed)
        elif config.noise_type == "salt_pepper":
            return cls.add_salt_pepper_noise(X, config.noise_probability, seed)
        elif config.noise_type == "speckle":
            return cls.add_speckle_noise(X, config.noise_level, seed)
        elif config.noise_type == "uniform":
            return cls.add_uniform_noise(X, config.noise_level, seed)
        else:
            raise ValueError(f"Unknown noise type: {config.noise_type}")
