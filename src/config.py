"""
Configuration module for MLP-MNIST Experimentation Framework
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np


@dataclass
class MLPConfig:
    """Configuration for MLP model"""
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 0.01
    max_epochs: int = 100
    activation: str = "sigmoid"  # sigmoid, tanh, relu
    batch_size: int = 32
    tolerance: float = 1e-4
    random_seed: int = 42
    
    def __str__(self):
        return (f"MLP[layers={self.hidden_layers}, lr={self.learning_rate}, "
                f"activation={self.activation}, epochs={self.max_epochs}]")


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Hyperparameter ranges to test
    hidden_layer_configs: List[List[int]] = field(default_factory=lambda: [
        [32],
        [64],
        [128],
        [64, 32],
        [128, 64],
        [128, 64, 32],
        [256, 128, 64],
        [512, 256, 128, 64]
    ])
    
    learning_rates: List[float] = field(default_factory=lambda: [
        0.001, 0.01, 0.1, 0.5, 0.75, 1.0
    ])
    
    activations: List[str] = field(default_factory=lambda: [
        "sigmoid", "tanh"
    ])
    
    neurons_per_layer: List[int] = field(default_factory=lambda: [
        32, 64, 128, 256, 512
    ])
    
    num_hidden_layers: List[int] = field(default_factory=lambda: [
        1, 2, 3, 4, 5
    ])
    
    max_epochs: int = 150
    batch_size: int = 64
    n_samples: int = 1000  # Number of samples to use (for quick testing)
    test_split: float = 0.2
    random_seed: int = 42


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    output_dir: Path = field(default_factory=lambda: Path("output"))
    images_dir: Path = field(default_factory=lambda: Path("output/images"))
    data_dir: Path = field(default_factory=lambda: Path("output/data"))
    
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "seaborn-v0_8-darkgrid"
    
    # Color schemes
    color_palette: str = "viridis"
    cmap_heatmap: str = "RdYlBu_r"
    cmap_confusion: str = "Blues"
    
    # Animation settings
    animation_fps: int = 10
    animation_duration: float = 0.1
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class NoiseConfig:
    """Configuration for noise addition"""
    noise_type: str = "gaussian"  # gaussian, salt_pepper, speckle
    noise_level: float = 0.1
    noise_probability: float = 0.05  # for salt & pepper
    
    available_noise_types: List[str] = field(default_factory=lambda: [
        "gaussian", "salt_pepper", "speckle", "uniform"
    ])


@dataclass
class DatasetConfig:
    """Configuration for MNIST dataset"""
    n_samples: int = 5000
    test_size: float = 0.2
    random_seed: int = 42
    normalize: bool = True
    flatten: bool = True
    
    # MNIST properties
    image_shape: Tuple[int, int] = (28, 28)
    n_classes: int = 10
    n_features: int = 784  # 28 * 28


# Global configurations
DEFAULT_MLP_CONFIG = MLPConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()
DEFAULT_NOISE_CONFIG = NoiseConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()


# Color themes for Rich UI
THEME_COLORS = {
    "primary": "cyan",
    "secondary": "magenta",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "blue",
    "neutral": "white",
    "dim": "bright_black"
}
