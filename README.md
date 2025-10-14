# 🧠 MLP-MNIST Experimentation Framework

<div align="center">

**Advanced Multi-Layer Perceptron Experimentation System for MNIST Digit Recognition**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](.)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📋 Overview

A comprehensive, visually stunning framework for experimenting with Multi-Layer Perceptrons (MLPs) on the MNIST dataset. This system provides:

- 🎨 **Beautiful terminal UI** powered by Rich
- 📊 **Advanced visualizations** including heatmaps, confusion matrices, decision boundaries, and loss landscapes
- 🔬 **Extensive experimentation** tools for hyperparameter exploration
- 🎯 **Noise robustness testing** with multiple noise types
- 📈 **Detailed mathematical reports** with statistical analysis
- 🎬 **Training animations** to visualize the learning process
- 🌐 **Interactive dashboards** with Plotly

## ✨ Features

### 🎯 Core Capabilities

- **Multiple Experiment Types**
  - Layer configuration analysis (1-5 hidden layers)
  - Learning rate exploration (0.001 - 1.5)
  - Activation function comparison (sigmoid, tanh, ReLU)
  - Comprehensive random search
  - Noise robustness testing

- **Advanced Visualizations**
  - 📊 Dataset sample visualization
  - �️ **MNIST Dataset Overview** (NEW! - comprehensive dataset analysis)
  - 🧠 **Network Topology Animation** (NEW! - animated neuron activation flow)
  - �📈 Training history curves (loss, accuracy, time)
  - 🔲 Confusion matrices (normalized & raw)
  - 🎯 Prediction samples with confidence scores
  - 🌡️ Probability heatmaps
  - ⚖️ Weight distribution analysis
  - 🗺️ Decision boundary plots (PCA projection)
  - 🏔️ Loss landscape visualization (2D & 3D)
  - 🎬 Training animations (GIF)
  - 📊 Interactive dashboards (HTML)
  - 🔊 Clean vs noisy data comparison

- **Noise Analysis**
  - Gaussian noise
  - Salt & Pepper noise
  - Speckle noise
  - Uniform noise
  - Configurable noise levels

- **Mathematical Reports**
  - Comprehensive architecture analysis
  - Weight statistics per layer
  - Training dynamics metrics
  - Performance evaluation
  - Detailed classification metrics
  - Confusion matrix analysis
  - Statistical tests
  - Convergence analysis

### 🎨 User Interface

- Beautiful ASCII art banner
- Color-coded terminal output
- Progress bars with time estimates
- Interactive menus with rich formatting
- Tabular result displays
- Real-time experiment tracking

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project**
   ```bash
   cd T2_MLP-MNIST
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### Option 1: Interactive Application

Run the main interactive application:

```bash
python main.py
```

This launches a beautiful terminal UI with a main menu offering:
1. Quick Experiment
2. Layer Configuration Analysis
3. Learning Rate Exploration
4. Activation Function Comparison
5. Comprehensive Grid Search
6. Noise Robustness Testing
7. Configure Settings
8. Load Previous Results

### Option 2: Quick Start Script

For a simple demonstration:

```bash
python scripts/quick_start.py
```

This will:
- Load 2000 MNIST samples
- Train an MLP with [128, 64] architecture
- Generate all visualizations
- Create a mathematical report

Expected output in `output/images/`:
- Dataset samples
- Training history
- Confusion matrix
- Prediction samples
- Probability heatmap
- Weight distributions
- Decision boundary
- Loss landscape

### Option 3: Advanced Experiment

For comprehensive analysis:

```bash
python scripts/advanced_experiment.py
```

This performs:
- Layer configuration experiments (8 configs)
- Learning rate experiments (5 rates)
- Activation function comparison (2 functions)
- Full visualization suite
- Comparative analysis report

### Option 4: New Topology & Dataset Demo 🆕

To explore the new visualization features:

```bash
python scripts/topology_demo.py
```

**What's New:**

1. **🧠 Network Topology Animation**
   - Animated visualization of neural network structure
   - Shows real-time neuron activation as data flows through layers
   - Color-coded activation levels (red=high, green=low)
   - Displays prediction confidence and process
   - Creates GIF animations for multiple predictions

2. **🖼️ MNIST Dataset Overview**
   - Comprehensive dataset visualization
   - Class distribution (bar & pie charts)
   - Dataset statistics (mean, std, min, max)
   - Sample images for all 10 digits
   - Visual quality assessment

**Demo Output:**
- `mnist_dataset_overview.png` - Complete dataset analysis
- `network_topology_animation_[1-5].gif` - 5 animated network predictions
- See neurons "light up" as they process information!

This demo is perfect for:
- Understanding how neural networks work internally
- Presenting the model architecture visually
- Analyzing dataset characteristics
- Educational demonstrations
- Academic presentations

## 📚 Documentation

### Project Structure

```
T2_MLP-MNIST/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration classes
│   ├── data_loader.py         # MNIST loading & noise generation
│   ├── mlp_model.py           # MLP implementation
│   ├── experiments.py         # Experiment runner
│   ├── visualizations.py      # Visualization suite
│   ├── reports.py             # Mathematical reporting
│   └── ui.py                  # Interactive UI
├── scripts/
│   ├── quick_start.py         # Quick demo
│   ├── advanced_experiment.py # Comprehensive experiments
│   ├── batch_experiment.py    # All experiments runner
│   ├── noise_demo.py          # Noise comparison demo
│   └── topology_demo.py       # NEW! Network topology & dataset visualization demo
├── output/
│   ├── images/                # Generated visualizations
│   └── data/                  # Experiment results & reports
├── main.py                    # Main application
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

### Configuration Examples

#### Dataset Configuration

```python
from src.config import DatasetConfig

config = DatasetConfig(
    n_samples=5000,        # Number of samples
    test_size=0.2,         # Test split ratio
    normalize=True,        # Normalize pixels [0-1]
    random_seed=42         # Reproducibility
)
```

#### MLP Configuration

```python
from src.config import MLPConfig

config = MLPConfig(
    hidden_layers=[128, 64],    # Architecture
    learning_rate=0.01,         # Learning rate
    activation="sigmoid",       # sigmoid, tanh, relu
    max_epochs=100,             # Maximum epochs
    batch_size=64,              # Mini-batch size
    tolerance=1e-4,             # Early stopping
    random_seed=42
)
```

## 🎨 Visualizations Generated

The framework creates beautiful, publication-ready visualizations:

1. **Dataset Samples** - Grid view of MNIST digits
2. **Training History** - Loss and accuracy curves
3. **Confusion Matrix** - Model performance breakdown
4. **Prediction Samples** - Predictions with confidence
5. **Probability Heatmap** - Class probability distributions
6. **Weight Distributions** - Layer weight histograms
7. **Decision Boundary** - 2D decision regions (PCA)
8. **Loss Landscape** - 3D loss surface visualization
9. **Training Animation** - Learning process GIF
10. **Interactive Dashboard** - HTML dashboard with Plotly

## 📊 Mathematical Reports

Comprehensive text reports include:

- Network architecture details
- Weight statistics per layer
- Training dynamics and convergence
- Performance metrics (accuracy, loss, time)
- Per-class classification metrics
- Confusion matrix analysis
- Prediction confidence statistics
- Overfitting detection

## 🎓 Academic Use

Perfect for:
- 📝 Research papers on neural network hyperparameters
- 🎓 Course projects demonstrating MLP concepts
- 🔬 Experimentation with various architectures
- 📊 Comparative analysis of training strategies

## 🛠️ Troubleshooting

**Issue**: MNIST download fails
- Solution: Framework will retry and cache data automatically

**Issue**: Out of memory
- Solution: Reduce n_samples in DatasetConfig

**Issue**: Slow training
- Solution: Reduce max_epochs or use smaller architecture

## 📈 Performance Tips

1. Start with 1000-2000 samples for initial experiments
2. Use batch size of 32-64 for optimal performance
3. Start with learning rate of 0.01, adjust based on convergence
4. Early stopping is enabled by default (tolerance=1e-4)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional activation functions
- More optimization algorithms
- Regularization techniques
- Other datasets (Fashion-MNIST, CIFAR-10)

## 📜 License

MIT License

## 👨‍💻 Author

**Eye of the Universe**
- Soft Computing Projects
- Universidad del Universo

---

<div align="center">

**Made with ❤️ for Neural Network Research**

[⬆ Back to Top](#-mlp-mnist-experimentation-framework)

</div>
