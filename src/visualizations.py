"""
Advanced visualization module for MLP experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json

from .config import VisualizationConfig
from .mlp_model import MLPClassifier, TrainingHistory


class MLPVisualizer:
    """Comprehensive visualization suite for MLP experiments"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_dataset_samples(self, X: np.ndarray, y: np.ndarray, 
                            n_samples: int = 25, title: str = "Dataset Samples",
                            save_name: Optional[str] = None):
        """Visualize random samples from the dataset"""
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(indices):
                image = X[indices[idx]].reshape(28, 28)
                ax.imshow(image, cmap='gray')
                ax.set_title(f'Label: {y[indices[idx]]}', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_noise_comparison(self, X_clean: np.ndarray, X_noisy: np.ndarray, 
                             y: np.ndarray, n_samples: int = 10,
                             noise_type: str = "gaussian",
                             save_name: Optional[str] = None):
        """Compare clean and noisy samples side by side"""
        fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
        fig.suptitle(f'Clean vs {noise_type.title()} Noise Comparison', 
                    fontsize=16, fontweight='bold')
        
        indices = np.random.choice(len(X_clean), n_samples, replace=False)
        
        for idx, sample_idx in enumerate(indices):
            # Clean image
            axes[idx, 0].imshow(X_clean[sample_idx].reshape(28, 28), cmap='gray')
            axes[idx, 0].set_title(f'Clean - Label: {y[sample_idx]}')
            axes[idx, 0].axis('off')
            
            # Noisy image
            axes[idx, 1].imshow(X_noisy[sample_idx].reshape(28, 28), cmap='gray')
            axes[idx, 1].set_title(f'With Noise - Label: {y[sample_idx]}')
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_training_history(self, history: TrainingHistory, 
                             save_name: Optional[str] = None):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = history.epochs
        
        # Loss curves
        axes[0, 0].plot(epochs, history.train_losses, label='Train Loss', 
                       linewidth=2, marker='o', markersize=3)
        if history.val_losses:
            axes[0, 0].plot(epochs, history.val_losses, label='Validation Loss', 
                          linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history.train_accuracies, label='Train Accuracy', 
                       linewidth=2, marker='o', markersize=3)
        if history.val_accuracies:
            axes[0, 1].plot(epochs, history.val_accuracies, label='Validation Accuracy', 
                          linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training time per epoch
        axes[1, 0].plot(epochs, history.training_times, linewidth=2, 
                       marker='o', markersize=3, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if it changes)
        axes[1, 1].plot(epochs, history.learning_rates, linewidth=2, 
                       marker='o', markersize=3, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             classes: Optional[List[str]] = None,
                             normalize: bool = False,
                             save_name: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=self.config.cmap_confusion,
                   xticklabels=classes or range(len(cm)),
                   yticklabels=classes or range(len(cm)),
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax)
        
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_probability_heatmap(self, model: MLPClassifier, X: np.ndarray, 
                                y: np.ndarray, n_samples: int = 20,
                                save_name: Optional[str] = None):
        """Plot probability heatmap for predictions"""
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        probas = model.predict_proba(X_subset)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sns.heatmap(probas.T, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f'S{i}\n(L:{y_subset[i]})' for i in range(len(indices))],
                   yticklabels=[f'Class {i}' for i in range(probas.shape[1])],
                   cbar_kws={'label': 'Probability'},
                   ax=ax)
        
        ax.set_xlabel('Sample (True Label)', fontweight='bold')
        ax.set_ylabel('Predicted Class', fontweight='bold')
        ax.set_title('Prediction Probability Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_prediction_samples(self, model: MLPClassifier, X: np.ndarray, 
                               y: np.ndarray, n_samples: int = 20,
                               save_name: Optional[str] = None):
        """Visualize predictions with confidence scores"""
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        predictions = model.predict(X[indices])
        probas = model.predict_proba(X[indices])
        confidences = np.max(probas, axis=1)
        
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        fig.suptitle('Prediction Samples with Confidence', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < n_samples:
                image = X[indices[idx]].reshape(28, 28)
                true_label = y[indices[idx]]
                pred_label = predictions[idx]
                confidence = confidences[idx]
                
                ax.imshow(image, cmap='gray')
                
                color = 'green' if true_label == pred_label else 'red'
                title = f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2%}'
                ax.set_title(title, color=color, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_decision_boundary_2d(self, model: MLPClassifier, X: np.ndarray, 
                                 y: np.ndarray, resolution: int = 100,
                                 save_name: Optional[str] = None):
        """Plot decision boundary using PCA for dimensionality reduction"""
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Create mesh
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_full = pca.inverse_transform(mesh_points)
        Z = model.predict(mesh_full)
        Z = Z.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='tab10', levels=9)
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', 
                           edgecolors='k', s=50, alpha=0.8)
        
        ax.set_xlabel('First Principal Component', fontweight='bold')
        ax.set_ylabel('Second Principal Component', fontweight='bold')
        ax.set_title('Decision Boundary (PCA Projection)', fontsize=14, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, label='Class')
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_loss_landscape_2d(self, model: MLPClassifier, X: np.ndarray, 
                              y: np.ndarray, layer_idx: int = 0,
                              param_range: float = 0.5, resolution: int = 50,
                              save_name: Optional[str] = None):
        """Visualize loss landscape around current weights"""
        if layer_idx >= len(model.weights):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        # Select two weights to vary
        w_shape = model.weights[layer_idx].shape
        if w_shape[0] < 2 or w_shape[1] < 2:
            print("Weight matrix too small for 2D landscape")
            return None
        
        w1_idx = (0, 0)
        w2_idx = (1, 0) if w_shape[0] > 1 else (0, 1)
        
        # Original values
        w1_orig = model.weights[layer_idx][w1_idx]
        w2_orig = model.weights[layer_idx][w2_idx]
        
        # Create grid
        w1_range = np.linspace(w1_orig - param_range, w1_orig + param_range, resolution)
        w2_range = np.linspace(w2_orig - param_range, w2_orig + param_range, resolution)
        
        W1, W2 = np.meshgrid(w1_range, w2_range)
        losses = np.zeros_like(W1)
        
        # Compute loss for each point
        original_weights = [w.copy() for w in model.weights]
        
        for i in range(resolution):
            for j in range(resolution):
                model.weights[layer_idx][w1_idx] = W1[i, j]
                model.weights[layer_idx][w2_idx] = W2[i, j]
                
                activations, _ = model._forward_pass(X)
                losses[i, j] = model._compute_loss(y, activations[-1])
        
        # Restore original weights
        model.weights = original_weights
        
        # Plot
        fig = plt.figure(figsize=(14, 6))
        
        # 3D surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(W1, W2, losses, cmap='viridis', alpha=0.8)
        ax1.scatter([w1_orig], [w2_orig], [losses[resolution//2, resolution//2]], 
                   color='red', s=100, label='Current')
        ax1.set_xlabel(f'Weight [{w1_idx}]')
        ax1.set_ylabel(f'Weight [{w2_idx}]')
        ax1.set_zlabel('Loss')
        ax1.set_title('3D Loss Landscape')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # 2D contour
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(W1, W2, losses, levels=20, cmap='viridis')
        ax2.plot(w1_orig, w2_orig, 'r*', markersize=15, label='Current')
        ax2.set_xlabel(f'Weight [{w1_idx}]')
        ax2.set_ylabel(f'Weight [{w2_idx}]')
        ax2.set_title('2D Loss Contour')
        ax2.legend()
        fig.colorbar(contour, ax=ax2)
        
        plt.suptitle(f'Loss Landscape - Layer {layer_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def create_training_animation(self, history: TrainingHistory,
                                 save_name: str = "training_animation.gif"):
        """Create animated visualization of training progress"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        def update(frame):
            axes[0].clear()
            axes[1].clear()
            
            # Loss plot
            axes[0].plot(history.epochs[:frame+1], history.train_losses[:frame+1], 
                        'b-', linewidth=2, label='Train Loss')
            if history.val_losses:
                axes[0].plot(history.epochs[:frame+1], history.val_losses[:frame+1], 
                           'r-', linewidth=2, label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'Loss Curve (Epoch {history.epochs[frame]})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[1].plot(history.epochs[:frame+1], history.train_accuracies[:frame+1], 
                        'b-', linewidth=2, label='Train Acc')
            if history.val_accuracies:
                axes[1].plot(history.epochs[:frame+1], history.val_accuracies[:frame+1], 
                           'r-', linewidth=2, label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title(f'Accuracy Curve (Epoch {history.epochs[frame]})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, update, frames=len(history.epochs), 
                           interval=100, repeat=True)
        
        save_path = self.config.images_dir / save_name
        writer = PillowWriter(fps=self.config.animation_fps)
        anim.save(save_path, writer=writer)
        print(f"Saved animation: {save_path}")
        
        plt.close()
        return save_path
    
    def plot_weight_distributions(self, model: MLPClassifier,
                                 save_name: Optional[str] = None):
        """Plot weight distributions for each layer"""
        n_layers = len(model.weights)
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, (weights, ax) in enumerate(zip(model.weights, axes)):
            ax.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'Layer {i+1} Weights\n(Shape: {weights.shape})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean = np.mean(weights)
            std = np.std(weights)
            ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
            ax.axvline(mean + std, color='g', linestyle='--', linewidth=1, label=f'Std: {std:.3f}')
            ax.axvline(mean - std, color='g', linestyle='--', linewidth=1)
            ax.legend()
        
        plt.suptitle('Weight Distributions by Layer', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.config.images_dir / save_name
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def create_interactive_dashboard(self, results: List[Dict], 
                                   save_name: str = "experiment_dashboard.html"):
        """Create interactive Plotly dashboard"""
        # Prepare data
        configs = [r['config'] for r in results]
        accuracies = [r['test_accuracy'] for r in results]
        losses = [r['final_loss'] for r in results]
        times = [r['training_time'] for r in results]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy by Configuration', 'Loss by Configuration',
                          'Training Time', 'Accuracy vs Time Trade-off'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        config_names = [f"Config {i+1}" for i in range(len(results))]
        
        # Accuracy bar chart
        fig.add_trace(
            go.Bar(x=config_names, y=accuracies, name='Accuracy',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # Loss bar chart
        fig.add_trace(
            go.Bar(x=config_names, y=losses, name='Loss',
                  marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Time bar chart
        fig.add_trace(
            go.Bar(x=config_names, y=times, name='Time',
                  marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=times, y=accuracies, mode='markers+text',
                      text=config_names, textposition='top center',
                      marker=dict(size=12, color=accuracies, colorscale='Viridis',
                                showscale=True),
                      name='Configs'),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Configuration", row=1, col=1)
        fig.update_xaxes(title_text="Configuration", row=1, col=2)
        fig.update_xaxes(title_text="Configuration", row=2, col=1)
        fig.update_xaxes(title_text="Training Time (s)", row=2, col=2)
        
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="MLP Experiment Dashboard",
                         title_font_size=20)
        
        save_path = self.config.images_dir / save_name
        fig.write_html(str(save_path))
        print(f"Saved interactive dashboard: {save_path}")
        
        return fig
