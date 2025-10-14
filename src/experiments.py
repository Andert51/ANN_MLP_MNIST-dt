"""
Experiment runner and hyperparameter exploration
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.layout import Layout
from rich import box
import itertools

from .config import MLPConfig, ExperimentConfig
from .mlp_model import MLPClassifier
from .visualizations import MLPVisualizer

console = Console()


class ExperimentRunner:
    """Run and manage MLP experiments"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.results = []
        
    def run_single_experiment(self, mlp_config: MLPConfig, 
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            experiment_name: str = "") -> Dict:
        """Run a single experiment with given configuration"""
        start_time = time.time()
        
        # Create and train model
        model = MLPClassifier(mlp_config)
        model.fit(X_train, y_train, X_test, y_test, verbose=False)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        training_time = time.time() - start_time
        
        # Get final metrics
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        result = {
            'experiment_name': experiment_name,
            'config': mlp_config,
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'final_loss': model.history.train_losses[-1] if model.history.train_losses else 0,
            'training_time': training_time,
            'n_epochs': len(model.history.epochs),
            'history': model.history,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'architecture_info': model.get_architecture_info()
        }
        
        return result
    
    def run_layer_experiment(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> List[Dict]:
        """Experiment with different numbers of hidden layers"""
        console.print("\n[bold cyan]═══ Experiment 1: Number of Hidden Layers ═══[/bold cyan]\n")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Testing layer configurations...", 
                total=len(self.config.hidden_layer_configs)
            )
            
            for i, layers in enumerate(self.config.hidden_layer_configs):
                config = MLPConfig(
                    hidden_layers=layers,
                    learning_rate=0.01,
                    max_epochs=self.config.max_epochs,
                    activation="sigmoid",
                    batch_size=self.config.batch_size
                )
                
                result = self.run_single_experiment(
                    config, X_train, y_train, X_test, y_test,
                    f"Layers: {layers}"
                )
                
                results.append(result)
                progress.update(task, advance=1, 
                              description=f"[cyan]Config {i+1}/{len(self.config.hidden_layer_configs)}: "
                                        f"Acc={result['test_accuracy']:.3f}")
        
        self._display_results_table(results, "Layer Configuration Results")
        return results
    
    def run_learning_rate_experiment(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    base_layers: List[int] = [128, 64]) -> List[Dict]:
        """Experiment with different learning rates"""
        console.print("\n[bold cyan]═══ Experiment 2: Learning Rates ═══[/bold cyan]\n")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Testing learning rates...", 
                total=len(self.config.learning_rates)
            )
            
            for i, lr in enumerate(self.config.learning_rates):
                config = MLPConfig(
                    hidden_layers=base_layers,
                    learning_rate=lr,
                    max_epochs=self.config.max_epochs,
                    activation="sigmoid",
                    batch_size=self.config.batch_size
                )
                
                result = self.run_single_experiment(
                    config, X_train, y_train, X_test, y_test,
                    f"LR: {lr}"
                )
                
                results.append(result)
                progress.update(task, advance=1,
                              description=f"[cyan]LR {i+1}/{len(self.config.learning_rates)}: "
                                        f"Acc={result['test_accuracy']:.3f}")
        
        self._display_results_table(results, "Learning Rate Results")
        return results
    
    def run_activation_experiment(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 base_layers: List[int] = [128, 64]) -> List[Dict]:
        """Experiment with different activation functions"""
        console.print("\n[bold cyan]═══ Experiment 3: Activation Functions ═══[/bold cyan]\n")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Testing activation functions...", 
                total=len(self.config.activations)
            )
            
            for i, activation in enumerate(self.config.activations):
                config = MLPConfig(
                    hidden_layers=base_layers,
                    learning_rate=0.01,
                    max_epochs=self.config.max_epochs,
                    activation=activation,
                    batch_size=self.config.batch_size
                )
                
                result = self.run_single_experiment(
                    config, X_train, y_train, X_test, y_test,
                    f"Activation: {activation}"
                )
                
                results.append(result)
                progress.update(task, advance=1,
                              description=f"[cyan]Activation {i+1}/{len(self.config.activations)}: "
                                        f"Acc={result['test_accuracy']:.3f}")
        
        self._display_results_table(results, "Activation Function Results")
        return results
    
    def run_comprehensive_experiment(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   n_configs: int = 20) -> List[Dict]:
        """Run comprehensive random search over hyperparameters"""
        console.print("\n[bold cyan]═══ Comprehensive Random Search ═══[/bold cyan]\n")
        
        results = []
        np.random.seed(self.config.random_seed)
        
        # Generate random configurations
        configs = []
        for _ in range(n_configs):
            n_layers = np.random.choice(self.config.num_hidden_layers)
            neurons = np.random.choice(self.config.neurons_per_layer, size=n_layers)
            lr = np.random.choice(self.config.learning_rates)
            activation = np.random.choice(self.config.activations)
            
            config = MLPConfig(
                hidden_layers=neurons.tolist(),
                learning_rate=lr,
                max_epochs=self.config.max_epochs,
                activation=activation,
                batch_size=self.config.batch_size
            )
            configs.append(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Running configurations...", total=n_configs)
            
            for i, config in enumerate(configs):
                result = self.run_single_experiment(
                    config, X_train, y_train, X_test, y_test,
                    f"Config {i+1}"
                )
                
                results.append(result)
                progress.update(task, advance=1,
                              description=f"[cyan]Config {i+1}/{n_configs}: "
                                        f"Acc={result['test_accuracy']:.3f}")
        
        # Sort by test accuracy
        results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
        self._display_results_table(results_sorted[:10], "Top 10 Configurations")
        
        return results_sorted
    
    def _display_results_table(self, results: List[Dict], title: str):
        """Display results in a formatted table"""
        table = Table(title=title, box=box.ROUNDED, show_header=True,
                     header_style="bold magenta")
        
        table.add_column("Config", style="cyan", no_wrap=True)
        table.add_column("Layers", style="yellow")
        table.add_column("LR", justify="right", style="green")
        table.add_column("Activation", style="blue")
        table.add_column("Train Acc", justify="right", style="green")
        table.add_column("Test Acc", justify="right", style="green")
        table.add_column("Loss", justify="right", style="red")
        table.add_column("Time (s)", justify="right", style="magenta")
        table.add_column("Epochs", justify="right", style="cyan")
        
        for result in results:
            config = result['config']
            layers_str = str(config.hidden_layers)
            if len(layers_str) > 20:
                layers_str = layers_str[:17] + "..."
            
            table.add_row(
                result['experiment_name'],
                layers_str,
                f"{config.learning_rate:.3f}",
                config.activation,
                f"{result['train_accuracy']:.4f}",
                f"{result['test_accuracy']:.4f}",
                f"{result['final_loss']:.4f}",
                f"{result['training_time']:.2f}",
                str(result['n_epochs'])
            )
        
        console.print(table)
        console.print()
    
    def save_results(self, results: List[Dict], filename: str = "experiment_results.json"):
        """Save experiment results to JSON"""
        save_path = Path("output/data") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'experiment_name': result['experiment_name'],
                'config': {
                    'hidden_layers': result['config'].hidden_layers,
                    'learning_rate': result['config'].learning_rate,
                    'activation': result['config'].activation,
                    'max_epochs': result['config'].max_epochs,
                    'batch_size': result['config'].batch_size
                },
                'train_accuracy': float(result['train_accuracy']),
                'test_accuracy': float(result['test_accuracy']),
                'final_loss': float(result['final_loss']),
                'training_time': float(result['training_time']),
                'n_epochs': int(result['n_epochs']),
                'architecture_info': result['architecture_info']
            }
            serializable_results.append(serializable_result)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        console.print(f"[green]✓ Results saved to: {save_path}[/green]")
    
    def find_best_configuration(self, results: List[Dict], 
                              metric: str = 'test_accuracy') -> Dict:
        """Find the best configuration based on a metric"""
        if metric in ['test_accuracy', 'train_accuracy']:
            best = max(results, key=lambda x: x[metric])
        else:  # For loss or time, lower is better
            best = min(results, key=lambda x: x[metric])
        
        console.print(Panel.fit(
            f"[bold green]Best Configuration (by {metric})[/bold green]\n\n"
            f"Layers: {best['config'].hidden_layers}\n"
            f"Learning Rate: {best['config'].learning_rate}\n"
            f"Activation: {best['config'].activation}\n"
            f"Test Accuracy: {best['test_accuracy']:.4f}\n"
            f"Training Time: {best['training_time']:.2f}s\n"
            f"Epochs: {best['n_epochs']}",
            border_style="green"
        ))
        
        return best
