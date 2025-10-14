"""
Interactive UI module using Rich for beautiful terminal interface
"""
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from typing import List, Tuple, Optional, Dict
import time

from .config import (
    MLPConfig, ExperimentConfig, NoiseConfig, DatasetConfig,
    VisualizationConfig, THEME_COLORS
)

console = Console()


class InteractiveUI:
    """Interactive terminal UI for MLP experimentation"""
    
    def __init__(self):
        self.console = console
        
    def show_banner(self):
        """Display application banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                   ║
║    ███╗   ███╗██╗     ██████╗         ███╗   ███╗███╗   ██╗██╗███████╗████████╗   ║
║    ████╗ ████║██║     ██╔══██╗        ████╗ ████║████╗  ██║██║██╔════╝╚══██╔══╝   ║
║    ██╔████╔██║██║     ██████╔╝        ██╔████╔██║██╔██╗ ██║██║███████╗   ██║      ║
║    ██║╚██╔╝██║██║     ██╔═══╝         ██║╚██╔╝██║██║╚██╗██║██║╚════██║   ██║      ║
║    ██║ ╚═╝ ██║███████╗██║             ██║ ╚═╝ ██║██║ ╚████║██║███████║   ██║      ║
║    ╚═╝     ╚═╝╚══════╝╚═╝             ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝   ╚═╝      ║
║                                                                                   ║
║              Multi-Layer Perceptron Experimentation Framework                     ║
║                      MNIST Digit Recognition System                               ║
║                   Made by ander_tdaemon UG University DICIS                       ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
        """
        
        self.console.print(banner, style="bold cyan")
        self.console.print(
            Panel.fit(
                "[bold yellow]Advanced Neural Network Training & Analysis System[/bold yellow]\n"
                "[dim]Featuring comprehensive hyperparameter exploration, "
                "noise analysis, and visual reporting[/dim]",
                border_style="cyan"
            )
        )
        self.console.print()
    
    def show_main_menu(self) -> str:
        """Display main menu and get user choice"""
        menu_items = [
            ("1", "📊 Quick Experiment", "Run a quick experiment with default settings"),
            ("2", "🔬 Layer Configuration Analysis", "Explore different layer architectures"),
            ("3", "📈 Learning Rate Exploration", "Test various learning rates"),
            ("4", "⚡ Activation Function Comparison", "Compare activation functions"),
            ("5", "🎯 Comprehensive Grid Search", "Exhaustive hyperparameter search"),
            ("6", "🔊 Noise Robustness Testing", "Test model with noisy data"),
            ("7", "⚙️  Configure Experiment Settings", "Customize experiment parameters"),
            ("8", "📁 Load & Analyze Previous Results", "View saved experiment results"),
            ("9", "❌ Exit", "Exit the application")
        ]
        
        table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
        table.add_column("Choice", style="bold cyan", width=8)
        table.add_column("Option", style="bold yellow", width=35)
        table.add_column("Description", style="dim white")
        
        for choice, option, description in menu_items:
            table.add_row(choice, option, description)
        
        self.console.print(Panel(table, title="[bold cyan]Main Menu[/bold cyan]", 
                                border_style="cyan"))
        
        choice = Prompt.ask(
            "\n[bold cyan]Select an option[/bold cyan]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            default="1"
        )
        
        return choice
    
    def configure_dataset(self) -> DatasetConfig:
        """Interactive dataset configuration"""
        self.console.print("\n[bold cyan]═══ Dataset Configuration ═══[/bold cyan]\n")
        
        n_samples = IntPrompt.ask(
            "[cyan]Number of samples to use[/cyan]",
            default=5000
        )
        
        test_size = FloatPrompt.ask(
            "[cyan]Test set proportion (0-1)[/cyan]",
            default=0.2
        )
        
        normalize = Confirm.ask(
            "[cyan]Normalize pixel values?[/cyan]",
            default=True
        )
        
        config = DatasetConfig(
            n_samples=n_samples,
            test_size=test_size,
            normalize=normalize
        )
        
        self.console.print("\n[green]✓ Dataset configuration complete![/green]\n")
        return config
    
    def configure_mlp(self) -> MLPConfig:
        """Interactive MLP configuration"""
        self.console.print("\n[bold cyan]═══ MLP Configuration ═══[/bold cyan]\n")
        
        # Hidden layers
        n_layers = IntPrompt.ask(
            "[cyan]Number of hidden layers[/cyan]",
            default=2
        )
        
        hidden_layers = []
        for i in range(n_layers):
            neurons = IntPrompt.ask(
                f"[cyan]Neurons in hidden layer {i+1}[/cyan]",
                default=128 if i == 0 else 64
            )
            hidden_layers.append(neurons)
        
        # Learning rate
        learning_rate = FloatPrompt.ask(
            "[cyan]Learning rate[/cyan]",
            default=0.01
        )
        
        # Activation function
        self.console.print("\n[yellow]Activation Functions:[/yellow]")
        self.console.print("  1. Sigmoid")
        self.console.print("  2. Tanh")
        self.console.print("  3. ReLU")
        
        activation_choice = Prompt.ask(
            "[cyan]Select activation function[/cyan]",
            choices=["1", "2", "3"],
            default="1"
        )
        
        activation_map = {"1": "sigmoid", "2": "tanh", "3": "relu"}
        activation = activation_map[activation_choice]
        
        # Max epochs
        max_epochs = IntPrompt.ask(
            "[cyan]Maximum epochs[/cyan]",
            default=100
        )
        
        # Batch size
        batch_size = IntPrompt.ask(
            "[cyan]Batch size[/cyan]",
            default=64
        )
        
        config = MLPConfig(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            activation=activation,
            max_epochs=max_epochs,
            batch_size=batch_size
        )
        
        # Display summary
        summary = Table(box=box.SIMPLE, show_header=False)
        summary.add_column("Parameter", style="cyan")
        summary.add_column("Value", style="yellow")
        
        summary.add_row("Hidden Layers", str(hidden_layers))
        summary.add_row("Learning Rate", str(learning_rate))
        summary.add_row("Activation", activation)
        summary.add_row("Max Epochs", str(max_epochs))
        summary.add_row("Batch Size", str(batch_size))
        
        self.console.print("\n[bold green]Configuration Summary:[/bold green]")
        self.console.print(summary)
        self.console.print()
        
        return config
    
    def configure_noise(self) -> NoiseConfig:
        """Interactive noise configuration"""
        self.console.print("\n[bold cyan]═══ Noise Configuration ═══[/bold cyan]\n")
        
        self.console.print("[yellow]Noise Types:[/yellow]")
        self.console.print("  1. Gaussian Noise")
        self.console.print("  2. Salt & Pepper Noise")
        self.console.print("  3. Speckle Noise")
        self.console.print("  4. Uniform Noise")
        
        noise_choice = Prompt.ask(
            "[cyan]Select noise type[/cyan]",
            choices=["1", "2", "3", "4"],
            default="1"
        )
        
        noise_map = {
            "1": "gaussian",
            "2": "salt_pepper",
            "3": "speckle",
            "4": "uniform"
        }
        noise_type = noise_map[noise_choice]
        
        if noise_type == "salt_pepper":
            noise_param = FloatPrompt.ask(
                "[cyan]Noise probability (0-1)[/cyan]",
                default=0.05
            )
            config = NoiseConfig(
                noise_type=noise_type,
                noise_probability=noise_param
            )
        else:
            noise_param = FloatPrompt.ask(
                "[cyan]Noise level (0-1)[/cyan]",
                default=0.1
            )
            config = NoiseConfig(
                noise_type=noise_type,
                noise_level=noise_param
            )
        
        self.console.print(f"\n[green]✓ Noise type: {noise_type}, Level: {noise_param}[/green]\n")
        return config
    
    def configure_experiment(self) -> ExperimentConfig:
        """Interactive experiment configuration"""
        self.console.print("\n[bold cyan]═══ Experiment Configuration ═══[/bold cyan]\n")
        
        max_epochs = IntPrompt.ask(
            "[cyan]Maximum epochs per experiment[/cyan]",
            default=150
        )
        
        batch_size = IntPrompt.ask(
            "[cyan]Batch size[/cyan]",
            default=64
        )
        
        # Quick or comprehensive
        mode = Prompt.ask(
            "[cyan]Experiment mode[/cyan]",
            choices=["quick", "comprehensive"],
            default="quick"
        )
        
        if mode == "quick":
            config = ExperimentConfig(
                max_epochs=max_epochs,
                batch_size=batch_size,
                n_samples=1000,
                hidden_layer_configs=[
                    [64], [128], [128, 64]
                ],
                learning_rates=[0.01, 0.1, 0.5],
                activations=["sigmoid", "tanh"]
            )
        else:
            config = ExperimentConfig(
                max_epochs=max_epochs,
                batch_size=batch_size,
                n_samples=5000
            )
        
        self.console.print(f"\n[green]✓ Experiment mode: {mode}[/green]\n")
        return config
    
    def show_progress_panel(self, title: str, details: Dict):
        """Show a progress panel with details"""
        content = "\n".join([f"[cyan]{k}:[/cyan] [yellow]{v}[/yellow]" 
                           for k, v in details.items()])
        
        panel = Panel(
            content,
            title=f"[bold green]{title}[/bold green]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def show_results_summary(self, results: List[Dict]):
        """Display results summary"""
        if not results:
            self.console.print("[red]No results to display[/red]")
            return
        
        # Sort by test accuracy
        results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
        
        table = Table(
            title="[bold cyan]Experiment Results Summary[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Configuration", style="yellow")
        table.add_column("Test Acc", style="green", justify="right")
        table.add_column("Train Acc", style="green", justify="right")
        table.add_column("Loss", style="red", justify="right")
        table.add_column("Time (s)", style="blue", justify="right")
        table.add_column("Epochs", style="magenta", justify="right")
        
        for i, result in enumerate(results_sorted[:10]):
            config = result['config']
            layers_str = str(config.hidden_layers)
            if len(layers_str) > 15:
                layers_str = layers_str[:12] + "..."
            
            config_str = f"{layers_str} | {config.activation[:3]} | lr={config.learning_rate}"
            
            table.add_row(
                str(i + 1),
                config_str,
                f"{result['test_accuracy']:.4f}",
                f"{result['train_accuracy']:.4f}",
                f"{result['final_loss']:.4f}",
                f"{result['training_time']:.2f}",
                str(result['n_epochs'])
            )
        
        self.console.print(table)
    
    def show_best_config(self, best_result: Dict):
        """Display best configuration"""
        config = best_result['config']
        
        content = f"""
[bold yellow]Architecture:[/bold yellow]
  Hidden Layers: {config.hidden_layers}
  Activation: {config.activation}
  
[bold yellow]Hyperparameters:[/bold yellow]
  Learning Rate: {config.learning_rate}
  Batch Size: {config.batch_size}
  Epochs Trained: {best_result['n_epochs']}
  
[bold yellow]Performance:[/bold yellow]
  Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)
  Train Accuracy: {best_result['train_accuracy']:.4f}
  Final Loss: {best_result['final_loss']:.4f}
  Training Time: {best_result['training_time']:.2f}s
  
[bold yellow]Status:[/bold yellow]
  Overfitting: {'⚠️  Yes' if best_result['train_accuracy'] - best_result['test_accuracy'] > 0.1 else '✓  No'}
  Generalization Gap: {abs(best_result['train_accuracy'] - best_result['test_accuracy']):.4f}
        """
        
        panel = Panel(
            content.strip(),
            title="[bold green]🏆 BEST CONFIGURATION[/bold green]",
            border_style="green",
            box=box.DOUBLE
        )
        
        self.console.print(panel)
    
    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Ask for confirmation"""
        return Confirm.ask(f"[cyan]{message}[/cyan]", default=default)
    
    def pause(self):
        """Pause and wait for user input"""
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")
    
    def show_visualization_menu(self) -> List[str]:
        """Show visualization options"""
        self.console.print("\n[bold cyan]═══ Visualization Options ═══[/bold cyan]\n")
        
        options = [
            ("dataset", "📊 Dataset Samples"),
            ("training", "📈 Training History"),
            ("confusion", "🔲 Confusion Matrix"),
            ("predictions", "🎯 Prediction Samples"),
            ("probabilities", "🌡️  Probability Heatmap"),
            ("weights", "⚖️  Weight Distributions"),
            ("decision", "🗺️  Decision Boundary"),
            ("loss_landscape", "🏔️  Loss Landscape"),
            ("animation", "🎬 Training Animation"),
            ("dashboard", "📊 Interactive Dashboard"),
            ("all", "🌟 Generate All Visualizations")
        ]
        
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Key", style="cyan", width=15)
        table.add_column("Visualization", style="yellow")
        
        for key, name in options:
            table.add_row(key, name)
        
        self.console.print(table)
        
        selected = Prompt.ask(
            "\n[cyan]Select visualizations (comma-separated)[/cyan]",
            default="all"
        )
        
        if selected == "all":
            return [opt[0] for opt in options if opt[0] != "all"]
        else:
            return [s.strip() for s in selected.split(",")]
    
    def show_error(self, message: str):
        """Display error message"""
        self.console.print(f"\n[bold red]❌ Error: {message}[/bold red]\n")
    
    def show_success(self, message: str):
        """Display success message"""
        self.console.print(f"\n[bold green]✓ {message}[/bold green]\n")
    
    def show_info(self, message: str):
        """Display info message"""
        self.console.print(f"\n[bold blue]ℹ️  {message}[/bold blue]\n")
    
    def show_warning(self, message: str):
        """Display warning message"""
        self.console.print(f"\n[bold yellow]⚠️  {message}[/bold yellow]\n")
