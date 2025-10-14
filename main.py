"""
Main application entry point
Multi-Layer Perceptron MNIST Experimentation Framework
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui import InteractiveUI
from src.config import (
    MLPConfig, ExperimentConfig, DatasetConfig, 
    NoiseConfig, VisualizationConfig
)
from src.data_loader import MNISTLoader, NoiseGenerator
from src.mlp_model import MLPClassifier
from src.experiments import ExperimentRunner
from src.visualizations import MLPVisualizer
from src.reports import MathematicalReporter
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

console = Console()


class MLPExperimentApp:
    """Main application class"""
    
    def __init__(self):
        self.ui = InteractiveUI()
        self.dataset_config = DatasetConfig()
        self.experiment_config = ExperimentConfig()
        self.viz_config = VisualizationConfig()
        
        self.mnist_loader = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.current_results = []
        
    def initialize_data(self):
        """Load and prepare data"""
        if self.mnist_loader is None:
            self.ui.show_progress_panel("Loading Dataset", {
                "Dataset": "MNIST",
                "Samples": self.dataset_config.n_samples,
                "Status": "Loading..."
            })
            
            self.mnist_loader = MNISTLoader(self.dataset_config)
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.mnist_loader.load_data()
            
            self.ui.show_success("Dataset loaded successfully!")
    
    def quick_experiment(self):
        """Run a quick experiment with default settings"""
        console.print("\n[bold cyan]═══ Quick Experiment ═══[/bold cyan]\n")
        
        self.initialize_data()
        
        # Use a subset for quick testing
        n_samples = 1000
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = \
            self.mnist_loader.get_subset(n_samples)
        
        # Default configuration
        config = MLPConfig(
            hidden_layers=[128, 64],
            learning_rate=0.01,
            activation="sigmoid",
            max_epochs=50,
            batch_size=32
        )
        
        self.ui.show_progress_panel("Training MLP", {
            "Architecture": str(config.hidden_layers),
            "Learning Rate": config.learning_rate,
            "Activation": config.activation,
            "Max Epochs": config.max_epochs
        })
        
        # Train model
        runner = ExperimentRunner(self.experiment_config)
        result = runner.run_single_experiment(
            config, X_train_sub, y_train_sub, X_test_sub, y_test_sub,
            "Quick Experiment"
        )
        
        self.current_results = [result]
        
        # Display results
        self.ui.show_best_config(result)
        
        # Generate visualizations
        if self.ui.confirm_action("Generate visualizations?"):
            self.generate_visualizations([result])
        
        # Mathematical report
        if self.ui.confirm_action("Generate mathematical report?"):
            self.generate_mathematical_report(result['model'])
    
    def layer_experiment(self):
        """Explore different layer configurations"""
        console.print("\n[bold cyan]═══ Layer Configuration Experiment ═══[/bold cyan]\n")
        
        self.initialize_data()
        
        # Use subset
        n_samples = 2000
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = \
            self.mnist_loader.get_subset(n_samples)
        
        runner = ExperimentRunner(self.experiment_config)
        results = runner.run_layer_experiment(
            X_train_sub, y_train_sub, X_test_sub, y_test_sub
        )
        
        self.current_results = results
        
        # Find best
        best = runner.find_best_configuration(results, 'test_accuracy')
        self.ui.show_best_config(best)
        
        # Save results
        runner.save_results(results, "layer_experiment_results.json")
        
        if self.ui.confirm_action("Generate visualizations for best model?"):
            self.generate_visualizations([best])
    
    def learning_rate_experiment(self):
        """Explore different learning rates"""
        console.print("\n[bold cyan]═══ Learning Rate Experiment ═══[/bold cyan]\n")
        
        self.initialize_data()
        
        n_samples = 2000
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = \
            self.mnist_loader.get_subset(n_samples)
        
        runner = ExperimentRunner(self.experiment_config)
        results = runner.run_learning_rate_experiment(
            X_train_sub, y_train_sub, X_test_sub, y_test_sub
        )
        
        self.current_results = results
        
        best = runner.find_best_configuration(results, 'test_accuracy')
        self.ui.show_best_config(best)
        
        runner.save_results(results, "learning_rate_experiment_results.json")
        
        if self.ui.confirm_action("Generate visualizations for best model?"):
            self.generate_visualizations([best])
    
    def activation_experiment(self):
        """Compare activation functions"""
        console.print("\n[bold cyan]═══ Activation Function Experiment ═══[/bold cyan]\n")
        
        self.initialize_data()
        
        n_samples = 2000
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = \
            self.mnist_loader.get_subset(n_samples)
        
        runner = ExperimentRunner(self.experiment_config)
        results = runner.run_activation_experiment(
            X_train_sub, y_train_sub, X_test_sub, y_test_sub
        )
        
        self.current_results = results
        
        best = runner.find_best_configuration(results, 'test_accuracy')
        self.ui.show_best_config(best)
        
        runner.save_results(results, "activation_experiment_results.json")
        
        if self.ui.confirm_action("Generate visualizations for best model?"):
            self.generate_visualizations([best])
    
    def comprehensive_experiment(self):
        """Run comprehensive grid search"""
        console.print("\n[bold cyan]═══ Comprehensive Experiment ═══[/bold cyan]\n")
        
        self.initialize_data()
        
        n_configs = IntPrompt.ask(
            "[cyan]Number of configurations to test[/cyan]",
            default=20
        )
        
        n_samples = 3000
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = \
            self.mnist_loader.get_subset(n_samples)
        
        runner = ExperimentRunner(self.experiment_config)
        results = runner.run_comprehensive_experiment(
            X_train_sub, y_train_sub, X_test_sub, y_test_sub,
            n_configs=n_configs
        )
        
        self.current_results = results
        
        best = runner.find_best_configuration(results, 'test_accuracy')
        self.ui.show_best_config(best)
        
        runner.save_results(results, "comprehensive_experiment_results.json")
        
        # Generate comparison report
        reporter = MathematicalReporter()
        comparison_report = reporter.generate_comparison_report(results)
        reporter.save_report(comparison_report, "comparison_report.txt")
        
        if self.ui.confirm_action("Generate visualizations?"):
            visualizer = MLPVisualizer(self.viz_config)
            visualizer.create_interactive_dashboard(results)
            self.generate_visualizations([best])
    
    def noise_experiment(self):
        """Test model robustness with noise"""
        console.print("\n[bold cyan]═══ Noise Robustness Experiment ═══[/bold cyan]\n")
        
        self.initialize_data()
        
        # Configure noise
        noise_config = self.ui.configure_noise()
        
        n_samples = 1000
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = \
            self.mnist_loader.get_subset(n_samples)
        
        # Train on clean data
        config = MLPConfig(
            hidden_layers=[128, 64],
            learning_rate=0.01,
            activation="sigmoid",
            max_epochs=50
        )
        
        self.ui.show_info("Training on clean data...")
        runner = ExperimentRunner(self.experiment_config)
        clean_result = runner.run_single_experiment(
            config, X_train_sub, y_train_sub, X_test_sub, y_test_sub,
            "Clean Data"
        )
        
        # Add noise to test data
        X_test_noisy = NoiseGenerator.apply_noise(X_test_sub, noise_config)
        
        # Evaluate on noisy data
        self.ui.show_info("Evaluating on noisy data...")
        model = clean_result['model']
        noisy_acc = model.score(X_test_noisy, y_test_sub)
        
        # Display results
        console.print(Panel.fit(
            f"[bold yellow]Noise Robustness Results[/bold yellow]\n\n"
            f"Clean Test Accuracy:  {clean_result['test_accuracy']:.4f}\n"
            f"Noisy Test Accuracy:  {noisy_acc:.4f}\n"
            f"Accuracy Drop:        {(clean_result['test_accuracy'] - noisy_acc):.4f}\n"
            f"Robustness Score:     {(noisy_acc / clean_result['test_accuracy']):.4f}",
            border_style="yellow"
        ))
        
        # Visualize noise comparison
        if self.ui.confirm_action("Generate noise comparison visualization?"):
            visualizer = MLPVisualizer(self.viz_config)
            
            # Show dataset samples
            visualizer.plot_dataset_samples(
                X_test_sub, y_test_sub, n_samples=25,
                title="Clean Test Samples",
                save_name="clean_samples.png"
            )
            
            # Show noisy samples
            visualizer.plot_dataset_samples(
                X_test_noisy, y_test_sub, n_samples=25,
                title=f"Noisy Test Samples ({noise_config.noise_type})",
                save_name="noisy_samples.png"
            )
            
            # Show comparison
            visualizer.plot_noise_comparison(
                X_test_sub, X_test_noisy, y_test_sub,
                n_samples=10, noise_type=noise_config.noise_type,
                save_name="noise_comparison.png"
            )
            
            self.ui.show_success("Visualizations saved!")
    
    def configure_settings(self):
        """Configure experiment settings"""
        console.print("\n[bold cyan]═══ Configuration Menu ═══[/bold cyan]\n")
        
        console.print("[yellow]1.[/yellow] Configure Dataset")
        console.print("[yellow]2.[/yellow] Configure MLP")
        console.print("[yellow]3.[/yellow] Configure Experiments")
        console.print("[yellow]4.[/yellow] Back to Main Menu")
        
        choice = Prompt.ask(
            "\n[cyan]Select option[/cyan]",
            choices=["1", "2", "3", "4"],
            default="4"
        )
        
        if choice == "1":
            self.dataset_config = self.ui.configure_dataset()
            self.mnist_loader = None  # Reset loader
        elif choice == "2":
            mlp_config = self.ui.configure_mlp()
            self.ui.show_info("Configuration saved for next experiment")
        elif choice == "3":
            self.experiment_config = self.ui.configure_experiment()
    
    def generate_visualizations(self, results: list):
        """Generate selected visualizations"""
        if not results:
            self.ui.show_error("No results available")
            return
        
        best_result = results[0]
        model = best_result['model']
        
        visualizer = MLPVisualizer(self.viz_config)
        selected = self.ui.show_visualization_menu()
        
        console.print("\n[cyan]Generating visualizations...[/cyan]\n")
        
        for viz_type in selected:
            try:
                if viz_type == "dataset":
                    visualizer.plot_dataset_samples(
                        self.X_test, self.y_test, n_samples=25,
                        save_name="dataset_samples.png"
                    )
                
                elif viz_type == "training":
                    visualizer.plot_training_history(
                        model.history,
                        save_name="training_history.png"
                    )
                
                elif viz_type == "confusion":
                    y_pred = model.predict(self.X_test)
                    visualizer.plot_confusion_matrix(
                        self.y_test, y_pred,
                        save_name="confusion_matrix.png"
                    )
                
                elif viz_type == "predictions":
                    visualizer.plot_prediction_samples(
                        model, self.X_test, self.y_test,
                        n_samples=20,
                        save_name="prediction_samples.png"
                    )
                
                elif viz_type == "probabilities":
                    visualizer.plot_probability_heatmap(
                        model, self.X_test, self.y_test,
                        n_samples=20,
                        save_name="probability_heatmap.png"
                    )
                
                elif viz_type == "weights":
                    visualizer.plot_weight_distributions(
                        model,
                        save_name="weight_distributions.png"
                    )
                
                elif viz_type == "decision":
                    # Use subset for speed
                    indices = np.random.choice(len(self.X_test), 500, replace=False)
                    visualizer.plot_decision_boundary_2d(
                        model, self.X_test[indices], self.y_test[indices],
                        save_name="decision_boundary.png"
                    )
                
                elif viz_type == "loss_landscape":
                    # Use subset
                    indices = np.random.choice(len(self.X_train), 200, replace=False)
                    visualizer.plot_loss_landscape_2d(
                        model, self.X_train[indices], self.y_train[indices],
                        save_name="loss_landscape.png"
                    )
                
                elif viz_type == "animation":
                    visualizer.create_training_animation(
                        model.history,
                        save_name="training_animation.gif"
                    )
                
                elif viz_type == "dashboard":
                    visualizer.create_interactive_dashboard(
                        results,
                        save_name="experiment_dashboard.html"
                    )
                
                console.print(f"[green]✓ Generated: {viz_type}[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error generating {viz_type}: {str(e)}[/red]")
        
        self.ui.show_success(f"Visualizations saved to: {self.viz_config.images_dir}")
    
    def generate_mathematical_report(self, model: MLPClassifier):
        """Generate and display mathematical report"""
        console.print("\n[cyan]Generating mathematical report...[/cyan]\n")
        
        reporter = MathematicalReporter()
        report = reporter.generate_model_report(
            model, self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Save report
        save_path = reporter.save_report(report, "mathematical_report.txt")
        
        # Display excerpt
        lines = report.split('\n')
        console.print("\n".join(lines[:50]))
        console.print("\n[dim]... (see full report in output/data)[/dim]\n")
        
        self.ui.show_success(f"Full report saved to: {save_path}")
    
    def run(self):
        """Main application loop"""
        self.ui.show_banner()
        
        while True:
            try:
                choice = self.ui.show_main_menu()
                
                if choice == "1":
                    self.quick_experiment()
                elif choice == "2":
                    self.layer_experiment()
                elif choice == "3":
                    self.learning_rate_experiment()
                elif choice == "4":
                    self.activation_experiment()
                elif choice == "5":
                    self.comprehensive_experiment()
                elif choice == "6":
                    self.noise_experiment()
                elif choice == "7":
                    self.configure_settings()
                elif choice == "8":
                    self.ui.show_info("Load results feature coming soon!")
                elif choice == "9":
                    console.print("\n[bold cyan]Thank you for using MLP-MNIST Framework![/bold cyan]\n")
                    break
                
                self.ui.pause()
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted by user[/yellow]")
                if self.ui.confirm_action("Exit application?"):
                    break
            except Exception as e:
                self.ui.show_error(f"An error occurred: {str(e)}")
                if not self.ui.confirm_action("Continue?"):
                    break


def main():
    """Entry point"""
    app = MLPExperimentApp()
    app.run()


if __name__ == "__main__":
    main()
