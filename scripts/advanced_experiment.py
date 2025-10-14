"""
Advanced experiment script with comprehensive analysis
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MLPConfig, ExperimentConfig, DatasetConfig, VisualizationConfig
from src.data_loader import MNISTLoader
from src.experiments import ExperimentRunner
from src.visualizations import MLPVisualizer
from src.reports import MathematicalReporter
from rich.console import Console

console = Console()


def main():
    console.print("\n[bold cyan]═══ Advanced Hyperparameter Exploration ═══[/bold cyan]\n")
    
    # Configuration
    dataset_config = DatasetConfig(n_samples=3000, test_size=0.2)
    experiment_config = ExperimentConfig(
        hidden_layer_configs=[
            [32], [64], [128],
            [64, 32], [128, 64], [256, 128],
            [128, 64, 32], [256, 128, 64]
        ],
        learning_rates=[0.001, 0.01, 0.05, 0.1, 0.5],
        activations=["sigmoid", "tanh"],
        max_epochs=100,
        batch_size=64
    )
    viz_config = VisualizationConfig()
    
    # Load data
    console.print("[cyan]Loading dataset...[/cyan]")
    loader = MNISTLoader(dataset_config)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Create experiment runner
    runner = ExperimentRunner(experiment_config)
    
    # Experiment 1: Layer configurations
    console.print("\n[bold yellow]Experiment 1: Layer Configurations[/bold yellow]")
    layer_results = runner.run_layer_experiment(X_train, y_train, X_test, y_test)
    runner.save_results(layer_results, "layer_experiment_results.json")
    
    # Experiment 2: Learning rates
    console.print("\n[bold yellow]Experiment 2: Learning Rates[/bold yellow]")
    lr_results = runner.run_learning_rate_experiment(
        X_train, y_train, X_test, y_test, base_layers=[128, 64]
    )
    runner.save_results(lr_results, "lr_experiment_results.json")
    
    # Experiment 3: Activation functions
    console.print("\n[bold yellow]Experiment 3: Activation Functions[/bold yellow]")
    act_results = runner.run_activation_experiment(
        X_train, y_train, X_test, y_test, base_layers=[128, 64]
    )
    runner.save_results(act_results, "activation_experiment_results.json")
    
    # Combine all results
    all_results = layer_results + lr_results + act_results
    
    # Find best configuration
    console.print("\n[bold green]Finding best configuration...[/bold green]")
    best = runner.find_best_configuration(all_results, 'test_accuracy')
    
    # Generate comprehensive visualizations for best model
    console.print("\n[cyan]Generating visualizations for best model...[/cyan]")
    visualizer = MLPVisualizer(viz_config)
    
    best_model = best['model']
    
    # All visualizations
    visualizations = [
        ("Dataset Samples", "dataset_samples.png", 
         lambda: visualizer.plot_dataset_samples(X_test, y_test, n_samples=25, save_name="dataset_samples.png")),
        
        ("Training History", "training_history_best.png",
         lambda: visualizer.plot_training_history(best_model.history, save_name="training_history_best.png")),
        
        ("Confusion Matrix", "confusion_matrix_best.png",
         lambda: visualizer.plot_confusion_matrix(y_test, best_model.predict(X_test), save_name="confusion_matrix_best.png")),
        
        ("Normalized Confusion Matrix", "confusion_matrix_normalized.png",
         lambda: visualizer.plot_confusion_matrix(y_test, best_model.predict(X_test), normalize=True, save_name="confusion_matrix_normalized.png")),
        
        ("Prediction Samples", "prediction_samples_best.png",
         lambda: visualizer.plot_prediction_samples(best_model, X_test, y_test, n_samples=20, save_name="prediction_samples_best.png")),
        
        ("Probability Heatmap", "probability_heatmap_best.png",
         lambda: visualizer.plot_probability_heatmap(best_model, X_test, y_test, n_samples=20, save_name="probability_heatmap_best.png")),
        
        ("Weight Distributions", "weight_distributions_best.png",
         lambda: visualizer.plot_weight_distributions(best_model, save_name="weight_distributions_best.png")),
        
        ("Decision Boundary", "decision_boundary_best.png",
         lambda: visualizer.plot_decision_boundary_2d(best_model, X_test[:500], y_test[:500], save_name="decision_boundary_best.png")),
        
        ("Loss Landscape", "loss_landscape_best.png",
         lambda: visualizer.plot_loss_landscape_2d(best_model, X_train[:200], y_train[:200], save_name="loss_landscape_best.png")),
        
        ("Training Animation", "training_animation_best.gif",
         lambda: visualizer.create_training_animation(best_model.history, save_name="training_animation_best.gif")),
        
        ("Interactive Dashboard", "experiment_dashboard.html",
         lambda: visualizer.create_interactive_dashboard(all_results, save_name="experiment_dashboard.html"))
    ]
    
    for name, filename, viz_func in visualizations:
        try:
            viz_func()
            console.print(f"  [green]✓ {name}[/green]")
        except Exception as e:
            console.print(f"  [red]✗ {name}: {str(e)}[/red]")
    
    # Generate mathematical reports
    console.print("\n[cyan]Generating mathematical reports...[/cyan]")
    reporter = MathematicalReporter()
    
    # Individual report for best model
    best_report = reporter.generate_model_report(
        best_model, X_train, y_train, X_test, y_test
    )
    reporter.save_report(best_report, "best_model_report.txt")
    console.print("  [green]✓ Best model report[/green]")
    
    # Comparison report
    comparison_report = reporter.generate_comparison_report(all_results)
    reporter.save_report(comparison_report, "experiment_comparison_report.txt")
    console.print("  [green]✓ Comparison report[/green]")
    
    # Summary
    console.print("\n[bold green]═══ Experiment Complete ═══[/bold green]\n")
    console.print(f"[green]Total configurations tested: {len(all_results)}[/green]")
    console.print(f"[green]Best test accuracy: {best['test_accuracy']:.4f}[/green]")
    console.print(f"[green]Best configuration: {best['config'].hidden_layers}[/green]")
    console.print(f"[green]Visualizations: {viz_config.images_dir}[/green]")
    console.print(f"[green]Reports: output/data/[/green]\n")


if __name__ == "__main__":
    main()
