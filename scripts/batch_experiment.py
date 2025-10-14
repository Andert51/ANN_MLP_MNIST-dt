"""
Batch experiment runner - run all experiments at once
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.config import MLPConfig, ExperimentConfig, DatasetConfig, VisualizationConfig
from src.data_loader import MNISTLoader
from src.experiments import ExperimentRunner
from src.visualizations import MLPVisualizer
from src.reports import MathematicalReporter
from rich.console import Console
from rich.panel import Panel
import time

console = Console()


def main():
    console.print("\n[bold cyan]═══ BATCH EXPERIMENT RUNNER ═══[/bold cyan]\n")
    
    start_time = time.time()
    
    # Configuration
    dataset_config = DatasetConfig(n_samples=5000, test_size=0.2)
    experiment_config = ExperimentConfig(
        max_epochs=100,
        batch_size=64
    )
    viz_config = VisualizationConfig()
    
    # Load data
    console.print("[bold yellow]Step 1: Loading Dataset[/bold yellow]")
    loader = MNISTLoader(dataset_config)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Create runner
    runner = ExperimentRunner(experiment_config)
    visualizer = MLPVisualizer(viz_config)
    reporter = MathematicalReporter()
    
    # All experiments
    all_results = []
    
    # Experiment 1: Layers
    console.print("\n[bold yellow]Step 2: Layer Configuration Experiment[/bold yellow]")
    layer_results = runner.run_layer_experiment(X_train, y_train, X_test, y_test)
    all_results.extend(layer_results)
    runner.save_results(layer_results, "batch_layer_results.json")
    
    # Experiment 2: Learning rates
    console.print("\n[bold yellow]Step 3: Learning Rate Experiment[/bold yellow]")
    lr_results = runner.run_learning_rate_experiment(
        X_train, y_train, X_test, y_test, base_layers=[128, 64]
    )
    all_results.extend(lr_results)
    runner.save_results(lr_results, "batch_lr_results.json")
    
    # Experiment 3: Activations
    console.print("\n[bold yellow]Step 4: Activation Function Experiment[/bold yellow]")
    act_results = runner.run_activation_experiment(
        X_train, y_train, X_test, y_test, base_layers=[128, 64]
    )
    all_results.extend(act_results)
    runner.save_results(act_results, "batch_activation_results.json")
    
    # Find best
    console.print("\n[bold yellow]Step 5: Analyzing Results[/bold yellow]")
    best = runner.find_best_configuration(all_results, 'test_accuracy')
    
    # Generate comprehensive visualizations
    console.print("\n[bold yellow]Step 6: Generating Visualizations[/bold yellow]")
    
    best_model = best['model']
    
    visualizations = [
        ("Dataset Samples", lambda: visualizer.plot_dataset_samples(
            X_test, y_test, n_samples=25, save_name="batch_dataset.png"
        )),
        ("Training History", lambda: visualizer.plot_training_history(
            best_model.history, save_name="batch_training_history.png"
        )),
        ("Confusion Matrix", lambda: visualizer.plot_confusion_matrix(
            y_test, best_model.predict(X_test), save_name="batch_confusion.png"
        )),
        ("Predictions", lambda: visualizer.plot_prediction_samples(
            best_model, X_test, y_test, n_samples=25, save_name="batch_predictions.png"
        )),
        ("Probabilities", lambda: visualizer.plot_probability_heatmap(
            best_model, X_test, y_test, n_samples=20, save_name="batch_probabilities.png"
        )),
        ("Weights", lambda: visualizer.plot_weight_distributions(
            best_model, save_name="batch_weights.png"
        )),
        ("Decision Boundary", lambda: visualizer.plot_decision_boundary_2d(
            best_model, X_test[:500], y_test[:500], save_name="batch_boundary.png"
        )),
        ("Loss Landscape", lambda: visualizer.plot_loss_landscape_2d(
            best_model, X_train[:200], y_train[:200], save_name="batch_landscape.png"
        )),
        ("Animation", lambda: visualizer.create_training_animation(
            best_model.history, save_name="batch_animation.gif"
        )),
        ("Dashboard", lambda: visualizer.create_interactive_dashboard(
            all_results, save_name="batch_dashboard.html"
        ))
    ]
    
    for name, viz_func in visualizations:
        try:
            viz_func()
            console.print(f"  [green]✓ {name}[/green]")
        except Exception as e:
            console.print(f"  [red]✗ {name}: {str(e)}[/red]")
    
    # Generate reports
    console.print("\n[bold yellow]Step 7: Generating Reports[/bold yellow]")
    
    # Best model report
    best_report = reporter.generate_model_report(
        best_model, X_train, y_train, X_test, y_test
    )
    reporter.save_report(best_report, "batch_best_model_report.txt")
    console.print("  [green]✓ Best model report[/green]")
    
    # Comparison report
    comparison_report = reporter.generate_comparison_report(all_results)
    reporter.save_report(comparison_report, "batch_comparison_report.txt")
    console.print("  [green]✓ Comparison report[/green]")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final summary
    console.print("\n" + "="*80)
    summary = Panel.fit(
        f"[bold green]BATCH EXPERIMENT COMPLETE![/bold green]\n\n"
        f"[cyan]Total Configurations Tested:[/cyan] {len(all_results)}\n"
        f"[cyan]Best Test Accuracy:[/cyan] {best['test_accuracy']:.4f} ({best['test_accuracy']*100:.2f}%)\n"
        f"[cyan]Best Configuration:[/cyan]\n"
        f"  • Layers: {best['config'].hidden_layers}\n"
        f"  • Learning Rate: {best['config'].learning_rate}\n"
        f"  • Activation: {best['config'].activation}\n"
        f"  • Training Time: {best['training_time']:.2f}s\n"
        f"  • Epochs: {best['n_epochs']}\n\n"
        f"[cyan]Total Execution Time:[/cyan] {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n"
        f"[yellow]Output Locations:[/yellow]\n"
        f"  • Visualizations: [green]output/images/[/green]\n"
        f"  • Reports: [green]output/data/[/green]\n"
        f"  • Results JSON: [green]output/data/batch_*_results.json[/green]",
        title="[bold cyan]Summary[/bold cyan]",
        border_style="green"
    )
    
    console.print(summary)
    console.print("="*80 + "\n")


if __name__ == "__main__":
    main()
