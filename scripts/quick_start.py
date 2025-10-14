"""
Quick start example script
Run this for a simple demonstration
"""
import numpy as np
from src.config import MLPConfig, DatasetConfig, VisualizationConfig
from src.data_loader import MNISTLoader, NoiseGenerator
from src.mlp_model import MLPClassifier
from src.visualizations import MLPVisualizer
from src.reports import MathematicalReporter
from rich.console import Console
from rich.progress import track

console = Console()

def main():
    console.print("\n[bold cyan]═══ MLP-MNIST Quick Start Example ═══[/bold cyan]\n")
    
    # Configuration
    dataset_config = DatasetConfig(n_samples=2000, test_size=0.2)
    mlp_config = MLPConfig(
        hidden_layers=[128, 64],
        learning_rate=0.01,
        activation="sigmoid",
        max_epochs=50,
        batch_size=32
    )
    viz_config = VisualizationConfig()
    
    # Load data
    console.print("[cyan]Loading MNIST dataset...[/cyan]")
    loader = MNISTLoader(dataset_config)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Visualize dataset
    console.print("[cyan]Visualizing dataset samples...[/cyan]")
    visualizer = MLPVisualizer(viz_config)
    visualizer.plot_dataset_samples(
        X_test, y_test, n_samples=25,
        title="MNIST Test Samples",
        save_name="01_dataset_samples.png"
    )
    
    # Train model
    console.print("\n[cyan]Training MLP...[/cyan]")
    model = MLPClassifier(mlp_config)
    model.fit(X_train, y_train, X_test, y_test, verbose=False)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    console.print(f"\n[green]Training Accuracy: {train_acc:.4f}[/green]")
    console.print(f"[green]Test Accuracy: {test_acc:.4f}[/green]\n")
    
    # Generate visualizations
    console.print("[cyan]Generating visualizations...[/cyan]\n")
    
    visualizations = [
        ("Training History", lambda: visualizer.plot_training_history(
            model.history, save_name="02_training_history.png"
        )),
        ("Confusion Matrix", lambda: visualizer.plot_confusion_matrix(
            y_test, model.predict(X_test), save_name="03_confusion_matrix.png"
        )),
        ("Prediction Samples", lambda: visualizer.plot_prediction_samples(
            model, X_test, y_test, n_samples=20, save_name="04_predictions.png"
        )),
        ("Probability Heatmap", lambda: visualizer.plot_probability_heatmap(
            model, X_test, y_test, n_samples=20, save_name="05_probability_heatmap.png"
        )),
        ("Weight Distributions", lambda: visualizer.plot_weight_distributions(
            model, save_name="06_weight_distributions.png"
        )),
        ("Decision Boundary", lambda: visualizer.plot_decision_boundary_2d(
            model, X_test[:500], y_test[:500], save_name="07_decision_boundary.png"
        )),
        ("Loss Landscape", lambda: visualizer.plot_loss_landscape_2d(
            model, X_train[:200], y_train[:200], save_name="08_loss_landscape.png"
        ))
    ]
    
    for name, viz_func in track(visualizations, description="Creating visualizations"):
        try:
            viz_func()
            console.print(f"  [green]✓ {name}[/green]")
        except Exception as e:
            console.print(f"  [red]✗ {name}: {str(e)}[/red]")
    
    # Generate mathematical report
    console.print("\n[cyan]Generating mathematical report...[/cyan]")
    reporter = MathematicalReporter()
    report = reporter.generate_model_report(
        model, X_train, y_train, X_test, y_test
    )
    reporter.save_report(report, "quick_start_report.txt")
    
    console.print("\n[bold green]✓ Quick start complete![/bold green]")
    console.print(f"[green]✓ Visualizations saved to: {viz_config.images_dir}[/green]")
    console.print(f"[green]✓ Report saved to: output/data/quick_start_report.txt[/green]\n")


if __name__ == "__main__":
    main()
