"""
Quick test script to verify bug fixes for:
1. mnist_overview visualization
2. topology animation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MLPConfig, DatasetConfig, VisualizationConfig
from src.data_loader import MNISTLoader
from src.mlp_model import MLPClassifier
from src.visualizations import MLPVisualizer
from rich.console import Console
import numpy as np

console = Console()

def main():
    console.print("\n[bold cyan]🔧 Testing Bug Fixes v2.0.1[/bold cyan]\n")
    
    # Load small dataset for quick testing
    console.print("[yellow]Loading dataset...[/yellow]")
    dataset_config = DatasetConfig(n_samples=500, test_size=0.2)
    loader = MNISTLoader(dataset_config)
    X_train, X_test, y_train, y_test = loader.load_data()
    console.print("[green]✓ Dataset loaded[/green]\n")
    
    # Initialize visualizer
    viz_config = VisualizationConfig()
    visualizer = MLPVisualizer(viz_config)
    
    # Test 1: MNIST Overview
    console.print("[bold cyan]Test 1: MNIST Dataset Overview[/bold cyan]")
    try:
        visualizer.plot_mnist_dataset_overview(
            X_train, y_train,
            save_name="test_mnist_overview.png"
        )
        console.print("[green]✓ MNIST overview generated successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Error in MNIST overview: {str(e)}[/red]\n")
        return False
    
    # Test 2: Train a small model
    console.print("[bold cyan]Test 2: Training Small MLP[/bold cyan]")
    config = MLPConfig(
        hidden_layers=[64, 32],
        learning_rate=0.01,
        activation="tanh",
        max_epochs=10,
        batch_size=32
    )
    
    model = MLPClassifier(config)
    console.print("[dim]Training...[/dim]")
    model.fit(X_train, y_train, X_test, y_test, verbose=False)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    console.print(f"[green]✓ Model trained! Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}[/green]\n")
    
    # Test 3: Network Topology Animation
    console.print("[bold cyan]Test 3: Network Topology Animation[/bold cyan]")
    try:
        # Select a random test sample
        idx = np.random.choice(len(X_test))
        sample_X = X_test[idx]
        sample_y = y_test[idx]
        
        prediction = model.predict(sample_X.reshape(1, -1))[0]
        console.print(f"[dim]Sample: True={sample_y}, Pred={prediction}[/dim]")
        
        visualizer.animate_network_topology(
            model, sample_X, sample_y,
            save_name="test_topology_animation.gif"
        )
        console.print("[green]✓ Topology animation generated successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Error in topology animation: {str(e)}[/red]\n")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    console.print("""
[bold green]╔═════════════════════════════════════════════════════╗
║           ✅ ALL TESTS PASSED!                      ║
╚═════════════════════════════════════════════════════╝[/bold green]

[cyan]Generated files:[/cyan]
  • test_mnist_overview.png
  • test_topology_animation.gif

[cyan]Location:[/cyan] output/images/

[green]Both bugs have been fixed successfully! 🎉[/green]
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
