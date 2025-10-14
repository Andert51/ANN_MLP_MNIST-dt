"""
Demo script to showcase new visualization features:
1. Network Topology Animation
2. MNIST Dataset Overview
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MLPConfig, DatasetConfig, VisualizationConfig
from src.data_loader import MNISTLoader
from src.mlp_model import MLPClassifier
from src.visualizations import MLPVisualizer
from rich.console import Console
from rich.panel import Panel
import numpy as np

console = Console()


def main():
    """Run topology and dataset visualization demo"""
    
    # Banner
    console.print("""
    [bold cyan]╔═══════════════════════════════════════════════════════════╗
    ║   🧠 Network Topology & Dataset Visualization Demo 🖼️    ║
    ╚═══════════════════════════════════════════════════════════╝[/bold cyan]
    """)
    
    console.print("[yellow]This demo showcases two new visualization features:[/yellow]")
    console.print("  1. [green]Network Topology Animation[/green] - See neurons activate in real-time")
    console.print("  2. [green]MNIST Dataset Overview[/green] - Comprehensive dataset visualization\n")
    
    # Configuration
    dataset_config = DatasetConfig(n_samples=2000, test_size=0.2)
    viz_config = VisualizationConfig()
    
    # Step 1: Load data
    console.print("[bold cyan]Step 1:[/bold cyan] Loading MNIST dataset...\n")
    loader = MNISTLoader(dataset_config)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    console.print(Panel.fit(
        f"[green]✓ Dataset loaded successfully![/green]\n\n"
        f"Training samples: {len(X_train)}\n"
        f"Test samples: {len(X_test)}\n"
        f"Features per sample: {X_train.shape[1]}\n"
        f"Classes: {len(np.unique(y_train))}",
        border_style="green",
        title="[bold]Dataset Info[/bold]"
    ))
    
    # Step 2: Generate MNIST Dataset Overview
    console.print("\n[bold cyan]Step 2:[/bold cyan] Generating MNIST Dataset Overview...\n")
    console.print("[dim]This creates a comprehensive visualization showing:[/dim]")
    console.print("  • Class distribution (bar chart & pie chart)")
    console.print("  • Dataset statistics")
    console.print("  • Sample images for each digit (0-9)")
    console.print("  • Visual overview of the entire dataset\n")
    
    visualizer = MLPVisualizer(viz_config)
    visualizer.plot_mnist_dataset_overview(
        X_train, y_train,
        save_name="mnist_dataset_overview.png"
    )
    
    console.print("[green]✓ MNIST Dataset Overview generated![/green]")
    console.print(f"[dim]Saved to: {viz_config.images_dir / 'mnist_dataset_overview.png'}[/dim]\n")
    
    # Step 3: Train a model
    console.print("[bold cyan]Step 3:[/bold cyan] Training MLP model...\n")
    
    config = MLPConfig(
        hidden_layers=[128, 64, 32],
        learning_rate=0.01,
        activation="tanh",
        max_epochs=30,
        batch_size=32
    )
    
    console.print(Panel.fit(
        f"[yellow]Model Configuration[/yellow]\n\n"
        f"Architecture: [784] → {config.hidden_layers} → [10]\n"
        f"Activation: {config.activation}\n"
        f"Learning Rate: {config.learning_rate}\n"
        f"Max Epochs: {config.max_epochs}\n"
        f"Batch Size: {config.batch_size}",
        border_style="yellow"
    ))
    
    model = MLPClassifier(config)
    
    console.print("\n[dim]Training in progress...[/dim]")
    model.fit(X_train, y_train, X_test, y_test, verbose=True)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    console.print(Panel.fit(
        f"[green]✓ Training completed![/green]\n\n"
        f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n"
        f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n"
        f"Total Epochs: {len(model.history.train_losses)}\n"
        f"Final Loss: {model.history.train_losses[-1]:.4f}",
        border_style="green",
        title="[bold]Training Results[/bold]"
    ))
    
    # Step 4: Generate Network Topology Animation
    console.print("\n[bold cyan]Step 4:[/bold cyan] Generating Network Topology Animation...\n")
    console.print("[dim]This creates an animated GIF showing:[/dim]")
    console.print("  • Network structure (input, hidden, output layers)")
    console.print("  • Neuron activation values in real-time")
    console.print("  • Information flow through the network")
    console.print("  • Prediction process visualization")
    console.print("  • Color-coded activation levels\n")
    
    # Select 5 random samples for different animations
    console.print("[yellow]Creating animations for 5 different predictions...[/yellow]\n")
    
    for i in range(5):
        idx = np.random.choice(len(X_test))
        sample_X = X_test[idx]
        sample_y = y_test[idx]
        
        prediction = model.predict(sample_X.reshape(1, -1))[0]
        confidence = model.predict_proba(sample_X.reshape(1, -1))[0][prediction]
        
        console.print(f"  [cyan]Animation {i+1}:[/cyan] True={sample_y}, Pred={prediction}, "
                     f"Confidence={confidence:.2%}")
        
        visualizer.animate_network_topology(
            model, sample_X, sample_y,
            save_name=f"network_topology_animation_{i+1}.gif"
        )
    
    console.print("\n[green]✓ All topology animations generated![/green]")
    console.print(f"[dim]Saved to: {viz_config.images_dir}/[/dim]\n")
    
    # Step 5: Summary
    console.print("""
    [bold green]╔═══════════════════════════════════════════════════════════╗
    ║                    🎉 Demo Completed! 🎉                  ║
    ╚═══════════════════════════════════════════════════════════╝[/bold green]
    """)
    
    console.print("[bold cyan]Generated Files:[/bold cyan]")
    console.print("  📊 [yellow]mnist_dataset_overview.png[/yellow]")
    console.print("     → Comprehensive MNIST dataset visualization")
    console.print()
    console.print("  🧠 [yellow]network_topology_animation_1.gif[/yellow]")
    console.print("  🧠 [yellow]network_topology_animation_2.gif[/yellow]")
    console.print("  🧠 [yellow]network_topology_animation_3.gif[/yellow]")
    console.print("  🧠 [yellow]network_topology_animation_4.gif[/yellow]")
    console.print("  🧠 [yellow]network_topology_animation_5.gif[/yellow]")
    console.print("     → Animated network topology showing neuron activations\n")
    
    console.print(f"[bold]All files saved to:[/bold] [cyan]{viz_config.images_dir}[/cyan]\n")
    
    console.print("[dim]Open the GIF files to see the network in action![/dim]")
    console.print("[dim]The neurons light up as data flows through the network.[/dim]\n")
    
    # Additional info
    console.print(Panel.fit(
        "[bold yellow]Understanding the Topology Animation:[/bold yellow]\n\n"
        "🔴 Red neurons = High activation (strongly activated)\n"
        "🟡 Yellow neurons = Medium activation\n"
        "🟢 Green neurons = Low activation\n"
        "⚪ Gray neurons = Not yet processed\n\n"
        "The animation shows the forward propagation process,\n"
        "where input data flows through hidden layers to produce\n"
        "the final classification output.",
        border_style="yellow",
        title="[bold]Legend[/bold]"
    ))
    
    console.print("\n[bold green]✓ Demo completed successfully![/bold green]\n")


if __name__ == "__main__":
    main()
