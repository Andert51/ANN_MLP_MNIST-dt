"""
Script to demonstrate noise comparison
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.config import DatasetConfig, NoiseConfig, VisualizationConfig
from src.data_loader import MNISTLoader, NoiseGenerator
from src.visualizations import MLPVisualizer
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print("\n[bold cyan]═══ Noise Comparison Demo ═══[/bold cyan]\n")
    
    # Load data
    config = DatasetConfig(n_samples=1000)
    loader = MNISTLoader(config)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Use test set samples
    n_samples = 10
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    
    # Visualizer
    viz = MLPVisualizer(VisualizationConfig())
    
    # Test different noise types
    noise_types = [
        ("gaussian", NoiseConfig(noise_type="gaussian", noise_level=0.2)),
        ("salt_pepper", NoiseConfig(noise_type="salt_pepper", noise_probability=0.1)),
        ("speckle", NoiseConfig(noise_type="speckle", noise_level=0.2)),
        ("uniform", NoiseConfig(noise_type="uniform", noise_level=0.2))
    ]
    
    console.print("[cyan]Generating noise comparisons...[/cyan]\n")
    
    for noise_name, noise_config in noise_types:
        # Apply noise
        X_noisy = NoiseGenerator.apply_noise(X_samples, noise_config)
        
        # Visualize comparison
        viz.plot_noise_comparison(
            X_samples, X_noisy, y_samples,
            n_samples=n_samples,
            noise_type=noise_name,
            save_name=f"noise_comparison_{noise_name}.png"
        )
        
        console.print(f"[green]✓ Generated {noise_name} comparison[/green]")
    
    # Summary visualization with all noise types
    console.print("\n[cyan]Creating comprehensive noise showcase...[/cyan]")
    
    # Show clean samples
    viz.plot_dataset_samples(
        X_samples, y_samples,
        n_samples=n_samples,
        title="Clean Samples",
        save_name="noise_clean_samples.png"
    )
    
    # Show each noise type
    for noise_name, noise_config in noise_types:
        X_noisy = NoiseGenerator.apply_noise(X_samples, noise_config)
        viz.plot_dataset_samples(
            X_noisy, y_samples,
            n_samples=n_samples,
            title=f"{noise_name.replace('_', ' ').title()} Noise",
            save_name=f"noise_{noise_name}_samples.png"
        )
    
    console.print("\n[bold green]✓ Noise comparison complete![/bold green]")
    console.print(f"[green]✓ Images saved to: output/images/[/green]\n")
    
    # Print summary
    summary = Panel.fit(
        "[bold yellow]Noise Types Demonstrated:[/bold yellow]\n\n"
        "1. [cyan]Gaussian Noise[/cyan]\n"
        "   - Random noise from normal distribution\n"
        "   - Simulates sensor noise\n\n"
        "2. [cyan]Salt & Pepper Noise[/cyan]\n"
        "   - Random black and white pixels\n"
        "   - Simulates transmission errors\n\n"
        "3. [cyan]Speckle Noise[/cyan]\n"
        "   - Multiplicative noise\n"
        "   - Common in medical imaging\n\n"
        "4. [cyan]Uniform Noise[/cyan]\n"
        "   - Uniformly distributed noise\n"
        "   - General purpose testing",
        title="[bold green]Summary[/bold green]",
        border_style="green"
    )
    
    console.print(summary)


if __name__ == "__main__":
    main()
