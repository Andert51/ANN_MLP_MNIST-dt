"""
Mathematical and statistical reporting module
"""
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import json

from .mlp_model import MLPClassifier
from .config import MLPConfig


class MathematicalReporter:
    """Generate detailed mathematical and statistical reports"""
    
    @staticmethod
    def generate_model_report(model: MLPClassifier, X_train: np.ndarray, 
                            y_train: np.ndarray, X_test: np.ndarray, 
                            y_test: np.ndarray) -> str:
        """Generate comprehensive model report"""
        report = []
        report.append("=" * 80)
        report.append("MATHEMATICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Architecture
        report.append("1. NETWORK ARCHITECTURE")
        report.append("-" * 80)
        arch_info = model.get_architecture_info()
        report.append(f"   Input Layer:        {arch_info['input_size']} neurons")
        
        for i, neurons in enumerate(arch_info['hidden_layers']):
            report.append(f"   Hidden Layer {i+1}:    {neurons} neurons")
        
        report.append(f"   Output Layer:       {arch_info['output_size']} neurons")
        report.append(f"   Activation Function: {arch_info['activation']}")
        report.append(f"   Total Parameters:   {arch_info['total_parameters']:,}")
        report.append("")
        
        # Weight Statistics
        report.append("2. WEIGHT STATISTICS")
        report.append("-" * 80)
        
        for i, (weights, biases) in enumerate(zip(model.weights, model.biases)):
            report.append(f"   Layer {i+1}:")
            report.append(f"      Weights Shape:      {weights.shape}")
            report.append(f"      Weights Mean:       {np.mean(weights):.6f}")
            report.append(f"      Weights Std:        {np.std(weights):.6f}")
            report.append(f"      Weights Min:        {np.min(weights):.6f}")
            report.append(f"      Weights Max:        {np.max(weights):.6f}")
            report.append(f"      Biases Shape:       {biases.shape}")
            report.append(f"      Biases Mean:        {np.mean(biases):.6f}")
            report.append(f"      Biases Std:         {np.std(biases):.6f}")
            report.append("")
        
        # Training History
        report.append("3. TRAINING DYNAMICS")
        report.append("-" * 80)
        history = model.history
        
        if history.train_losses:
            report.append(f"   Total Epochs:           {len(history.epochs)}")
            report.append(f"   Initial Training Loss:  {history.train_losses[0]:.6f}")
            report.append(f"   Final Training Loss:    {history.train_losses[-1]:.6f}")
            report.append(f"   Loss Reduction:         {(history.train_losses[0] - history.train_losses[-1]):.6f}")
            report.append(f"   Loss Reduction Rate:    {((history.train_losses[0] - history.train_losses[-1]) / history.train_losses[0] * 100):.2f}%")
            report.append("")
            
            report.append(f"   Initial Training Acc:   {history.train_accuracies[0]:.6f}")
            report.append(f"   Final Training Acc:     {history.train_accuracies[-1]:.6f}")
            report.append(f"   Accuracy Improvement:   {(history.train_accuracies[-1] - history.train_accuracies[0]):.6f}")
            report.append("")
            
            if history.val_losses:
                report.append(f"   Final Validation Loss:  {history.val_losses[-1]:.6f}")
                report.append(f"   Final Validation Acc:   {history.val_accuracies[-1]:.6f}")
                report.append("")
            
            total_time = sum(history.training_times)
            avg_time = np.mean(history.training_times)
            report.append(f"   Total Training Time:    {total_time:.2f} seconds")
            report.append(f"   Average Epoch Time:     {avg_time:.4f} seconds")
            report.append(f"   Training Throughput:    {len(X_train) / avg_time:.2f} samples/sec")
        report.append("")
        
        # Performance Metrics
        report.append("4. PERFORMANCE METRICS")
        report.append("-" * 80)
        
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_acc = np.mean(train_predictions == y_train)
        test_acc = np.mean(test_predictions == y_test)
        
        report.append(f"   Training Accuracy:      {train_acc:.6f} ({train_acc*100:.2f}%)")
        report.append(f"   Test Accuracy:          {test_acc:.6f} ({test_acc*100:.2f}%)")
        report.append(f"   Generalization Gap:     {(train_acc - test_acc):.6f}")
        
        # Check for overfitting/underfitting
        if train_acc - test_acc > 0.1:
            report.append(f"   Status:                 ⚠️  OVERFITTING DETECTED")
        elif train_acc < 0.7 and test_acc < 0.7:
            report.append(f"   Status:                 ⚠️  UNDERFITTING DETECTED")
        else:
            report.append(f"   Status:                 ✓  GOOD GENERALIZATION")
        report.append("")
        
        # Classification Report
        report.append("5. DETAILED CLASSIFICATION METRICS (TEST SET)")
        report.append("-" * 80)
        class_report = classification_report(y_test, test_predictions, 
                                            target_names=[f"Class {i}" for i in range(10)],
                                            zero_division=0)
        report.append(class_report)
        report.append("")
        
        # Confusion Matrix Analysis
        report.append("6. CONFUSION MATRIX ANALYSIS")
        report.append("-" * 80)
        cm = confusion_matrix(y_test, test_predictions)
        
        # Most confused pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        report.append("   Most Common Misclassifications:")
        for i, (true_class, pred_class, count) in enumerate(confused_pairs[:5]):
            report.append(f"      {i+1}. True: {true_class} → Predicted: {pred_class} ({count} times)")
        report.append("")
        
        # Per-class accuracy
        report.append("   Per-Class Accuracy:")
        for i in range(len(cm)):
            class_acc = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
            report.append(f"      Class {i}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        report.append("")
        
        # Statistical Tests
        report.append("7. STATISTICAL ANALYSIS")
        report.append("-" * 80)
        
        # Prediction confidence
        probas = model.predict_proba(X_test)
        max_probas = np.max(probas, axis=1)
        
        report.append(f"   Average Confidence:     {np.mean(max_probas):.6f}")
        report.append(f"   Confidence Std:         {np.std(max_probas):.6f}")
        report.append(f"   Min Confidence:         {np.min(max_probas):.6f}")
        report.append(f"   Max Confidence:         {np.max(max_probas):.6f}")
        report.append("")
        
        # Correct vs incorrect confidence
        correct_mask = test_predictions == y_test
        correct_confidence = max_probas[correct_mask]
        incorrect_confidence = max_probas[~correct_mask]
        
        if len(incorrect_confidence) > 0:
            report.append(f"   Avg Confidence (Correct):   {np.mean(correct_confidence):.6f}")
            report.append(f"   Avg Confidence (Incorrect): {np.mean(incorrect_confidence):.6f}")
            report.append(f"   Confidence Difference:      {(np.mean(correct_confidence) - np.mean(incorrect_confidence)):.6f}")
        report.append("")
        
        # Convergence Analysis
        if len(history.train_losses) > 1:
            report.append("8. CONVERGENCE ANALYSIS")
            report.append("-" * 80)
            
            # Loss gradient
            loss_gradient = np.diff(history.train_losses)
            report.append(f"   Final Loss Gradient:        {loss_gradient[-1]:.6f}")
            report.append(f"   Avg Loss Gradient (last 10): {np.mean(loss_gradient[-10:]):.6f}")
            
            # Check convergence
            if abs(loss_gradient[-1]) < 0.001:
                report.append(f"   Convergence Status:         ✓  CONVERGED")
            else:
                report.append(f"   Convergence Status:         ⚠️  NOT FULLY CONVERGED")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_comparison_report(results: List[Dict]) -> str:
        """Generate comparison report for multiple configurations"""
        report = []
        report.append("=" * 80)
        report.append("CONFIGURATION COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Configurations: {len(results)}")
        report.append("")
        
        # Sort by test accuracy
        results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        
        accuracies = [r['test_accuracy'] for r in results]
        losses = [r['final_loss'] for r in results]
        times = [r['training_time'] for r in results]
        
        report.append(f"Test Accuracy:")
        report.append(f"   Best:    {np.max(accuracies):.6f}")
        report.append(f"   Worst:   {np.min(accuracies):.6f}")
        report.append(f"   Mean:    {np.mean(accuracies):.6f}")
        report.append(f"   Std:     {np.std(accuracies):.6f}")
        report.append(f"   Median:  {np.median(accuracies):.6f}")
        report.append("")
        
        report.append(f"Final Loss:")
        report.append(f"   Best:    {np.min(losses):.6f}")
        report.append(f"   Worst:   {np.max(losses):.6f}")
        report.append(f"   Mean:    {np.mean(losses):.6f}")
        report.append(f"   Std:     {np.std(losses):.6f}")
        report.append("")
        
        report.append(f"Training Time:")
        report.append(f"   Fastest: {np.min(times):.2f}s")
        report.append(f"   Slowest: {np.max(times):.2f}s")
        report.append(f"   Mean:    {np.mean(times):.2f}s")
        report.append(f"   Std:     {np.std(times):.2f}s")
        report.append("")
        
        # Top configurations
        report.append("TOP 5 CONFIGURATIONS")
        report.append("-" * 80)
        
        for i, result in enumerate(results_sorted[:5]):
            config = result['config']
            report.append(f"\n{i+1}. {result['experiment_name']}")
            report.append(f"   Layers:          {config.hidden_layers}")
            report.append(f"   Learning Rate:   {config.learning_rate}")
            report.append(f"   Activation:      {config.activation}")
            report.append(f"   Test Accuracy:   {result['test_accuracy']:.6f}")
            report.append(f"   Final Loss:      {result['final_loss']:.6f}")
            report.append(f"   Training Time:   {result['training_time']:.2f}s")
            report.append(f"   Epochs:          {result['n_epochs']}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def save_report(report: str, filename: str = "mathematical_report.txt"):
        """Save report to file"""
        save_path = Path("output/data") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {save_path}")
        return save_path
