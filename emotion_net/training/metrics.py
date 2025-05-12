"""Metrics calculation and visualization for emotion recognition."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred, labels):
    """Calculate metrics using confusion matrix."""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    # Calculate metrics from confusion matrix
    metrics = {}
    total_samples = cm.sum()
    
    # Calculate per-class metrics
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - (tp + fp + fn)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': cm[i, :].sum()
        }
    
    # Calculate macro and weighted averages
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    # Weighted averages
    weights = np.array([m['support'] for m in metrics.values()])
    weights = weights / weights.sum()
    weighted_precision = np.sum([m['precision'] * w for m, w in zip(metrics.values(), weights)])
    weighted_recall = np.sum([m['recall'] * w for m, w in zip(metrics.values(), weights)])
    weighted_f1 = np.sum([m['f1'] * w for m, w in zip(metrics.values(), weights)])
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("True ↓       " + " ".join(f"{label:8}" for label in labels))
    print("-" * (12 + 9 * len(labels)))
    for i, label in enumerate(labels):
        row = f"{label:8} "
        row += " ".join(f"{cm[i,j]:8}" for j in range(len(labels)))
        print(row)
    
    return {
        'f1': macro_f1,  # Use macro F1 as the main metric
        'precision': macro_precision,
        'recall': macro_recall,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'per_class': metrics,
        'confusion_matrix': cm
    }

def save_training_history(history, save_dir):
    """Save training history to file."""
    metrics_file = os.path.join(save_dir, 'training_history.txt')
    with open(metrics_file, 'w') as f:
        f.write("Training History:\n")
        f.write("-" * 50 + "\n")
        for epoch, (train_loss, train_f1, val_loss, val_f1) in enumerate(zip(
            history['train_loss'], history['train_f1'],
            history['val_loss'], history['val_f1']
        )):
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"  Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}\n")
            f.write(f"  Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}\n")
            f.write("-" * 50 + "\n")

def plot_training_history(history, save_dir):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close() 