"""Metrics calculation and visualization utilities for training."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from emotion_net.config.constants import EMOTIONS

def calculate_metrics(predictions, targets, phase="train", save_dir=None):
    """Calculate and print detailed metrics."""
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Calculate per-class metrics
    report = classification_report(targets, predictions, 
                                 target_names=list(EMOTIONS.values()),
                                 output_dict=True)
    
    # Calculate F1 scores
    macro_f1 = f1_score(targets, predictions, average='macro')
    weighted_f1 = f1_score(targets, predictions, average='weighted')
    
    # Print metrics
    print(f"\n{phase.capitalize()} Metrics:")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("\nPer-class metrics:")
    for emotion, metrics in report.items():
        if emotion in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"{emotion}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(EMOTIONS.values()),
                yticklabels=list(EMOTIONS.values()))
    plt.title(f'{phase.capitalize()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{phase}_confusion_matrix.png'))
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(os.path.join(save_dir, f'{phase}_metrics.csv'))
    else:
        plt.close()
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(f"{phase}_metrics.csv")
    
    return cm, report, macro_f1, weighted_f1

def save_training_history(history, save_dir):
    """Save training history to CSV file."""
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, "training_history.csv"))
    
def plot_training_history(history, save_dir):
    """Plot and save training history curves."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close() 