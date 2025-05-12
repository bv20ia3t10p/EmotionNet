"""Main training script for the emotion recognition model."""

import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from emotion_net.training.metrics import (
    calculate_metrics, save_training_history, plot_training_history
)
from emotion_net.training.model_manager import (
    setup_training, calculate_class_weights, create_criterion, save_model
)
from emotion_net.training.training_loops import train_epoch, validate
from emotion_net.config.constants import CHECKPOINT_DIR, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE

def calculate_metrics(y_true, y_pred, labels, save_dir=None, prefix=''):
    """Calculate metrics using confusion matrix for more reliable F1 scores."""
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
            'f1-score': f1,
            'support': cm[i, :].sum()
        }
    
    # Calculate macro and weighted averages
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1-score'] for m in metrics.values()])
    
    # Weighted averages
    weights = np.array([m['support'] for m in metrics.values()])
    weights = weights / weights.sum()
    weighted_precision = np.sum([m['precision'] * w for m, w in zip(metrics.values(), weights)])
    weighted_recall = np.sum([m['recall'] * w for m, w in zip(metrics.values(), weights)])
    weighted_f1 = np.sum([m['f1-score'] * w for m, w in zip(metrics.values(), weights)])
    
    # Save confusion matrix plot if save_dir is provided
    if save_dir:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{prefix} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'))
        plt.close()
        
        # Save metrics to file
        metrics_file = os.path.join(save_dir, f'{prefix}_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"{prefix} Metrics:\n")
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
            f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
            f.write("\nPer-class metrics:\n")
            for label, m in metrics.items():
                f.write(f"{label}:\n")
                f.write(f"  Precision: {m['precision']:.4f}\n")
                f.write(f"  Recall: {m['recall']:.4f}\n")
                f.write(f"  F1-score: {m['f1-score']:.4f}\n")
                f.write(f"  Support: {m['support']}\n")
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': metrics,
        'confusion_matrix': cm
    }

class EmotionTrainer:
    """Trainer class for emotion recognition model."""
    
    def __init__(self, model, train_dataset, val_dataset, device, config):
        """Initialize trainer with model and datasets."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.config = config
        
        # Create data loaders with appropriate batch sizes
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', DEFAULT_BATCH_SIZE),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', DEFAULT_BATCH_SIZE) * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup training components
        self.model, self.criterion, self.optimizer, self.scheduler = setup_training(
            model=model,
            train_labels=[label for _, label in train_dataset],
            num_classes=len(train_dataset.classes),
            device=device,
            learning_rate=config.get('learning_rate', 1e-4)
        )
        
        # Training state
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience = config.get('patience', 15)
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_f1': [],
            'val_loss': [], 'val_f1': []
        }
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    def evaluate(self, dataset):
        """Evaluate model on a dataset."""
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', DEFAULT_BATCH_SIZE) * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Run validation
        metrics = validate(
            self.model, loader, self.criterion,
            self.device, dataset.classes
        )
        
        return metrics
    
    def train(self):
        """Main training loop with improved early stopping."""
        num_epochs = self.config.get('num_epochs', DEFAULT_NUM_EPOCHS)
        
        print("\nStarting training...")
        print(f"Training on {len(self.train_dataset)} samples")
        print(f"Validating on {len(self.val_dataset)} samples")
        print(f"Using device: {self.device}")
        print(f"Number of classes: {len(self.train_dataset.classes)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_metrics = train_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scheduler
            )
            
            # Validation phase
            val_metrics = validate(
                self.model, self.val_loader, self.criterion,
                self.device, self.train_dataset.classes
            )
            
            # Update learning rate
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['f1'])
            
            # Store metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Print epoch metrics
            print(f"\nEpoch {epoch + 1} Metrics:")
            print(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save checkpoint
                save_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
                save_model(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, save_path
                )
                
                print(f"\nNew best model saved! (Validation F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                print(f"\nNo improvement for {self.patience_counter} epochs. "
                      f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch + 1}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}. "
                      f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch + 1}")
                break
        
        # Save training history
        save_training_history(self.history, CHECKPOINT_DIR)
        plot_training_history(self.history, CHECKPOINT_DIR)
        
        print("\nTraining completed successfully!")
        return self.best_val_f1 