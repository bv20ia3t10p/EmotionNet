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
from torch.cuda.amp import GradScaler
from torch.utils.data.sampler import WeightedRandomSampler

from emotion_net.training.metrics import (
    calculate_metrics, save_training_history, plot_training_history
)
from emotion_net.training.model_manager import (
    setup_training, calculate_class_weights, create_criterion, save_model
)
from emotion_net.training.training_loops import train_epoch, validate
from emotion_net.config.constants import CHECKPOINT_DIR, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE
from emotion_net.models.ema import EMA

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
    
    def __init__(self, model, train_dataset, val_dataset, config, device, test_dataset=None, data_manager=None):
        """Initialize trainer with model and datasets."""
        self.model = model.to(device)
        self.train_labels = config.get('train_labels', [])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device
        self.data_manager = data_manager  # Store reference to data manager
        
        # Store data paths from the data manager if available
        self.temp_dir = getattr(data_manager, 'temp_dir', None)
        if self.temp_dir:
            print(f"EmotionTrainer using temp directory: {self.temp_dir}")
            if hasattr(data_manager, 'data_dir'):
                print(f"Data manager's data_dir: {data_manager.data_dir}")
        
        self.history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}

        # Calculate class weights and counts for sampler
        if self.train_labels:
            print("Setting up WeightedRandomSampler...")
            # Get class counts using the function (we need counts here, not weights)
            _, class_counts = calculate_class_weights(self.train_labels, len(train_dataset.classes))
            # Calculate weights for each sample: 1 / count of sample's class
            # Avoid division by zero if a class has 0 samples (though unlikely in train set)
            class_weights_inv = [1.0 / count if count > 0 else 0 for count in class_counts]
            sample_weights = [class_weights_inv[label] for label in self.train_labels]
            
            # Create sampler
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            print(f"Sampler created for {len(sample_weights)} samples.")
            # Shuffle must be False when using a sampler
            shuffle_train = False 
        else:
            print("WARNING: No train_labels provided for WeightedRandomSampler. Using standard shuffling.")
            self.train_sampler = None
            shuffle_train = True

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', DEFAULT_BATCH_SIZE), 
            # Use sampler if created, otherwise shuffle based on flag
            sampler=self.train_sampler, 
            shuffle=shuffle_train, # Must be False if sampler is not None
            num_workers=config.get('num_workers', 4), 
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
            model=self.model,
            train_labels=self.train_labels,
            num_classes=len(train_dataset.classes),
            device=device,
            learning_rate=config.get('learning_rate', 1e-4),
            num_epochs=config.get('num_epochs', DEFAULT_NUM_EPOCHS),
            steps_per_epoch=len(self.train_loader),
            label_smoothing_factor=config.get('label_smoothing', 0.1),
            loss_type=config.get('loss_type', 'cross_entropy'),
            focal_loss_gamma=config.get('focal_gamma', 2.0),
            scheduler_type=config.get('scheduler_type', 'one_cycle'),
            dataset_name=config.get('dataset_name', None),
            weight_decay=config.get('weight_decay', 0.0001),
            optimizer_type=config.get('optimizer', 'adam'),
            warmup_epochs=config.get('warmup_epochs', 0),
            use_class_weights=config.get('class_weights', False),
            sad_class_weight=config.get('sad_class_weight', 1.0),
            triplet_margin=config.get('triplet_margin', 0.3),
            custom_loss_fn=config.get('custom_loss_fn', None)
        )
        
        # Initialize EMA if enabled
        if self.config.get('use_ema', True):
            self.ema = EMA(self.model, decay=self.config.get('ema_decay', 0.999))
        else:
            self.ema = None

        # Training state
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience = config.get('patience', 15)
        self.patience_counter = 0
        
        # Get model directory from config or fall back to default
        self.model_dir = config.get('model_dir', CHECKPOINT_DIR)
        
        # Create checkpoint directory
        os.makedirs(self.model_dir, exist_ok=True)
    
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
        val_loss, metrics = validate(
            self.model, loader, self.criterion,
            self.device, dataset.classes
        )
        
        # Add loss to metrics
        metrics['loss'] = val_loss
        return metrics
    
    def train(self):
        """Main training loop with improved early stopping."""
        num_epochs = self.config.get('num_epochs', DEFAULT_NUM_EPOCHS)
        
        print("\nStarting training...")
        print(f"Training on {len(self.train_dataset)} samples")
        print(f"Validating on {len(self.val_dataset)} samples")
        print(f"Using device: {self.device}")
        print(f"Number of classes: {len(self.train_dataset.classes)}")
        
        # Verify first few training paths exist
        if hasattr(self.train_dataset, 'image_paths'):
            print("Verifying first 5 training image paths...")
            for i in range(min(5, len(self.train_dataset.image_paths))):
                path = self.train_dataset.image_paths[i]
                if os.path.exists(path):
                    print(f"  Path exists: {path}")
                else:
                    print(f"  Path does not exist: {path}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Update epoch for adaptive loss if used
            if hasattr(self.criterion, 'update_epoch'):
                self.criterion.update_epoch(epoch)
                print(f"Updated adaptive loss weights for epoch {epoch + 1}")
            
            # Training phase
            train_metrics = train_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scheduler,
                ema=self.ema,
                mixup_alpha=self.config.get('mixup_alpha', 0.0)
            )
            
            # Validation phase
            if self.ema:
                self.ema.apply_shadow()
            
            val_loss, val_metrics = validate(
                self.model, self.val_loader, self.criterion,
                self.device, self.train_dataset.classes
            )
            
            # Add loss to validation metrics
            val_metrics['loss'] = val_loss

            if self.ema:
                self.ema.restore()
            
            # Update learning rate scheduler (if not OneCycleLR, which is stepped per batch)
            if self.scheduler and self.config.get('scheduler_type', 'one_cycle') == 'cosine_annealing':
                self.scheduler.step()

            # Store metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_f1'].append(train_metrics.get('f1', train_metrics.get('macro_f1', 0.0)))  # Use either f1 or macro_f1
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics.get('f1', val_metrics.get('macro_f1', 0.0)))  # Use either f1 or macro_f1
            
            # Print epoch metrics
            print(f"\nEpoch {epoch + 1} Metrics:")
            print(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics.get('f1', train_metrics.get('macro_f1', 0.0)):.4f}")
            print(f"Val Loss: {val_loss:.4f}, F1: {val_metrics.get('f1', val_metrics.get('macro_f1', 0.0)):.4f}")
            
            # Save best model based on macro F1
            if val_metrics.get('f1', val_metrics.get('macro_f1', 0.0)) > self.best_val_f1:
                self.best_val_f1 = val_metrics.get('f1', val_metrics.get('macro_f1', 0.0))
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save checkpoint (with EMA weights if enabled)
                if self.ema:
                    self.ema.apply_shadow()
                
                save_path = os.path.join(self.model_dir, 'best_model.pth')
                save_model(
                    self.model, self.optimizer, epoch, save_path,
                    scheduler=self.scheduler, best_metrics=val_metrics
                )

                if self.ema:
                    self.ema.restore()
                
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
        save_training_history(self.history, self.model_dir)
        plot_training_history(self.history, self.model_dir)
        
        print("\nTraining completed successfully!")
        return self.best_val_f1 