"""
Checkpoint management for EmotionNet training.
Handles saving model checkpoints and training history.
"""

import os
import json
import datetime
import torch
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages model checkpoints and training history."""
    
    def __init__(self, output_dir, model=None, optimizer=None, scheduler=None, ema_model=None, 
                best_metric='accuracy', save_dir=None):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints and stats
            model: The model to checkpoint
            optimizer: The optimizer
            scheduler: The learning rate scheduler
            ema_model: Exponential moving average model (optional)
            best_metric: Metric to track for best model ('accuracy' or 'f1')
            save_dir: Legacy parameter, use output_dir instead
        """
        self.save_dir = output_dir if save_dir is None else save_dir
        self.epoch_stats_dir = os.path.join(self.save_dir, 'epoch_stats')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
        # Store model references
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema_model = ema_model
        self.best_metric = best_metric
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.epoch_stats_dir, exist_ok=True)
    
    def save_epoch_stats(self, epoch: int, train_metrics: Dict[str, Any], 
                        val_metrics: Dict[str, Any], epoch_time: float, lr: float) -> None:
        """Save detailed statistics for the current epoch."""
        epoch_stats = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'epoch_time': float(epoch_time),
            'learning_rate': float(lr),
            'train': self._serialize_metrics(train_metrics),
            'val': self._serialize_metrics(val_metrics),
            'confusion_matrices': {
                'train': train_metrics.get('confusion_matrix', []),
                'val': val_metrics.get('confusion_matrix', [])
            }
        }
        
        # Save epoch stats
        stats_file = os.path.join(self.epoch_stats_dir, f'epoch_{epoch:03d}_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(epoch_stats, f, indent=2)
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler=None, metrics=None, is_best=False):
        """
        Save model checkpoint.
        
        Supports both standard interface and compatibility interface.
        """
        # Use provided instances or fall back to stored instances
        model = model if model is not None else self.model
        optimizer = optimizer if optimizer is not None else self.optimizer
        scheduler = scheduler if scheduler is not None else self.scheduler
        ema_model = self.ema_model
        
        if model is None:
            raise ValueError("No model provided for checkpoint saving")
            
        # Extract metrics from val_metrics if provided
        if metrics is not None:
            val_acc = metrics.get('accuracy', 0.0)
            val_f1 = metrics.get('f1_score', 0.0)
        else:
            val_f1 = 0.0
            
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'best_val_acc': self.best_val_acc
        }
        
        # Add EMA model if it exists
        if ema_model is not None:
            checkpoint['ema_model_state_dict'] = ema_model.state_dict()
            
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            self.best_val_acc = val_acc
            self.best_val_f1 = val_f1
            self.best_epoch = epoch
            return True
        
        # Save periodic checkpoint
        if epoch % 15 == 0:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        return False
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best validation metrics."""
        return {
            'val_acc': self.best_val_acc,
            'val_f1': self.best_val_f1,
            'epoch': self.best_epoch
        }
    
    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable format."""
        if not metrics:
            return {}
            
        serialized = {
            'loss': float(metrics.get('loss', 0.0)),
            'accuracy': float(metrics.get('accuracy', 0.0)),
            'f1_score': float(metrics.get('f1_score', 0.0)),
            'per_class': {}
        }
        
        # Add per-class metrics if available
        if 'class_report' in metrics:
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            for label in emotion_labels:
                if label in metrics['class_report']:
                    class_metrics = metrics['class_report'][label]
                    serialized['per_class'][label] = {
                        'precision': float(class_metrics['precision']),
                        'recall': float(class_metrics['recall']),
                        'f1_score': float(class_metrics['f1-score']),
                        'support': int(class_metrics['support'])
                    }
        
        return serialized 