"""
Checkpoint management for EmotionNet training.
Handles saving and loading model checkpoints and training history.
"""

import os
import json
import datetime
import torch
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages model checkpoints and training history."""
    
    def __init__(self, save_dir: str = 'checkpoints'):
        self.save_dir = save_dir
        self.epoch_stats_dir = os.path.join(save_dir, 'epoch_stats')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
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
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int, val_metrics: Dict[str, Any], config: Dict[str, Any],
                       is_best: bool = False) -> bool:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score'],
            'config': config
        }
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            self.best_val_acc = val_metrics['accuracy']
            self.best_val_f1 = val_metrics['f1_score']
            self.best_epoch = epoch
            return True
        
        # Save periodic checkpoint
        if epoch % 15 == 0:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        return False
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       checkpoint_path: str) -> int:
        """Load model checkpoint and return last epoch."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best validation metrics."""
        return {
            'val_acc': self.best_val_acc,
            'val_f1': self.best_val_f1,
            'epoch': self.best_epoch
        }
    
    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable format."""
        serialized = {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'f1_score': float(metrics['f1_score']),
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