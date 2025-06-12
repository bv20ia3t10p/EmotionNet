import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, confusion_matrix

class MetricsTracker:
    def __init__(
        self,
        num_classes: int,
        output_dir: str,
        class_names: List[str] = None
    ):
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.metrics_path = os.path.join(output_dir, "metrics.json")
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "train_f1_per_class": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_f1_per_class": [],
            "best_val_acc": 0.0,
            "best_val_f1": 0.0
        }
        
    def update_metrics(
        self,
        split: str,
        metrics: Dict,
        epoch: int
    ) -> None:
        """Update metrics for the given split (train/val)."""
        # Store metrics
        self.history[f"{split}_loss"].append(metrics['loss'])
        self.history[f"{split}_acc"].append(metrics['accuracy'])
        self.history[f"{split}_f1"].append(metrics['f1_macro'])
        self.history[f"{split}_f1_per_class"].append(
            [metrics['f1_per_class'][name] for name in self.class_names]
        )
        
        # Update best metrics for validation
        if split == "val":
            if metrics['accuracy'] > self.history["best_val_acc"]:
                self.history["best_val_acc"] = metrics['accuracy']
            if metrics['f1_macro'] > self.history["best_val_f1"]:
                self.history["best_val_f1"] = metrics['f1_macro']
    
    def print_epoch_metrics(self, epoch: int):
        """Print metrics for the current epoch."""
        print(f"\nEpoch {epoch + 1} Results:")
        
        for split in ['train', 'val']:
            print(f"\n{split.capitalize()}:")
            print(f"  Loss: {self.history[f'{split}_loss'][-1]:.4f}")
            print(f"  Accuracy: {self.history[f'{split}_acc'][-1]:.2f}%")
            print(f"  F1 Score (Macro): {self.history[f'{split}_f1'][-1]:.2f}%")
            print("\nF1 Scores per class:")
            for class_name, f1 in zip(self.class_names, self.history[f'{split}_f1_per_class'][-1]):
                print(f"  {class_name}: {f1:.2f}%")
            
        print(f"\nBest Validation Accuracy: {self.history['best_val_acc']:.2f}%")
        print(f"Best Validation F1 Score: {self.history['best_val_f1']:.2f}%")
    
    def save_metrics(self):
        """Save metrics history to JSON file."""
        with open(self.metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def should_save_model(self, current_val_acc: float) -> bool:
        """Check if current model should be saved based on validation accuracy."""
        return current_val_acc >= self.history["best_val_acc"]

class BatchMetricsAggregator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.total_loss = 0
        self.predictions = []
        self.targets = []
        self.num_batches = 0
        
    def update(
        self,
        loss: float,
        batch_predictions: torch.Tensor,
        batch_targets: torch.Tensor
    ):
        self.total_loss += loss
        self.predictions.extend(batch_predictions.cpu().numpy())
        self.targets.extend(batch_targets.cpu().numpy())
        self.num_batches += 1
        
    def get_aggregated_metrics(self) -> Dict:
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Calculate metrics
        avg_loss = self.total_loss / self.num_batches
        accuracy = (predictions == targets).mean() * 100
        f1_macro = f1_score(targets, predictions, average='macro') * 100
        f1_per_class = f1_score(targets, predictions, average=None) * 100
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_per_class': {
                f'class_{i}': f1_per_class[i]
                for i in range(self.num_classes)
            }
        } 