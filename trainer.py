"""
Clean SOTA Trainer for EmotionNet
Simplified trainer focused on essential functionality with clean separation of concerns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from typing import Dict, Any

from loss_functions import FocalLoss
from metrics import calculate_metrics, print_epoch_results, get_class_weights_tensor
from checkpoint_manager import CheckpointManager


class SOTATrainer:
    """Clean, focused trainer for SOTA EmotionNet."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_loss()
        self.checkpoint_manager = CheckpointManager(config['checkpoint_dir'])
        
        print(f"✅ SOTA Trainer initialized")
        print(f"   - Learning Rate: {config['lr']}")
        print(f"   - Focal Gamma: {config['focal_gamma']}")
        print(f"   - Class Weights: Optimized")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup AdamW optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['step_size'],
            gamma=self.config['gamma']
        )
    
    def _setup_loss(self) -> nn.Module:
        """Setup focal loss with class weights."""
        class_weights = get_class_weights_tensor(self.config['class_weights'], self.device)
        return FocalLoss(
            gamma=self.config['focal_gamma'],
            alpha=class_weights,
            reduction='mean'
        )
    
    def train_epoch(self, train_loader) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress
            pbar.set_postfix({
                'loss': total_loss / (len(all_preds) // len(inputs) + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / len(train_loader)
        return calculate_metrics(all_targets, all_preds, avg_loss)
    
    def validate(self, val_loader) -> Dict[str, Any]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        return calculate_metrics(all_targets, all_preds, avg_loss)
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """Main training loop."""
        print("\n🚀 Starting SOTA Training")
        print(f"   - Epochs: {self.config['epochs']}")
        print(f"   - Batch Size: {self.config['batch_size']}")
        print("="*50)
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training and validation
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            
            # Save epoch statistics
            self.checkpoint_manager.save_epoch_stats(
                epoch, train_metrics, val_metrics, epoch_time, lr
            )
            
            # Print results
            print_epoch_results(epoch, self.config['epochs'], train_metrics, val_metrics, lr, epoch_time)
            
            # Save checkpoints
            is_best = val_metrics['accuracy'] > self.checkpoint_manager.best_val_acc
            saved_best = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, val_metrics, self.config, is_best
            )
            
            if saved_best:
                print(f"💾 Saved best model - Val Acc: {val_metrics['accuracy']:.2f}%")
        
        print(f"\n✅ Training completed!")
        best_metrics = self.checkpoint_manager.get_best_metrics()
        print(f"   - Best Val Acc: {best_metrics['val_acc']:.2f}% (Epoch {best_metrics['epoch']})")
        
        return best_metrics 