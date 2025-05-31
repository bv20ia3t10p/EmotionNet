"""
Enhanced SOTA Trainer for EmotionNet - Refactored for clarity
"""

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from typing import Dict, Any
import collections

from .optimizer import create_optimizer, create_scheduler
from .losses import LabelSmoothingCrossEntropy, FocalLoss, ClassBalancedLoss
from .checkpoint_manager import CheckpointManager
from .augmentation_handler import TrainingAugmentationHandler
from .forward_pass import ForwardPassHandler
from ..data.augmentation import TestTimeAugmentation
from ..utils.metrics import calculate_metrics, print_epoch_results
from ..config.base import Config


class Trainer:
    """EmotionNet Trainer with advanced techniques."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Setup training components
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # Setup loss function based on configuration
        has_loss_config = hasattr(config, 'loss')
        
        # Configure loss function - use advanced losses for better handling of class imbalance
        if has_loss_config and getattr(config.loss, 'use_focal_loss', False):
            # Focal loss with class weights for handling class imbalance
            gamma = getattr(config.loss, 'gamma', 2.0)
            self.criterion = FocalLoss(
                gamma=gamma, 
                weight=self._get_class_weights(),
                temperature=getattr(config.loss, 'temperature', 1.0)
            )
            print(f"   - Using FocalLoss with gamma={gamma}")
        elif has_loss_config and getattr(config.loss, 'use_class_balanced_loss', False):
            # Class balanced loss that automatically adapts to class frequencies
            self.criterion = self._create_class_balanced_loss()
            print(f"   - Using ClassBalancedLoss")
        elif has_loss_config and hasattr(config.loss, 'label_smoothing'):
            # Label smoothing for better generalization
            label_smoothing = config.loss.label_smoothing
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            print(f"   - Using LabelSmoothingCrossEntropy with smoothing={label_smoothing}")
        else:
            # Default to CrossEntropy with class weights
            self.criterion = nn.CrossEntropyLoss(weight=self._get_class_weights())
            print(f"   - Using weighted CrossEntropyLoss")
        
        self.checkpoint_manager = CheckpointManager(config.data.checkpoint_dir)
        
        # Setup handlers
        self.aug_handler = TrainingAugmentationHandler(config)
        self.forward_handler = ForwardPassHandler(self.criterion)
        
        # Safe access to TTA configuration
        tta_transforms = 5  # Default value
        if hasattr(config, 'augmentation') and hasattr(config.augmentation, 'tta_transforms'):
            tta_transforms = config.augmentation.tta_transforms
        self.tta = TestTimeAugmentation(tta_transforms)
        
        # Training settings with safe defaults
        self.accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        
        self._print_initialization_info()
    
    def _get_class_weights(self):
        """
        Calculate class weights based on configuration or default to None.
        Higher weights for minority classes to handle class imbalance.
        """
        if hasattr(self.config, 'class_weights'):
            # If specific weights are provided in config
            return torch.tensor(self.config.class_weights).float().to(self.device)
        
        if hasattr(self.config.data, 'class_counts'):
            # If class frequency information is available
            counts = np.array(self.config.data.class_counts)
            weights = 1.0 / counts
            # Normalize weights
            weights = weights / weights.sum() * len(counts)
            return torch.tensor(weights).float().to(self.device)
        
        # Default weights for 7 classes (FER2013) - calculated from dataset distribution
        # Gives higher weight to minority classes like Disgust
        if hasattr(self.config.model, 'num_classes') and self.config.model.num_classes == 7:
            weights = torch.tensor([1.0, 3.0, 1.2, 1.0, 1.2, 1.5, 1.0]).to(self.device)
            return weights
        
        # Default weights for 8 classes (FERPlus) - based on typical distribution
        # Gives higher weight to minority classes like Disgust and Contempt
        if hasattr(self.config.model, 'num_classes') and self.config.model.num_classes == 8:
            weights = torch.tensor([1.0, 1.0, 1.2, 1.3, 1.5, 3.0, 1.5, 2.0]).to(self.device)
            return weights
        
        return None
    
    def _create_class_balanced_loss(self):
        """
        Create a class balanced loss that adjusts automatically to class distribution.
        Falls back to FocalLoss with manual weights if class counts not available.
        """
        if hasattr(self.config.data, 'class_counts'):
            samples_per_class = torch.tensor(self.config.data.class_counts)
            beta = getattr(self.config.loss, 'cb_beta', 0.9999)
            gamma = getattr(self.config.loss, 'gamma', 2.0)
            loss_type = getattr(self.config.loss, 'cb_loss_type', 'focal')
            
            return ClassBalancedLoss(
                samples_per_class=samples_per_class,
                beta=beta,
                gamma=gamma,
                loss_type=loss_type
            )
        else:
            # Fall back to focal loss with manual weights
            gamma = getattr(self.config.loss, 'gamma', 2.0)
            return FocalLoss(gamma=gamma, weight=self._get_class_weights())
    
    def _print_initialization_info(self):
        """Print trainer initialization information."""
        print(f"✅ Trainer initialized")
        # Get actual optimizer and scheduler types
        optimizer_name = "SGD" if hasattr(self.optimizer, 'param_groups') else "SAM"
        scheduler_name = "Plateau" if hasattr(self.scheduler, 'patience') else "OneCycleLR"
        print(f"   - Optimizer: {optimizer_name}")
        print(f"   - Scheduler: {scheduler_name}")
        
        # Print loss function details
        loss_name = self.criterion.__class__.__name__
        print(f"   - Loss function: {loss_name}")
        
        # Show class weights if available
        if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
            weights = [f"{w:.2f}" for w in self.criterion.weight.cpu().numpy()]
            print(f"   - Class weights: {weights}")
        
        # Safe access to config values for printing
        if hasattr(self.config, 'loss') and hasattr(self.config.loss, 'label_smoothing'):
            label_smoothing = self.config.loss.label_smoothing
            print(f"   - Label Smoothing: {label_smoothing}")
        
        print(f"   - Gradient Accumulation: {self.accumulation_steps}")
        
        # Safe access to TTA configuration for printing
        use_tta = False
        tta_transforms = 5
        if hasattr(self.config, 'augmentation'):
            use_tta = getattr(self.config.augmentation, 'use_tta', False)
            tta_transforms = getattr(self.config.augmentation, 'tta_transforms', 5)
        print(f"   - TTA: {'Enabled' if use_tta else 'Disabled'} ({tta_transforms} transforms)")
    
    def train_epoch(self, train_loader) -> Dict[str, Any]:
        """Train for one epoch with advanced techniques."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        accumulation_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply augmentations
            inputs = self.aug_handler.apply_class_specific_augmentation(inputs, targets)
            inputs, targets_a, targets_b, lam, is_mixed = self.aug_handler.apply_mixup_cutmix(inputs, targets)
            
            # Forward pass and loss computation
            loss, outputs = self._process_batch(inputs, targets_a, targets_b, lam, is_mixed, accumulation_count)
            
            # Update accumulation count and optimizer
            accumulation_count = self._update_optimizer(
                inputs, targets_a, targets_b, lam, is_mixed, accumulation_count
            )
            
            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            self._update_metrics(outputs, targets_a, targets_b, lam, is_mixed, all_preds, all_targets)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        return calculate_metrics(all_targets, all_preds, total_loss / len(train_loader))
    
    def _process_batch(self, inputs, targets_a, targets_b, lam, is_mixed, accumulation_count):
        """Process a single batch: forward pass and backward pass."""
        if accumulation_count == 0:
            self.optimizer.zero_grad()
        
        loss, outputs = self.forward_handler.compute_loss(
            self.model, inputs, targets_a, targets_b, lam, is_mixed, self.accumulation_steps
        )
        loss.backward()
        
        return loss, outputs
    
    def _update_optimizer(self, inputs, targets_a, targets_b, lam, is_mixed, accumulation_count):
        """Update optimizer when gradient accumulation is complete."""
        accumulation_count += 1
        
        if accumulation_count >= self.accumulation_steps:
            # Check if using SAM or regular optimizer
            if hasattr(self.optimizer, 'step') and hasattr(self.optimizer, 'base_optimizer'):
                # SAM optimizer
                closure = self.forward_handler.create_sam_closure(
                    self.model, inputs, targets_a, targets_b, lam, is_mixed, self.accumulation_steps
                )
                self.optimizer.step(closure)
            else:
                # Regular optimizer (SGD)
                self.optimizer.step()
            return 0
        
        return accumulation_count
    
    def _update_metrics(self, outputs, targets_a, targets_b, lam, is_mixed, all_preds, all_targets):
        """Update prediction and target lists for metrics calculation."""
        with torch.no_grad():
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            
            if is_mixed:
                # For mixed samples, use the dominant label based on lambda
                dominant_targets = targets_a if torch.rand(1) < lam else targets_b
                all_targets.extend(dominant_targets.cpu().numpy())
            else:
                all_targets.extend(targets_a.cpu().numpy())
    
    def validate_epoch(self, val_loader, use_tta=False) -> Dict[str, Any]:
        """Validate for one epoch with optional TTA."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Safely check if TTA should be used
                use_tta_enabled = False
                if hasattr(self.config, 'augmentation'):
                    use_tta_enabled = getattr(self.config.augmentation, 'use_tta', False)
                
                # Use TTA only if specified (for final evaluation)
                if use_tta and use_tta_enabled:
                    outputs = self.tta(self.model, inputs)
                else:
                    # Fast validation without TTA
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Use main output for multi-output models
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return calculate_metrics(all_targets, all_preds, total_loss / len(val_loader))
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """Main training loop."""
        import time
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        best_metrics = {}
        
        epochs = self.config.training.epochs
        start_time = time.time()
        
        try:
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                
                # Train and validate
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate_epoch(val_loader)
                
                # Update learning rate
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Print epoch results
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}/{epochs} - Time: {epoch_time:.2f}s")
                print_epoch_results(train_metrics, val_metrics)
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Save checkpoint
                is_best = val_metrics['accuracy'] > best_val_acc
                if is_best:
                    best_val_acc = val_metrics['accuracy']
                    best_val_f1 = val_metrics['f1_score']
                    best_epoch = epoch
                    best_metrics = val_metrics.copy()
                    best_metrics['epoch'] = epoch
                    best_metrics['val_acc'] = best_val_acc
                    best_metrics['val_f1'] = best_val_f1
                    
                self.checkpoint_manager.save_checkpoint(
                    epoch, self.model, self.optimizer, self.scheduler, val_metrics, is_best=is_best
                )
                
                # Save detailed epoch stats
                self.checkpoint_manager.save_epoch_stats(
                    epoch, train_metrics, val_metrics, epoch_time, current_lr
                )
                    
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time/60:.2f} minutes")
            print(f"Best Validation Accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            
            return best_metrics
            
        except KeyboardInterrupt:
            print("\nTraining interrupted")
            return {
                'epoch': epoch,
                'val_acc': best_val_acc,
                'val_f1': best_val_f1
            } 