"""
Enhanced SOTA Trainer for EmotionNet - 79%+ Target
Advanced training with SAM optimizer, OneCycleLR, Label Smoothing, and all optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from typing import Dict, Any
import numpy as np

from loss_functions import FocalLoss
from metrics import calculate_metrics, print_epoch_results, get_class_weights_tensor
from checkpoint_manager import CheckpointManager
from sam_optimizer import SAM


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = torch.log_softmax(pred, dim=1)
        return torch.mean(torch.sum(-one_hot * log_prob, dim=1))


class RandAugment:
    """RandAugment implementation for emotion recognition"""
    def __init__(self, n=2, m=9):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations
        
    def __call__(self, x, labels=None):
        # Apply n random augmentations with magnitude m
        batch_size = x.size(0)
        device = x.device
        
        # Define augmentation operations
        ops = [
            self.rotate, self.translate, self.brightness, 
            self.contrast, self.sharpness, self.gaussian_noise
        ]
        
        for _ in range(self.n):
            op = np.random.choice(ops)
            x = op(x)
            
        return torch.clamp(x, 0, 1)
    
    def rotate(self, x):
        """Random rotation"""
        angle = (self.m / 10) * 30  # Max 30 degrees
        angle = np.random.uniform(-angle, angle)
        return torch.rot90(x, k=int(angle // 90), dims=[2, 3])
    
    def translate(self, x):
        """Random translation"""
        shift = int((self.m / 10) * 4)  # Max 4 pixels
        shift_x = np.random.randint(-shift, shift + 1)
        shift_y = np.random.randint(-shift, shift + 1)
        return torch.roll(x, shifts=(shift_x, shift_y), dims=(2, 3))
    
    def brightness(self, x):
        """Random brightness adjustment"""
        factor = 1 + (self.m / 10) * 0.3 * np.random.uniform(-1, 1)
        return x * factor
    
    def contrast(self, x):
        """Random contrast adjustment"""
        factor = 1 + (self.m / 10) * 0.3 * np.random.uniform(-1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - mean) * factor + mean
    
    def sharpness(self, x):
        """Random sharpness adjustment"""
        if np.random.random() < 0.5:
            kernel = torch.tensor([[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]).float().to(x.device)
            return torch.nn.functional.conv2d(x, kernel, padding=1)
        return x
    
    def gaussian_noise(self, x):
        """Add Gaussian noise"""
        noise_level = (self.m / 10) * 0.1
        noise = torch.randn_like(x) * noise_level
        return x + noise


class TestTimeAugmentation:
    """Test Time Augmentation for improved inference"""
    def __init__(self, n_transforms=5):
        self.n_transforms = n_transforms
        
    def __call__(self, model, x):
        """Apply TTA and return averaged predictions"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            pred = model(x)
            predictions.append(torch.softmax(pred, dim=1))
            
            # Augmented predictions
            for _ in range(self.n_transforms - 1):
                # Apply simple augmentations
                aug_x = x.clone()
                
                # Random horizontal flip
                if torch.rand(1) < 0.5:
                    aug_x = torch.flip(aug_x, dims=[3])
                
                # Random rotation
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    aug_x = torch.rot90(aug_x, k=k, dims=[2, 3])
                
                # Small random translation
                shift_x = torch.randint(-2, 3, (1,)).item()
                shift_y = torch.randint(-2, 3, (1,)).item()
                aug_x = torch.roll(aug_x, shifts=(shift_x, shift_y), dims=(2, 3))
                
                pred = model(aug_x)
                predictions.append(torch.softmax(pred, dim=1))
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)


class Trainer:
    """Enhanced SOTA Trainer with advanced techniques for 79%+ accuracy."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_loss()
        self.checkpoint_manager = CheckpointManager(config['checkpoint_dir'])
        
        # Setup augmentation
        self.randaugment = RandAugment(
            n=config.get('randaugment_n', 2),
            m=config.get('randaugment_m', 9)
        )
        self.tta = TestTimeAugmentation(config.get('tta_transforms', 5))
        
        # Gradient accumulation
        self.accumulation_steps = config.get('gradient_accumulation_steps', 2)
        
        print(f"✅ Enhanced SOTA Trainer initialized")
        print(f"   - Optimizer: SAM")
        print(f"   - Scheduler: OneCycleLR") 
        print(f"   - Label Smoothing: {config.get('label_smoothing', 0.0)}")
        print(f"   - Gradient Accumulation: {self.accumulation_steps}")
        print(f"   - TTA Transforms: {config.get('tta_transforms', 5)}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup SAM optimizer."""
        return SAM(
            self.model.parameters(),
            optim.AdamW,
            rho=self.config.get('sam_rho', 0.05),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup OneCycleLR scheduler."""
        total_steps = self.config['epochs'] * self.config.get('steps_per_epoch', 100)
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer.base_optimizer,
            max_lr=self.config.get('max_lr', 0.01),
            total_steps=total_steps,
            pct_start=self.config.get('pct_start', 0.3),
            anneal_strategy=self.config.get('anneal_strategy', 'cos'),
            div_factor=self.config.get('div_factor', 25),
            final_div_factor=self.config.get('final_div_factor', 10000)
        )
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function with label smoothing."""
        return LabelSmoothingCrossEntropy(smoothing=self.config['label_smoothing'])
    
    def train_epoch(self, train_loader) -> Dict[str, Any]:
        """Train for one epoch with advanced techniques."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Import augmentation classes
        from dataset import MixUp, CutMix
        mixup = MixUp(alpha=self.config.get('mixup_alpha', 0.4))
        cutmix = CutMix(alpha=self.config.get('cutmix_alpha', 0.5))
        
        # Class-specific augmentation probabilities
        class_aug_probs = {
            0: 0.5,   # Angry - reduced from 0.8
            1: 0.3,   # Disgust
            2: 0.6,   # Fear - reduced from 0.95 for initial training
            3: 0.3,   # Happy
            4: 0.5,   # Sad - reduced from 0.90 for initial training
            5: 0.3,   # Surprise
            6: 0.5    # Neutral - reduced from 0.8
        }
        
        pbar = tqdm(train_loader, desc='Training')
        accumulation_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply class-specific augmentation
            for i in range(inputs.size(0)):
                label = targets[i].item()
                aug_prob = class_aug_probs.get(label, 0.5)
                if torch.rand(1) < aug_prob:
                    inputs[i:i+1] = self.randaugment(inputs[i:i+1])
            
            # Apply MixUp or CutMix
            use_mixup = self.config.get('mixup_alpha', 0) > 0 and torch.rand(1) < 0.5
            use_cutmix = self.config.get('cutmix_alpha', 0) > 0 and torch.rand(1) < 0.5
            
            lam = 1.0
            targets_a = targets
            targets_b = targets
            
            if use_mixup and not use_cutmix:
                inputs, targets_a, targets_b, lam = mixup(inputs, targets)
            elif use_cutmix and not use_mixup:
                inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            
            # Forward pass
            model_output = self.model(inputs)
            if isinstance(model_output, tuple):
                outputs, aux_outputs = model_output
                # Main loss
                if use_mixup or use_cutmix:
                    main_loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                    aux_loss = lam * self.criterion(aux_outputs, targets_a) + (1 - lam) * self.criterion(aux_outputs, targets_b)
                else:
                    main_loss = self.criterion(outputs, targets)
                    aux_loss = self.criterion(aux_outputs, targets)
                loss = main_loss + 0.1 * aux_loss  # Reduced auxiliary loss weight from 0.4
            else:
                outputs = model_output
                if use_mixup or use_cutmix:
                    loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Accumulate gradients
            if accumulation_count == 0:
                self.optimizer.zero_grad()
            
            # Always call backward to accumulate gradients
            loss.backward()
            accumulation_count += 1
            
            # Only call optimizer step when accumulation is complete
            if accumulation_count >= self.accumulation_steps:
                # Define closure for SAM
                def closure():
                    # Clear gradients and recompute for SAM
                    self.optimizer.zero_grad()
                    
                    # Recompute forward pass and loss for SAM
                    model_output_sam = self.model(inputs)
                    if isinstance(model_output_sam, tuple):
                        outputs_sam, aux_outputs_sam = model_output_sam
                        if use_mixup or use_cutmix:
                            main_loss_sam = lam * self.criterion(outputs_sam, targets_a) + (1 - lam) * self.criterion(outputs_sam, targets_b)
                            aux_loss_sam = lam * self.criterion(aux_outputs_sam, targets_a) + (1 - lam) * self.criterion(aux_outputs_sam, targets_b)
                        else:
                            main_loss_sam = self.criterion(outputs_sam, targets)
                            aux_loss_sam = self.criterion(aux_outputs_sam, targets)
                        loss_sam = main_loss_sam + 0.1 * aux_loss_sam
                    else:
                        outputs_sam = model_output_sam
                        if use_mixup or use_cutmix:
                            loss_sam = lam * self.criterion(outputs_sam, targets_a) + (1 - lam) * self.criterion(outputs_sam, targets_b)
                        else:
                            loss_sam = self.criterion(outputs_sam, targets)
                    
                    loss_sam = loss_sam / self.accumulation_steps
                    loss_sam.backward()
                    return loss_sam
                
                # Step with SAM
                self.optimizer.step(closure)
                accumulation_count = 0
            
            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            with torch.no_grad():
                _, predicted = outputs.max(1)
                if use_mixup or use_cutmix:
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets_a.cpu().numpy() if torch.rand(1) < lam else targets_b.cpu().numpy())
                else:
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Update progress
            current_lr = self.optimizer.base_optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'lr': f"{current_lr:.6f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        return calculate_metrics(all_targets, all_preds, avg_loss)
    
    def validate(self, val_loader, use_tta=False) -> Dict[str, Any]:
        """Validate the model with optional TTA."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if use_tta and self.config.get('use_tta', False):
                    # Use Test Time Augmentation
                    outputs = self.tta(self.model, inputs)
                    # Convert back to logits for loss calculation
                    outputs = torch.log(outputs + 1e-8)
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Use main output
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        return calculate_metrics(all_targets, all_preds, avg_loss)
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """Main training loop with all enhancements."""
        print("\n🚀 Starting Enhanced SOTA Training for 79%+ Accuracy")
        print(f"   - Epochs: {self.config['epochs']}")
        print(f"   - Effective Batch Size: {self.config['batch_size'] * self.accumulation_steps}")
        print(f"   - SAM Optimizer + OneCycleLR")
        print(f"   - Label Smoothing: {self.config.get('label_smoothing', 0.0)}")
        print("="*60)
        
        # Set steps per epoch for OneCycleLR
        self.config['steps_per_epoch'] = len(train_loader)
        if hasattr(self.scheduler, 'total_steps'):
            self.scheduler.total_steps = self.config['epochs'] * len(train_loader)
        
        best_val_acc = 0.0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation (use TTA every 10 epochs for speed)
            use_tta = epoch % 10 == 0 and self.config.get('use_tta', False)
            val_metrics = self.validate(val_loader, use_tta=use_tta)
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.base_optimizer.param_groups[0]['lr']
            
            # Save epoch statistics
            self.checkpoint_manager.save_epoch_stats(
                epoch, train_metrics, val_metrics, epoch_time, current_lr
            )
            
            # Print results
            print_epoch_results(epoch, self.config['epochs'], train_metrics, val_metrics, current_lr, epoch_time)
            
            # Save checkpoints
            is_best = val_metrics['accuracy'] > self.checkpoint_manager.best_val_acc
            saved_best = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, val_metrics, self.config, is_best
            )
            
            if saved_best:
                best_val_acc = val_metrics['accuracy']
                print(f"💾 New Best Model! Val Acc: {best_val_acc:.2f}%")
                
                # Additional TTA validation for best model
                if self.config.get('use_tta', False):
                    tta_metrics = self.validate(val_loader, use_tta=True)
                    print(f"🎯 TTA Val Acc: {tta_metrics['accuracy']:.2f}%")
        
        print(f"\n✅ Enhanced Training Completed!")
        best_metrics = self.checkpoint_manager.get_best_metrics()
        print(f"   - Best Val Acc: {best_metrics['val_acc']:.2f}% (Epoch {best_metrics['epoch']})")
        
        return best_metrics 