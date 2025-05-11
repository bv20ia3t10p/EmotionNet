"""Model management and optimization utilities."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from emotion_net.models.ema import EMA
from emotion_net.config.constants import EMOTIONS

def setup_training(model, learning_rate, device, use_amp=True, use_ema=True):
    """Setup model, optimizer, scheduler, and other training components."""
    # Setup EMA if enabled
    ema_model = EMA(model, decay=0.999) if use_ema else None
    
    # Setup AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Split parameters for different learning rates
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained backbone
        {'params': other_params, 'lr': learning_rate}
    ], weight_decay=1e-4)
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    return ema_model, scaler, optimizer, scheduler

def calculate_class_weights(train_loader, class_weights_boost=1.5, device=None):
    """Calculate class weights for imbalanced dataset."""
    # Calculate class counts
    class_counts = torch.zeros(len(EMOTIONS))
    for _, label in train_loader.dataset:
        class_counts[label] += 1
    
    # Calculate weights using inverse frequency with smoothing
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * (class_counts + 1e-6))
    
    # Apply class weight boosting for important classes
    class_weights[0] *= class_weights_boost  # angry
    class_weights[6] *= class_weights_boost  # neutral
    
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    
    print("Class weights:", class_weights.numpy())
    
    if device:
        class_weights = class_weights.to(device)
    
    return class_weights

def create_criterion(class_weights, device):
    """Create loss criterion with class weights."""
    return nn.CrossEntropyLoss(weight=class_weights.to(device))

def save_model(model, save_path, use_ema=False, ema_model=None):
    """Save model checkpoint."""
    if use_ema and ema_model is not None:
        torch.save(model.state_dict(), save_path)
        ema_model.restore()
    else:
        torch.save(model.state_dict(), save_path) 