"""Model management utilities for emotion recognition training."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from emotion_net.config.constants import CHECKPOINT_DIR
from .losses import FocalLoss
import numpy as np

def setup_training(model, train_labels, num_classes, device, 
                   learning_rate=1e-4, # This will be max_lr for OneCycleLR
                   num_epochs=100, 
                   steps_per_epoch=1, # Need steps_per_epoch for OneCycleLR
                   label_smoothing_factor=0.0, 
                   loss_type='cross_entropy', 
                   focal_loss_gamma=2.0,
                   scheduler_type='one_cycle'): # Add scheduler_type
    """Setup model, criterion, optimizer and scheduler for training."""
    # Calculate class weights and counts
    # We only need weights for the loss function here
    class_weights, _ = calculate_class_weights(train_labels, num_classes) # Unpack weights
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create criterion with class weights and label smoothing or Focal Loss
    if loss_type == 'focal':
        print(f"Using Focal Loss with gamma: {focal_loss_gamma} and NO alpha (class imbalance handled by gamma)")
        # Pass alpha=None to rely on gamma for imbalance, or pass class_weights for explicit alpha weighting
        criterion = FocalLoss(alpha=None, gamma=focal_loss_gamma, reduction='mean')
    elif loss_type == 'cross_entropy':
        print(f"Using Cross Entropy Loss with label smoothing: {label_smoothing_factor} and class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing_factor)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'cross_entropy' or 'focal'.")
    
    # Create optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate, # For OneCycleLR, this will be treated as max_lr by the scheduler call
        weight_decay=0.01
    )
    
    # Create learning rate scheduler
    if scheduler_type == 'one_cycle':
        print(f"Using OneCycleLR scheduler with max_lr: {learning_rate}")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate, 
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2, # Changed from 0.3 to 0.2 for shorter warmup
            anneal_strategy='cos' # Can be 'linear' or 'cos'
        )
    elif scheduler_type == 'cosine_annealing': # Keep CosineAnnealingLR as an option
        print(f"Using CosineAnnealingLR scheduler with T_max: {num_epochs}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-7 
        )
    else:
        print("No scheduler or unsupported scheduler_type specified.")
        scheduler = None
    
    return model, criterion, optimizer, scheduler

def calculate_class_weights(labels, num_classes):
    """Calculate class weights inversely proportional to class frequencies. Returns weights and counts."""
    labels_np = np.array(labels)
    class_counts = np.bincount(labels_np, minlength=num_classes)
    
    # Avoid division by zero for classes with no samples
    total_samples = float(sum(class_counts))
    class_weights = np.zeros(num_classes)
    non_zero_indices = class_counts > 0
    class_weights[non_zero_indices] = total_samples / (num_classes * class_counts[non_zero_indices])
    
    # Print class distribution
    print("\nClass Distribution:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        # Use EMOTIONS list for label name if available, otherwise use index
        try:
            from emotion_net.config.constants import EMOTIONS
            label_name = EMOTIONS[i]
        except (ImportError, IndexError):
            label_name = f"Class {i}"
        print(f"{label_name:>10s}: {int(count)} samples, weight: {weight:.2f}")
    
    # Return both weights and counts
    return class_weights, class_counts

def create_criterion(class_weights, label_smoothing=0.0):
    """DEPRECATED: This function is no longer directly called by setup_training.
    Create weighted cross entropy loss criterion with optional label smoothing.
    """
    print(f"Using label smoothing: {label_smoothing}")
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

def save_model(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint."""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, save_path)
    
    print(f"Model saved to {save_path}") 