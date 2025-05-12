"""Model management utilities for emotion recognition training."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from emotion_net.config.constants import CHECKPOINT_DIR
from .losses import FocalLoss
import numpy as np

def calculate_class_weights(labels, num_classes, dataset_name=None):
    """Calculate class weights and counts.
    
    Args:
        labels: List of integer labels
        num_classes: Number of classes
        dataset_name: Optional dataset name for dataset-specific weighting
        
    Returns:
        tuple: (class_weights, class_counts)
            - class_weights: Array of weights for each class (for loss function)
            - class_counts: Array of counts for each class
    """
    # Count occurrences of each class
    counts = np.bincount(labels, minlength=num_classes)
    
    # Handle empty classes to avoid division by zero
    counts = np.maximum(counts, 1)
    
    # For RAF-DB, use a simpler inverse frequency weighting with sqrt to reduce extremes
    if dataset_name == 'rafdb':
        # sqrt(N/n_i) weighting - reduces extreme weights while still balancing
        # This prevents the model from focusing too much on minority classes
        N = sum(counts)
        weights = np.sqrt(N / counts)
        
        # Normalize weights so they sum to num_classes
        weights = weights / weights.sum() * num_classes
        
        # Apply small manual adjustments to prevent class collapse
        # We need to prevent overweighting and causing class collapse
        # Cap weights to prevent extremes - max weight should be 3x the min weight
        max_weight = weights.max()
        min_weight = weights.min()
        if max_weight / min_weight > 3.0:
            # Scale weights to have a max/min ratio of 3
            weights = (weights - min_weight) / (max_weight - min_weight) * (3 * min_weight - min_weight) + min_weight
    else:
        # Default effective number of samples weighting
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights
        weights = weights / weights.sum() * num_classes
    
    print("Class weights for loss function:")
    for i in range(num_classes):
        print(f"  Class {i}: {weights[i]:.4f} (count: {counts[i]})")
        
    return weights, counts

def create_criterion(class_weights, loss_type='cross_entropy', focal_gamma=2.0, label_smoothing=0.1, device='cuda'):
    """Create a criterion based on loss type."""
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    if loss_type == 'focal':
        print(f"Using Focal Loss with gamma: {focal_gamma}")
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
    elif loss_type == 'cross_entropy':
        print(f"Using Cross Entropy Loss with label smoothing: {label_smoothing} and class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'cross_entropy' or 'focal'.")
    
    return criterion

def setup_training(model, train_labels, num_classes, device, 
                   learning_rate=1e-4, # This will be max_lr for OneCycleLR
                   num_epochs=100, 
                   steps_per_epoch=1, # Need steps_per_epoch for OneCycleLR
                   label_smoothing_factor=0.0, 
                   loss_type='cross_entropy', 
                   focal_loss_gamma=2.0,
                   scheduler_type='one_cycle',
                   dataset_name=None): # Add dataset_name parameter
    """Setup model, criterion, optimizer and scheduler for training."""
    # Calculate class weights and counts
    # We only need weights for the loss function here
    class_weights, _ = calculate_class_weights(train_labels, num_classes, dataset_name) # Pass dataset_name
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create criterion with class weights and label smoothing or Focal Loss
    if loss_type == 'focal':
        print(f"Using Focal Loss with gamma: {focal_loss_gamma} and class weights as alpha")
        # Pass class_weights as alpha for explicit weighting
        criterion = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma, reduction='mean')
    elif loss_type == 'cross_entropy':
        print(f"Using Cross Entropy Loss with label smoothing: {label_smoothing_factor} and class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing_factor)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'cross_entropy' or 'focal'.")
    
    # Create optimizer with AdamW and weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Create scheduler
    if scheduler_type == 'one_cycle':
        print(f"Using OneCycleLR scheduler with max_lr: {learning_rate}")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,  # Spend 30% of time warming up
            div_factor=25.0,  # Initial lr is max_lr/25
            final_div_factor=1000.0  # Final lr is max_lr/1000
        )
    elif scheduler_type == 'cosine_annealing':
        print(f"Using CosineAnnealingLR scheduler with T_max: {num_epochs}")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate / 100
        )
    else:  # 'none' or any other value
        print("No scheduler used")
        scheduler = None
    
    return model, criterion, optimizer, scheduler

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