"""Model management utilities for emotion recognition training."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, LinearLR
import math

# Import our custom loss
from emotion_net.training.losses import SOTAEmotionLoss, StableFocalLoss, FocalLoss

# Calculate statistics from training data
def calculate_class_weights(labels, num_classes, class_weight_factor=1.0):
    """Calculate class weights for imbalanced datasets."""
    # Count occurrences of each class
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1
    
    # Convert counts to weights (inversely proportional to count)
    max_count = max(class_counts)
    class_weights = []
    for count in class_counts:
        if count == 0:
            # Avoid division by zero; set weight to 1.0 for classes with no samples
            class_weights.append(1.0)
        else:
            # Scale by the ratio of occurrences relative to the most common class
            class_weights.append((max_count / count) ** class_weight_factor)
    
    # Normalize weights (optional)
    total_weight = sum(class_weights)
    normalized_weights = [weight / total_weight * num_classes for weight in class_weights]
    
    return normalized_weights, class_counts

# Create loss function
def create_criterion(loss_type, num_classes, device, 
                     class_weights=None, focal_gamma=2.0, 
                     label_smoothing=0.0, sad_class_weight=1.0, 
                     custom_loss_fn=None, triplet_margin=0.3,
                     dataset_name=None):
    """Create loss function based on configuration."""
    if custom_loss_fn is not None:
        print("Using custom loss function.")
        return custom_loss_fn

    # If class weights are provided, convert to tensor
    if class_weights is not None:
        print(f"Using class weights: {class_weights}")
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        weight = None
    
    # Apply additional weight to 'sad' class if required
    if sad_class_weight != 1.0 and weight is not None:
        # Find the index of 'sad' class based on dataset
        sad_idx = 4  # Default for FER2013, RAFDB
        if dataset_name == 'rafdb':
            sad_idx = 4  # In RAFDB, sad is index 4 (0-indexed)
        elif dataset_name == 'fer2013':
            sad_idx = 4  # In FER2013, sad is index
        
        # Multiply sad class weight
        if 0 <= sad_idx < len(weight):
            print(f"Applying additional weight {sad_class_weight} to sad class (index {sad_idx})")
            weight[sad_idx] *= sad_class_weight
    
    # Select loss function based on type
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    elif loss_type == 'focal':
        # Use the stable version of Focal Loss
        return StableFocalLoss(alpha=weight, gamma=focal_gamma, reduction='mean', label_smoothing=label_smoothing)
    elif loss_type == 'hybrid':
        from emotion_net.training.losses import HybridLoss
        return HybridLoss(alpha=weight, gamma=focal_gamma, label_smoothing=label_smoothing)
    elif loss_type == 'sota_emotion' or loss_type == 'sota':
        # Use our custom SOTAEmotionLoss
        print("Using SOTA Emotion Loss for training")
        return SOTAEmotionLoss(
            num_classes=num_classes,
            embedding_size=1024,  # Default value, should match model's embedding size
            label_smoothing=label_smoothing,
            aux_weight=0.4
        )
    elif loss_type == 'adaptive' or loss_type == 'adaptive_emotion':
        # Use our specialized AdaptiveEmotionLoss for 80%+ accuracy
        from emotion_net.training.losses import AdaptiveEmotionLoss
        print("Using Adaptive Emotion Loss for 80%+ accuracy training")
        return AdaptiveEmotionLoss(
            num_classes=num_classes,
            embedding_size=1024,  # Should match model's embedding size
            label_smoothing=label_smoothing,
            max_epochs=300  # Maximum epochs for adaptive weighting
        )
    else:
        print(f"Warning: Unknown loss type '{loss_type}', falling back to cross_entropy")
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

# Create optimizer
def create_optimizer(model, learning_rate, optimizer_type='adam', weight_decay=0.0):
    """Create optimizer based on configuration."""
    if optimizer_type == 'adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == 'sgd':
        return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Available options: 'adam', 'adamw', 'sgd'")

# Create learning rate scheduler with warmup
def create_scheduler(optimizer, scheduler_type, num_epochs, steps_per_epoch, warmup_epochs=0):
    """Create learning rate scheduler with optional warmup."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Create a composite scheduler with warmup if requested
    if warmup_epochs > 0:
        print(f"Using {warmup_epochs} epochs of warmup")
        # First create the main scheduler
        if scheduler_type == 'cosine_annealing':
            main_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
        elif scheduler_type == 'cosine_warmup_restarts':
            main_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=steps_per_epoch * 10,  # Restart every 10 epochs
                T_mult=1,  # Don't increase restart period
                eta_min=1e-6
            )
        elif scheduler_type == 'one_cycle':
            # OneCycleLR already includes warmup, so we'll use it directly
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=warmup_epochs / num_epochs,  # Percentage of steps for warmup
                div_factor=25.0,  # Initial lr = max_lr / div_factor
                final_div_factor=10000.0  # Final lr = max_lr / final_div_factor
            )
        else:
            raise ValueError(f"Unsupported scheduler type with warmup: {scheduler_type}")
        
        # Create a linear warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Start with 10% of the target lr
            end_factor=1.0,    # End with the target lr
            total_iters=warmup_steps
        )
        
        # Combine the schedulers using a chain wrapper
        from torch.optim.lr_scheduler import SequentialLR
        schedulers = [warmup_scheduler, main_scheduler]
        milestones = [warmup_steps]
        
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    
    # If no warmup, create a simple scheduler
    if scheduler_type == 'cosine_annealing':
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif scheduler_type == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=0.3,  # Default warmup percentage
            div_factor=25.0,
            final_div_factor=10000.0
        )
    elif scheduler_type == 'cosine_warmup_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch * 10,  # Restart every 10 epochs
            T_mult=1,
            eta_min=1e-6
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def setup_training(model, train_labels, num_classes, device, learning_rate=1e-4, 
                 num_epochs=50, steps_per_epoch=None, loss_type='cross_entropy', 
                 focal_loss_gamma=2.0, label_smoothing_factor=0.1, scheduler_type='one_cycle',
                 dataset_name=None, weight_decay=0.0, optimizer_type='adam', warmup_epochs=0, 
                 use_class_weights=False, sad_class_weight=1.0, triplet_margin=0.3, custom_loss_fn=None):
    """Set up model, criterion, optimizer, and scheduler for training."""
    model = model.to(device)
    
    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        normalized_weights, class_counts = calculate_class_weights(train_labels, num_classes, class_weight_factor=0.5)
        # Log class counts and weights for debugging
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {normalized_weights}")
        class_weights = normalized_weights
    
    # Create criterion
    criterion = create_criterion(
        loss_type=loss_type,
        num_classes=num_classes,
        device=device,
        class_weights=class_weights,
        focal_gamma=focal_loss_gamma,
        label_smoothing=label_smoothing_factor,
        sad_class_weight=sad_class_weight,
        custom_loss_fn=custom_loss_fn,
        triplet_margin=triplet_margin,
        dataset_name=dataset_name
    )
    
    # Create optimizer with decoupled weight decay
    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay
    )
    
    # Create scheduler if steps_per_epoch is provided
    scheduler = None
    if steps_per_epoch is not None:
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs
        )
    
    return model, criterion, optimizer, scheduler

# Save model checkpoint
def save_model(model, optimizer, epoch, save_path, scheduler=None, best_metrics=None):
    """Save model checkpoint to disk."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if best_metrics is not None:
        checkpoint['best_metrics'] = best_metrics
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}") 