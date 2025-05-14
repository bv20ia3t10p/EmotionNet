"""Model management utilities for emotion recognition training."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Import our custom loss
from emotion_net.models.sota_resemote import SOTAEmotionLoss

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
        from emotion_net.training.losses import FocalLoss
        return FocalLoss(alpha=weight, gamma=focal_gamma, reduction='mean')
    elif loss_type == 'hybrid':
        from emotion_net.training.losses import HybridLoss
        return HybridLoss(alpha=weight, gamma=focal_gamma, label_smoothing=label_smoothing)
    elif loss_type == 'sota_emotion':
        # Use our custom SOTAEmotionLoss
        return SOTAEmotionLoss(
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            aux_weight=0.4
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# Function to set up training components
def setup_training(model, train_labels, num_classes, device, 
                  learning_rate=1e-4, num_epochs=50, steps_per_epoch=None,
                  loss_type='cross_entropy', focal_loss_gamma=2.0, label_smoothing_factor=0.1,
                  dataset_name=None, scheduler_type='cosine_annealing',
                  weight_decay=0.0001, optimizer_type='adam', warmup_epochs=0,
                  use_class_weights=False, sad_class_weight=1.0, triplet_margin=0.3,
                  custom_loss_fn=None):
    """Set up training components: model, criterion, optimizer, scheduler."""
    # Initialize weights for imbalanced classes if requested
    if use_class_weights and train_labels:
        class_weights, _ = calculate_class_weights(train_labels, num_classes)
    else:
        class_weights = None
    
    # Create loss function
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
    
    # Create optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Create learning rate scheduler
    if scheduler_type == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'one_cycle' and steps_per_epoch:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
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