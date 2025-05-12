"""Model management utilities for emotion recognition training."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from emotion_net.config.constants import CHECKPOINT_DIR

def setup_training(model, train_labels, num_classes, device, learning_rate=1e-4):
    """Setup model, criterion, optimizer and scheduler for training."""
    # Calculate class weights for weighted loss
    class_weights = calculate_class_weights(train_labels, num_classes)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create criterion with class weights
    criterion = create_criterion(class_weights)
    
    # Create optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return model, criterion, optimizer, scheduler

def calculate_class_weights(labels, num_classes):
    """Calculate class weights for weighted loss."""
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for label in labels:
        class_counts[label] += 1
    
    # Calculate weights (inverse of frequency)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Print class distribution
    print("\nClass Distribution:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"Class {i}: {int(count)} samples, weight: {weight:.2f}")
    
    return class_weights

def create_criterion(class_weights):
    """Create weighted cross entropy loss criterion."""
    return nn.CrossEntropyLoss(weight=class_weights)

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