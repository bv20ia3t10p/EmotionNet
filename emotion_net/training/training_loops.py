"""Training and validation loops for emotion recognition model."""

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
from emotion_net.training.metrics import calculate_metrics
import os

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Training loop with progress bar
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        train_loader.dataset.classes
    )
    
    # Add loss to metrics
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def validate(model, val_loader, criterion, device, labels):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Validation loop with progress bar
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate validation metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        val_loader.dataset.classes
    )
    
    # Add loss to metrics
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics 