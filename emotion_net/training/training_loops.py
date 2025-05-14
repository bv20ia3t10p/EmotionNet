"""Training and validation loops for emotion recognition model."""

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
from emotion_net.training.metrics import calculate_metrics
import os

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Apply mixup to criterion with mixed labels.
    
    Args:
        criterion: Loss function
        pred: Model predictions (can be tensor, list, or dict)
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixup parameter
        
    Returns:
        Mixed loss value
    """
    if isinstance(pred, list) and len(pred) > 1:
        # Handle SOTAEmotionLoss specially - apply mixup to each element
        # This properly mixes both the main and auxiliary outputs
        main_loss = lam * criterion([pred[0], pred[1]], y_a) + (1 - lam) * criterion([pred[0], pred[1]], y_b)
        return main_loss
    elif isinstance(pred, dict):
        # For models that return a dictionary of outputs
        # Apply mixup to the main logits
        key = 'direct_logits' if 'direct_logits' in pred else 'group_logits'
        return lam * criterion(pred[key], y_a) + (1 - lam) * criterion(pred[key], y_b)
    else:
        # Standard case for a single tensor output
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, ema=None, mixup_alpha=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    nan_count = 0  # Track NaN count
    total_batches = len(train_loader)
    
    # Training loop with progress bar
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply Mixup with reduced alpha if many NaNs are detected
        effective_mixup = mixup_alpha
        if nan_count > total_batches * 0.05:  # If >5% batches have NaNs
            effective_mixup = 0.0  # Disable mixup
        
        images, targets_a, targets_b, lam = mixup_data(images, labels, effective_mixup, device)
        
        # Forward pass with mixed precision
        with autocast():
            # Handle different model output formats
            outputs = model(images)
            
            # For SOTA ResEmote models with auxiliary outputs
            if isinstance(outputs, list) and len(outputs) > 1:
                logits = outputs[0]  # Take main outputs for prediction
            # Expert model returns a dictionary
            elif isinstance(outputs, dict):
                # Use direct_logits for classification if available, otherwise first item
                if 'direct_logits' in outputs:
                    logits = outputs['direct_logits']
                else:
                    # Fall back to group_logits if direct_logits not available
                    logits = outputs['group_logits']
            else:
                # Handle case where model returns a tuple of (outputs, _)
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    logits = outputs[0]
                else:
                    # Just use the outputs directly
                    logits = outputs
            
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        # Check for NaN in loss and skip this batch if found
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: NaN or Inf detected in loss at batch {batch_idx}. Skipping batch.")
            nan_count += 1
            
            # If too many NaNs, reduce the learning rate
            if nan_count > total_batches * 0.1:  # If >10% batches have NaNs
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8  # Reduce learning rate by 20%
                print(f"WARNING: Too many NaNs detected. Reducing learning rate to {optimizer.param_groups[0]['lr']:.8f}")
                nan_count = 0  # Reset counter after reducing
            
            continue
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Very aggressive gradient clipping - dynamically adjust based on NaN frequency
        max_norm = 0.5  # Default
        if nan_count > 5:
            max_norm = 0.1  # More aggressive if NaNs are common
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        # Check for NaN in gradients and skip update if found
        skip_update = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"WARNING: NaN or Inf detected in gradients at batch {batch_idx}. Skipping parameter update.")
                    skip_update = True
                    nan_count += 1
                    break
        
        if not skip_update:
            optimizer.step()
            
            # Step the scheduler if it's OneCycleLR (batch-wise)
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            # Update EMA if provided
            if ema:
                ema.update()
        
        # Store predictions for metrics (use original labels before mixup for F1 calc)
        with torch.no_grad():
            # Handle NaN values in logits
            if torch.isnan(logits).any():
                # Replace NaN with very negative number to ensure 0 probability after softmax
                logits = torch.nan_to_num(logits, nan=-1e9)
                
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar with safe loss value
        loss_value = loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else float('nan')
        pbar.set_postfix({'loss': f'{loss_value:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.8f}'})
    
    print(f"NaN batches: {nan_count}/{total_batches} ({nan_count/total_batches*100:.2f}%)")
    
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
        for images, labels_batch in pbar:
            images, labels_batch = images.to(device), labels_batch.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # For SOTA ResEmote models with auxiliary outputs
            if isinstance(outputs, list) and len(outputs) > 1:
                logits = outputs[0]  # Take main outputs for prediction
            # Expert model returns a dictionary
            elif isinstance(outputs, dict):
                # Use direct_logits for classification if available, otherwise first item
                if 'direct_logits' in outputs:
                    logits = outputs['direct_logits']
                else:
                    # Fall back to group_logits if direct_logits not available
                    logits = outputs['group_logits']
            else:
                # Handle case where model returns a tuple of (outputs, _)
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    logits = outputs[0]
                else:
                    # Just use the outputs directly
                    logits = outputs
            
            with autocast():
                loss = criterion(outputs, labels_batch)
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
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