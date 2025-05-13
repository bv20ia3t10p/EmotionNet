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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, ema=None, mixup_alpha=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Training loop with progress bar
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha, device)
        
        # Forward pass with mixed precision
        with autocast():
            # Handle different model output formats
            outputs = model(images)
            
            # Expert model returns a dictionary
            if isinstance(outputs, dict):
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
            
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step the scheduler if it's OneCycleLR (batch-wise)
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Update EMA if provided
        if ema:
            ema.update()
        
        # Store predictions for metrics (use original labels before mixup for F1 calc)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update metrics
        total_loss += loss.item()
        
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
        for images, labels_batch in pbar:
            images, labels_batch = images.to(device), labels_batch.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Expert model returns a dictionary
            if isinstance(outputs, dict):
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
                loss = criterion(logits, labels_batch)
            
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