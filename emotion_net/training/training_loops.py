"""Training and validation loops for emotion recognition model."""

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
from emotion_net.training.metrics import calculate_metrics
import os
import torch.nn.functional as F

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # If alpha not positive, return original data
    if alpha <= 0:
        return x, y, y, 1.0

    # Verify x is a tensor before trying to use size()
    if not isinstance(x, torch.Tensor):
        print(f"ERROR: mixup_data received {type(x)} instead of tensor. Returning original data.")
        return x, y, y, 1.0
    
    try:
        # Get batch size safely
        batch_size = x.size(0)  # Using size(0) instead of size()[0] for robustness
        
        # Generate mixup coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Create permutation for mixing
        index = torch.randperm(batch_size).to(device)
        
        # Mix data
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    except Exception as e:
        print(f"ERROR in mixup_data: {str(e)}. Returning original data.")
        return x, y, y, 1.0

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
    # If lam is 1.0, no mixing happened, use standard loss calculation
    if lam == 1.0:
        return criterion(pred, y_a)
    
    # Check if this is a SOTAEmotionLoss by looking at criterion's class name
    is_sota_loss = 'SOTAEmotionLoss' in criterion.__class__.__name__
    
    try:
            
        if isinstance(pred, (list, tuple)) and len(pred) >= 2:
            # Handle SOTAEmotionLoss specially
            if is_sota_loss:
                try:
                    # For SOTAEmotionLoss that expects a tuple, we don't try to unpack
                    # Instead pass the tuple directly to the loss function
                    loss_a = criterion(pred, y_a)
                    loss_b = criterion(pred, y_b)
                    return lam * loss_a + (1 - lam) * loss_b
                except Exception as e:
                    # If that fails, fall back to using just the main output
                    print(f"Warning: Error in mixup SOTAEmotionLoss: {e}. Falling back to standard approach.")
                    return lam * criterion(pred[0], y_a) + (1 - lam) * criterion(pred[0], y_b)
            else:
                # For other loss types with list outputs, use the first element
                return lam * criterion(pred[0], y_a) + (1 - lam) * criterion(pred[0], y_b)
        elif isinstance(pred, dict):
            # For models that return a dictionary of outputs
            # Apply mixup to the main logits
            key = 'logits' if 'logits' in pred else 'direct_logits'
            return lam * criterion(pred[key], y_a) + (1 - lam) * criterion(pred[key], y_b)
        else:
            # Standard case for a single tensor output
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    except Exception as e:
        print(f"Error in mixup_criterion: {str(e)}. Using standard loss.")
        return criterion(pred, y_a)

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, ema=None, mixup_alpha=0.0, gradient_accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    nan_count = 0  # Track NaN count
    total_batches = len(train_loader)
    
    # Initialize running loss avg for dynamic gradient clipping
    running_loss_avg = torch.tensor(1.0, device=device)
    running_grad_norm = torch.tensor(1.0, device=device)
    
    # Determine initial gradient clip value (more conservative)
    grad_clip_value = 0.5
    
    # Set up gradient accumulation
    optimizer.zero_grad()
    accumulated_batches = 0
    
    # Training loop with progress bar
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Basic sanity checks on input data
        if not isinstance(images, torch.Tensor):
            print(f"WARNING: Input images are not a tensor but {type(images)}. Skipping batch.")
            continue
        
        # Apply Mixup with reduced alpha if many NaNs are detected
        effective_mixup = 0.0 if nan_count > total_batches * 0.03 else mixup_alpha
        
        # Completely disable mixup if we're seeing too many NaNs
        if nan_count > total_batches * 0.05:
            effective_mixup = 0.0
        
        # Apply mixup if enabled (which is now always disabled due to the force disable above)
        if effective_mixup > 0:
            images, targets_a, targets_b, lam = mixup_data(images, labels, effective_mixup, device)
            use_mixup = True
        else:
            targets_a, targets_b, lam = labels, labels, 1.0
            use_mixup = False
        
        # Forward pass with mixed precision and gradient checkpointing
        with autocast():
            try:
                # Set gradient checkpointing for memory efficiency
                if hasattr(model, 'set_grad_checkpointing'):
                    model.set_grad_checkpointing(True)
                    
                # Forward pass
                outputs = model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    # Model returns a dictionary with logits, aux_logits, and features
                    logits = outputs['logits']
                    aux_logits = outputs.get('aux_logits', None)
                elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                    # Model returns a tuple/list of (logits, aux_logits, features)
                    logits = outputs[0]
                    aux_logits = outputs[1] if len(outputs) > 1 else None
                else:
                    # Model returns just the logits
                    logits = outputs
                    aux_logits = None
                
                # Check for NaN in outputs and replace with zeros
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Calculate loss
                if use_mixup:
                    # For SOTA Loss that expects a tuple, pass the full outputs
                    try:
                        if 'SOTAEmotionLoss' in criterion.__class__.__name__ and isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                            if aux_logits is not None:
                                aux_loss = mixup_criterion(criterion, aux_logits, targets_a, targets_b, lam)
                                loss = loss + 0.4 * aux_loss  # Add auxiliary loss with weight
                    except Exception as e:
                        print(f"Error in mixup loss calculation: {e}")
                        # Fallback to non-mixup loss if mixup fails
                        loss = criterion(logits, labels)
                        if aux_logits is not None:
                            aux_loss = criterion(aux_logits, labels)
                            loss = loss + 0.4 * aux_loss
                else:
                    # For SOTA Loss that expects a tuple, pass the full outputs
                    try:
                        if 'SOTAEmotionLoss' in criterion.__class__.__name__ and isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                            loss = criterion(outputs, labels)
                        else:
                            loss = criterion(logits, labels)
                            if aux_logits is not None:
                                aux_loss = criterion(aux_logits, labels)
                                loss = loss + 0.4 * aux_loss  # Add auxiliary loss with weight
                    except Exception as e:
                        print(f"Error in loss calculation: {e}")
                        # Fallback to simpler loss if complex loss fails
                        loss = F.cross_entropy(logits, labels)
                
                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
            except Exception as e:
                print(f"ERROR in forward/loss pass: {str(e)}")
                nan_count += 1
                continue
        
        # Check for NaN in loss and skip this batch if found
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: NaN or Inf detected in loss at batch {batch_idx}. Skipping batch.")
            nan_count += 1
            
            # If too many NaNs, reduce the learning rate
            if nan_count > total_batches * 0.1:  # If >10% batches have NaNs
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5  # Reduce learning rate by 50%
                print(f"WARNING: Too many NaNs detected. Reducing learning rate to {optimizer.param_groups[0]['lr']:.8f}")
                nan_count = 0  # Reset counter after reducing
            
            continue
            
        # Update running loss average for adaptive clipping
        with torch.no_grad():
            if torch.isfinite(loss):
                running_loss_avg = 0.95 * running_loss_avg + 0.05 * loss.detach() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
        
        # Backward pass
        loss.backward()
        
        # Increment accumulated batches counter
        accumulated_batches += 1
        
        # Only update weights after accumulating enough gradients or at the end of the loader
        if accumulated_batches >= gradient_accumulation_steps or batch_idx == len(train_loader) - 1:
            # Dynamically adjust gradient clipping based on loss value and NaN count
            # Scale down if we have NaNs, scale up if loss is below average
            if nan_count > 0:
                # More aggressive clipping if we've seen NaNs
                grad_clip_value = min(0.1, grad_clip_value * 0.9)
            elif loss.item() * gradient_accumulation_steps < running_loss_avg.item() * 0.8:
                # Gradually relax clipping if we're doing well
                grad_clip_value = min(1.0, grad_clip_value * 1.01)
            
            # Compute grad norm before clipping for statistics
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Update running gradient norm
            running_grad_norm = 0.95 * running_grad_norm + 0.05 * torch.tensor(total_norm, device=device)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            
            # Check for NaN in gradients and skip update if found
            skip_update = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"WARNING: NaN or Inf detected in gradients at batch {batch_idx}. Zeroing bad gradients.")
                        param.grad.data.zero_()
                        nan_count += 1
                        
                        # Only skip if too many NaN gradients
                        if nan_count > total_batches * 0.2:
                            skip_update = True
            
            if not skip_update:
                optimizer.step()
                
                # Step the scheduler if it's OneCycleLR (batch-wise)
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                # Update EMA if provided
                if ema:
                    ema.update()
            
            # Reset gradients and accumulation counter
            optimizer.zero_grad()
            accumulated_batches = 0
        
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
        if torch.isfinite(loss):
            total_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
        
        # Update progress bar with safe loss value
        loss_value = loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1) if torch.isfinite(loss) else float('nan')
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}', 
            'lr': f'{optimizer.param_groups[0]["lr"]:.8f}',
            'grad_clip': f'{grad_clip_value:.3f}'
        })
    
    print(f"NaN batches: {nan_count}/{total_batches} ({nan_count/total_batches*100:.2f}%)")
    
    # Calculate epoch metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        train_loader.dataset.classes
    )
    
    # Add loss to metrics
    valid_batches = total_batches - nan_count
    metrics['loss'] = total_loss / max(1, valid_batches)
    
    return metrics

def validate(model, val_loader, criterion, device, labels):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # Model returns a dictionary with logits, aux_logits, and features
                logits = outputs['logits']
                aux_logits = outputs.get('aux_logits', None)
            elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                # Model returns a tuple/list of (logits, aux_logits, features)
                logits = outputs[0]
                aux_logits = outputs[1] if len(outputs) > 1 else None
            else:
                # Model returns just the logits
                logits = outputs
                aux_logits = None
            
            # Calculate loss
            try:
                if 'SOTAEmotionLoss' in criterion.__class__.__name__ and isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                    # For SOTA Loss that expects a tuple, pass the full outputs
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(logits, targets)
                    if aux_logits is not None:
                        aux_loss = criterion(aux_logits, targets)
                        loss = loss + 0.4 * aux_loss  # Add auxiliary loss with weight
            except Exception as e:
                print(f"Error in validation loss calculation: {e}")
                # Fallback to using cross entropy if the loss function fails
                loss = F.cross_entropy(logits, targets)
            
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(logits.data, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        labels=labels
    )
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    # Add loss to metrics dict
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics 