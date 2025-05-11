"""Main training script for the emotion recognition model."""

import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

from emotion_net.training.metrics import (
    calculate_metrics, save_training_history, plot_training_history
)
from emotion_net.training.model_manager import (
    setup_training, calculate_class_weights, create_criterion, save_model
)

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, ema_model=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with AMP if enabled
        if scaler is not None:
            with autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward and optimize with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Update EMA if enabled
        if ema_model is not None:
            ema_model.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss = running_loss / (i + 1)
        progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
    
    return running_loss / len(train_loader), all_preds, all_targets

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    return val_loss / len(val_loader), all_preds, all_targets

def train(model, train_loader, val_loader, epochs, patience, learning_rate, device, 
          use_amp=True, use_ema=True, save_path="model.pth", class_weights_boost=1.5):
    """Train the model with advanced techniques for high accuracy."""
    # Setup training components
    ema_model, scaler, optimizer, scheduler = setup_training(
        model, learning_rate, device, use_amp, use_ema
    )
    
    # Calculate class weights and create criterion
    class_weights = calculate_class_weights(train_loader, class_weights_boost, device)
    criterion = create_criterion(class_weights, device)
    
    # Create save directory
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve_count = 0
    
    # Initialize metrics history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ema_model
        )
        
        # Calculate training metrics
        train_cm, train_report, train_macro_f1, train_weighted_f1 = calculate_metrics(
            train_preds, train_targets, phase="train", save_dir=save_dir
        )
        
        # Validation phase
        if use_ema:
            ema_model.apply_shadow()
        
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        # Calculate validation metrics
        val_cm, val_report, val_macro_f1, val_weighted_f1 = calculate_metrics(
            val_preds, val_targets, phase="val", save_dir=save_dir
        )
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_report['accuracy'])
        history['train_f1'].append(train_macro_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_report['accuracy'])
        history['val_f1'].append(val_macro_f1)
        
        # Save best model based on validation F1 score
        if val_macro_f1 > best_val_f1:
            print(f"Validation F1 improved from {best_val_f1:.4f} to {val_macro_f1:.4f}")
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_count = 0
            
            # Save model and metrics
            save_model(model, save_path, use_ema, ema_model)
            save_training_history(history, save_dir)
            plot_training_history(history, save_dir)
            
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs. Best F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
            break
    
    return best_val_f1 