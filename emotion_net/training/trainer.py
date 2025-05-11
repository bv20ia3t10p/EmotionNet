"""Training utilities for the emotion recognition model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

from emotion_net.models.ema import EMA
from emotion_net.utils.visualization import plot_confusion_matrix

def train(model, train_loader, val_loader, epochs, patience, learning_rate, device, 
          use_amp=True, use_ema=True, save_path="model.pth", class_weights_boost=1.5):
    """Train the model with advanced techniques for high accuracy."""
    # Setup EMA if enabled
    ema_model = EMA(model, decay=0.999) if use_ema else None
    
    # Calculate class weights
    class_counts = np.array([0] * len(train_loader.dataset.dataset.labels))
    for _, label in train_loader.dataset:
        class_counts[label] += 1
    
    # Calculate class weights inversely proportional to count
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    # Boost angry and neutral classes further
    class_weights[0] *= class_weights_boost  # angry
    class_weights[6] *= class_weights_boost  # neutral
    
    print("Class weights:", class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different learning rates for different parts of the model
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained backbone
        {'params': other_params, 'lr': learning_rate}
    ], weight_decay=1e-4)
    
    # Learning rate scheduler with cosine annealing and warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    # Setup AMP if enabled
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            if use_amp:
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
            if use_ema:
                ema_model.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (i + 1)
            acc = 100 * correct / total
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{acc:.2f}%"
            })
        
        scheduler.step()
        
        # Validation phase
        if use_ema:
            ema_model.apply_shadow()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate per-class accuracies
        class_correct = [0] * len(train_loader.dataset.dataset.labels)
        class_total = [0] * len(train_loader.dataset.dataset.labels)
        
        for pred, target in zip(all_preds, all_targets):
            class_correct[target] += int(pred == target)
            class_total[target] += 1
        
        # Print validation results
        val_acc = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
        
        # Print per-class accuracies
        for i in range(len(class_correct)):
            if class_total[i] > 0:
                print(f"Class {i}: {100*class_correct[i]/class_total[i]:.2f}%")
            else:
                print(f"Class {i}: N/A (no samples)")
        
        # Save best model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0
            
            # Save model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if use_ema:
                torch.save(model.state_dict(), save_path)
                ema_model.restore()
            else:
                torch.save(model.state_dict(), save_path)
                
            # Generate and save confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            cm_path = save_path.replace(".pth", "_confusion_matrix.png")
            plot_confusion_matrix(cm, cm_path)
            
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs. Best accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
            break
    
    return best_val_acc 