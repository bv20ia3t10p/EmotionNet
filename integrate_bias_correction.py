#!/usr/bin/env python3
# Integration of bias correction with existing emotion recognition models

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy
from torch.optim.lr_scheduler import StepLR  # Add proper import for learning rate scheduler

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

# Import from existing modules
try:
    from balanced_training import (
        EnhancedResEmoteNet, EmotionDataset, load_data, EMOTIONS,
        create_training_plots, create_confusion_matrix
    )
    from bias_correction import BiasCorrector
    print("Successfully imported from balanced_training and bias_correction")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure balanced_training.py and bias_correction.py are in the current directory.")
    sys.exit(1)

def integrate_bias_correction(
    model_path=None,
    data_dir="./extracted/emotion/train",
    test_dir="./extracted/emotion/test",
    model_dir="./models",
    backbone="resnet18",
    batch_size=32,
    epochs=30,
    patience=7,
    learning_rate=0.0001,
    correction_strength=3.0,  # Higher value for stronger correction
    correction_frequency=1,
    save_path="integrated_model.pth"
):
    """
    Integrate bias correction with existing emotion recognition training pipeline.
    
    This function will:
    1. Load or create a model
    2. Setup training with the existing balanced_training pipeline
    3. Apply bias correction during training
    4. Evaluate and save the model
    
    Args:
        model_path: Path to existing model to continue training (optional)
        data_dir: Path to training data
        test_dir: Path to test data
        model_dir: Directory to save models
        backbone: Model backbone architecture
        batch_size: Batch size for training
        epochs: Number of training epochs
        patience: Early stopping patience
        learning_rate: Base learning rate
        correction_strength: Strength of bias correction (higher = more aggressive)
        correction_frequency: How often to update bias weights (in epochs)
        save_path: Path to save final model
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    from torchvision import transforms
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load training and test data
    print(f"Loading training data from {data_dir}...")
    train_paths, train_labels = load_data(data_dir)
    
    print(f"Loading test data from {test_dir}...")
    test_paths, test_labels = load_data(test_dir)
    
    # Create datasets with face-preserving augmentations
    train_dataset = EmotionDataset(
        train_paths, train_labels, transform=transform, 
        augmentation_level=2, target_classes=[0, 1, 6]  # Targeted augmentation for underrepresented classes
    )
    test_dataset = EmotionDataset(
        test_paths, test_labels, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    try:
        model = EnhancedResEmoteNet(
            num_classes=7, backbone=backbone,
            dropout_rate=0.5  # Increase dropout to prevent overfitting
        )
    except Exception as e:
        print(f"Error creating model with specified parameters: {e}")
        print("Trying with minimal parameters...")
        model = EnhancedResEmoteNet(num_classes=7)
    
    # Move model to device
    model = model.to(device)
    
    # Load pretrained model if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model instead.")
    
    # Calculate initial class weights based on class distribution
    class_counts = np.bincount(train_labels, minlength=len(EMOTIONS))
    total_samples = np.sum(class_counts)
    print("Class distribution in training data:")
    for i, count in enumerate(class_counts):
        print(f"  {EMOTIONS[i]}: {count} samples ({100*count/total_samples:.2f}%)")
    
    # Initial class weights based on inverse frequency with enhancement for problem classes
    initial_weights = np.ones(len(EMOTIONS), dtype=np.float32)
    class_freq = class_counts / total_samples
    
    # Set weights inversely proportional to frequency
    for i in range(len(EMOTIONS)):
        if class_counts[i] > 0:
            initial_weights[i] = 1.0 / (class_freq[i] + 1e-6)
        else:
            initial_weights[i] = 1.0
    
    # Normalize weights
    initial_weights = initial_weights / np.mean(initial_weights)
    
    # Extra boost for classes that typically have low accuracy
    boost_classes = {0: 4.0, 6: 4.0}  # Angry and Neutral
    for cls, boost in boost_classes.items():
        initial_weights[cls] *= boost
    
    # Convert to tensor
    initial_weights = torch.tensor(initial_weights, dtype=torch.float32, device=device)
    
    print("Initial class weights:")
    for i, emotion in EMOTIONS.items():
        print(f"  {emotion}: {initial_weights[i].item():.4f}")
    
    # Create optimizer with different learning rates for different parts of the model
    # Feature extractor: low learning rate
    # Classifier: high learning rate
    params = []
    for name, param in model.named_parameters():
        lr_multiplier = 1.0
        
        # Lower learning rate for backbone/feature extractor
        if any(x in name for x in ['backbone', 'feature_extractor', 'conv', 'stem', 'stage']):
            lr_multiplier = 0.1
        # Standard learning rate for middle layers
        elif any(x in name for x in ['fpn', 'middle', 'enhance', 'attn']):
            lr_multiplier = 0.5
        # Higher learning rate for classifier
        elif any(x in name for x in ['classifier', 'fc', 'linear', 'out', 'emotion_classifier']):
            lr_multiplier = 10.0
            
        params.append({
            'params': param,
            'lr': learning_rate * lr_multiplier
        })
    
    optimizer = torch.optim.Adam(params)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Use directly from torch.optim.lr_scheduler
    
    # Initialize bias corrector
    bias_corrector = BiasCorrector(
        model, train_loader, test_loader, device,
        num_classes=len(EMOTIONS),
        initial_weights=initial_weights,
        correction_strength=correction_strength,
        min_weight=1.0,
        max_weight=30.0,  # Allow higher max weights
        smooth_factor=0.6  # Less smoothing for more aggressive correction
    )
    
    # Get initial criterion with bias correction
    criterion = bias_corrector.get_criterion()
    
    # Train with bias correction
    print("\n" + "="*50)
    print("Starting training with bias correction")
    print("="*50)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Track metrics for plotting
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    f1_scores = []
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_preds = []
        train_labels = []
        
        # Training progress bar
        pbar = tqdm(train_loader, desc=f"Training")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            
            # Store for per-class metrics
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), 
                            acc=torch.sum(preds == labels).double().item() / inputs.size(0))
        
        # Calculate epoch training statistics
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)
        
        # Print per-class training accuracy
        print("Per-class training accuracy:")
        for class_idx, class_name in EMOTIONS.items():
            class_mask = np.array(train_labels) == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(np.array(train_preds)[class_mask] == class_idx) * 100
                print(f"  {class_name}: {class_acc:.2f}%")
            else:
                print(f"  {class_name}: No samples")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        valid_preds = []
        valid_labels = []
        
        # Validation progress bar
        pbar = tqdm(test_loader, desc=f"Validating")
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                
                # Store for per-class metrics
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch validation statistics
        valid_loss = running_loss / len(test_loader.dataset)
        valid_acc = running_corrects / len(test_loader.dataset)
        
        # Store metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        # Calculate per-class F1 scores
        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(valid_labels, valid_preds, average='macro')
        f1_scores.append(macro_f1)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc*100:.2f}%, F1: {macro_f1*100:.2f}%")
        
        # Print per-class prediction counts
        pred_counts = np.bincount(valid_preds, minlength=len(EMOTIONS))
        print("\nPrediction distribution:")
        for idx, count in enumerate(pred_counts):
            pct = 100 * count / len(valid_preds)
            print(f"  {EMOTIONS[idx]}: {count} ({pct:.1f}%)")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for class_idx in range(len(EMOTIONS)):
            class_mask = np.array(valid_labels) == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(np.array(valid_preds)[class_mask] == class_idx) * 100
                print(f"  {EMOTIONS[class_idx]}: {class_acc:.2f}%")
            else:
                print(f"  {EMOTIONS[class_idx]}: No samples")
        
        # Print classification report
        print("\nClassification Report:")
        target_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]
        print(classification_report(valid_labels, valid_preds, target_names=target_names))
        
        # Check for bias correction update
        if epoch % correction_frequency == 0:
            print("\nUpdating bias correction weights...")
            bias_corrector.update_weights()
            criterion = bias_corrector.get_criterion()
            bias_corrector.print_current_status()
        
        # Check if this is the best model based on F1 score
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"New best model! F1: {best_f1*100:.2f}%")
            
            # Save the best model
            torch.save(model.state_dict(), os.path.join(model_dir, save_path))
            print(f"Saved best model to {os.path.join(model_dir, save_path)}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update learning rate
        scheduler.step()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Loaded best model from epoch {best_epoch+1} with F1: {best_f1*100:.2f}%")
    
    # Create final visualization
    bias_corrector.visualize_bias_correction(os.path.join(model_dir, "bias_correction.png"))
    
    # Create training curves
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot([acc * 100 for acc in train_accs], label='Train Accuracy')
    plt.plot([acc * 100 for acc in valid_accs], label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 score plot
    plt.subplot(1, 3, 3)
    plt.plot([f1 * 100 for f1 in f1_scores], label='Macro F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (%)')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'integrated_training_curves.png'))
    print(f"Training curves saved to {os.path.join(model_dir, 'integrated_training_curves.png')}")
    
    # Create confusion matrix
    create_confusion_matrix(model, test_loader, device)
    
    return model, bias_corrector

def main():
    parser = argparse.ArgumentParser(description='Integrate bias correction with emotion recognition training')
    parser.add_argument('--data_dir', type=str, default='./extracted/emotion/train', help='Path to training data')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test data')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--model_path', type=str, default=None, help='Path to existing model to continue training')
    parser.add_argument('--save_path', type=str, default='integrated_model.pth', help='Filename to save model')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                      choices=['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                      help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--correction_strength', type=float, default=3.0, 
                      help='Bias correction strength (higher = more aggressive)')
    parser.add_argument('--correction_frequency', type=int, default=1, 
                      help='How often to update bias weights (in epochs)')
    
    args = parser.parse_args()
    
    # Run the integrated training
    model, bias_corrector = integrate_bias_correction(
        model_path=args.model_path,
        data_dir=args.data_dir,
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        correction_strength=args.correction_strength,
        correction_frequency=args.correction_frequency,
        save_path=args.save_path
    )
    
    print("\nTraining with bias correction completed!")

if __name__ == "__main__":
    main() 