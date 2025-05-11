#!/usr/bin/env python3
# Automatic Bias Detection and Correction for Emotion Recognition

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import copy
from tqdm import tqdm
import sys

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

# Try to import necessary modules
try:
    from balanced_training import EnhancedResEmoteNet, EmotionDataset, load_data, EMOTIONS
    print("Successfully imported classes from balanced_training")
except ImportError:
    print("Error importing from balanced_training. Please make sure balanced_training.py is in the current directory.")
    exit(1)

class BiasCorrector:
    """
    Automatic bias detection and correction system for neural networks.
    Dynamically adjusts class weights during training to combat prediction bias.
    """
    
    def __init__(self, model, train_loader, valid_loader, device, 
                 num_classes=7, initial_weights=None, correction_strength=2.0,
                 min_weight=1.0, max_weight=20.0, smooth_factor=0.7):
        """
        Initialize the bias corrector.
        
        Args:
            model: Neural network model to correct
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            device: Compute device (cpu or cuda)
            num_classes: Number of classes
            initial_weights: Starting class weights (default: equal weights)
            correction_strength: How aggressively to correct bias (higher = more aggressive)
            min_weight: Minimum allowed class weight
            max_weight: Maximum allowed class weight
            smooth_factor: Smoothing factor for weight updates (0-1, higher = more smoothing)
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_classes = num_classes
        self.correction_strength = correction_strength
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.smooth_factor = smooth_factor
        
        # Initialize class weights
        if initial_weights is None:
            self.class_weights = torch.ones(num_classes, device=device)
        else:
            self.class_weights = initial_weights.clone().to(device)
        
        # Track prediction distributions and bias metrics
        self.pred_distributions = []
        self.weight_history = [self.class_weights.cpu().numpy().copy()]
        self.bias_metrics = []
        
    def detect_bias(self, data_loader):
        """
        Detect bias in model predictions and calculate correction factors.
        
        Returns:
            pred_dist: Class prediction distribution
            correction_factors: Weight correction factors for each class
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # Collect predictions
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                
                # Handle different model output formats
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate prediction distribution
        pred_counts = np.bincount(all_preds, minlength=self.num_classes)
        pred_dist = pred_counts / np.sum(pred_counts)
        
        # Calculate ideal distribution (based on validation set)
        label_counts = np.bincount(all_labels, minlength=self.num_classes)
        ideal_dist = label_counts / np.sum(label_counts)
        
        # Calculate class-specific accuracy
        class_accuracy = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            class_mask = np.array(all_labels) == i
            if np.sum(class_mask) > 0:
                class_accuracy[i] = np.mean(np.array(all_preds)[class_mask] == i)
        
        # Calculate correction factors
        # 1. Identify underrepresented classes in predictions
        dist_ratio = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if ideal_dist[i] > 0:
                dist_ratio[i] = min(pred_dist[i] / max(ideal_dist[i], 1e-5), 1.0)
            else:
                dist_ratio[i] = 1.0
        
        # 2. Adjust more aggressively for classes with poor accuracy
        correction_factors = (1.0 - class_accuracy) * (1.0 - dist_ratio) + 1.0
        
        # 3. Apply correction strength factor
        correction_factors = correction_factors ** self.correction_strength
        
        # Calculate bias metric (lower is better)
        # This measures how far the prediction distribution is from ideal
        bias_metric = np.sum(np.abs(pred_dist - ideal_dist)) / 2.0
        
        return pred_dist, correction_factors, bias_metric
        
    def update_weights(self):
        """
        Update class weights based on detected bias.
        
        Returns:
            new_weights: Updated class weights tensor
        """
        # Detect bias and get correction factors
        pred_dist, correction_factors, bias_metric = self.detect_bias(self.valid_loader)
        
        # Store metrics for analysis
        self.pred_distributions.append(pred_dist)
        self.bias_metrics.append(bias_metric)
        
        # Update weights with temporal smoothing
        old_weights = self.class_weights.cpu().numpy()
        raw_new_weights = old_weights * correction_factors
        
        # Apply smoothing
        smoothed_weights = self.smooth_factor * old_weights + (1 - self.smooth_factor) * raw_new_weights
        
        # Clip weights to reasonable range
        clipped_weights = np.clip(smoothed_weights, self.min_weight, self.max_weight)
        
        # Normalize weights to prevent overall loss scaling issues
        norm_factor = np.sum(old_weights) / np.sum(clipped_weights)
        normalized_weights = clipped_weights * norm_factor
        
        # Convert back to tensor
        new_weights = torch.tensor(normalized_weights, device=self.device, dtype=torch.float32)
        
        # Store weight history
        self.weight_history.append(new_weights.cpu().numpy().copy())
        
        # Update internal weights
        self.class_weights = new_weights
        
        return new_weights
    
    def get_criterion(self):
        """
        Get loss criterion with current class weights.
        
        Returns:
            criterion: Weighted CrossEntropyLoss with current weights
        """
        return nn.CrossEntropyLoss(weight=self.class_weights)
    
    def visualize_bias_correction(self, save_path="bias_correction.png"):
        """
        Generate visualization showing bias correction progress.
        
        Args:
            save_path: Path to save the visualization image
        """
        if len(self.pred_distributions) < 2:
            print("Not enough data for visualization. Run more epochs first.")
            return
        
        # Convert to numpy arrays
        pred_dists = np.array(self.pred_distributions)
        weight_history = np.array(self.weight_history)
        bias_metrics = np.array(self.bias_metrics)
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # Plot prediction distributions over time
        plt.subplot(3, 1, 1)
        for i in range(self.num_classes):
            plt.plot(pred_dists[:, i], label=f"Class {i} ({EMOTIONS[i]})")
        
        plt.xlabel("Correction Iteration")
        plt.ylabel("Prediction Proportion")
        plt.title("Evolution of Prediction Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot weight adjustments over time
        plt.subplot(3, 1, 2)
        for i in range(self.num_classes):
            plt.plot(weight_history[:, i], label=f"Class {i} ({EMOTIONS[i]})")
        
        plt.xlabel("Correction Iteration")
        plt.ylabel("Class Weight")
        plt.title("Evolution of Class Weights")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot bias metric over time
        plt.subplot(3, 1, 3)
        plt.plot(bias_metrics, 'r-', label="Bias Metric")
        
        plt.xlabel("Correction Iteration")
        plt.ylabel("Bias Metric")
        plt.title("Reduction in Prediction Bias")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Bias correction visualization saved to {save_path}")
    
    def print_current_status(self):
        """
        Print the current bias correction status.
        """
        if len(self.pred_distributions) == 0:
            print("No bias measurements available yet.")
            return
        
        # Get latest prediction distribution and bias metric
        pred_dist = self.pred_distributions[-1]
        bias_metric = self.bias_metrics[-1]
        
        print("\nCurrent Bias Status:")
        print(f"Bias Metric: {bias_metric:.4f} (Lower is better)")
        
        print("\nPrediction Distribution:")
        for i in range(self.num_classes):
            print(f"  {EMOTIONS[i]}: {pred_dist[i]*100:.2f}%")
        
        print("\nCurrent Class Weights:")
        for i in range(self.num_classes):
            print(f"  {EMOTIONS[i]}: {self.class_weights[i].item():.4f}")


def train_with_bias_correction(model, train_loader, valid_loader, optimizer, scheduler, 
                              device, num_epochs=50, early_stopping_patience=5,
                              correction_frequency=1, initial_weights=None,
                              correction_strength=2.0, save_path=None):
    """
    Train model with automatic bias correction.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        optimizer: Optimizer to use
        scheduler: Learning rate scheduler
        device: Compute device (cpu or cuda)
        num_epochs: Number of training epochs
        early_stopping_patience: Patience for early stopping
        correction_frequency: How often to update weights (in epochs)
        initial_weights: Initial class weights
        correction_strength: Bias correction strength
        save_path: Path to save the best model
        
    Returns:
        model: Trained model
        bias_corrector: BiasCorrector instance with correction history
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Initialize bias corrector
    bias_corrector = BiasCorrector(
        model, train_loader, valid_loader, device,
        num_classes=len(EMOTIONS),
        initial_weights=initial_weights,
        correction_strength=correction_strength
    )
    
    # Get initial criterion with bias correction
    criterion = bias_corrector.get_criterion()
    
    # Track metrics for plotting
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    f1_scores = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
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
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        valid_preds = []
        valid_labels = []
        
        # Validation progress bar
        pbar = tqdm(valid_loader, desc=f"Validating")
        
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
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_acc = running_corrects / len(valid_loader.dataset)
        
        # Store metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        # Calculate per-class F1 scores
        from sklearn.metrics import f1_score
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
            
            # Save the best model if a path is provided
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update learning rate
        scheduler.step()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Loaded best model from epoch {best_epoch+1} with F1: {best_f1*100:.2f}%")
    
    # Create final visualization
    bias_corrector.visualize_bias_correction()
    
    # Create training curve plots
    plot_training_curves(train_losses, valid_losses, train_accs, valid_accs, f1_scores)
    
    return model, bias_corrector


def plot_training_curves(train_losses, valid_losses, train_accs, valid_accs, f1_scores):
    """
    Plot training curves for loss, accuracy, and F1 score.
    """
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
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model with automatic bias correction')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test data directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model (optional)')
    parser.add_argument('--save_path', type=str, default='./bias_corrected_model.pth', help='Path to save the best model')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'], help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=str, default='0.0001', help='Learning rate')
    parser.add_argument('--correction_strength', type=float, default=2.0, help='Bias correction strength (higher = more aggressive)')
    parser.add_argument('--correction_frequency', type=int, default=1, help='How often to update bias correction (in epochs)')
    
    args = parser.parse_args()
    
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
    
    print(f"Loading training data from {args.data_dir}...")
    train_paths, train_labels = load_data(args.data_dir)
    
    print(f"Loading test data from {args.test_dir}...")
    test_paths, test_labels = load_data(args.test_dir)
    
    # Create datasets
    train_dataset = EmotionDataset(train_paths, train_labels, transform=transform)
    test_dataset = EmotionDataset(test_paths, test_labels, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    try:
        model = EnhancedResEmoteNet(num_classes=7, backbone=args.backbone)
    except Exception as e:
        print(f"Error creating model with backbone {args.backbone}: {e}")
        print("Trying with minimal parameters...")
        model = EnhancedResEmoteNet(num_classes=7)
    
    model = model.to(device)
    
    # Load pretrained model if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pretrained model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Set initial class weights based on class distribution
    class_counts = np.bincount(train_labels, minlength=len(EMOTIONS))
    total_samples = np.sum(class_counts)
    base_weights = total_samples / (len(EMOTIONS) * class_counts + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Normalize weights
    base_weights = base_weights / np.mean(base_weights) * 3.0  # Scale by 3 for stronger initial weighting
    
    print("Initial class weights (based on class distribution):")
    for i, emotion in EMOTIONS.items():
        print(f"  {emotion}: {base_weights[i]:.4f} (count: {class_counts[i]})")
    
    # Handle very imbalanced classes
    for i, count in enumerate(class_counts):
        if count <= 10:  # Very few samples
            base_weights[i] = min(base_weights[i] * 2, 10.0)  # Cap at 10
            print(f"  Boosting weight for {EMOTIONS[i]} due to very low sample count: {base_weights[i]:.4f}")
    
    initial_weights = torch.tensor(base_weights, dtype=torch.float32, device=device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Train with bias correction
    model, bias_corrector = train_with_bias_correction(
        model, train_loader, test_loader, optimizer, scheduler, device,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        correction_frequency=args.correction_frequency,
        initial_weights=initial_weights,
        correction_strength=args.correction_strength,
        save_path=args.save_path
    )
    
    print("\nFinal Bias Correction Status:")
    bias_corrector.print_current_status()
    print("\nTraining completed!")


if __name__ == "__main__":
    main() 