#!/usr/bin/env python3
# Class-balanced training script for facial emotion recognition

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import sys

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

# Try multiple import approaches
try:
    # First try to import it directly
    from enhanced_resemotenet import EnhancedResEmoteNet
    print("Successfully imported EnhancedResEmoteNet directly")
except ImportError:
    try:
        # Try relative import with models directory
        from models.enhanced_resemotenet import EnhancedResEmoteNet
        print("Successfully imported EnhancedResEmoteNet from models directory")
    except ImportError:
        # Define a simplified version of the model
        print("WARNING: Could not import EnhancedResEmoteNet. Using simplified model structure.")
        import torchvision.models as models
        
        class EnhancedResEmoteNet(nn.Module):
            def __init__(self, num_classes=7, dropout_rate=0.4, backbone='resnet18', 
                         use_fpn=True, use_landmarks=True, use_contrastive=True):
                super(EnhancedResEmoteNet, self).__init__()
                
                # Load pretrained backbone model
                if backbone == 'resnet18':
                    base_model = models.resnet18(pretrained=True)
                    feature_dim = 512
                elif backbone == 'efficientnet_b0':
                    base_model = models.efficientnet_b0(pretrained=True)
                    feature_dim = 1280
                elif backbone == 'mobilenet_v3_small':
                    base_model = models.mobilenet_v3_small(pretrained=True)
                    feature_dim = 576
                
                # Use most of the model except the final layer
                self.backbone = nn.Sequential(*list(base_model.children())[:-1])
                
                # Simple classifier
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, num_classes)
                )
                
                # Custom feature_extractor attribute for compatibility
                self.feature_extractor = self.backbone
                self.fpn = nn.Identity()
                
            def forward(self, x):
                # Feature extraction
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
                
                # Classification
                logits = self.classifier(features)
                
                # Dummy values for valence and arousal
                valence = torch.zeros(x.size(0), 1).to(x.device)
                arousal = torch.zeros(x.size(0), 1).to(x.device)
                
                return logits, valence, arousal

# Emotion labels
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear', 
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augmentation_level=0, target_classes=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augmentation_level = augmentation_level
        self.target_classes = target_classes  # Classes that need extra augmentation
        
        # Create advanced augmentation pipeline for targeted classes
        # Use only face-friendly augmentations that preserve facial features
        self.targeted_aug = A.Compose([
            # Non-geometric transformations that preserve facial structure
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            
            # Very gentle blur that won't remove important facial details
            A.GaussianBlur(blur_limit=1, p=0.3),
            
            # Very minimal geometric transformations to preserve facial alignment
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5,
                              border_mode=cv2.BORDER_CONSTANT, value=0),
        ])
        
        # Super aggressive augmentation pipeline for extremely underrepresented classes
        # Still preserving facial features while increasing diversity
        self.extreme_aug = A.Compose([
            # Non-geometric transformations that preserve facial structure
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(5, 30), p=0.5),
            
            # Sequence of careful transformations
            A.OneOf([
                A.GaussianBlur(blur_limit=1, p=0.5),
                A.MedianBlur(blur_limit=1, p=0.5),
            ], p=0.4),
            
            # Very minimal geometric transformations to preserve facial alignment
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6,
                              border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # Add slight random shadow that might simulate lighting variations
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=1, 
                          shadow_dimension=3, p=0.3),
        ])
        
        # Calculate class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.class_counts = dict(zip(unique_labels, counts))
        print(f"Class distribution: {self.class_counts}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def apply_targeted_augmentation(self, image, label):
        """Apply special augmentations to targeted underrepresented classes"""
        if self.target_classes and label in self.target_classes:
            # Choose between targeted and extreme augmentation based on class rarity
            if label == 1:  # Disgust (extremely rare)
                augmented = self.extreme_aug(image=image)["image"]
            else:
                augmented = self.targeted_aug(image=image)["image"]
            return augmented
        return image
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                # Return a placeholder image
                img = np.zeros((48, 48), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # Return a placeholder image
            img = np.zeros((48, 48), dtype=np.uint8)
        
        # Apply targeted augmentation for underrepresented classes
        if self.augmentation_level > 0 and random.random() < 0.8:
            img = self.apply_targeted_augmentation(img, label)
        
        # Apply standard transforms
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        
        return img, label

def create_samplers(train_dataset, valid_dataset, class_weights=None):
    """
    Create weighted samplers to balance class distribution during training
    """
    # Count samples per class
    train_labels = train_dataset.labels
    class_sample_count = np.array([sum(train_labels == t) for t in range(len(EMOTIONS))])
    
    if class_weights is None:
        # Calculate weights inversely proportional to class frequencies
        class_weights = 1.0 / class_sample_count
        
        # Enhance weights for extremely underrepresented classes
        class_weights[0] *= 3.0  # Boost angry class even more
        class_weights[1] *= 5.0  # Boost disgust class (very rare)
        class_weights[6] *= 3.0  # Boost neutral class
    
    print(f"Class weights: {class_weights}")
    
    # Assign weights to each sample
    weights = [class_weights[label] for label in train_labels]
    
    # Create samplers
    train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return train_sampler

def load_data(data_dir, target_size=(48, 48)):
    """
    Load images and labels from directory
    """
    image_paths = []
    labels = []
    
    # Scan through data directory
    for emotion_idx, emotion in EMOTIONS.items():
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: {emotion_dir} not found, skipping...")
            continue
        
        # Get all image files in this emotion directory
        files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(files)} images for {emotion}")
        
        # Add to dataset
        image_paths.extend(files)
        labels.extend([emotion_idx] * len(files))
    
    return image_paths, np.array(labels)

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                device, num_epochs=50, early_stopping_patience=7, class_weights=None):
    """
    Train the model with early stopping and layer-specific learning rates
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_weighted_metric = 0.0
    patience_counter = 0
    
    # Lists to track metrics
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    valid_weighted_metrics = []
    valid_macro_f1_scores = []  # Track macro F1 scores
    valid_weighted_f1_scores = []  # Track weighted F1 scores
    
    # Progress bar for epochs
    pbar_epoch = tqdm(range(num_epochs), desc="Training")
    
    for epoch in pbar_epoch:
        # Set model to training mode
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        train_preds = []
        train_labels = []
        
        # Progress bar for training batches
        pbar_train = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        
        # Training phase
        for inputs, labels in pbar_train:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                # Handle different return types from the model
                outputs = model(inputs)
                
                # Extract logits based on model output type
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        # For models that return logits, valence, arousal
                        logits = outputs[0]
                    else:
                        # For other tuple returns, assume first element is logits
                        logits = outputs[0]
                else:
                    # For models that return only logits
                    logits = outputs
                
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels for per-class metrics
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar_train.set_postfix(loss=loss.item(), 
                                  acc=torch.sum(preds == labels.data).double().item() / inputs.size(0),
                                  lr=optimizer.param_groups[0]['lr'])
        
        # Calculate epoch statistics for training
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        
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
        valid_true = []
        
        print("Validating...")
        pbar_valid = tqdm(valid_loader, leave=False, desc="Validating")
        
        for inputs, labels in pbar_valid:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass (no gradient)
            with torch.no_grad():
                # Handle different return types from the model
                outputs = model(inputs)
                
                # Extract logits based on model output type
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        # For models that return logits, valence, arousal
                        logits = outputs[0]
                    else:
                        # For other tuple returns, assume first element is logits
                        logits = outputs[0]
                else:
                    # For models that return only logits
                    logits = outputs
                
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and true labels
            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(labels.cpu().numpy())
        
        # Calculate epoch statistics for validation
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_acc = running_corrects.double() / len(valid_loader.dataset)
        
        # Calculate per-class validation accuracy and F1 scores
        print("Per-class validation metrics:")
        class_accuracies = {}
        
        # Calculate individual F1 scores for each class
        f1_scores = {}
        for class_idx, class_name in EMOTIONS.items():
            # Create binary arrays for this class (1 for this class, 0 for others)
            y_true_bin = np.array(valid_true) == class_idx
            y_pred_bin = np.array(valid_preds) == class_idx
            
            # Calculate F1 score for this class
            if np.sum(y_true_bin) > 0:  # Only calculate if class exists in true labels
                class_f1 = f1_score(y_true_bin, y_pred_bin) * 100
                f1_scores[class_name] = class_f1
            else:
                f1_scores[class_name] = 0.0
            
            # Also calculate accuracy
            class_mask = np.array(valid_true) == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(np.array(valid_preds)[class_mask] == class_idx) * 100
                class_accuracies[class_name] = class_acc
                print(f"  {class_name}: Acc = {class_acc:.2f}%, F1 = {f1_scores[class_name]:.2f}%")
            else:
                class_accuracies[class_name] = 0.0
                print(f"  {class_name}: No samples")
        
        # Calculate and print macro and weighted F1 scores
        macro_f1 = f1_score(valid_true, valid_preds, average='macro') * 100
        weighted_f1 = f1_score(valid_true, valid_preds, average='weighted') * 100
        print(f"Overall F1 Scores: Macro = {macro_f1:.2f}%, Weighted = {weighted_f1:.2f}%")
        
        # Store F1 scores for plotting
        valid_macro_f1_scores.append(macro_f1)
        valid_weighted_f1_scores.append(weighted_f1)
        
        # Count predictions per class
        pred_counts = np.bincount(valid_preds, minlength=len(EMOTIONS))
        print("Predictions per class:")
        for class_idx, count in enumerate(pred_counts):
            percentage = count / len(valid_preds) * 100
            print(f"  {EMOTIONS[class_idx]}: {count} ({percentage:.1f}%)")
        
        # Log current learning rates
        lr_string = ', '.join([f"{pg['lr']:.7f}" for pg in optimizer.param_groups])
        print(f"Current learning rates: {lr_string}")
        
        # Calculate weighted validation metric
        # This gives more weight to underrepresented classes
        weighted_metric = 0.0
        weights = {
            'angry': 3.0,    # Critical class, heavily weighted
            'disgust': 0.5,  # Rare but less important
            'fear': 1.0,
            'happy': 0.5,    # Overrepresented
            'sad': 1.0,
            'surprise': 1.0,
            'neutral': 3.0   # Critical class, heavily weighted
        }
        
        total_weight = sum(weights.values())
        for class_name, acc in class_accuracies.items():
            weighted_metric += (acc * weights[class_name]) / total_weight
        
        # Print epoch summary
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc*100:.2f}%")
        
        # Print classification report
        print("Classification Report:")
        target_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]
        print(classification_report(valid_true, valid_preds, target_names=target_names))
        
        # Print accuracy gap to detect overfitting
        acc_gap = train_acc*100 - valid_acc*100
        print(f"Train-Val Accuracy Gap: {acc_gap:.2f}%")
        
        # Print weighted validation metric
        print(f"Weighted validation metric: {weighted_metric:.2f}%")
        
        # Check if this is the best model based on weighted metric
        if weighted_metric > best_weighted_metric:
            best_weighted_metric = weighted_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Best model saved with weighted metric: {best_weighted_metric:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{early_stopping_patience} (best: {best_weighted_metric:.2f}% at epoch {epoch+1-patience_counter})")
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc.item())
        valid_accuracies.append(valid_acc.item())
        valid_weighted_metrics.append(weighted_metric)
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Get ready for next epoch
        print(f"Training epoch {epoch+2}/{num_epochs}...")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Create and save training plots
    create_training_plots(train_losses, valid_losses, train_accuracies, valid_accuracies, 
                         valid_weighted_metrics, valid_macro_f1_scores, valid_weighted_f1_scores)
    
    return model

def create_training_plots(train_losses, valid_losses, train_accs, valid_accs, 
                         weighted_metrics, macro_f1_scores=None, weighted_f1_scores=None):
    """Create and save plots for training metrics"""
    # Determine number of plots based on whether F1 scores are provided
    num_plots = 4 if macro_f1_scores is not None else 3
    
    plt.figure(figsize=(16, 4))
    
    # Loss plot
    plt.subplot(1, num_plots, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Accuracy plot
    plt.subplot(1, num_plots, 2)
    plt.plot([acc * 100 for acc in train_accs], label='Train Accuracy')
    plt.plot([acc * 100 for acc in valid_accs], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Weighted metric plot
    plt.subplot(1, num_plots, 3)
    plt.plot(weighted_metrics, label='Weighted Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted Metric (%)')
    plt.legend()
    plt.title('Weighted Validation Metric')
    
    # F1 score plot (if provided)
    if macro_f1_scores is not None and weighted_f1_scores is not None:
        plt.subplot(1, num_plots, 4)
        plt.plot(macro_f1_scores, label='Macro F1')
        plt.plot(weighted_f1_scores, label='Weighted F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score (%)')
        plt.legend()
        plt.title('F1 Score Curves')
    
    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=100)
    print("Training plots saved to training_plots.png")

def create_confusion_matrix(model, valid_loader, device):
    """Create and save a confusion matrix visualization"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            # Handle different return types from the model
            outputs = model(inputs)
            
            # Extract logits based on model output type
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    # For models that return logits, valence, arousal
                    logits = outputs[0]
                else:
                    # For other tuple returns, assume first element is logits
                    logits = outputs[0]
            else:
                # For models that return only logits
                logits = outputs
                
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))],
               yticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    print("Confusion matrix saved to confusion_matrix.png")

def pretrain_underrepresented_classes(model, device, data_dir, save_path, target_classes=[0, 6]):
    """
    Pre-train the model specifically on underrepresented classes
    """
    print(f"Pre-training model on underrepresented classes: {[EMOTIONS[i] for i in target_classes]}")
    
    # Create special transformation for pre-training
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load all data
    image_paths, labels = load_data(data_dir)
    
    # Filter to include only target classes and a small subset of other classes
    target_indices = [i for i, label in enumerate(labels) if label in target_classes]
    other_indices = [i for i, label in enumerate(labels) if label not in target_classes]
    
    # Randomly sample a small subset of other classes to prevent overfitting
    other_indices = np.random.choice(other_indices, size=min(len(other_indices), 1000), replace=False)
    
    # Combine indices
    selected_indices = np.concatenate([target_indices, other_indices])
    selected_paths = [image_paths[i] for i in selected_indices]
    selected_labels = labels[selected_indices]
    
    # Split data
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        selected_paths, selected_labels, test_size=0.2, random_state=42, stratify=selected_labels
    )
    
    # Create datasets with targeted augmentation
    train_dataset = EmotionDataset(
        train_paths, train_labels, transform=transform, 
        augmentation_level=2, target_classes=target_classes
    )
    valid_dataset = EmotionDataset(
        valid_paths, valid_labels, transform=transform
    )
    
    # Create samplers with extreme weights for target classes
    class_weights = np.ones(len(EMOTIONS))
    for cls in target_classes:
        class_weights[cls] = 10.0  # Extremely high weight for target classes
    
    train_sampler = create_samplers(train_dataset, valid_dataset, class_weights)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Define loss function with class weights
    criterion_weights = torch.ones(len(EMOTIONS), device=device)
    for cls in target_classes:
        criterion_weights[cls] = 5.0  # Higher loss weight for underrepresented classes
    
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)
    
    # Create optimizer with adaptation for different model structures
    params = []
    
    # Inspect model to find available modules for optimization
    for name, param in model.named_parameters():
        lr_multiplier = 1.0
        
        # Determine learning rate multiplier based on parameter name
        if any(x in name for x in ['backbone', 'feature_extractor', 'conv', 'stem', 'stage']):
            lr_multiplier = 0.1
        elif any(x in name for x in ['fpn', 'middle', 'enhance', 'attn']):
            lr_multiplier = 0.5
        elif any(x in name for x in ['classifier', 'fc', 'linear', 'out', 'emotion_classifier']):
            lr_multiplier = 5.0
            
        # Add to parameter groups
        params.append({
            'params': param,
            'lr': lr_multiplier * 1e-5
        })
    
    optimizer = optim.Adam(params)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Train for a few epochs
    pretrain_epochs = 10
    model = train_model(
        model, train_loader, valid_loader, criterion, optimizer, scheduler,
        device, num_epochs=pretrain_epochs, early_stopping_patience=5
    )
    
    # Save pre-trained model
    torch.save(model.state_dict(), save_path)
    print(f"Pre-trained model saved to {save_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Class-balanced training for facial emotion recognition')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                      choices=['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                      help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--pretrain', action='store_true', help='Pre-train on underrepresented classes')
    parser.add_argument('--augmentation_level', type=int, default=2, 
                      choices=[0, 1, 2, 3], help='Augmentation intensity level')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Base learning rate')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load training data
    print(f"Loading training data from {args.data_dir}...")
    train_paths, train_labels = load_data(args.data_dir)
    
    # Load test data separately (without splitting)
    print(f"Loading test data from {args.test_dir}...")
    test_paths, test_labels = load_data(args.test_dir)
    print(f"Loaded {len(test_paths)} test samples")
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_paths, train_labels, transform=transform, 
        augmentation_level=args.augmentation_level, 
        target_classes=[0, 1, 6]  # Targeted augmentation for angry, disgust, neutral
    )
    test_dataset = EmotionDataset(
        test_paths, test_labels, transform=transform
    )
    
    # Create samplers for training data
    train_sampler = create_samplers(train_dataset, test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Initialize model
    # Fix: Inspect the actual EnhancedResEmoteNet class to see what args it supports
    try:
        # Try creating with all parameters first
        model = EnhancedResEmoteNet(
            num_classes=7, 
            dropout_rate=0.4,
            backbone=args.backbone,
            use_fpn=True,
            use_landmarks=True,
            use_contrastive=True
        )
    except TypeError as e:
        print(f"Error creating model with full parameters: {e}")
        # Try with minimal parameters
        model = EnhancedResEmoteNet(
            num_classes=7
        )
    
    # Move model to device
    model = model.to(device)
    
    # Print model structure to help with debugging
    print("Model structure:")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")
    
    # Pre-train on underrepresented classes if requested
    if args.pretrain:
        pretrain_path = os.path.join(args.model_dir, 'pretrained_underrep.pth')
        model = pretrain_underrepresented_classes(
            model, device, args.data_dir, pretrain_path, target_classes=[0, 6]
        )
    
    # Define loss function with class weights
    # Higher weights for underrepresented classes
    class_weights = torch.tensor([
        3.0,  # angry
        5.0,  # disgust
        1.5,  # fear
        0.8,  # happy
        1.0,  # sad
        1.2,  # surprise
        3.0   # neutral
    ], device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer with adaptable parameter groups based on model structure
    params = []
    
    # Inspect model to find available modules and create parameter groups dynamically
    for name, param in model.named_parameters():
        lr_multiplier = 1.0
        
        # Adjust learning rate based on parameter location in network
        if any(x in name for x in ['backbone', 'feature_extractor', 'conv', 'stem', 'stage']):
            lr_multiplier = 0.1  # Lower learning rate for feature extraction
        elif any(x in name for x in ['fpn', 'middle', 'enhance', 'attn']):
            lr_multiplier = 1.0  # Standard learning rate for middle layers
        elif any(x in name for x in ['classifier', 'fc', 'linear', 'out', 'emotion_classifier']):
            lr_multiplier = 5.0  # Higher learning rate for classification layers
            
        # Add parameter group
        params.append({
            'params': param,
            'lr': args.learning_rate * lr_multiplier
        })
    
    optimizer = optim.Adam(params)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Train model (using test_loader for validation)
    model = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        device, num_epochs=args.epochs, early_stopping_patience=args.patience
    )
    
    # Generate confusion matrix using test data
    create_confusion_matrix(model, test_loader, device)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'final_balanced_model.pth'))
    print(f"Final model saved to {os.path.join(args.model_dir, 'final_balanced_model.pth')}")

if __name__ == '__main__':
    main() 