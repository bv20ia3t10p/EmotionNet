#!/usr/bin/env python3
# High-performance emotion recognition training to reach 80% accuracy

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import random
import cv2
from torch.cuda.amp import autocast, GradScaler

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

# Try to import from existing modules for compatibility
try:
    from balanced_training import EMOTIONS, load_data
    print("Successfully imported constants from balanced_training")
except ImportError:
    # Define constants if imports fail
    EMOTIONS = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }
    
    def load_data(data_dir):
        """Load image paths and labels from directory structure."""
        paths = []
        labels = []
        for emotion_idx, emotion_name in EMOTIONS.items():
            emotion_dir = os.path.join(data_dir, emotion_name)
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} does not exist.")
                continue
            
            for img_file in os.listdir(emotion_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_dir, img_file)
                    paths.append(img_path)
                    labels.append(emotion_idx)
        
        return paths, labels

# Advanced dataset class with strong augmentations
class AdvancedEmotionDataset(Dataset):
    def __init__(self, image_paths, labels, mode='train', image_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        self.image_size = image_size
        
        # Define strong augmentations for training
        if mode == 'train':
            self.transform = A.Compose([
                # Resize with padding to maintain aspect ratio
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=0),
                
                # Mild spatial transformations that preserve facial features
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, 
                                  border_mode=cv2.BORDER_CONSTANT, p=0.5),
                
                # Carefully tuned intensity augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                    A.CLAHE(clip_limit=2.0, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.5),
                
                # Very mild noise that won't distort facial features
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                ], p=0.3),
                
                # Normalize and convert to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Minimal transforms for validation and test
            self.transform = A.Compose([
                # Preserve aspect ratio in test mode
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        try:
            # Try to use OpenCV for faster loading
            img = cv2.imread(img_path)
            if img is None:
                raise Exception("Image loading failed")
                
            # Check if the image is grayscale (1 channel) and convert to RGB if needed
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback to PIL
            try:
                pil_img = Image.open(img_path)
                # Ensure RGB mode (convert from grayscale if needed)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img = np.array(pil_img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image and the label if loading fails
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        
        return img, label

class EnsembleModel(nn.Module):
    """Ensemble of multiple backbone models with attention fusion"""
    def __init__(self, num_classes=7, backbones=None):
        super(EnsembleModel, self).__init__()
        if backbones is None:
            backbones = ['efficientnet_b0', 'resnet18']
        
        self.backbone_names = backbones
        self.backbones = nn.ModuleList()
        self.neck = nn.ModuleList()
        backbone_out_dims = []
        
        # Create backbones
        for backbone_name in backbones:
            if 'efficientnet' in backbone_name:
                model = timm.create_model(backbone_name, pretrained=True)
                out_dim = model.classifier.in_features
                model.classifier = nn.Identity()
                
            elif 'resnet' in backbone_name:
                model = timm.create_model(backbone_name, pretrained=True)
                out_dim = model.fc.in_features
                model.fc = nn.Identity()
                
            elif 'vit' in backbone_name:
                model = timm.create_model(backbone_name, pretrained=True)
                out_dim = model.head.in_features
                model.head = nn.Identity()
                
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            self.backbones.append(model)
            backbone_out_dims.append(out_dim)
            
            # Create neck for each backbone (reduce dimensions and add regularization)
            neck = nn.Sequential(
                nn.Linear(out_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.neck.append(neck)
        
        # Attention mechanism to weight models
        self.attention = nn.Sequential(
            nn.Linear(512 * len(backbones), len(backbones)),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Get features from each backbone
        features = []
        for i, backbone in enumerate(self.backbones):
            feat = backbone(x)
            feat = self.neck[i](feat)
            features.append(feat)
        
        # Concatenate features for attention
        concat_features = torch.cat(features, dim=1)
        
        # Get attention weights
        weights = self.attention(concat_features)
        
        # Apply attention weights
        weighted_features = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            weighted_features += feat * weights[:, i].unsqueeze(1)
        
        # Final classification
        logits = self.classifier(weighted_features)
        
        return logits, weighted_features

def train_high_performance_model(
    data_dir="./extracted/emotion/train",
    test_dir="./extracted/emotion/test",
    model_dir="./models",
    model_path=None,
    backbones=['efficientnet_b0', 'resnet18'],
    batch_size=32,
    image_size=224,
    epochs=50,
    patience=10,
    learning_rate=0.0001,
    use_amp=True,
    use_ema=True,
    save_path="high_accuracy_model.pth",
    class_weights_boost=1.5  # Parameter to control extra weighting for problem classes
):
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load data
    print(f"Loading training data from {data_dir}...")
    train_paths, train_labels = load_data(data_dir)
    
    print(f"Loading test data from {test_dir}...")
    test_paths, test_labels = load_data(test_dir)
    
    # Calculate class distribution
    class_counts = np.bincount(train_labels, minlength=len(EMOTIONS))
    print("Class distribution in training data:")
    for i, count in enumerate(class_counts):
        print(f"  {EMOTIONS[i]}: {count} samples")
    
    # Create datasets
    train_dataset = AdvancedEmotionDataset(
        train_paths, train_labels, mode='train', image_size=image_size
    )
    test_dataset = AdvancedEmotionDataset(
        test_paths, test_labels, mode='test', image_size=image_size
    )
    
    # Create data loaders with weighted sampling
    class_weights = 1.0 / np.bincount(train_labels, minlength=len(EMOTIONS))
    class_weights = class_weights / np.sum(class_weights) * len(EMOTIONS)
    sample_weights = torch.tensor([class_weights[label] for label in train_labels], dtype=torch.float)
    
    # Balance problematic classes more aggressively
    for label_idx in [0, 6]:  # angry and neutral
        indices = [i for i, label in enumerate(train_labels) if label == label_idx]
        for idx in indices:
            sample_weights[idx] *= class_weights_boost  # Configurable boost
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    print(f"Creating ensemble model with backbones: {backbones}")
    model = EnsembleModel(num_classes=len(EMOTIONS), backbones=backbones)
    
    # Move model to device
    model = model.to(device)
    
    # Load pretrained model if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create optimizer with different learning rates
    # Higher learning rate for classifier, lower for backbones
    param_groups = [
        {'params': model.classifier.parameters(), 'lr': learning_rate * 10},
        {'params': model.attention.parameters(), 'lr': learning_rate * 5},
    ]
    
    # Add backbone parameters with lower learning rates
    for i, backbone_name in enumerate(model.backbone_names):
        param_groups.append({
            'params': model.backbones[i].parameters(),
            'lr': learning_rate * 0.1  # Lower learning rate for pretrained backbones
        })
        param_groups.append({
            'params': model.neck[i].parameters(),
            'lr': learning_rate
        })
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate/100
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize automatic mixed precision (AMP) scaler
    scaler = GradScaler() if use_amp else None
    
    # Initialize EMA model
    if use_ema:
        ema_model = copy.deepcopy(model)
        ema_decay = 0.999
    
    # Training variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
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
        train_labels_list = []
        
        # Training progress bar
        pbar = tqdm(train_loader, desc=f"Training")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            if use_amp:
                with autocast():
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)
                    
                # Backward + optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass without mixed precision
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update EMA model
            if use_ema:
                with torch.no_grad():
                    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1-ema_decay)
            
            # Track statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            
            # Store predictions and labels for metrics
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), 
                          acc=torch.sum(preds == labels).double().item() / inputs.size(0))
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch training statistics
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Print per-class training accuracy
        train_preds = np.array(train_preds)
        train_labels_list = np.array(train_labels_list)
        print("Per-class training accuracy:")
        for class_idx, class_name in EMOTIONS.items():
            class_mask = train_labels_list == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(train_preds[class_mask] == class_idx) * 100
                print(f"  {class_name}: {class_acc:.2f}%")
            else:
                print(f"  {class_name}: No samples")
        
        # Validation phase - use EMA model if enabled
        eval_model = ema_model if use_ema else model
        eval_model.eval()
        running_loss = 0.0
        running_corrects = 0
        valid_preds = []
        valid_labels_list = []
        
        # Validation progress bar
        pbar = tqdm(test_loader, desc=f"Validating")
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs, _ = eval_model(inputs)
                loss = criterion(outputs, labels)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                
                # Store for per-class metrics
                valid_preds.extend(preds.cpu().numpy())
                valid_labels_list.extend(labels.cpu().numpy())
        
        # Calculate epoch validation statistics
        valid_loss = running_loss / len(test_loader.dataset)
        valid_acc = running_corrects / len(test_loader.dataset)
        
        # Store metrics
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # Calculate per-class F1 scores
        from sklearn.metrics import f1_score
        valid_preds = np.array(valid_preds)
        valid_labels_list = np.array(valid_labels_list)
        macro_f1 = f1_score(valid_labels_list, valid_preds, average='macro')
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
            class_mask = valid_labels_list == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(valid_preds[class_mask] == class_idx) * 100
                print(f"  {EMOTIONS[class_idx]}: {class_acc:.2f}%")
            else:
                print(f"  {EMOTIONS[class_idx]}: No samples")
        
        # Print classification report
        print("\nClassification Report:")
        target_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]
        report = classification_report(valid_labels_list, valid_preds, target_names=target_names)
        print(report)
        
        # Check if this is the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_f1 = macro_f1
            best_epoch = epoch
            best_model_wts = copy.deepcopy(eval_model.state_dict())
            patience_counter = 0
            print(f"New best model! Accuracy: {best_acc*100:.2f}%, F1: {best_f1*100:.2f}%")
            
            # Save the best model
            torch.save(eval_model.state_dict(), os.path.join(model_dir, save_path))
            print(f"Saved best model to {os.path.join(model_dir, save_path)}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Loaded best model from epoch {best_epoch+1} with Accuracy: {best_acc*100:.2f}%, F1: {best_f1*100:.2f}%")
    
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
    plt.savefig(os.path.join(model_dir, 'high_accuracy_training_curves.png'))
    print(f"Training curves saved to {os.path.join(model_dir, 'high_accuracy_training_curves.png')}")
    
    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(valid_labels_list, valid_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(EMOTIONS))
    plt.xticks(tick_marks, [EMOTIONS[i] for i in range(len(EMOTIONS))], rotation=45)
    plt.yticks(tick_marks, [EMOTIONS[i] for i in range(len(EMOTIONS))])
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(model_dir, 'high_accuracy_confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(model_dir, 'high_accuracy_confusion_matrix.png')}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='High-performance emotion recognition training')
    parser.add_argument('--data_dir', type=str, default='./extracted/emotion/train', help='Path to training data')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test data')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--model_path', type=str, default=None, help='Path to existing model to continue training')
    parser.add_argument('--save_path', type=str, default='high_accuracy_model.pth', help='Filename to save model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for resizing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--backbones', type=str, nargs='+', 
                      default=['efficientnet_b0', 'resnet18'],
                      help='Backbone architectures to use in ensemble')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--no_ema', action='store_true', help='Disable exponential moving average model')
    parser.add_argument('--class_weights_boost', type=float, default=1.5, help='Extra weighting for problem classes')
    
    args = parser.parse_args()
    
    # Run the high-performance training
    model = train_high_performance_model(
        data_dir=args.data_dir,
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        model_path=args.model_path,
        backbones=args.backbones,
        batch_size=args.batch_size,
        image_size=args.image_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        save_path=args.save_path,
        class_weights_boost=args.class_weights_boost
    )
    
    print("\nHigh-performance training completed!")

if __name__ == "__main__":
    main() 