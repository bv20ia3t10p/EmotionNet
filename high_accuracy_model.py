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

# Define constants
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

# Exponential Moving Average model for more stable results
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def train(model, train_loader, val_loader, epochs, patience, learning_rate, device, 
          use_amp=True, use_ema=True, save_path="model.pth", class_weights_boost=1.5):
    """Train the model with advanced techniques for high accuracy"""
    # Setup EMA if enabled
    ema_model = EMA(model, decay=0.999) if use_ema else None
    
    # Loss function with class weights that boost underrepresented emotions
    class_counts = np.array([0, 0, 0, 0, 0, 0, 0])  # Initialize with zeros
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
        class_correct = [0] * len(EMOTIONS)
        class_total = [0] * len(EMOTIONS)
        
        for pred, target in zip(all_preds, all_targets):
            class_correct[target] += int(pred == target)
            class_total[target] += 1
        
        # Print validation results
        val_acc = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
        
        # Print per-class accuracies
        for i in range(len(EMOTIONS)):
            if class_total[i] > 0:
                print(f"Class {EMOTIONS[i]}: {100*class_correct[i]/class_total[i]:.2f}%")
            else:
                print(f"Class {EMOTIONS[i]}: N/A (no samples)")
        
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
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Normalized Confusion Matrix')
            plt.colorbar()
            
            tick_marks = np.arange(len(EMOTIONS))
            plt.xticks(tick_marks, list(EMOTIONS.values()), rotation=45)
            plt.yticks(tick_marks, list(EMOTIONS.values()))
            
            fmt = '.2f'
            thresh = cm_normalized.max() / 2.
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    plt.text(j, i, format(cm_normalized[i, j], fmt),
                             ha="center", va="center",
                             color="white" if cm_normalized[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            cm_path = save_path.replace(".pth", "_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs. Best accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
            break
    
    return best_val_acc

def predict(model, image_path, device, image_size=224):
    """Make a prediction on a single image"""
    # Load and preprocess the image
    transform = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs, _ = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Return emotion and confidence
    emotion = EMOTIONS[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    return emotion, confidence, probabilities.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description="High-performance emotion recognition training")
    parser.add_argument("--data_dir", type=str, default="./extracted/emotion/train", 
                        help="Path to training data directory")
    parser.add_argument("--test_dir", type=str, default="./extracted/emotion/test", 
                        help="Path to test data directory")
    parser.add_argument("--model_dir", type=str, default="./models", 
                        help="Directory to save models")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to pre-trained model to resume from")
    parser.add_argument("--backbones", type=str, nargs="+", 
                        default=["efficientnet_b0", "resnet18"],
                        help="Backbone architectures to use")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=224, 
                        help="Size to resize images to")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, 
                        help="Base learning rate")
    parser.add_argument("--no_amp", action="store_true", 
                        help="Disable automatic mixed precision")
    parser.add_argument("--no_ema", action="store_true", 
                        help="Disable exponential moving average model")
    parser.add_argument("--predict", type=str, default=None,
                        help="Make a prediction on a specific image")
    parser.add_argument("--save_path", type=str, default="models/high_accuracy_model.pth",
                        help="Path to save the best model")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # If prediction mode, load model and make prediction
    if args.predict:
        # Create model
        model = EnsembleModel(num_classes=len(EMOTIONS), backbones=args.backbones)
        
        # Load model weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        # Make prediction
        emotion, confidence, probabilities = predict(model, args.predict, device, args.image_size)
        
        print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
        
        # Print all probabilities
        for i, prob in enumerate(probabilities):
            print(f"{EMOTIONS[i]}: {prob:.4f}")
        
        return
    
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data
    train_paths, train_labels = load_data(args.data_dir)
    test_paths, test_labels = load_data(args.test_dir)
    
    # Split training data into train and validation sets
    indices = np.arange(len(train_paths))
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    
    train_indices = indices[:split]
    valid_indices = indices[split:]
    
    train_dataset = AdvancedEmotionDataset(
        [train_paths[i] for i in train_indices],
        [train_labels[i] for i in train_indices],
        mode='train',
        image_size=args.image_size
    )
    
    valid_dataset = AdvancedEmotionDataset(
        [train_paths[i] for i in valid_indices],
        [train_labels[i] for i in valid_indices],
        mode='val',
        image_size=args.image_size
    )
    
    test_dataset = AdvancedEmotionDataset(
        test_paths, test_labels, mode='test', image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=min(os.cpu_count(), 4), pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(os.cpu_count(), 4), pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(os.cpu_count(), 4), pin_memory=True
    )
    
    # Create model
    model = EnsembleModel(num_classes=len(EMOTIONS), backbones=args.backbones)
    
    # Load pre-trained weights if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model = model.to(device)
    
    # Print model summary
    print(f"Model created with backbones: {args.backbones}")
    
    # Train model
    best_acc = train(
        model, train_loader, valid_loader,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=device,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        save_path=args.save_path
    )
    
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    # Test best model
    print("Testing best model...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate per-class accuracies
    class_correct = [0] * len(EMOTIONS)
    class_total = [0] * len(EMOTIONS)
    
    for pred, target in zip(all_preds, all_targets):
        class_correct[target] += int(pred == target)
        class_total[target] += 1
    
    # Print test results
    test_acc = 100 * test_correct / test_total
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Print per-class accuracies
    for i in range(len(EMOTIONS)):
        if class_total[i] > 0:
            print(f"Class {EMOTIONS[i]}: {100*class_correct[i]/class_total[i]:.2f}%")
        else:
            print(f"Class {EMOTIONS[i]}: N/A (no samples)")
    
    # Generate and save classification report
    report = classification_report(all_targets, all_preds, 
                                  target_names=list(EMOTIONS.values()),
                                  output_dict=True)
    
    # Convert to DataFrame and save as CSV
    report_df = pd.DataFrame(report).transpose()
    report_path = args.save_path.replace(".pth", "_classification_report.csv")
    report_df.to_csv(report_path)
    
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    main() 