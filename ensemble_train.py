"""
High-accuracy FER2013 training script using model ensemble and advanced augmentation
Targeting 80%+ accuracy using model ensemble approach with advanced techniques
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast, GradScaler
import copy
from collections import OrderedDict
import torch.nn.functional as F
import urllib.request
import bz2
import shutil

# Suppress albumentations update warning
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Constants
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMAGE_SIZE = 48
BATCH_SIZE = 128  # Smaller batch size to allow for larger models
NUM_EPOCHS = 300  # More epochs for better convergence
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_MODELS = 7  # Increased ensemble size
FEATURE_DIM = 512  # Define feature dimension globally

# Set random seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True



class FER2013Dataset(Dataset):
    """Dataset class for FER2013 emotion recognition"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Images are already normalized to 0-255 and face-cropped
        image = self.images[idx].reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
        label = self.labels[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, label

def load_fer2013(csv_file='data/fer2013/icml_face_data.csv'):
    """Load FER2013 dataset from CSV file"""
    print(f"Loading data from {csv_file}")
    
    try:
        # Load dataset
        data = pd.read_csv(csv_file)
        
        # Check if split by Usage column is available
        if ' Usage' in data.columns:
            print("Using ' Usage' column to split data")
            
            # Split by predefined sets
            train_data = data[data[' Usage'] == 'Training']
            val_data = data[data[' Usage'] == 'PublicTest']
            test_data = data[data[' Usage'] == 'PrivateTest']
            
            # Extract pixels and labels
            train_pixels = np.array([np.fromstring(pixel, dtype=int, sep=' ') 
                                    for pixel in train_data[' pixels']])
            train_labels = train_data['emotion'].values
            
            val_pixels = np.array([np.fromstring(pixel, dtype=int, sep=' ') 
                                 for pixel in val_data[' pixels']])
            val_labels = val_data['emotion'].values
            
            test_pixels = np.array([np.fromstring(pixel, dtype=int, sep=' ') 
                                  for pixel in test_data[' pixels']])
            test_labels = test_data['emotion'].values
            
            print(f"Split by Usage: {len(train_pixels)} training, {len(val_pixels)} validation, {len(test_pixels)} test samples")
            
        else:
            # If no Usage column, split the data ourselves
            pixels = np.array([np.fromstring(pixel, dtype=int, sep=' ') 
                              for pixel in data[' pixels']])
            labels = data['emotion'].values
            
            # Split into train/val/test sets (80/10/10)
            train_pixels, temp_pixels, train_labels, temp_labels = train_test_split(
                pixels, labels, test_size=0.2, random_state=42)
            
            val_pixels, test_pixels, val_labels, test_labels = train_test_split(
                temp_pixels, temp_labels, test_size=0.5, random_state=42)
            
            print(f"Split by random: {len(train_pixels)} training, {len(val_pixels)} validation, {len(test_pixels)} test samples")
        
        # Print class distribution
        print("Class distribution:")
        print("  Train:")
        for i, emotion in enumerate(EMOTIONS):
            count = np.sum(train_labels == i)
            percentage = count / len(train_labels) * 100
            print(f"    {emotion}: {count} samples ({percentage:.1f}%)")
            
        print("  Validation:")
        for i, emotion in enumerate(EMOTIONS):
            count = np.sum(val_labels == i)
            percentage = count / len(val_labels) * 100
            print(f"    {emotion}: {count} samples ({percentage:.1f}%)")
            
        print("  Test:")
        for i, emotion in enumerate(EMOTIONS):
            count = np.sum(test_labels == i)
            percentage = count / len(test_labels) * 100
            print(f"    {emotion}: {count} samples ({percentage:.1f}%)")
        
        return train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

class AdvancedModel(nn.Module):
    """Advanced CNN for emotion recognition using different architectures"""
    def __init__(self, model_name, num_classes=7, dropout_rate=0.3):
        super(AdvancedModel, self).__init__()
        
        # Model type selection with different architectures for diversity
        if model_name == 'efficientnet_b0':
            self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
            self.base_model.conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.feature_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'resnet18':
            self.base_model = timm.create_model('resnet18', pretrained=True)
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'mobilenetv3_small':
            self.base_model = timm.create_model('mobilenetv3_small_100', pretrained=True)
            self.base_model.conv_stem = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.feature_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'convnext_tiny':
            self.base_model = timm.create_model('convnext_tiny', pretrained=True)
            # Modify first conv for grayscale
            orig_weight = self.base_model.stem[0].weight
            self.base_model.stem[0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, bias=False)
            with torch.no_grad():
                self.base_model.stem[0].weight = nn.Parameter(orig_weight.sum(dim=1, keepdim=True))
            self.feature_dim = self.base_model.head.fc.in_features
            self.base_model.head.fc = nn.Identity()
        elif model_name == 'vit_tiny_patch16':
            # Vision Transformer - tiny version
            self.base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
            # Modify for grayscale input
            orig_weight = self.base_model.patch_embed.proj.weight
            self.base_model.patch_embed.proj = nn.Conv2d(1, 192, kernel_size=16, stride=16)
            with torch.no_grad():
                self.base_model.patch_embed.proj.weight = nn.Parameter(orig_weight.sum(dim=1, keepdim=True))
            self.feature_dim = self.base_model.head.in_features
            self.base_model.head = nn.Identity()
        elif model_name == 'swin_tiny':
            # Swin Transformer - tiny version
            self.base_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            # Modify for grayscale input
            orig_weight = self.base_model.patch_embed.proj.weight
            self.base_model.patch_embed.proj = nn.Conv2d(1, 96, kernel_size=4, stride=4)
            with torch.no_grad():
                self.base_model.patch_embed.proj.weight = nn.Parameter(orig_weight.sum(dim=1, keepdim=True))
            self.feature_dim = self.base_model.head.in_features
            self.base_model.head = nn.Identity()
        elif model_name == 'resnest50d':
            # ResNeSt with dilated convolutions
            self.base_model = timm.create_model('resnest50d', pretrained=True)
            # Modify for grayscale input
            orig_weight = self.base_model.conv1[0].weight
            self.base_model.conv1[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            with torch.no_grad():
                self.base_model.conv1[0].weight = nn.Parameter(orig_weight.sum(dim=1, keepdim=True))
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Store model name for reference
        self.model_name = model_name
        
        # Enhanced classifier with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Classifier with appropriate regularization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate/3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # He/Kaiming initialization for ReLU-based networks
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_features(self, x):
        """Extract features before the final classifier"""
        return self.base_model(x)
        
    def forward(self, x):
        # Get base features
        features = self.base_model(x)
        
        # Apply attention - optional weighted feature path
        att_weights = self.attention(features)
        enhanced_features = features * att_weights
        
        # Classification
        return self.classifier(enhanced_features)

# New FocalLoss implementation for handling class imbalance better
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# New center loss for better feature discrimination
class CenterLoss(nn.Module):
    def __init__(self, num_classes=7, feat_dim=FEATURE_DIM, device=DEVICE):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Centers for each class
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        batch_size = x.size(0)
        
        # Check feature dimension
        if x.size(1) != self.feat_dim:
            print(f"Warning: Feature dimension mismatch in CenterLoss. Expected {self.feat_dim}, got {x.size(1)}")
            # Return a dummy loss that won't impact training
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Extract feature centers for the current labels
        centers_batch = self.centers.index_select(0, labels)
        
        # Calculate l2 distance between features and their centers
        diff = x - centers_batch
        dist = torch.pow(diff, 2).sum(dim=1)
        
        # Return mean distance as loss
        return dist.mean()

def train_single_model(model_idx, model_type, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """Train a single model for ensemble with advanced training strategies"""
    print(f"\nTraining model {model_idx+1}/{NUM_MODELS} - Type: {model_type}")
    
    # Create model
    model = AdvancedModel(model_type).to(DEVICE)
    
    # Get the feature dimension from the model
    feature_dim = model.feature_dim
    print(f"Model {model_type} feature dimension: {feature_dim}")
    
    # Calculate class weights to handle imbalance
    class_distribution = np.bincount(train_loader.dataset.labels)
    effective_num = 1.0 - np.power(0.99, class_distribution)
    class_weights = (1.0 - 0.99) / np.array(effective_num)
    class_weights = class_weights / np.sum(class_weights) * len(class_distribution)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    # Focal loss with class weights for imbalance
    focal_criterion = FocalLoss(gamma=2.0)
    
    # Center loss for better feature separation
    center_criterion = CenterLoss(num_classes=len(EMOTIONS), feat_dim=feature_dim).to(DEVICE)
    
    # Multi-loss approach with label smoothing
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Reduced label smoothing
    
    # Extract feature parameters and classifier parameters for different learning rates
    backbone_params = [p for n, p in model.named_parameters() if 'base_model' in n]
    attention_params = [p for n, p in model.named_parameters() if 'attention' in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    center_params = list(center_criterion.parameters())
    
    # Optimizer with weight decay and discriminative learning rates
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': attention_params, 'lr': LEARNING_RATE * 0.5},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)  # Increased weight decay
    
    # Separate optimizer for center loss
    optimizer_center = optim.SGD(center_params, lr=LEARNING_RATE * 0.5)
    
    # One cycle learning rate schedule for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LEARNING_RATE * 0.1, LEARNING_RATE * 0.5, LEARNING_RATE],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # Initial lr is max_lr/25
        final_div_factor=1e4  # Final lr is max_lr/10000
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler()
    
    # Training loop with progressive difficulty
    best_val_acc = 0.0
    patience = 7  # Reduced patience for early stopping
    early_stop_counter = 0
    best_model_weights = None
    
    # List to track training progress
    train_losses = []
    val_accuracies = []
    
    # Progressive training phases
    progressive_phase = 0
    phase_epochs = {
        0: 0.3 * num_epochs,  # First 30% - basic training
        1: 0.6 * num_epochs,  # Next 30% - harder augmentation
        2: num_epochs         # Final 40% - full difficulty
    }
    
    # Lambda for center loss weight
    center_loss_weight = lambda epoch: min(0.001 * epoch / 10, 0.1)
    
    # Set augmentation strength based on training phase
    def get_augmentation_strength(phase):
        if phase == 0:
            return 0.5  # Increased base augmentation
        elif phase == 1:
            return 0.8  # Increased medium augmentation
        else:
            return 1.0  # Full augmentation
    
    # Enable mixup in later phases with higher probability
    def use_mixup(phase, epoch):
        if phase == 0:
            return random.random() < 0.3  # Some mixup from start
        elif phase == 1:
            return random.random() < 0.5
        else:
            return random.random() < 0.7
    
    for epoch in range(num_epochs):
        # Check if we should move to next progressive phase
        if epoch >= phase_epochs[progressive_phase] and progressive_phase < 2:
            progressive_phase += 1
            print(f"Moving to progressive phase {progressive_phase+1}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        aug_strength = get_augmentation_strength(progressive_phase)
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Apply mixup with increasing probability in later phases
            if use_mixup(progressive_phase, epoch):
                # Beta distribution with concentration on ends for less ambiguity
                lam = np.random.beta(0.4, 0.4)
                idx = torch.randperm(images.size(0)).to(DEVICE)
                mixed_images = lam * images + (1 - lam) * images[idx]
                labels_a, labels_b = labels, labels[idx]
                
                # Forward pass with mixed precision
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(mixed_images)
                    
                    # Extract features for center loss
                    features = model.get_features(mixed_images)
                    
                    # Compute mixed loss for classification
                    ce_loss = lam * ce_criterion(outputs, labels_a) + (1 - lam) * ce_criterion(outputs, labels_b)
                    focal_loss = lam * focal_criterion(outputs, labels_a) + (1 - lam) * focal_criterion(outputs, labels_b)
                    
                    # Center loss can't easily use mixup, so we'll apply to first label only
                    c_loss = center_criterion(features, labels_a)
                    
                    # Weighted loss combination
                    loss = ce_loss * 0.4 + focal_loss * 0.4 + center_loss_weight(epoch) * c_loss
            else:
                # Standard forward pass with mixed precision
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    
                    # Extract features for center loss
                    features = model.get_features(images)
                    
                    # Compute losses
                    ce_loss = ce_criterion(outputs, labels)
                    focal_loss = focal_criterion(outputs, labels)
                    c_loss = center_criterion(features, labels)
                    
                    # Weighted loss combination
                    loss = ce_loss * 0.4 + focal_loss * 0.4 + center_loss_weight(epoch) * c_loss
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step with both optimizers
            scaler.step(optimizer)
            scaler.step(optimizer_center)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            
            # For mixup, we can only approximately count correct predictions
            if not use_mixup(progressive_phase, epoch):
                train_correct += (predicted == labels).sum().item()
            
            # Update learning rate after optimizer step
            scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation phase with test-time augmentation
        val_acc = evaluate_with_tta(model, val_loader)
        
        # Track progress
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  New best model! Val Acc: {val_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best weights
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), f'model_{model_idx+1}.pth')
    print(f"Model {model_idx+1} saved with validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Model {model_idx+1} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title(f'Model {model_idx+1} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'model_{model_idx+1}_training_curves.png')
    plt.close()
    
    return model, best_val_acc

def evaluate_with_tta(model, data_loader, num_augmentations=5):
    """Evaluation with test-time augmentation"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            all_labels.extend(labels.numpy())
            batch_size = images.size(0)
            
            # Storage for predictions
            probs = torch.zeros(batch_size, len(EMOTIONS)).to(DEVICE)
            
            # Original images
            images_orig = images.to(DEVICE)
            outputs = model(images_orig)
            probs += outputs.softmax(dim=1)
            
            # Horizontal flip
            images_flip = torch.flip(images, [3]).to(DEVICE)
            outputs = model(images_flip)
            probs += outputs.softmax(dim=1)
            
            # Random brightness/contrast variations
            for _ in range(num_augmentations - 2):
                brightness = 1.0 + random.uniform(-0.1, 0.1)
                contrast = 1.0 + random.uniform(-0.1, 0.1)
                images_aug = torch.clamp(brightness * images + contrast, 0, 1).to(DEVICE)
                outputs = model(images_aug)
                probs += outputs.softmax(dim=1)
                
            # Average predictions
            probs = probs / num_augmentations
            _, predicted = torch.max(probs, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy
    return accuracy_score(all_labels, all_predictions)

def predict_ensemble(models, data_loader, weights=None):
    """Make predictions using ensemble of models"""
    predictions = []
    true_labels = []
    
    # Default to equal weighting if not provided
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Ensemble prediction"):
            true_labels.extend(labels.numpy())
            batch_size = images.size(0)
            
            # Storage for ensemble predictions
            ensemble_probs = torch.zeros(batch_size, len(EMOTIONS)).to(DEVICE)
            
            # Accumulate predictions from each model
            for i, model in enumerate(models):
                model.eval()
                
                # TTA for each model
                probs = torch.zeros(batch_size, len(EMOTIONS)).to(DEVICE)
                
                # Original
                outputs = model(images.to(DEVICE))
                probs += outputs.softmax(dim=1)
                
                # Flip
                outputs = model(torch.flip(images, [3]).to(DEVICE))
                probs += outputs.softmax(dim=1)
                
                # Random augmentations
                for _ in range(3):
                    brightness = 1.0 + random.uniform(-0.1, 0.1)
                    contrast = 1.0 + random.uniform(-0.1, 0.1)
                    images_aug = torch.clamp(brightness * images + contrast, 0, 1).to(DEVICE)
                    outputs = model(images_aug)
                    probs += outputs.softmax(dim=1)
                
                # Average TTA predictions
                probs = probs / 5
                
                # Add to ensemble with weight
                ensemble_probs += weights[i] * probs
            
            # Get final predictions
            _, predicted = torch.max(ensemble_probs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    
    return acc, cm, predictions, true_labels

def plot_confusion_matrix(cm, normalize=True):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(EMOTIONS))
    plt.xticks(tick_marks, EMOTIONS, rotation=45)
    plt.yticks(tick_marks, EMOTIONS)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('ensemble_confusion_matrix.png')
    plt.close()

def main():
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels = load_fer2013()
    
    # Check the installed albumentations version
    try:
        import pkg_resources
        albu_version = pkg_resources.get_distribution("albumentations").version
        print(f"Using albumentations version: {albu_version}")
    except:
        print("Could not determine albumentations version")
    
    # Create advanced augmentations that preserve face structure
    train_transform = A.Compose([
        # Basic transforms
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        
        # Geometric transforms that preserve face structure
        A.Affine(
            scale=(0.92, 1.08),  # Scale limit
            translate_percent=(-0.08, 0.08),  # Shift limit
            rotate=(-15, 15),  # Rotation limit
            p=0.7
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),  # Subtle distortion
        
        # Pixel-level transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.7),
        
        # Noise and blur for robustness
        A.OneOf([
            A.GaussNoise(mean=0, per_channel=True, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.5),
        
        # Cutout/holes for regularization using Dropout
        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.3),
        
        # Normalization
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    
    # Simpler transforms for validation, only normalization
    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = FER2013Dataset(train_pixels, train_labels, transform=train_transform)
    val_dataset = FER2013Dataset(val_pixels, val_labels, transform=val_transform)
    test_dataset = FER2013Dataset(test_pixels, test_labels, transform=val_transform)
    
    # Create data loaders with augmented datasets
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Define model types for ensemble (diverse architectures)
    model_types = [
        'efficientnet_b0',
        'resnet18',
        'mobilenetv3_small',
        'convnext_tiny',
        'vit_tiny_patch16',        # Vision Transformer
        'swin_tiny',               # Swin Transformer
        'resnest50d'               # ResNeSt with dilated convolutions
    ]
    
    # Train ensemble of models
    models = []
    val_accuracies = []
    
    for i, model_type in enumerate(model_types):
        model, val_acc = train_single_model(i, model_type, train_loader, val_loader)
        models.append(model)
        val_accuracies.append(val_acc)
        
    # Calculate weights using softmax for weighted ensemble (better models get higher weights)
    temp = 1.0  # Temperature for softmax - lower for more concentration on best models
    val_accuracies_array = np.array(val_accuracies)
    exp_accuracies = np.exp(val_accuracies_array / temp)
    weights = exp_accuracies / np.sum(exp_accuracies)
    
    print("\nEnsemble weights based on validation accuracy:")
    for i, (model_type, weight) in enumerate(zip(model_types, weights)):
        print(f"  Model {i+1} ({model_type}): {weight:.4f}")
    
    # Enhanced ensemble prediction technique
    test_acc, cm, predictions, true_labels = predict_stacked_ensemble(models, test_loader, weights)
    print(f"\nEnsemble Test Accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Save individual model predictions for further analysis
    np.save('ensemble_predictions.npy', predictions)
    np.save('true_labels.npy', true_labels)
    
    # Print performance metrics by class
    print("\nPer-class accuracy:")
    for i, emotion in enumerate(EMOTIONS):
        class_correct = np.sum((predictions == i) & (true_labels == i))
        class_total = np.sum(true_labels == i)
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {emotion}: {class_acc:.4f} ({class_correct}/{class_total})")
    
    print("Done!")

def predict_stacked_ensemble(models, data_loader, weights=None):
    """Make predictions using stacked ensemble of models"""
    predictions = []
    true_labels = []
    all_probs = []
    
    # Default to equal weighting if not provided
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    with torch.no_grad():
        # First pass: collect all model predictions
        for images, labels in tqdm(data_loader, desc="Collecting ensemble predictions"):
            true_labels.extend(labels.numpy())
            batch_size = images.size(0)
            
            # Storage for model predictions
            batch_probs = []
            
            # Get predictions from each model
            for i, model in enumerate(models):
                model.eval()
                
                # Use TTA for each model
                probs = torch.zeros(batch_size, len(EMOTIONS)).to(DEVICE)
                
                # Original
                outputs = model(images.to(DEVICE))
                probs += outputs.softmax(dim=1)
                
                # Flip
                outputs = model(torch.flip(images, [3]).to(DEVICE))
                probs += outputs.softmax(dim=1)
                
                # Random brightness/contrast variations
                for _ in range(3):
                    brightness = 1.0 + random.uniform(-0.1, 0.1)
                    contrast = 1.0 + random.uniform(-0.1, 0.1)
                    images_aug = torch.clamp(brightness * images + contrast, 0, 1).to(DEVICE)
                    outputs = model(images_aug)
                    probs += outputs.softmax(dim=1)
                
                # Average TTA predictions
                probs = probs / 5
                batch_probs.append(probs.cpu().numpy())
            
            # Store all model probabilities
            all_probs.append(np.array(batch_probs))
        
        # Concatenate all batches
        all_probs = np.concatenate([np.stack(p, axis=1) for p in all_probs], axis=0)
        
        # Now use weighted voting for final predictions
        for i in range(len(all_probs)):
            model_preds = all_probs[i]
            
            # Apply weights to each model's prediction
            weighted_sum = np.zeros(len(EMOTIONS))
            for j, weight in enumerate(weights):
                weighted_sum += weight * model_preds[j]
            
            # Get final prediction
            predictions.append(np.argmax(weighted_sum))
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate accuracy and confusion matrix
    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    
    return acc, cm, predictions, true_labels

if __name__ == '__main__':
    main() 