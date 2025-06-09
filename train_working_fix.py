#!/usr/bin/env python3
"""
WORKING FER Training - Fixed Architecture
Addresses the model collapse issues found in train_upgraded_fixed.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import pandas as pd
import numpy as np
from PIL import Image
import os
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FERPlusDataset(torch.utils.data.Dataset):
    """FERPlus dataset loader"""
    def __init__(self, csv_file, fer2013_csv_file, split='Training', transform=None):
        # Load FERPlus annotations
        self.ferplus_df = pd.read_csv(csv_file)
        self.ferplus_df = self.ferplus_df[self.ferplus_df['Usage'] == split].reset_index(drop=True)
        
        # Load original FER2013 images
        self.fer2013_df = pd.read_csv(fer2013_csv_file)
        self.fer2013_df = self.fer2013_df[self.fer2013_df['Usage'] == split].reset_index(drop=True)
        
        self.transform = transform
        
        # Emotion mapping
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        # Filter out samples where sum of votes is 0
        vote_sums = self.ferplus_df[self.emotion_columns].sum(axis=1)
        valid_indices = vote_sums > 0
        self.ferplus_df = self.ferplus_df[valid_indices].reset_index(drop=True)
        self.fer2013_df = self.fer2013_df[valid_indices].reset_index(drop=True)
        
        print(f"Loaded {len(self.ferplus_df)} {split} samples")
        
        # Print class distribution
        hard_labels = self.ferplus_df[self.emotion_columns].values.argmax(axis=1)
        unique, counts = np.unique(hard_labels, return_counts=True)
        print(f"Class distribution in {split}:")
        for emotion_id, count in zip(unique, counts):
            print(f"  {self.emotion_columns[emotion_id]}: {count} samples")
    
    def __len__(self):
        return len(self.ferplus_df)
    
    def __getitem__(self, idx):
        # Get image from FER2013
        pixels = self.fer2013_df.iloc[idx]['pixels']
        image = np.array([int(p) for p in pixels.split()]).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels from FERPlus
        emotion_votes = self.ferplus_df.iloc[idx][self.emotion_columns].values.astype(np.float32)
        
        # Hard label (majority vote)
        hard_label = emotion_votes.argmax()
        
        # Soft label (normalized votes)
        vote_sum = emotion_votes.sum()
        soft_label = emotion_votes / vote_sum if vote_sum > 0 else emotion_votes
        
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)

class SimpleConvNet(nn.Module):
    """Simple but effective CNN for emotion recognition"""
    def __init__(self, num_classes=8):
        super(SimpleConvNet, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            nn.Dropout2d(0.3),
        )
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_data_transforms():
    """Create data transforms with face-preserving augmentations"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=5, fill=128),  # Small rotation
        transforms.RandomHorizontalFlip(p=0.3),  # 30% flip chance
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_weighted_sampler(dataset):
    """Create weighted sampler to handle class imbalance"""
    # Get class counts
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    hard_labels = []
    for i in range(len(dataset)):
        _, hard_label, _ = dataset[i]
        hard_labels.append(hard_label)
    
    hard_labels = np.array(hard_labels)
    class_counts = np.bincount(hard_labels, minlength=8)
    
    # Calculate weights (inverse frequency)
    total_samples = len(hard_labels)
    class_weights = total_samples / (8 * class_counts + 1e-6)  # Add small epsilon
    
    # Apply extra boost to very rare classes
    class_weights[5] *= 3  # disgust boost
    class_weights[7] *= 3  # contempt boost
    class_weights[6] *= 2  # fear boost
    
    print("Class weights:")
    for i, (emotion, weight) in enumerate(zip(emotion_columns, class_weights)):
        print(f"  {emotion}: {weight:.3f} (count: {class_counts[i]})")
    
    # Create sample weights
    sample_weights = class_weights[hard_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    return sampler

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, hard_labels, soft_labels) in enumerate(pbar):
        images = images.to(device)
        hard_labels = hard_labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, hard_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for images, hard_labels, soft_labels in pbar:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, hard_labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform, val_transform = create_data_transforms()
    
    # Datasets
    print("Creating datasets...")
    train_dataset = FERPlusDataset(
        csv_file='./FERPlus-master/fer2013new.csv',
        fer2013_csv_file='./fer2013.csv',
        split='Training',
        transform=train_transform
    )
    
    val_dataset = FERPlusDataset(
        csv_file='./FERPlus-master/fer2013new.csv',
        fer2013_csv_file='./fer2013.csv',
        split='PublicTest',
        transform=val_transform
    )
    
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    print("Creating Simple ConvNet model...")
    model = SimpleConvNet(num_classes=8).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss function with class weights
    class_weights = torch.tensor([1.0, 1.0, 1.5, 1.5, 1.5, 5.0, 3.0, 5.0]).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, 500):  # 100 epochs max
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch}/100")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"🎯 New best validation accuracy: {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_working_fer.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Stop if we reach good accuracy
        if val_acc > 82.0:
            print(f"Reached target accuracy of {val_acc:.2f}%!")
            break
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main() 