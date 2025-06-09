#!/usr/bin/env python3
"""
UPGRADED WORKING FER Training - Enhanced Architecture
Builds upon the successful SimpleConvNet from train_working_fix.py
Adds ResNet-style connections, SE blocks, and multi-scale features
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
    """FERPlus dataset loader - same as working version"""
    def __init__(self, csv_file, fer2013_csv_file, split='Training', transform=None):
        self.ferplus_df = pd.read_csv(csv_file)
        self.ferplus_df = self.ferplus_df[self.ferplus_df['Usage'] == split].reset_index(drop=True)
        
        self.fer2013_df = pd.read_csv(fer2013_csv_file)
        self.fer2013_df = self.fer2013_df[self.fer2013_df['Usage'] == split].reset_index(drop=True)
        
        self.transform = transform
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        vote_sums = self.ferplus_df[self.emotion_columns].sum(axis=1)
        valid_indices = vote_sums > 0
        self.ferplus_df = self.ferplus_df[valid_indices].reset_index(drop=True)
        self.fer2013_df = self.fer2013_df[valid_indices].reset_index(drop=True)
        
        print(f"Loaded {len(self.ferplus_df)} {split} samples")
        
        hard_labels = self.ferplus_df[self.emotion_columns].values.argmax(axis=1)
        unique, counts = np.unique(hard_labels, return_counts=True)
        print(f"Class distribution in {split}:")
        for emotion_id, count in zip(unique, counts):
            print(f"  {self.emotion_columns[emotion_id]}: {count} samples")
    
    def __len__(self):
        return len(self.ferplus_df)
    
    def __getitem__(self, idx):
        pixels = self.fer2013_df.iloc[idx]['pixels']
        image = np.array([int(p) for p in pixels.split()]).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        emotion_votes = self.ferplus_df.iloc[idx][self.emotion_columns].values.astype(np.float32)
        hard_label = emotion_votes.argmax()
        vote_sum = emotion_votes.sum()
        soft_label = emotion_votes / vote_sum if vote_sum > 0 else emotion_votes
        
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)

class SqueezeExcitation(nn.Module):
    """SE block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualSEBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        out += residual
        return F.relu(out, inplace=True)

class UpgradedEmotionNet(nn.Module):
    """Enhanced architecture with ResNet + SE blocks"""
    def __init__(self, num_classes=8):
        super(UpgradedEmotionNet, self).__init__()
        
        # Initial conv (64x64 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 1: 32x32 -> 16x16
        self.stage1 = nn.Sequential(
            ResidualSEBlock(64, 64, stride=1),
            ResidualSEBlock(64, 128, stride=2)
        )
        
        # Stage 2: 16x16 -> 8x8
        self.stage2 = nn.Sequential(
            ResidualSEBlock(128, 128, stride=1),
            ResidualSEBlock(128, 256, stride=2)
        )
        
        # Stage 3: 8x8 -> 4x4
        self.stage3 = nn.Sequential(
            ResidualSEBlock(256, 256, stride=1),
            ResidualSEBlock(256, 512, stride=2)
        )
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # 512 avg + 512 max
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Dual pooling
        avg_feat = self.avgpool(x).flatten(1)
        max_feat = self.maxpool(x).flatten(1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        
        return self.classifier(x)

def create_data_transforms():
    """Enhanced transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=10, fill=128),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_weighted_sampler(dataset):
    """Enhanced weighted sampler"""
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    hard_labels = []
    for i in range(len(dataset)):
        _, hard_label, _ = dataset[i]
        hard_labels.append(hard_label)
    
    hard_labels = np.array(hard_labels)
    class_counts = np.bincount(hard_labels, minlength=8)
    total_samples = len(hard_labels)
    class_weights = total_samples / (8 * class_counts + 1e-6)
    
    # Enhanced minority class boosting
    class_weights[5] *= 5  # disgust
    class_weights[7] *= 5  # contempt  
    class_weights[6] *= 3  # fear
    class_weights[3] *= 2  # sadness
    
    print("Enhanced class weights:")
    for i, (emotion, weight) in enumerate(zip(emotion_columns, class_weights)):
        print(f"  {emotion}: {weight:.3f} (count: {class_counts[i]})")
    
    sample_weights = class_weights[hard_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

class EnhancedFocalLoss(nn.Module):
    """Enhanced Focal Loss"""
    def __init__(self, alpha=None, gamma=3.0, label_smoothing=0.05):
        super(EnhancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=1)
            loss = F.kl_div(log_probs, smooth_targets, reduction='none').sum(dim=1)
        else:
            loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-loss)
        focal_loss = (1 - pt) ** self.gamma * loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, hard_labels, soft_labels) in enumerate(pbar):
        images, hard_labels = images.to(device), hard_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, hard_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """Validation epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for images, hard_labels, soft_labels in pbar:
            images, hard_labels = images.to(device), hard_labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, hard_labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    train_transform, val_transform = create_data_transforms()
    
    print("📊 Creating datasets...")
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
    
    train_sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=48, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print("🏗️  Creating Upgraded Emotion Recognition Network...")
    model = UpgradedEmotionNet(num_classes=8).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Total parameters: {total_params:,}")
    
    # Enhanced loss and optimizer
    class_weights = torch.tensor([1.0, 1.0, 2.0, 2.5, 2.0, 8.0, 4.0, 8.0]).to(device)
    criterion = EnhancedFocalLoss(alpha=class_weights, gamma=3.0, label_smoothing=0.05)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4, betas=(0.9, 0.999))
    
    # Cosine annealing with warm restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    print("🎯 Starting enhanced training...")
    best_val_acc = 0.0
    patience = 30
    patience_counter = 0
    
    for epoch in range(1, 500):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/100:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Best:  {best_val_acc:.2f}%")
        print(f"  LR:    {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            improvement = val_acc - 67.64
            print(f"  🎉 NEW BEST! Improvement: +{improvement:.2f}%")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, 'best_upgraded_working_fer.pth')
        else:
            patience_counter += 1
        
        print(f"  ⏳ Patience: {patience_counter}/{patience}")
        print("-" * 60)
        
        # if patience_counter >= patience:
        #     print(f"\n⏹️  Early stopping after {patience} epochs without improvement")
        #     break
        
        # if val_acc > 85.0:
        #     print(f"\n🏆 Reached excellent accuracy of {val_acc:.2f}%!")
        #     break
    
    final_improvement = best_val_acc - 67.64
    print(f"\n🏁 Training completed!")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📈 Improvement over baseline: +{final_improvement:.2f}%")
    print(f"💾 Best model saved as: best_upgraded_working_fer.pth")

if __name__ == "__main__":
    main() 