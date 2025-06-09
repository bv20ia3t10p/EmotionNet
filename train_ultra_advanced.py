#!/usr/bin/env python3
"""
Ultra-Advanced EmotionNet Training Script
Target: 75%+ validation accuracy with advanced architectures and techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ===== DATASET =====

class FERPlusDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, fer2013_csv_file, split='Training', transform=None):
        self.ferplus_df = pd.read_csv(csv_file)
        self.ferplus_df = self.ferplus_df[self.ferplus_df['Usage'] == split].reset_index(drop=True)
        
        self.fer2013_df = pd.read_csv(fer2013_csv_file)
        self.fer2013_df = self.fer2013_df[self.fer2013_df['Usage'] == split].reset_index(drop=True)
        
        self.transform = transform
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        # Filter valid samples
        vote_sums = self.ferplus_df[self.emotion_columns].sum(axis=1)
        valid_indices = vote_sums > 0
        self.ferplus_df = self.ferplus_df[valid_indices].reset_index(drop=True)
        self.fer2013_df = self.fer2013_df[valid_indices].reset_index(drop=True)
        
        print(f"Loaded {len(self.ferplus_df)} {split} samples")
    
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

# ===== ADVANCED COMPONENTS =====

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature enhancement"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim) 
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to (B, H*W, C) for attention
        x_flat = x.view(B, C, H*W).transpose(1, 2)
        
        q = self.q_linear(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, H*W, C)
        out = self.out_linear(out)
        
        # Reshape back to (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out + x  # Residual connection

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_pool = self.avgpool(x).view(b, c)
        avg_out = self.fc(avg_pool).view(b, c, 1, 1)
        
        # Max pooling branch  
        max_pool = self.maxpool(x).view(b, c)
        max_out = self.fc(max_pool).view(b, c, 1, 1)
        
        # Combine both branches
        att = avg_out + max_out
        return x * att.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(concat))
        return x * att

class AdvancedResidualBlock(nn.Module):
    """Advanced residual block with multiple attention mechanisms"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super().__init__()
        
        # Main convolutional path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention mechanisms
        self.use_attention = use_attention
        if use_attention:
            self.channel_att = ChannelAttention(out_channels)
            self.spatial_att = SpatialAttention()
            if out_channels >= 256:  # Only use self-attention for higher dimensions
                self.self_att = MultiHeadSelfAttention(out_channels, num_heads=8)
            else:
                self.self_att = None
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Stochastic depth for regularization
        self.drop_path = nn.Dropout2d(0.1) if stride == 1 and in_channels == out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # Apply attention mechanisms
        if self.use_attention:
            out = self.channel_att(out)
            out = self.spatial_att(out)
            if self.self_att is not None:
                out = self.self_att(out)
        
        out = self.drop_path(out)
        out += residual
        return F.relu(out, inplace=True)

class PyramidPooling(nn.Module):
    """Pyramid pooling for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(2),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            )
        ])
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            pyramid = stage(x)
            pyramid = F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(pyramid)
            
        return torch.cat(pyramids, dim=1)

# ===== ULTRA-ADVANCED MODEL =====

class UltraAdvancedEmotionNet(nn.Module):
    """Ultra-advanced emotion recognition network"""
    def __init__(self, num_classes=8):
        super().__init__()
        
        # Stem with enhanced feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Progressive stages with attention
        self.stage1 = nn.Sequential(
            AdvancedResidualBlock(64, 128, stride=1, use_attention=True),
            AdvancedResidualBlock(128, 128, stride=1, use_attention=True),
        )
        
        self.stage2 = nn.Sequential(
            AdvancedResidualBlock(128, 256, stride=2, use_attention=True),
            AdvancedResidualBlock(256, 256, stride=1, use_attention=True),
            AdvancedResidualBlock(256, 256, stride=1, use_attention=True),
        )
        
        self.stage3 = nn.Sequential(
            AdvancedResidualBlock(256, 512, stride=2, use_attention=True),
            AdvancedResidualBlock(512, 512, stride=1, use_attention=True),
            AdvancedResidualBlock(512, 512, stride=1, use_attention=True),
        )
        
        # Pyramid pooling for multi-scale features
        self.pyramid = PyramidPooling(512, 512)
        
        # Global pooling with both average and max
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        # Advanced classifier with multiple dropout layers
        # Features: 512 (avg) + 512 (max) + 1024 (pyramid) = 2048 total
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),  
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features through stages
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Multi-scale pyramid features
        pyramid_feat = self.pyramid(x)
        pyramid_global = self.avgpool(pyramid_feat).flatten(1)
        
        # Global pooling features
        avg_feat = self.avgpool(x).flatten(1)
        max_feat = self.maxpool(x).flatten(1)
        
        # Combine all features
        combined = torch.cat([avg_feat, max_feat, pyramid_global], dim=1)
        
        return self.classifier(combined)

# ===== LOSS FUNCTIONS =====

class FocalLoss(nn.Module):
    """Enhanced Focal Loss"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # For focal loss, we'll use the standard cross entropy without label smoothing
        # Label smoothing will be handled by the separate SoftTargetCrossEntropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SoftTargetCrossEntropy(nn.Module):
    """Cross entropy with soft targets for label smoothing"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

# ===== TRAINING UTILITIES =====

def get_class_weights(dataset):
    """Calculate class weights for handling imbalance"""
    labels = []
    for _, hard_label, _ in dataset:
        labels.append(hard_label)
    
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Inverse frequency weighting with smoothing
    weights = total_samples / (len(class_counts) * (class_counts + 1))
    weights = weights / weights.sum() * len(class_counts)  # Normalize
    
    print("Class distribution and weights:")
    emotions = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    for i, (emotion, count, weight) in enumerate(zip(emotions, class_counts, weights)):
        print(f"  {emotion}: {count} samples (weight: {weight:.3f})")
    
    return torch.FloatTensor(weights)

def create_weighted_sampler(dataset):
    """Create weighted sampler for balanced training"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts and warmup"""
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1., last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2 for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# ===== TRAINING LOOP =====

def train_ultra_advanced():
    """Train the ultra-advanced emotion recognition model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on device: {device}")
    
    # Enhanced data transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Slightly larger for more context
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("📊 Loading datasets...")
    train_dataset = FERPlusDataset('./FERPlus-master/fer2013new.csv', './fer2013.csv', 'Training', train_transform)
    val_dataset = FERPlusDataset('./FERPlus-master/fer2013new.csv', './fer2013.csv', 'PublicTest', val_transform)
    
    # Create samplers and loaders
    train_sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model and loss
    model = UltraAdvancedEmotionNet(num_classes=8).to(device)
    class_weights = get_class_weights(train_dataset).to(device)
    
    # Multi-loss approach
    focal_loss = FocalLoss(alpha=0.75, gamma=1.25)
    soft_ce_loss = SoftTargetCrossEntropy()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Model parameters: {total_params:,}")
    
    # Advanced optimizer with different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': classifier_params, 'lr': 2e-3, 'weight_decay': 1e-3}
    ])
    
    # Advanced scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=30,
        cycle_mult=2,
        max_lr=2e-3,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.8
    )
    
    # Training settings
    num_epochs = 150
    best_val_acc = 0.0
    patience = 50
    patience_counter = 0
    
    print(f"🎯 Starting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for images, hard_labels, soft_labels in pbar:
            images = images.to(device, non_blocking=True)
            hard_labels = hard_labels.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Combined loss
            loss1 = focal_loss(outputs, hard_labels)
            loss2 = soft_ce_loss(outputs, soft_labels) * 0.3
            loss = loss1 + loss2
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += hard_labels.size(0)
            train_correct += predicted.eq(hard_labels).sum().item()
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validating')
            for images, hard_labels, soft_labels in val_pbar:
                images = images.to(device, non_blocking=True)
                hard_labels = hard_labels.to(device, non_blocking=True)
                soft_labels = soft_labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = focal_loss(outputs, hard_labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += hard_labels.size(0)
                val_correct += predicted.eq(hard_labels).sum().item()
        
        # Calculate accuracies
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        print(f"  Best:  {best_val_acc:.2f}%")
        print(f"  LR:    {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc if best_val_acc > 0 else val_acc
            print(f"  🎉 NEW BEST! Improvement: +{improvement:.2f}%")
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, 'best_ultra_advanced_fer.pth')
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        print("-" * 80)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"💤 Early stopping triggered after {patience} epochs without improvement")
            break
    
    print(f"\n✅ Training completed!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved as: best_ultra_advanced_fer.pth")

if __name__ == "__main__":
    train_ultra_advanced()