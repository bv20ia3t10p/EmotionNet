#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Supports multiple architectures: SimpleConvNet, UpgradedEmotionNet, EffectiveFER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import argparse
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# ===== DATASET =====

class FERPlusDataset(torch.utils.data.Dataset):
    """FERPlus dataset loader for evaluation"""
    def __init__(self, csv_file, fer2013_csv_file, split='PublicTest', transform=None):
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

# ===== MODEL ARCHITECTURES =====

# SimpleConvNet (Working Model)
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleConvNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Components for UpgradedEmotionNet
class SqueezeExcitation(nn.Module):
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

# UpgradedEmotionNet
class UpgradedEmotionNet(nn.Module):
    def __init__(self, num_classes=8):
        super(UpgradedEmotionNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.stage1 = nn.Sequential(
            ResidualSEBlock(64, 64, stride=1),
            ResidualSEBlock(64, 128, stride=2)
        )
        
        self.stage2 = nn.Sequential(
            ResidualSEBlock(128, 128, stride=1),
            ResidualSEBlock(128, 256, stride=2)
        )
        
        self.stage3 = nn.Sequential(
            ResidualSEBlock(256, 256, stride=1),
            ResidualSEBlock(256, 512, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        avg_feat = self.avgpool(x).flatten(1)
        max_feat = self.maxpool(x).flatten(1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        
        return self.classifier(x)

# Components for EffectiveFER
class EfficientPatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_encoding
        return self.dropout(self.norm(x))

class EffectiveFER(nn.Module):
    def __init__(self, img_size=64, patch_size=16, num_classes=8, embed_dim=512, 
                 depth=8, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        self.patch_embed = EfficientPatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        return self.head(cls_output)

# Components for UltraAdvancedEmotionNet (Matching training script)
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

# UltraAdvancedEmotionNet (Exact match to training script)
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

# ===== EVALUATION FUNCTIONS =====

def load_model(model_path, model_type, device):
    """Load model with proper architecture"""
    print(f"🔧 Loading {model_type} from {model_path}")
    
    if model_type.lower() in ['simple', 'simpleconvnet']:
        model = SimpleConvNet(num_classes=8)
    elif model_type.lower() in ['upgraded', 'upgradedemotionnet']:
        model = UpgradedEmotionNet(num_classes=8)
    elif model_type.lower() in ['effective', 'effectivefer']:
        model = EffectiveFER(num_classes=8)
    elif model_type.lower() in ['ultra', 'ultraadvanced', 'ultraadvancedemotionnet']:
        model = UltraAdvancedEmotionNet(num_classes=8)
    else:
        raise ValueError(f"❌ Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_acc' in checkpoint:
            print(f"📊 Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Model parameters: {total_params:,}")
    
    return model

def evaluate_model(model, dataloader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("🔍 Evaluating model...")
    with torch.no_grad():
        for images, hard_labels, soft_labels in tqdm(dataloader, desc='Evaluating'):
            images, hard_labels = images.to(device), hard_labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def create_detailed_report(predictions, labels, probabilities, emotion_names, save_dir):
    """Create comprehensive evaluation report with visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
    
    print(f"\n📊 Per-Class Performance:")
    print("=" * 80)
    print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("=" * 80)
    
    for i, emotion in enumerate(emotion_names):
        print(f"{emotion:<12} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")
    
    # Summary metrics
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    print("=" * 80)
    print(f"{'Macro F1':<12} {macro_f1:<10.3f}")
    print(f"{'Weighted F1':<12} {weighted_f1:<10.3f}")
    print("=" * 80)
    
    # Save text report
    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f"Model Evaluation Report\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(classification_report(labels, predictions, target_names=emotion_names))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names,
                cbar_kws={'label': 'Recall'})
    plt.title('Normalized Confusion Matrix (Recall)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Per-class performance charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Precision
    bars1 = axes[0, 0].bar(emotion_names, precision, color=colors[0], alpha=0.8)
    axes[0, 0].set_title('Precision by Emotion', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Precision', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    for bar, val in zip(bars1, precision):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Recall
    bars2 = axes[0, 1].bar(emotion_names, recall, color=colors[1], alpha=0.8)
    axes[0, 1].set_title('Recall by Emotion', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Recall', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    for bar, val in zip(bars2, recall):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # F1-Score
    bars3 = axes[1, 0].bar(emotion_names, f1, color=colors[2], alpha=0.8)
    axes[1, 0].set_title('F1-Score by Emotion', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    for bar, val in zip(bars3, f1):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Support (sample count)
    bars4 = axes[1, 1].bar(emotion_names, support, color=colors[3], alpha=0.8)
    axes[1, 1].set_title('Test Samples by Emotion', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    for bar, val in zip(bars4, support):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}', 
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Model performance summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall metrics
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    values = [accuracy, macro_f1, weighted_f1]
    bars = ax1.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_title('Overall Model Performance', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Class distribution
    ax2.pie(support, labels=emotion_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Test Set Distribution', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm
    }

def main():
    """Interactive evaluation - edit parameters below"""
    
    # ===== EDIT THESE PARAMETERS =====
    MODEL_PATH = "best_ultra_advanced_fer.pth"      # Your model checkpoint
    MODEL_TYPE = "ultraadvanced"                    # simple, upgraded, or effective  
    SPLIT = "PublicTest"                     # PublicTest or PrivateTest
    SAVE_DIR = "evaluation_results"          # Results directory
    BATCH_SIZE = 64                          # Evaluation batch size
    # =================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"🏗️  Architecture: {MODEL_TYPE}")
    print(f"📊 Split: {SPLIT}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"\n📊 Loading {SPLIT} dataset...")
    dataset = FERPlusDataset(
        csv_file='./FERPlus-master/fer2013new.csv',
        fer2013_csv_file='./fer2013.csv',
        split=SPLIT,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load and evaluate model
    model = load_model(MODEL_PATH, MODEL_TYPE, device)
    predictions, labels, probabilities = evaluate_model(model, dataloader, device)
    
    # Generate report
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 
                    'Anger', 'Disgust', 'Fear', 'Contempt']
    
    save_dir = f"{SAVE_DIR}_{MODEL_TYPE}_{SPLIT}"
    results = create_detailed_report(predictions, labels, probabilities, emotion_names, save_dir)
    
    # Final summary
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"🎯 Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"📈 Macro F1: {results['macro_f1']:.3f}")
    print(f"⚖️  Weighted F1: {results['weighted_f1']:.3f}")
    
    # Performance analysis
    print(f"\n📋 Performance Analysis:")
    best_classes = np.argsort(results['f1'])[-3:][::-1]  # Top 3
    worst_classes = np.argsort(results['f1'])[:3]        # Bottom 3
    
    print(f"🏆 Best performing emotions:")
    for i in best_classes:
        print(f"   {emotion_names[i]}: F1={results['f1'][i]:.3f}")
    
    print(f"📉 Challenging emotions:")
    for i in worst_classes:
        print(f"   {emotion_names[i]}: F1={results['f1'][i]:.3f}")

if __name__ == "__main__":
    main() 