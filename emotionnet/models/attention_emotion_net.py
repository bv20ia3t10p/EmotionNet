#!/usr/bin/env python3
"""
AttentionEmotionNet Implementation

A deep neural network architecture for facial emotion recognition
with specialized attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) attention block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EfficientLocalGlobalAttention(nn.Module):
    """
    Efficient Local-Global Attention (ELGA)
    
    Captures both local and global dependencies efficiently through:
    - Local neighborhood attention
    - Global sparse attention
    - Feature aggregation mechanism
    """
    def __init__(self, channels, local_window=7, global_points=16):
        super(EfficientLocalGlobalAttention, self).__init__()
        self.channels = channels
        self.local_window = local_window
        self.global_points = global_points
        
        # Local attention
        self.local_conv = nn.Conv2d(
            channels, channels, 
            kernel_size=local_window, 
            padding=local_window//2, 
            groups=channels
        )
        
        # Global attention through key points
        self.global_pool = nn.AdaptiveAvgPool2d(global_points)
        self.global_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Fusion layers
        self.local_weight = nn.Parameter(torch.ones(1))
        self.global_weight = nn.Parameter(torch.ones(1))
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Local attention
        local_out = self.local_conv(x)
        
        # Global attention
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_out = F.interpolate(
            global_feat, 
            size=(x.size(2), x.size(3)), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Dynamic weighting
        local_out = local_out * self.local_weight
        global_out = global_out * self.global_weight
        
        # Fusion
        concat = torch.cat([local_out, global_out], dim=1)
        attention = self.fusion(concat)
        
        return x * attention


class DynamicRegionAwareAttention(nn.Module):
    """
    Dynamic Region-Aware Attention (DRAA)
    
    Focuses on emotionally salient facial regions through:
    - Region proposal mechanism
    - Adaptive importance weighting
    - Region-specific feature enhancement
    """
    def __init__(self, channels, num_regions=7):  # 7 for basic emotions
        super(DynamicRegionAwareAttention, self).__init__()
        self.channels = channels
        self.num_regions = num_regions
        
        # Region proposal network
        self.region_conv = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_regions, kernel_size=1)
        )
        
        # Region-specific feature enhancement
        self.region_enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // num_regions, kernel_size=1),
                nn.BatchNorm2d(channels // num_regions),
                nn.ReLU()
            ) for _ in range(num_regions)
        ])
        
        # Attention fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate region proposals
        region_maps = self.region_conv(x)  # B, num_regions, H, W
        region_weights = F.softmax(region_maps, dim=1)  # B, num_regions, H, W
        
        # Enhanced features per region
        enhanced_features = []
        for i in range(self.num_regions):
            # Extract region weight
            region_weight = region_weights[:, i:i+1]  # B, 1, H, W
            
            # Apply region-specific enhancement
            enhanced = self.region_enhancement[i](x)  # B, C//num_regions, H, W
            weighted_feature = enhanced * region_weight
            enhanced_features.append(weighted_feature)
        
        # Concatenate enhanced features
        concat_features = torch.cat(enhanced_features, dim=1)  # B, C, H, W
        
        # Generate attention map
        attention = self.fusion(concat_features)
        
        return x * attention


class CrossScalePyramidalAttention(nn.Module):
    """
    Cross-Scale Pyramidal Attention (CSPA)
    
    Enables multi-scale feature interactions through:
    - Feature pyramid decomposition
    - Cross-scale attention mechanisms
    - Adaptive feature fusion
    """
    def __init__(self, channels, scales=[1, 2, 4]):
        super(CrossScalePyramidalAttention, self).__init__()
        self.scales = scales
        num_scales = len(scales)
        
        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(channels, channels // num_scales, kernel_size=1, bias=False)
            for _ in range(num_scales)
        ])
        
        # Cross-scale attention
        self.cross_scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels // num_scales, channels // num_scales, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels // num_scales),
                nn.ReLU()
            ) for scale in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Scale-specific projections
        scale_features = [proj(x) for proj in self.scale_projs]
        
        # Cross-scale attention
        pyramid_features = []
        for i, (feature, attn) in enumerate(zip(scale_features, self.cross_scale_attn)):
            pooled = attn(feature)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # Concatenate pyramid features
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        # Generate attention weights
        attention = self.fusion(pyramid_concat)
        
        return x * attention


class AttentionEmotionNet(nn.Module):
    """
    AttentionEmotionNet for facial emotion recognition.
    
    Features:
    1. Deep CNN architecture
    2. Advanced attention mechanisms
    3. Enhanced regularization techniques
    4. Support for various training modes
    5. Configurable output for different datasets (7 or 8 classes)
    """
    def __init__(self, num_classes=8, dropout=0.4, stochastic_depth=True, training_mode='majority'):
        super(AttentionEmotionNet, self).__init__()
        
        # Stem block: initial convolution layers
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # First convolutional block with ELGA
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            EfficientLocalGlobalAttention(256, local_window=5, global_points=8),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Second convolutional block with Cross-Scale Attention
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            CrossScalePyramidalAttention(384, scales=[1, 2, 4]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Residual blocks with attention
        res_blocks = []
        channels = 384
        for i in range(3):
            drop_prob = 0.1 * (i + 1) if stochastic_depth else 0
            res_blocks.append(self._make_attention_res_block(channels, drop_prob))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Dynamic Region-Aware Attention for facial emotion regions
        self.region_attention = DynamicRegionAwareAttention(
            channels=384, 
            num_regions=8 if num_classes == 8 else 7  # Match num_regions to emotion classes
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate the feature size after convolutions and pooling
        # For 48x48 input:
        # After stem: 24x24x128
        # After conv_block1: 12x12x256
        # After conv_block2: 6x6x384
        # After global_pool: 1x1x384
        self.features_dim = 384
        
        # Classifier head with improved dropout pattern
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.features_dim, 512),
            nn.LayerNorm(512),  # LayerNorm instead of BatchNorm for better generalization
            nn.GELU(),  # GELU instead of ReLU for better gradient flow
            nn.Dropout(dropout/2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        self.training_mode = training_mode
    
    def _make_attention_res_block(self, channels, drop_prob=0.0):
        """Create a residual block with attention"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            SEBlock(channels),  # Squeeze-and-Excitation attention
            nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity(),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.res_blocks(x)
        
        # Apply region-aware attention for emotion-specific regions
        x = self.region_attention(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        # Apply appropriate activation based on training mode
        if not self.training:
            if self.training_mode == 'majority' or self.training_mode == 'multi_target':
                return torch.sigmoid(x)
            elif self.training_mode == 'probability':
                return F.softmax(x, dim=1)
        
        return x


def create_attention_emotion_net(num_classes=8, dropout=0.4, stochastic_depth=True, training_mode='majority'):
    """
    Create an AttentionEmotionNet model instance.
    
    Args:
        num_classes: Number of emotion classes (8 for FERPlus, 7 for FER2013)
        dropout: Dropout rate
        stochastic_depth: Whether to use stochastic depth for regularization
        training_mode: 'majority', 'probability', or 'multi_target'
    
    Returns:
        AttentionEmotionNet model instance
    """
    return AttentionEmotionNet(
        num_classes=num_classes,
        dropout=dropout,
        stochastic_depth=stochastic_depth,
        training_mode=training_mode
    ) 