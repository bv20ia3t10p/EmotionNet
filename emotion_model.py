"""
EmotionNet Model Architecture - SOTA Optimized
Enhanced multi-scale attention architecture for superior emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_modules import SEBlock, CBAM, ECABlock, CoordinateAttention, MultiHeadSelfAttention


class SimpleResidualBlock(nn.Module):
    """Simplified Residual Block focused on performance"""
    def __init__(self, in_ch, out_ch, stride=1):
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class SOTAEmotionNet(nn.Module):
    """SOTA Optimized EmotionNet - Multi-Scale Attention Architecture"""
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(SOTAEmotionNet, self).__init__()
        
        # Initial convolution optimized for 48x48
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 48x48 -> 24x24
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 24x24 -> 12x12
        
        # Multi-scale feature extraction with progressive attention
        self.layer1 = self._make_layer(64, 128, 2, stride=1)    # 12x12 -> 12x12
        self.layer2 = self._make_layer(128, 256, 2, stride=2)   # 12x12 -> 6x6
        self.layer3 = self._make_layer(256, 512, 2, stride=2)   # 6x6 -> 3x3
        
        # Multi-Scale Attention Strategy (6 attention mechanisms)
        # Layer 1 attention (128 channels, 12x12) - Spatial focus for basic features
        self.layer1_attention = ECABlock(128, gamma=2, b=1)
        
        # Layer 2 attention (256 channels, 6x6) - Coordinate attention for spatial relationships
        self.layer2_attention = CoordinateAttention(256, reduction=16)
        
        # Layer 3 attention stack (512 channels, 3x3) - Multiple attention for complex emotions
        self.layer3_se = SEBlock(512, reduction=16)
        self.layer3_cbam = CBAM(512, reduction=16)
        self.layer3_self_attn = MultiHeadSelfAttention(512, num_heads=8, dropout=0.1)
        
        # Cross-scale feature fusion with attention
        self.cross_scale_fusion = nn.ModuleDict({
            'layer1_proj': nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'layer2_proj': nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'layer3_proj': nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        })
        
        # Adaptive pooling for multi-scale fusion
        self.adaptive_pools = nn.ModuleDict({
            'pool_12_to_3': nn.AdaptiveAvgPool2d(3),
            'pool_6_to_3': nn.AdaptiveAvgPool2d(3),
            'pool_3_to_3': nn.Identity()
        })
        
        # Final fusion with attention
        self.final_fusion = nn.Sequential(
            nn.Conv2d(256 * 3, 511, kernel_size=1, bias=False),  # 768 -> 511 (divisible by 7)
            nn.BatchNorm2d(511),
            nn.ReLU(inplace=True)
        )
        
        # Emotion-specific feature enhancement
        self.emotion_enhancement = nn.Sequential(
            nn.Conv2d(511, 511, kernel_size=3, padding=1, groups=7, bias=False),  # 511/7 = 73 channels per group
            nn.BatchNorm2d(511),
            nn.ReLU(inplace=True),
            nn.Conv2d(511, 512, kernel_size=1, bias=False),  # Project back to 512 for consistency
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced classifier with progressive dropout
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2, 512),  # *2 for avg+max pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(SimpleResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(SimpleResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Multi-scale feature learning with progressive attention
        x1 = self.layer1(x)                    # 128 channels, 12x12
        x1_att = self.layer1_attention(x1)     # ECA attention for efficient channel focus
        
        x2 = self.layer2(x1_att)               # 256 channels, 6x6  
        x2_att = self.layer2_attention(x2)     # Coordinate attention for spatial awareness
        
        x3 = self.layer3(x2_att)               # 512 channels, 3x3
        
        # Triple attention at final layer for complex emotion discrimination
        x3_se = self.layer3_se(x3)             # Squeeze-and-Excitation
        x3_cbam = self.layer3_cbam(x3_se)      # Convolutional Block Attention
        x3_self = self.layer3_self_attn(x3_cbam)  # Multi-Head Self-Attention
        
        # Cross-scale feature fusion
        # Project all layers to same channel dimension (256)
        x1_proj = self.cross_scale_fusion['layer1_proj'](x1_att)
        x2_proj = self.cross_scale_fusion['layer2_proj'](x2_att)
        x3_proj = self.cross_scale_fusion['layer3_proj'](x3_self)
        
        # Resize to same spatial dimensions (3x3)
        x1_pooled = self.adaptive_pools['pool_12_to_3'](x1_proj)
        x2_pooled = self.adaptive_pools['pool_6_to_3'](x2_proj)
        x3_pooled = self.adaptive_pools['pool_3_to_3'](x3_proj)
        
        # Fuse multi-scale features
        fused = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)  # 768 channels
        fused = self.final_fusion(fused)  # 768 -> 511 channels
        
        # Emotion-specific enhancement
        enhanced = self.emotion_enhancement(fused)
        
        # Dual global pooling for richer feature representation
        avg_pooled = self.global_avg_pool(enhanced).view(enhanced.size(0), -1)
        max_pooled = self.global_max_pool(enhanced).view(enhanced.size(0), -1)
        combined_pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # 1024 features
        
        # Final classification
        output = self.emotion_classifier(combined_pooled)
        
        return output


def create_emotion_model(num_classes=7, dropout_rate=0.3, backbone=None, img_size=48, extreme_aug=False):
    """Factory function to create enhanced multi-scale SOTA model"""
    return SOTAEmotionNet(num_classes=num_classes, dropout_rate=dropout_rate) 