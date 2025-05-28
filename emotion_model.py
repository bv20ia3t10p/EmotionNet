"""
EmotionNet Model Architecture - SOTA Optimized
Simplified ResEmoteNet focusing on proven components for maximum performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_modules import SEBlock, CBAM


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
    """SOTA Optimized EmotionNet - Simplified for maximum performance"""
    def __init__(self, num_classes=7, dropout_rate=0.25):
        super(SOTAEmotionNet, self).__init__()
        
        # Initial convolution optimized for 48x48
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 48x48 -> 24x24
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 24x24 -> 12x12
        
        # Simplified feature extraction
        self.layer1 = self._make_layer(64, 128, 2, stride=1)    # 12x12 -> 12x12
        self.layer2 = self._make_layer(128, 256, 2, stride=2)   # 12x12 -> 6x6
        self.layer3 = self._make_layer(256, 512, 2, stride=2)   # 6x6 -> 3x3
        
        # Only 2 proven attention mechanisms
        self.se_attention = SEBlock(512, reduction=16)
        self.cbam_attention = CBAM(512, reduction=16)
        
        # Simple fusion
        self.fusion = nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(512)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Simplified classifier - fewer parameters
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
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
        
        # Feature learning
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Apply attention mechanisms
        se_out = self.se_attention(x)
        cbam_out = self.cbam_attention(x)
        
        # Fuse outputs
        fused = torch.cat([se_out, cbam_out], dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_bn(fused)
        fused = self.relu(fused)
        
        # Classification
        pooled = self.global_pool(fused).view(fused.size(0), -1)
        output = self.classifier(pooled)
        
        return output


def create_emotion_model(num_classes=7, dropout_rate=0.25, backbone=None, img_size=48, extreme_aug=False):
    """Factory function to create simplified SOTA model"""
    return SOTAEmotionNet(num_classes=num_classes, dropout_rate=dropout_rate) 