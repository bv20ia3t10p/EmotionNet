"""
Enhanced SOTA EmotionNet Model - 79%+ Target
Advanced architecture with state-of-the-art components for superior emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from attention_modules import SEBlock, CBAM, ECABlock, CoordinateAttention, MultiHeadSelfAttention


class EnhancedResidualBlock(nn.Module):
    """Enhanced Residual Block with Stochastic Depth and Squeeze-Excitation"""
    def __init__(self, in_ch, out_ch, stride=1, drop_path_rate=0.1):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Add SE block for channel attention
        self.se = SEBlock(out_ch, reduction=16)
        
        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
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
        out = self.se(out)  # Apply SE attention
        out = self.drop_path(out) + identity
        out = F.relu(out)
        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module for multi-scale context"""
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPooling, self).__init__()
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        for size in sizes[1:]:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = []
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        return torch.cat(out, dim=1)


class EmotionSpecificHead(nn.Module):
    """Emotion-specific classification head with multi-task learning"""
    def __init__(self, in_features, num_classes=7):
        super(EmotionSpecificHead, self).__init__()
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Emotion-specific branches
        self.emotion_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ) for _ in range(num_classes)
        ])
        
        # Global classifier
        self.global_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        shared = self.shared_fc(x)
        
        # Get emotion-specific predictions
        emotion_scores = []
        for branch in self.emotion_branches:
            emotion_scores.append(branch(shared))
        emotion_scores = torch.cat(emotion_scores, dim=1)
        
        # Get global predictions
        global_scores = self.global_classifier(shared)
        
        # Combine predictions
        combined_scores = 0.7 * global_scores + 0.3 * emotion_scores
        
        return combined_scores


class EnhancedSOTAEmotionNet(nn.Module):
    """Enhanced SOTA EmotionNet for 79%+ accuracy"""
    def __init__(self, num_classes=7, dropout_rate=0.35, use_pretrained_backbone=True):
        super(EnhancedSOTAEmotionNet, self).__init__()
        
        if use_pretrained_backbone:
            # Use EfficientNet-B0 backbone pretrained on ImageNet
            self.backbone = create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=0)
            backbone_dim = 1280
        else:
            # Custom backbone
            backbone_dim = 512
            self.backbone = self._create_custom_backbone()
        
        # Multi-scale context module
        pyramid_out_channels = backbone_dim // 8  # Reduce to prevent dimension explosion
        self.pyramid_pooling = PyramidPooling(backbone_dim, pyramid_out_channels)
        
        # Calculate pyramid dimensions dynamically
        # We need to test with a dummy input to get actual dimensions
        self.backbone_dim = backbone_dim
        self.pyramid_out_channels = pyramid_out_channels
        
        # Initialize feature reduction as None, will be created in first forward pass
        self.feature_reduce = None
        
        # Advanced attention mechanisms with consistent dimensions
        self.channel_attention = None
        self.spatial_attention = None
        self.self_attention = None
        
        # Feature fusion
        self.feature_fusion = None
        
        # Dual pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Advanced classifier with emotion-specific heads
        self.classifier = EmotionSpecificHead(256 * 2, num_classes)
        
        # Auxiliary outputs for deep supervision
        self.aux_classifier = nn.Linear(backbone_dim, num_classes)
        
        self._initialize_weights()
        self._initialized = False
    
    def _create_custom_backbone(self):
        """Create custom backbone when not using pretrained"""
        layers = []
        
        # Initial layers
        layers.extend([
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        
        # Residual stages with increasing drop path rate
        channels = [64, 128, 256, 512]
        blocks = [2, 3, 4, 3]
        drop_path_rates = torch.linspace(0, 0.2, sum(blocks))
        
        idx = 0
        for i in range(len(channels) - 1):
            for j in range(blocks[i]):
                stride = 2 if j == 0 and i > 0 else 1
                layers.append(EnhancedResidualBlock(
                    channels[i] if j == 0 else channels[i+1],
                    channels[i+1],
                    stride=stride,
                    drop_path_rate=drop_path_rates[idx].item()
                ))
                idx += 1
        
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
    
    def _initialize_layers(self, pyramid_features):
        """Initialize layers based on actual pyramid feature dimensions"""
        actual_pyramid_dim = pyramid_features.size(1)
        
        # Feature reduction before attention to prevent dimension issues
        self.feature_reduce = nn.Sequential(
            nn.Conv2d(actual_pyramid_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ).to(pyramid_features.device)
        
        # Advanced attention mechanisms with consistent dimensions
        self.channel_attention = SEBlock(512, reduction=8).to(pyramid_features.device)
        self.spatial_attention = CBAM(512, reduction=8).to(pyramid_features.device)
        self.self_attention = MultiHeadSelfAttention(512, num_heads=8, dropout=0.1).to(pyramid_features.device)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ).to(pyramid_features.device)
        
        self._initialized = True
        print(f"✅ Dynamic layer initialization: {actual_pyramid_dim} → 512 → 256 channels")
    
    def forward(self, x):
        # Extract features
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            # Get spatial dimensions
            if len(features.shape) == 2:
                # Global pooling already applied
                aux_out = self.aux_classifier(features)
                # Reshape for spatial operations
                B = features.shape[0]
                features = features.view(B, -1, 1, 1)
            else:
                aux_out = self.aux_classifier(self.avg_pool(features).view(features.size(0), -1))
        else:
            features = self.backbone(x)
            aux_out = self.aux_classifier(self.avg_pool(features).view(features.size(0), -1))
        
        # Multi-scale context
        pyramid_features = self.pyramid_pooling(features)
        
        # Initialize layers on first forward pass
        if not self._initialized:
            self._initialize_layers(pyramid_features)
        
        # Reduce dimensions before attention
        reduced_features = self.feature_reduce(pyramid_features)
        
        # Apply attention mechanisms
        attended_features = self.channel_attention(reduced_features)
        attended_features = self.spatial_attention(attended_features)
        attended_features = self.self_attention(attended_features)
        
        # Feature fusion
        fused_features = self.feature_fusion(attended_features)
        
        # Dual pooling
        avg_pooled = self.avg_pool(fused_features).view(fused_features.size(0), -1)
        max_pooled = self.max_pool(fused_features).view(fused_features.size(0), -1)
        combined_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        # Return main output and auxiliary output for training
        if self.training:
            return output, aux_out
        else:
            return output


def create_emotion_model(num_classes=7, dropout_rate=0.35, use_pretrained_backbone=True, **kwargs):
    """Create the enhanced SOTA emotion recognition model."""
    print("🚀 Creating Enhanced SOTA EmotionNet for 79%+ accuracy")
    return EnhancedSOTAEmotionNet(
        num_classes=num_classes, 
        dropout_rate=dropout_rate,
        use_pretrained_backbone=use_pretrained_backbone,
        **kwargs
    ) 