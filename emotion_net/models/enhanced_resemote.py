"""Enhanced ResEmote architecture for high-accuracy emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ROIAttention(nn.Module):
    """Region of Interest attention specifically for facial landmarks."""
    def __init__(self, in_channels):
        super().__init__()
        # Eyes region attention
        self.eyes_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Mouth region attention
        self.mouth_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Combined attention weights
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # For a 7x7 feature map, we can approximate regions
        b, c, h, w = x.shape
        
        # Extract eyes region (upper part of the feature map)
        upper_region = x[:, :, :h//2, :]
        eyes_mask = F.interpolate(
            self.eyes_attention(upper_region), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Extract mouth region (lower part of the feature map)
        lower_region = x[:, :, h//2:, :]
        mouth_mask = F.interpolate(
            self.mouth_attention(lower_region), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Combine attentions
        combined_features = torch.cat([eyes_mask * x, mouth_mask * x], dim=1)
        attention_weights = self.combine(combined_features)
        
        # Apply attention and add residual connection
        return x * attention_weights + x

class GEMPool(nn.Module):
    """Generalized Mean Pooling with learnable p-norm."""
    
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return self.gem(x, p=self.p)
    
    def gem(self, x, p):
        # Apply clamp for numerical stability
        x = x.clamp(min=self.eps)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p.data.tolist()[0]:.4f})'

class EmotionGroup(nn.Module):
    """Specialized module for processing each emotion group."""
    
    def __init__(self, in_dim, out_dim, num_emotions_in_group):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions_in_group),
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        features = self.fc(x)
        attention = self.attn(x)
        return self.norm(features), attention

class MultiHeadFusion(nn.Module):
    """Multi-head fusion module for different emotion groups."""
    
    def __init__(self, in_dim, embedding_size, num_classes):
        super().__init__()
        # Define emotion groups based on psychological similarity
        # Group 1: Anger, Disgust (negative, strong)
        # Group 2: Fear, Surprise (arousal, alertness)
        # Group 3: Happy (positive)
        # Group 4: Sad, Neutral (negative, low energy)
        
        self.group1 = EmotionGroup(in_dim, embedding_size, 2)  # anger, disgust
        self.group2 = EmotionGroup(in_dim, embedding_size, 2)  # fear, surprise
        self.group3 = EmotionGroup(in_dim, embedding_size, 1)  # happy
        self.group4 = EmotionGroup(in_dim, embedding_size, 2)  # sad, neutral
        
        # Global head (processes all emotions)
        self.global_head = nn.Sequential(
            nn.Linear(in_dim, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU()
        )
        
        # Final classifiers
        self.classifier_g1 = nn.Linear(embedding_size, 2)  # anger, disgust
        self.classifier_g2 = nn.Linear(embedding_size, 2)  # fear, surprise
        self.classifier_g3 = nn.Linear(embedding_size, 1)  # happy
        self.classifier_g4 = nn.Linear(embedding_size, 2)  # sad, neutral
        
        # Combined classifier
        self.classifier_global = nn.Linear(embedding_size, num_classes)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_size * 5, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Process through each emotion group
        g1_feat, g1_attn = self.group1(x)
        g2_feat, g2_attn = self.group2(x)
        g3_feat, g3_attn = self.group3(x)
        g4_feat, g4_attn = self.group4(x)
        global_feat = self.global_head(x)
        
        # Group-specific predictions
        g1_out = self.classifier_g1(g1_feat)
        g2_out = self.classifier_g2(g2_feat)
        g3_out = self.classifier_g3(g3_feat)
        g4_out = self.classifier_g4(g4_feat)
        
        # Fuse all features
        fused_features = torch.cat([g1_feat, g2_feat, g3_feat, g4_feat, global_feat], dim=1)
        fused_embedding = self.fusion(fused_features)
        
        # Final prediction
        global_out = self.classifier_global(fused_embedding)
        
        # Map group outputs to global class indices (for auxiliary losses)
        # FER2013 order: angry(0), disgust(1), fear(2), happy(3), sad(4), surprise(5), neutral(6)
        group_outputs = torch.zeros_like(global_out)
        
        # Map group1 outputs (anger, disgust) to positions 0,1
        group_outputs[:, 0:2] = g1_out
        
        # Map group2 outputs (fear, surprise) to positions 2,5
        group_outputs[:, 2] = g2_out[:, 0]
        group_outputs[:, 5] = g2_out[:, 1]
        
        # Map group3 output (happy) to position 3
        group_outputs[:, 3] = g3_out[:, 0]
        
        # Map group4 outputs (sad, neutral) to positions 4,6
        group_outputs[:, 4] = g4_out[:, 0]
        group_outputs[:, 6] = g4_out[:, 1]
        
        return global_out, group_outputs, fused_embedding

class EnhancedResEmote(nn.Module):
    """Enhanced ResEmote architecture with flexible backbone support."""
    
    def __init__(self, num_classes=7, pretrained=True, embedding_size=1024, 
                 drop_path_rate=0.2, use_gem_pooling=True, enforce_input_shape=None,
                 backbone_name='resnext50_32x4d', return_features=False):
        super().__init__()
        self.return_features = return_features
        
        # Initialize backbone
        if backbone_name == 'resnext50_32x4d':
            weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnext50_32x4d(weights=weights)
            backbone_out_dim = 2048
        else:
            # Use timm for other backbones
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                drop_path_rate=drop_path_rate
            )
            # Get output dimensions from the backbone
            out_indices = self.backbone.feature_info.info
            backbone_out_dim = out_indices[-1]['num_chs']
        
        # Remove the original classifier
        if hasattr(self.backbone, 'fc'):
            delattr(self.backbone, 'fc')
        if hasattr(self.backbone, 'head'):
            delattr(self.backbone, 'head')
        
        # Pooling layer
        self.pool = GEMPool() if use_gem_pooling else nn.AdaptiveAvgPool2d(1)
        
        # ROI attention for facial features
        self.roi_attention = ROIAttention(backbone_out_dim)
        
        # Channel and spatial attention
        self.channel_attention = ChannelAttention(backbone_out_dim)
        self.spatial_attention = SpatialAttention()
        
        # Multi-head fusion for emotion groups
        self.fusion = MultiHeadFusion(backbone_out_dim, embedding_size, num_classes)
        
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
    
    def extract_features(self, x):
        """Extract features before classification."""
        # Get backbone features
        if hasattr(self.backbone, 'forward_features'):
            # For timm models with forward_features method
            features = self.backbone.forward_features(x)
        elif 'resnext' in str(self.backbone.__class__.__name__).lower():
            # For torchvision ResNeXt models
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)
        elif hasattr(self.backbone, 'features') and callable(getattr(self.backbone, 'features')):
            # For models with features method
            features = self.backbone.features(x)
        else:
            # Generic approach for other models - access layers directly
            # This works for most torchvision models
            # Go through the backbone layers except for the final fc layer
            x = self.backbone.conv1(x) if hasattr(self.backbone, 'conv1') else x
            x = self.backbone.bn1(x) if hasattr(self.backbone, 'bn1') else x
            x = self.backbone.relu(x) if hasattr(self.backbone, 'relu') else x
            x = self.backbone.maxpool(x) if hasattr(self.backbone, 'maxpool') else x

            x = self.backbone.layer1(x) if hasattr(self.backbone, 'layer1') else x
            x = self.backbone.layer2(x) if hasattr(self.backbone, 'layer2') else x
            x = self.backbone.layer3(x) if hasattr(self.backbone, 'layer3') else x
            features = self.backbone.layer4(x) if hasattr(self.backbone, 'layer4') else x
        
        # Apply ROI attention
        features = self.roi_attention(features)
        
        # Apply channel attention
        channel_weights = self.channel_attention(features)
        features = features * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(features)
        features = features * spatial_weights
        
        # Pool features
        pooled = self.pool(features)
        return torch.flatten(pooled, 1)
    
    def forward(self, x):
        features = self.extract_features(x)
        
        # Multi-head fusion returns (global_out, group_outputs, fused_embedding)
        global_out, group_outputs, fused_embedding = self.fusion(features)
        
        # Return a tuple of (logits, aux_logits, features) for SOTAEmotionLoss
        # This format works with the SOTAEmotionLoss definition
        return global_out, group_outputs, fused_embedding

def enhanced_resemote(num_classes=7, pretrained=True, embedding_size=1024, 
                     use_gem_pooling=True, drop_path_rate=0.2, enforce_input_shape=None,
                     backbone_name='resnext50_32x4d', return_features=True):
    """Factory function for EnhancedResEmote model."""
    model = EnhancedResEmote(
        num_classes=num_classes,
        pretrained=pretrained,
        embedding_size=embedding_size,
        drop_path_rate=drop_path_rate,
        use_gem_pooling=use_gem_pooling,
        enforce_input_shape=enforce_input_shape,
        backbone_name=backbone_name,
        return_features=return_features
    )
    return model 