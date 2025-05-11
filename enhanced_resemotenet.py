import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Multi-head Channel-Spatial Attention Module
class CSAM(nn.Module):
    """Channel-Spatial Attention Module for better feature focus"""
    def __init__(self, channels):
        super(CSAM, self).__init__()
        
        # Channel attention with multi-scale pooling
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.LayerNorm(channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention with convolutional layers
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Channel attention
        channel_avg = self.channel_avg_pool(x).view(batch_size, channels)
        channel_max = self.channel_max_pool(x).view(batch_size, channels)
        
        channel_avg_attn = self.channel_mlp(channel_avg).view(batch_size, channels, 1, 1)
        channel_max_attn = self.channel_mlp(channel_max).view(batch_size, channels, 1, 1)
        
        # Combined channel attention
        channel_attn = channel_avg_attn + channel_max_attn
        x_channel = x * channel_attn
        
        # Spatial attention
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_attn = self.spatial_conv(spatial_concat)
        
        # Apply spatial attention
        x_spatial = x_channel * spatial_attn
        
        return x_spatial

# Emotion-Specific Feature Enhancement Module
class EmotionFeatureEnhancer(nn.Module):
    """Module that enhances features specifically for emotion recognition"""
    def __init__(self, in_channels, reduction=4):
        super(EmotionFeatureEnhancer, self).__init__()
        
        # Global context encoding
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.LayerNorm([in_channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Local feature enhancement
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Enhance with global context
        context = self.global_context(x)
        global_enhanced = x * context
        
        # Enhance with local features
        local_weights = self.local_enhance(global_enhanced)
        
        # Residual connection
        output = global_enhanced * local_weights + x
        
        return output

# Multi-Branch Emotion Recognition Network
class MultiPathEmotionNet(nn.Module):
    """Advanced emotion recognition model with multi-path architecture"""
    def __init__(self, num_classes=7, dropout_rate=0.5, backbone='resnet18'):
        super(MultiPathEmotionNet, self).__init__()
        
        # Select backbone architecture
        if backbone == 'resnet18':
            base = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = 1280
        elif backbone == 'mobilenet_v3_small':
            base = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            feature_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create grayscale-compatible input layer
        if backbone == 'resnet18':
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                base.bn1,
                base.relu,
                base.maxpool
            )
            
            # Feature extraction stages
            self.stage1 = base.layer1  # 64 channels
            self.stage2 = base.layer2  # 128 channels
            self.stage3 = base.layer3  # 256 channels
            self.stage4 = base.layer4  # 512 channels
            
        elif backbone == 'efficientnet_b0':
            # Modify first conv for grayscale
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True)
            )
            
            # Get the main feature extraction blocks
            features = list(base.features.children())[1:]  # Skip first conv
            
            # Split into stages
            self.stage1 = nn.Sequential(*features[0:2])   # 16 channels
            self.stage2 = nn.Sequential(*features[2:4])   # 24 channels
            self.stage3 = nn.Sequential(*features[4:10])  # 112 channels
            self.stage4 = nn.Sequential(*features[10:])   # 1280 channels
            
        elif backbone == 'mobilenet_v3_small':
            # Modify first conv for grayscale
            self.stem = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.Hardswish(inplace=True)
            )
            
            # Get feature extraction blocks
            features = list(base.features.children())[1:]  # Skip first conv
            
            # Split into stages
            self.stage1 = nn.Sequential(*features[0:3])   # 24 channels 
            self.stage2 = nn.Sequential(*features[3:6])   # 40 channels
            self.stage3 = nn.Sequential(*features[6:9])   # 96 channels
            self.stage4 = nn.Sequential(*features[9:])    # 576 channels
        
        # Freeze early stages to prevent overfitting
        self._freeze_stages()
        
        # Feature enhancement and attention modules
        self.enhance_s3 = EmotionFeatureEnhancer(256 if backbone == 'resnet18' else 112)
        self.enhance_s4 = EmotionFeatureEnhancer(feature_dim)
        
        self.attn_s3 = CSAM(256 if backbone == 'resnet18' else 112)
        self.attn_s4 = CSAM(feature_dim)
        
        # Global context pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2d = nn.Dropout2d(0.3)
        
        # Specialized pathways for different emotion groups
        # Negative emotions: angry, disgust, fear, sad
        self.negative_path = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.9)
        )
        
        # Positive emotions: happy, surprise
        self.positive_path = nn.Sequential(
            nn.Linear(feature_dim, 256), 
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.9)
        )
        
        # Neutral emotions: neutral
        self.neutral_path = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.9)
        )
        
        # Common feature extractor
        self.common_path = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4, 512),  # Combine all pathways
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 1.3)
        )
        
        # Add class token attention to focus on important emotion features
        self.class_attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8, 
            dropout=dropout_rate * 0.5,
            batch_first=True
        )
        
        # Class tokens (learnable emotion prototypes)
        self.class_tokens = nn.Parameter(torch.zeros(num_classes, 512))
        nn.init.normal_(self.class_tokens, std=0.02)
        
        # Final classifiers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            # No activation - will be applied in loss function
        )
        
        # Valence-arousal regression heads
        self.valence_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()  # Bound between -1 and 1
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()  # Bound between -1 and 1
        )
        
        # Mid-level supervision
        self.mid_classifier = nn.Linear(256 if backbone == 'resnet18' else 112, num_classes)
        self.mid_valence = nn.Linear(256 if backbone == 'resnet18' else 112, 1)
        self.mid_arousal = nn.Linear(256 if backbone == 'resnet18' else 112, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _freeze_stages(self):
        """Freeze early stages to prevent overfitting"""
        # Freeze stem and early stages
        for param in self.stem.parameters():
            param.requires_grad = False
            
        for param in self.stage1.parameters():
            param.requires_grad = False
            
        for param in self.stage2.parameters():
            param.requires_grad = False
    
    def _initialize_weights(self):
        """Initialize new layers with better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.stem(x)
        
        # Feature extraction stages
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Mid-level features with enhancement
        mid_features = self.stage3(x)
        mid_features = self.enhance_s3(mid_features)
        mid_features = self.attn_s3(mid_features)
        
        # Mid-level supervision
        mid_pool = self.global_pool(mid_features).view(mid_features.size(0), -1)
        mid_pool = self.dropout(mid_pool)
        
        aux_logits = self.mid_classifier(mid_pool)
        aux_valence = self.mid_valence(mid_pool)
        aux_arousal = self.mid_arousal(mid_pool)
        
        # Final features
        x = self.stage4(mid_features)
        x = self.enhance_s4(x)
        x = self.attn_s4(x)
        
        # Apply 2D dropout for spatial regularization
        x = self.dropout2d(x)
        
        # Global pooling
        x = self.global_pool(x).view(x.size(0), -1)
        
        # Apply emotion-specific pathways in parallel
        negative_features = self.negative_path(x)
        positive_features = self.positive_path(x)
        neutral_features = self.neutral_path(x)
        common_features = self.common_path(x)
        
        # Feature fusion
        fused_features = torch.cat([
            negative_features, 
            positive_features, 
            neutral_features, 
            common_features
        ], dim=1)
        
        fused_features = self.fusion(fused_features)
        
        # Apply class token attention
        batch_size = fused_features.size(0)
        # Expand class tokens for the batch
        batch_tokens = self.class_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        # Reshape features for attention
        fused_features_expanded = fused_features.unsqueeze(1)
        
        # Apply self-attention between features and class tokens
        attn_output, _ = self.class_attention(
            batch_tokens,  # Query: class tokens
            fused_features_expanded,  # Key: features
            fused_features_expanded   # Value: features
        )
        
        # Mean across class dimension to get final features
        final_features = attn_output.mean(dim=1)
        
        # Final layer dropout
        if self.training:
            final_features = F.dropout(final_features, p=0.2, training=True)
        
        # Classification and regression heads
        logits = self.emotion_classifier(final_features)
        valence = self.valence_head(final_features)
        arousal = self.arousal_head(final_features)
        
        # Return both main and auxiliary outputs during training
        if self.training:
            return logits, valence, arousal, aux_logits, aux_valence, aux_arousal
        else:
            return logits, valence, arousal

# Maintain backward compatibility with the original class name
class EnhancedResEmoteNet(MultiPathEmotionNet):
    """Wrapper class for backward compatibility"""
    def __init__(self, num_classes=7, dropout_rate=0.65, backbone='resnet18'):
        super(EnhancedResEmoteNet, self).__init__(
            num_classes=num_classes, 
            dropout_rate=dropout_rate,
            backbone=backbone
        ) 