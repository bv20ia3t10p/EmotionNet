import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Dict, List, Optional, Tuple

class AttentionBlock(nn.Module):
    def __init__(self, in_features, reduction_ratio=8):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction_ratio, in_features, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pool
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pool
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        y = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * y

class TripleAttention(nn.Module):
    """
    Triple Attention module that combines channel, spatial, and relation attention
    to enhance feature representation across multiple dimensions.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(TripleAttention, self).__init__()
        self.in_channels = in_channels
        
        # Channel Attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Relation Attention (self-attention mechanism)
        self.relation_query = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relation_key = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relation_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Channel Attention
        channel_att = self.channel_gate(x)
        channel_out = x * channel_att
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_gate(spatial_in)
        spatial_out = x * spatial_att
        
        # Relation Attention (simplified self-attention)
        q = self.relation_query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B, HW, C/r
        k = self.relation_key(x).view(batch_size, -1, H * W)  # B, C/r, HW
        v = self.relation_value(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B, HW, C
        
        # Compute attention map (B, HW, HW)
        energy = torch.bmm(q, k)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to value (B, HW, C)
        out = torch.bmm(attention, v)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, C, H, W)
        relation_out = self.gamma * out + x
        
        # Concatenate and fuse all attention outputs
        combined = torch.cat([channel_out, spatial_out, relation_out], dim=1)
        output = self.fusion(combined)
        
        return output

class DynamicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super(DynamicResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = TripleAttention(out_channels)  # Using Triple Attention now
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out

class BackboneFactory:
    """
    Factory for creating different backbone architectures
    """
    @staticmethod
    def create_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        """
        Create a backbone model with the given name.
        
        Args:
            name: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            
        Returns:
            Tuple of (backbone model, output features)
        """
        # ResNet variants
        if name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            return model, 512
        elif name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            return model, 512
        elif name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            return model, 2048
            
        # EfficientNet variants
        elif name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            return model, 1280
        elif name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=pretrained)
            return model, 1280
        elif name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=pretrained)
            return model, 1408
            
        # Vision Transformers (timm library)
        elif name == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            return model, 384
        elif name == 'vit_base':
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            return model, 768
            
        # MobileNet variants
        elif name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            return model, 1280
        elif name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)
            return model, 576
        elif name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            return model, 960
            
        # ConvNext variants
        elif name == 'convnext_tiny':
            model = models.convnext_tiny(pretrained=pretrained)
            return model, 768
        elif name == 'convnext_small':
            model = models.convnext_small(pretrained=pretrained)
            return model, 768
        elif name == 'convnext_base':
            model = models.convnext_base(pretrained=pretrained)
            return model, 1024
        elif name == 'convnext_large':
            model = models.convnext_large(pretrained=pretrained)
            return model, 1536
        elif name == 'convnext_xlarge':
            # ConvNeXt XLarge is not in torchvision, use timm instead
            model = timm.create_model('convnext_xlarge.in22k_ft_in1k', pretrained=pretrained)
            return model, 2048
            
        else:
            raise ValueError(f"Unsupported backbone: {name}")

class EmotionClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int = 7, 
        backbone_name: str = 'resnet34',
        dropout_rate: float = 0.5, 
        use_attention: bool = True,
        pretrained: bool = True
    ):
        super(EmotionClassifier, self).__init__()
        
        # Create backbone
        backbone, feature_size = BackboneFactory.create_backbone(backbone_name, pretrained)
        
        # Remove classification head from backbone
        if 'resnet' in backbone_name:
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif 'efficientnet' in backbone_name:
            self.backbone = backbone.features
        elif 'mobilenet' in backbone_name:
            if 'v2' in backbone_name:
                self.backbone = backbone.features
            else:  # v3
                self.backbone = backbone.features
        elif 'vit' in backbone_name:
            # For ViT, we keep the whole model but replace the head later
            self.backbone = backbone
            self.is_vit = True
        elif 'convnext' in backbone_name:
            # For ConvNeXt from timm
            if 'xlarge' in backbone_name:
                self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            else:
                self.backbone = backbone.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.is_vit = 'vit' in backbone_name
        self.feature_size = feature_size
        self.backbone_name = backbone_name
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1) if not self.is_vit else nn.Identity()
        
        # Additional processing blocks (only for CNN backbones)
        if not self.is_vit:
            self.residual_block = DynamicResidualBlock(feature_size, feature_size, dropout_rate=dropout_rate)
            self.attention = TripleAttention(feature_size) if use_attention else nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights for the new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize the new layers
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.is_vit:
            # For ViT, extract the classification token
            x = self.backbone.forward_features(x)
        else:
            # For CNN backbones
            x = self.backbone(x)
            x = self.residual_block(x)
            x = self.attention(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        
        # Classification head
        x = self.classifier(x)
        return x

class ModelEnsemble(nn.Module):
    """
    An ensemble of multiple emotion classification models
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Initialize weights if not provided
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = torch.tensor(weights) / sum(weights)
    
    def update_weights(self, val_accuracies: List[float]):
        """
        Update model weights based on validation accuracies
        """
        # Convert to tensor
        accuracies = torch.tensor(val_accuracies)
        
        # Softmax to convert to probabilities
        self.weights = F.softmax(accuracies, dim=0)
        
        print(f"Updated ensemble weights: {self.weights.tolist()}")
    
    def forward(self, x):
        outputs = []
        
        # Get outputs from each model
        for i, model in enumerate(self.models):
            model_output = model(x)
            outputs.append(model_output)
        
        # Stack and apply weights
        stacked = torch.stack(outputs, dim=0)
        weighted = stacked * self.weights.view(-1, 1, 1)
        
        # Sum along the model dimension
        return torch.sum(weighted, dim=0)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """
        Add a new model to the ensemble
        """
        self.models.append(model)
        
        # Update weights
        weights = self.weights.tolist()
        weights.append(weight)
        total = sum(weights)
        self.weights = torch.tensor([w / total for w in weights])

class EmotionSpecificBlock(nn.Module):
    """Emotion-specific feature learning block"""
    def __init__(self, in_channels, num_emotions=7, reduction_ratio=8):
        super().__init__()
        self.num_emotions = num_emotions
        
        # Emotion-specific attention
        self.emotion_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_emotions)
        ])
        
        # Emotion-specific feature transformation
        self.emotion_transform = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_emotions)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_emotions, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        emotion_features = []
        
        # Process each emotion-specific branch
        for i in range(self.num_emotions):
            # Apply emotion-specific attention
            att = self.emotion_attention[i](x)
            attended = x * att
            
            # Apply emotion-specific transformation
            transformed = self.emotion_transform[i](attended)
            emotion_features.append(transformed)
        
        # Concatenate and fuse all emotion-specific features
        combined = torch.cat(emotion_features, dim=1)
        output = self.fusion(combined)
        
        return output

class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion module with improved feature handling"""
    def __init__(self, channels_list, fusion_channels=256):
        super().__init__()
        self.channels_list = channels_list
        
        # Feature reduction layers for each scale
        self.reduction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, fusion_channels, 1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            ) for channels in channels_list
        ])
        
        # Spatial attention for each scale
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, padding=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ) for _ in channels_list
        ])
        
        # Channel attention for each scale
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(fusion_channels, fusion_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(fusion_channels // 4, fusion_channels, 1),
                nn.Sigmoid()
            ) for _ in channels_list
        ])
        
        # Final fusion with residual connection
        self.final_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels * len(channels_list), fusion_channels, 1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list):
        # Process each scale
        fused_features = []
        target_size = None
        
        # First pass: process all features and find target size
        for i, (feature, reduction_layer, spatial_att, channel_att) in enumerate(
            zip(features_list, self.reduction_layers, self.spatial_attention, self.channel_attention)
        ):
            # Reduce channels
            x = reduction_layer(feature)
            
            # Apply spatial attention
            avg_pool = torch.mean(x, dim=1, keepdim=True)
            max_pool, _ = torch.max(x, dim=1, keepdim=True)
            spatial_in = torch.cat([avg_pool, max_pool], dim=1)
            spatial_weight = spatial_att(spatial_in)
            x = x * spatial_weight
            
            # Apply channel attention
            channel_weight = channel_att(x)
            x = x * channel_weight
            
            # For the first feature map, store its size as target
            if i == 0:
                target_size = x.shape[-2:]
            
            # Ensure all features have the same spatial dimensions
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
            fused_features.append(x)
        
        # Concatenate all features along channel dimension
        combined = torch.cat(fused_features, dim=1)
        
        # Apply final fusion with residual connection
        fused = self.final_fusion(combined)
        
        # Project to output
        output = self.output_proj(fused)
        
        return output

class CustomFER(nn.Module):
    """
    Enhanced custom CNN architecture for facial emotion recognition
    without using pre-trained backbones, optimized for 64x64 input size.
    """
    def __init__(self, num_classes=7, dropout_rate=0.4):
        super().__init__()
        
        # Initial feature extraction with adjusted channel sizes for 64x64 input
        self.features = nn.ModuleList([
            # First conv block - capture basic features (64x64 -> 32x32)
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate * 0.5)
            ),
            
            # Second conv block - capture more complex features (32x32 -> 16x16)
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate * 0.5)
            ),
            
            # Third conv block - capture high-level features (16x16 -> 8x8)
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate)
            ),
            
            # Fourth conv block - capture fine-grained features (8x8 -> 4x4)
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate)
            )
        ])
        
        # Multiple attention blocks at different scales
        self.attention_blocks = nn.ModuleList([
            TripleAttention(64, reduction_ratio=8),    # After first block
            TripleAttention(128, reduction_ratio=8),   # After second block
            TripleAttention(256, reduction_ratio=8),   # After third block
            TripleAttention(512, reduction_ratio=8)    # After fourth block
        ])
        
        # Emotion-specific feature learning
        self.emotion_blocks = nn.ModuleList([
            EmotionSpecificBlock(64, num_classes),     # After first block
            EmotionSpecificBlock(128, num_classes),    # After second block
            EmotionSpecificBlock(256, num_classes),    # After third block
            EmotionSpecificBlock(512, num_classes)     # After fourth block
        ])
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            channels_list=[64, 128, 256, 512],
            fusion_channels=256
        )
        
        # Global context with adjusted capacity
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Emotion-specific feature enhancement
        self.emotion_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction with attention and emotion-specific processing at each stage
        features = []
        for i, (feature_block, attention_block, emotion_block) in enumerate(
            zip(self.features, self.attention_blocks, self.emotion_blocks)
        ):
            # Extract features
            x = feature_block(x)
            
            # Apply attention
            x = attention_block(x)
            
            # Apply emotion-specific processing
            x = emotion_block(x)
            
            # Store features for multi-scale fusion
            features.append(x)
        
        # Multi-scale feature fusion
        x = self.feature_fusion(features)
        
        # Global context
        x = self.global_context(x)
        
        # Emotion-specific features
        x = self.emotion_features(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Factory function to create a model
def create_model(
    model_type: str = 'single', 
    backbone_name: str = 'resnet34',
    num_classes: int = 7, 
    dropout_rate: float = 0.5,
    ensemble_size: int = 3,
    **kwargs
) -> nn.Module:
    """
    Create a model based on the specified type.
    
    Args:
        model_type: 'single', 'ensemble', or 'custom'
        backbone_name: Name of the backbone architecture (ignored for custom model)
        num_classes: Number of output classes
        dropout_rate: Dropout rate to use
        ensemble_size: Number of models in the ensemble (only for ensemble type)
        
    Returns:
        The created model
    """
    if model_type == 'custom':
        return CustomFER(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs
        )
    elif model_type == 'single':
        return EmotionClassifier(
            num_classes=num_classes, 
            backbone_name=backbone_name,
            dropout_rate=dropout_rate,
            **kwargs
        )
    elif model_type == 'ensemble':
        # Create ensemble with different backbones
        backbones = [
            'resnet34',
            'efficientnet_b0',
            'mobilenet_v3_large',
            'convnext_tiny',
            'vit_small'
        ]
        
        # Select the specified number of backbones
        selected_backbones = backbones[:ensemble_size]
        
        # Create the models
        models = [
            EmotionClassifier(
                num_classes=num_classes,
                backbone_name=bb,
                dropout_rate=dropout_rate,
                **kwargs
            )
            for bb in selected_backbones
        ]
        
        return ModelEnsemble(models)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 