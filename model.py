import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from config import *
import timm # type: ignore
import math
import os


# ==============================================================================
# CONSTANTS
# ==============================================================================
# Gradient Accumulation Steps - to handle large batch sizes effectively
ACCUMULATION_STEPS = int(os.environ.get('ACCUMULATION_STEPS', max(1, BATCH_SIZE // 512)))


# ==============================================================================
# ATTENTION MODULES
# ==============================================================================
class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) module.
    
    Improves standard channel attention by using 1D convolution
    instead of fully connected layers, making it more efficient.
    
    Args:
        channels: Number of input channels
        gamma, b: Parameters to determine kernel size
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Calculate kernel size based on channel dimensions
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Perform global average pooling
        y = self.avg_pool(x)
        # Reshape for 1D convolution
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # Reshape back to original format
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        # Apply attention scaling
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention.
    
    Captures channel-wise dependencies to recalibrate feature maps.
    
    Args:
        channel: Number of input channels
        reduction: Reduction ratio for the bottleneck
    """
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling for better feature capture
        
        # Feature dimension reduction and expansion
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
            nn.SiLU(inplace=True),  # Using SiLU (Swish) for better gradient flow
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Both average and max pooling for better feature representation
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        # Concatenate and process features
        y = self.fc(torch.cat([avg_out, max_out], dim=1)).view(b, c, 1, 1)
        # Apply attention scaling
        return x * y.expand_as(x)


# ==============================================================================
# VISION TRANSFORMER MODEL
# ==============================================================================
class EmotionViT(nn.Module):
    """Vision Transformer model optimized for emotion recognition.
    
    Features:
    - Self-attention module for better feature refinement
    - Enhanced classification head with improved normalization
    - Multi-head attention for token refinement
    
    Args:
        backbone: Base vision transformer model from timm
        pretrained: Whether to use pretrained weights
    """
    def __init__(self, backbone="vit_base_patch16_224", pretrained=True):
        super(EmotionViT, self).__init__()
        from config import NUM_CLASSES, HEAD_DROPOUT
        
        # Enable memory format optimization if CUDA is available
        self.channels_last = torch.cuda.is_available()
        
        # Load Vision Transformer backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Get backbone output features
        feature_dim = self.backbone.num_features

        # Enhanced classifier head with normalization layers
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 768),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.LayerNorm(768),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT * 0.8),
            nn.LayerNorm(384),
            nn.Linear(384, NUM_CLASSES)
        )
        
        # Additional self-attention for token refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for attention outputs
        self.attention_norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights with appropriate distributions."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                if m == list(self.head.modules())[-1]:  # Last layer
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """Forward pass through the model."""
        # Optimize memory format if on CUDA
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
            
        # Extract features from backbone
        features = self.backbone.forward_features(x)
        
        # Process features if they include token embeddings (ViT output)
        if len(features.shape) == 3:
            # Apply self-attention with residual connection
            attn_output, _ = self.self_attention(features, features, features)
            features = features + attn_output
            features = self.attention_norm(features)
            
            # Get class token (first token)
            features = features[:, 0]
        
        # Apply classification head
        x = self.head(features)
        return x


# ==============================================================================
# CONVNEXT MODEL
# ==============================================================================
class ConvNeXtEmoteNet(nn.Module):
    """ConvNeXt-based model enhanced for emotion recognition.
    
    Features:
    - Multi-scale attention mechanism (ECA + SE + CBAM)
    - Weighted feature pyramid for multi-scale feature aggregation
    - Residual connections for better gradient flow
    - Dynamic channel calibration
    - SiLU activations in classifier head
    
    Args:
        backbone: Base ConvNeXt model from timm
        pretrained: Whether to use pretrained weights
    """
    def __init__(self, backbone="convnext_base", pretrained=True):
        super(ConvNeXtEmoteNet, self).__init__()
        from config import NUM_CLASSES, HEAD_DROPOUT, FEATURE_DROPOUT
        
        # Enable memory optimization for CUDA
        self.channels_last = torch.cuda.is_available()
        
        # Load ConvNeXt backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''   # Remove pooling
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Determine feature dimensions based on backbone variant
        if 'tiny' in backbone:
            feature_dim = 768
        elif 'small' in backbone:
            feature_dim = 768
        elif 'base' in backbone:
            feature_dim = 1024
        else:  # large
            feature_dim = 1536
        
        # CBAM attention module (combined channel and spatial attention)
        self.cbam = CBAM(feature_dim, reduction_ratio=8)
        
        # Multi-scale attention mechanisms for better feature focus
        self.eca_attention = ECABlock(feature_dim)
        self.se_attention = SEBlock(feature_dim)
        
        # Dynamic channel calibration
        self.channel_calibration = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim // 4, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature refinement with batch normalization
        self.feature_refine = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),  # Using SiLU instead of GELU
            nn.Dropout2d(FEATURE_DROPOUT)
        )
        
        # Multi-scale feature pyramid
        self.pyramid_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, 512, kernel_size=3, padding=i+1, dilation=i+1),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True)
            ) for i in range(3)  # 3 parallel paths with different receptive fields
        ])
        
        # Pyramid fusion
        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True)
        )
        
        # Residual connection for feature refinement
        self.feature_proj = nn.Conv2d(feature_dim, 1024, kernel_size=1) if feature_dim != 1024 else nn.Identity()
        
        # Global pooling operations
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Improved pooling with learnable weights
        self.pool_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        
        # Enhanced classifier with layer normalization and SiLU
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 768),
            nn.LayerNorm(768),
            nn.SiLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT * 0.8),
            
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT * 0.6),
            
            nn.Linear(192, NUM_CLASSES)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights with appropriate distributions."""
        # Initialize the new layers we added
        for m in [self.feature_refine, self.classifier, self.feature_proj, 
                 self.pyramid_fusion, self.channel_calibration, *self.pyramid_layers]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    # Use specialized initialization for final layer
                    if layer == list(self.classifier.modules())[-1]:
                        nn.init.normal_(layer.weight, 0, 0.001)
                    else:
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the model."""
        # Optimize memory format if on CUDA
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply multi-scale attention mechanisms
        cbam_attended = self.cbam(features)
        eca_attended = self.eca_attention(features)
        se_attended = self.se_attention(features)
        
        # Combine attention mechanisms using learned weights
        # Add residual connection from original features
        features = cbam_attended + eca_attended + se_attended + features
        
        # Apply dynamic channel calibration
        channel_weights = self.channel_calibration(features)
        features = features * channel_weights
        
        # Apply multi-scale feature pyramid
        pyramid_features = []
        for layer in self.pyramid_layers:
            pyramid_features.append(layer(features))
        
        # Fuse pyramid features
        pyramid_output = self.pyramid_fusion(torch.cat(pyramid_features, dim=1))
        
        # Apply feature refinement with residual connection
        refined = self.feature_refine(features)
        residual = self.feature_proj(features)
        refined = refined + residual + pyramid_output
        
        # Apply weighted pooling with learnable weights
        pool_weights = F.softmax(self.pool_weights, dim=0)
        avg_pool = self.global_avg_pool(refined).view(refined.size(0), -1)
        max_pool = self.global_max_pool(refined).view(refined.size(0), -1)
        
        # Weighted combination of pooling methods
        x = torch.cat([
            pool_weights[0] * avg_pool + pool_weights[1] * max_pool,
            max_pool
        ], dim=1)
        
        # Apply dropout and classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


# CBAM: Convolutional Block Attention Module
class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention for better feature refinement.
    
    Args:
        channels: Number of input channels
        reduction_ratio: Channel reduction ratio for efficiency
    """
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Apply spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        
        # Apply both attentions
        return x * spatial_weights


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================
class FocalLoss(nn.Module):
    """Enhanced Focal Loss with label smoothing.
    
    This loss function focuses more on hard examples and
    reduces overfitting through label smoothing.
    
    Args:
        alpha: Weighting factor
        gamma: Focusing parameter (higher = more focus on hard examples)
        class_weights: Optional weights for class imbalance
        label_smoothing: Label smoothing factor (0-1)
    """
    def __init__(self, alpha=1, gamma=2, class_weights=None, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        from config import NUM_CLASSES
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.device = None
        self.label_smoothing = label_smoothing
        self.num_classes = NUM_CLASSES

    def forward(self, inputs, targets):
        """Forward pass to compute loss."""
        if self.device is None:
            self.device = inputs.device
            
        # Handle both one-hot encoded and class index targets
        if len(targets.shape) > 1 and targets.shape[1] == self.num_classes:
            # For mixup/cutmix (soft targets)
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -(targets * log_probs).sum(dim=1)
            
            # Apply focal weighting
            pt = torch.exp(-loss)
            focal_weight = self.alpha * (1 - pt) ** self.gamma
            return (focal_weight * loss).mean()
        else:
            # For standard class indices
            if self.class_weights is not None:
                weight = torch.FloatTensor(self.class_weights).to(self.device)
                BCE_loss = F.cross_entropy(inputs, targets, weight=weight, 
                                        reduction='none', label_smoothing=self.label_smoothing)
            else:
                BCE_loss = F.cross_entropy(inputs, targets, 
                                        reduction='none', label_smoothing=self.label_smoothing)
                
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            return F_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function with:
    1. Focal loss for handling class imbalance
    2. Label smoothing for regularization
    3. KL divergence for soft targets
    """
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, label_smoothing=0.1, kl_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma  # Higher gamma means more focus on hard examples
        self.label_smoothing = label_smoothing
        self.kl_weight = kl_weight
        
        # Setup class weights for handling class imbalance
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights).to(self.get_device())
        else:
            self.class_weights = None
    
    def get_device(self):
        # Helper to get current device
        if hasattr(self, 'weight') and self.weight is not None:
            return self.weight.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, inputs, targets):
        # If targets are one-hot encoded (for mixup/cutmix), use KL divergence
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            ce_loss = -torch.sum(F.log_softmax(inputs, dim=1) * targets, dim=1)
            if self.class_weights is not None:
                # Weight each sample based on the dominant class
                _, dominant_class = targets.max(1)
                weights = self.class_weights[dominant_class]
                ce_loss = ce_loss * weights
            
            # Apply focal weighting to focus on hard examples
            probs = torch.sum(F.softmax(inputs, dim=1) * targets, dim=1)
            focal_weight = (1 - probs) ** self.gamma
            focal_loss = focal_weight * ce_loss
            
            return focal_loss.mean()
        
        # For standard classification with integer labels
        else:
            # CrossEntropy with label smoothing
            smoothed_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smoothed_targets = smoothed_targets * (1.0 - self.label_smoothing) + \
                              self.label_smoothing / inputs.shape[1]
            
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -torch.sum(log_probs * smoothed_targets, dim=1)
            
            if self.class_weights is not None:
                # Apply class weights
                weights = self.class_weights[targets]
                ce_loss = ce_loss * weights
            
            # Apply focal weighting for hard examples
            probs = torch.gather(F.softmax(inputs, dim=1), 1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - probs) ** self.gamma
            focal_loss = focal_weight * ce_loss
            
            return focal_loss.mean()


class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss for model distillation.
    
    Enables learning from a teacher model's outputs.
    
    Args:
        alpha: Balance between hard and soft targets (0-1)
        temperature: Temperature for softening probability distributions
    """
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels=None, criterion=None):
        """Compute distillation loss between student and teacher outputs."""
        # Calculate soft target loss (KL divergence)
        soft_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_targets = soft_targets.detach()  # No gradient through teacher
        
        distill_loss = self.criterion_kl(soft_log_probs, soft_targets) * (self.temperature ** 2)
        
        # Combine with hard target loss if provided
        if labels is not None and criterion is not None:
            hard_loss = criterion(student_logits, labels)
            return self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss
        else:
            return distill_loss