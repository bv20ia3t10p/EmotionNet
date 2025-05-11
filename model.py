import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from config import *
import timm # type: ignore
import math
import os
import numpy as np


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
    def __init__(self, backbone="convnext_large", pretrained=True):
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
        
        # Global Context Attention Module - captures long-range dependencies
        self.global_context = GlobalContextBlock(feature_dim, reduction_ratio=16)
        
        # Multi-scale attention mechanisms for better feature focus
        self.eca_attention = ECABlock(feature_dim)
        self.se_attention = SEBlock(feature_dim, reduction=4)
        self.cbam = CBAM(feature_dim, reduction_ratio=8)
        
        # Improved Dynamic Channel Calibration with bottleneck
        self.channel_calibration = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim // 4, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced Feature Refinement with normalization and residual connections
        self.feature_refine = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout2d(FEATURE_DROPOUT)
        )
        
        # Multi-scale feature pyramid with dilated convolutions for larger receptive field
        self.pyramid_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, 512, kernel_size=3, padding=i+1, dilation=i+1),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True)
            ) for i in range(3)  # 3 parallel paths with different receptive fields
        ])
        
        # Bottleneck-based Pyramid fusion with improved efficiency
        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=1),  # Reduce channels first
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=1),     # Then expand to final dimension
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True)
        )
        
        # Residual connection for feature refinement
        self.feature_proj = nn.Conv2d(feature_dim, 1024, kernel_size=1, bias=False) if feature_dim != 1024 else nn.Identity()
        
        # Enhanced global pooling with learnable weights
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        
        # Advanced classifier with layer normalization
        classifier_dim = 1024 * 2  # Doubled for concatenated pooling methods
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, 768),
            nn.LayerNorm(768),
            nn.SiLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT * 0.8),
            
            nn.Linear(384, NUM_CLASSES)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights with proper distributions for stable training."""
        for m in [self.feature_refine, self.classifier, self.feature_proj, 
                 self.pyramid_fusion, self.channel_calibration, self.global_context, 
                 *self.pyramid_layers]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(layer.weight)
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
        batch_size = x.shape[0]
        
        # Optimize memory format if on CUDA
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply multi-scale attention mechanisms with weighted combination
        cbam_features = self.cbam(features)
        eca_features = self.eca_attention(features)
        se_features = self.se_attention(features)
        gc_features = self.global_context(features)
        
        # Combine attention features with residual connection
        features = features + cbam_features + eca_features + se_features + gc_features
        
        # Apply dynamic channel calibration
        channel_weights = self.channel_calibration(features)
        calibrated_features = features * channel_weights
        
        # Apply multi-scale feature pyramid
        pyramid_features = []
        for pyramid_layer in self.pyramid_layers:
            pyramid_features.append(pyramid_layer(calibrated_features))
        
        # Fuse pyramid features
        pyramid_output = self.pyramid_fusion(torch.cat(pyramid_features, dim=1))
        
        # Apply feature refinement with residual connection
        refined_features = self.feature_refine(calibrated_features)
        residual_features = self.feature_proj(calibrated_features)
        fused_features = refined_features + residual_features + pyramid_output
        
        # Apply weighted global pooling
        pool_weights = F.softmax(self.pool_weights, dim=0)
        avg_pooled = self.global_avg_pool(fused_features).view(batch_size, -1)
        max_pooled = self.global_max_pool(fused_features).view(batch_size, -1)
        
        # Weighted combination of pooling outputs plus concatenation
        weighted_pool = pool_weights[0] * avg_pooled + pool_weights[1] * max_pooled
        pooled_features = torch.cat([weighted_pool, max_pooled], dim=1)
        
        # Apply classification
        logits = self.classifier(pooled_features)
        
        return logits

    def extract_features(self, x, return_features=True):
        """Extract intermediate features for knowledge distillation.
        
        Returns both the final predictions and intermediate feature maps
        that can be used for attention transfer in distillation.
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
            
        Returns:
            tuple: (logits, features) if return_features=True, else logits
        """
        batch_size = x.shape[0]
        
        # Optimize memory format if on CUDA
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features from backbone
        backbone_features = self.backbone(x)
        
        # Apply attention mechanisms
        cbam_features = self.cbam(backbone_features)
        eca_features = self.eca_attention(backbone_features)
        se_features = self.se_attention(backbone_features)
        gc_features = self.global_context(backbone_features)
        
        # Combine attention features with residual connection
        features = backbone_features + cbam_features + eca_features + se_features + gc_features
        
        # Apply dynamic channel calibration
        channel_weights = self.channel_calibration(features)
        calibrated_features = features * channel_weights
        
        # Apply multi-scale feature pyramid
        pyramid_features = []
        for pyramid_layer in self.pyramid_layers:
            pyramid_features.append(pyramid_layer(calibrated_features))
        
        # Fuse pyramid features
        pyramid_output = self.pyramid_fusion(torch.cat(pyramid_features, dim=1))
        
        # Apply feature refinement with residual connection
        refined_features = self.feature_refine(calibrated_features)
        residual_features = self.feature_proj(calibrated_features)
        fused_features = refined_features + residual_features + pyramid_output
        
        # Apply weighted global pooling
        pool_weights = F.softmax(self.pool_weights, dim=0)
        avg_pooled = self.global_avg_pool(fused_features).view(batch_size, -1)
        max_pooled = self.global_max_pool(fused_features).view(batch_size, -1)
        
        # Weighted combination of pooling outputs plus concatenation
        weighted_pool = pool_weights[0] * avg_pooled + pool_weights[1] * max_pooled
        pooled_features = torch.cat([weighted_pool, max_pooled], dim=1)
        
        # Apply classification
        logits = self.classifier(pooled_features)
        
        if return_features:
            # Return intermediate features at multiple levels for attention transfer
            return logits, [backbone_features, calibrated_features, fused_features]
        else:
            return logits


# Global Context Block to capture long-range dependencies 
class GlobalContextBlock(nn.Module):
    """Global Context Block captures long-range dependencies.
    
    Inspired by GCNet and Non-local Neural Networks, but optimized
    for emotion recognition tasks with improved efficiency.
    
    Args:
        inplanes: Number of input channels
        reduction_ratio: Channel reduction ratio for bottleneck
    """
    def __init__(self, inplanes, reduction_ratio=16):
        super(GlobalContextBlock, self).__init__()
        self.inplanes = inplanes
        self.reduction_ratio = reduction_ratio
        self.bottleneck_dim = inplanes // reduction_ratio
        
        # Context modeling with spatial attention
        self.context_modeling = nn.Sequential(
            nn.Conv2d(inplanes, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel transform for feature aggregation
        self.channel_transform = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck_dim, kernel_size=1),
            nn.LayerNorm([self.bottleneck_dim, 1, 1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.bottleneck_dim, inplanes, kernel_size=1)
        )
        
    def forward(self, x):
        # Generate attention map
        context_mask = self.context_modeling(x)
        
        # Apply attention and pool globally
        context = x * context_mask
        context = torch.sum(context, dim=(2, 3), keepdim=True)
        
        # Transform channels and apply to input as residual
        transformed = self.channel_transform(context)
        
        return x + transformed


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
    """Enhanced Focal Loss with label smoothing and class balancing.
    
    This loss function focuses more on hard examples and
    reduces overfitting through label smoothing.
    
    Args:
        alpha: Weighting factor
        gamma: Focusing parameter (higher = more focus on hard examples)
        class_weights: Optional weights for class imbalance
        label_smoothing: Label smoothing factor (0-1)
        scale_pos_weight: Factor to scale positive class weights for better handling of imbalance
    """
    def __init__(self, alpha=1, gamma=2, class_weights=None, label_smoothing=0.1, scale_pos_weight=1.0):
        super(FocalLoss, self).__init__()
        from config import NUM_CLASSES
        self.alpha = alpha
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight
        self.device = None
        self.label_smoothing = label_smoothing
        self.num_classes = NUM_CLASSES
        
        # If class weights provided, convert to tensor and normalize
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
            # Scale the weights for better numeric stability
            self.class_weights = self.class_weights / self.class_weights.mean()
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        """Forward pass to compute loss."""
        if self.device is None:
            self.device = inputs.device
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(self.device)
            
        # Handle both one-hot encoded and class index targets
        if len(targets.shape) > 1 and targets.shape[1] == self.num_classes:
            # For mixup/cutmix (soft targets)
            log_probs = F.log_softmax(inputs, dim=1)
            probs = F.softmax(inputs, dim=1)
            
            # Apply class weights if provided
            if self.class_weights is not None:
                # Expand weights to batch size
                weights = self.class_weights.expand(targets.size(0), -1)
                # Apply weights to targets
                weighted_targets = targets * weights
                # Normalize targets back to sum to 1
                weighted_targets = weighted_targets / weighted_targets.sum(dim=1, keepdim=True)
                loss = -(weighted_targets * log_probs).sum(dim=1)
            else:
                loss = -(targets * log_probs).sum(dim=1)
            
            # Calculate focal weight based on prediction probability for each class
            focal_weights = torch.zeros_like(probs)
            for c in range(self.num_classes):
                focal_weights[:, c] = (1 - probs[:, c]).pow(self.gamma) * targets[:, c]
            focal_weight = focal_weights.sum(dim=1)
            
            return (self.alpha * focal_weight * loss).mean()
        else:
            # For standard class indices
            # Apply label smoothing - convert to one-hot first
            targets_one_hot = F.one_hot(targets, self.num_classes).float()
            
            # Apply label smoothing
            smoothed_targets = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            
            # Calculate loss with smoothed targets
            log_probs = F.log_softmax(inputs, dim=1)
            probs = F.softmax(inputs, dim=1)
            
            # Apply class weights if provided
            if self.class_weights is not None:
                # Get weights for each sample based on its class
                sample_weights = self.class_weights[targets]
                weighted_targets = smoothed_targets * sample_weights.unsqueeze(1)
                loss = -(weighted_targets * log_probs).sum(dim=1)
            else:
                loss = -(smoothed_targets * log_probs).sum(dim=1)
            
            # Calculate focal weight based on prediction probability of true class
            batch_size = inputs.size(0)
            focal_weight = torch.zeros(batch_size, device=self.device)
            
            for i in range(batch_size):
                focal_weight[i] = (1 - probs[i, targets[i]]).pow(self.gamma)
                
            # Apply scale_pos_weight for minority classes if class weights are provided
            if self.class_weights is not None:
                # Scale focal weight by the class weight to emphasize minority classes
                focal_weight = focal_weight * self.scale_pos_weight
                
            return (self.alpha * focal_weight * loss).mean()


class ClassBalancedLoss(nn.Module):
    """Class Balanced Loss for handling severe class imbalance.
    
    Implements effective number of samples to better handle class imbalance,
    particularly useful for datasets with long-tailed distributions.
    
    Args:
        samples_per_class: List of sample counts for each class
        beta: Hyperparameter for controlling effective number of samples (0-1)
        gamma: Focusing parameter for focal loss component
        loss_type: Base loss type ('focal', 'ce', 'bce')
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, loss_type='focal'):
        super(ClassBalancedLoss, self).__init__()
        from config import NUM_CLASSES
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_classes = NUM_CLASSES
        self.device = None
        
        # Calculate effective number of samples and CB weights
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        # Normalize weights
        weights = weights / np.sum(weights) * NUM_CLASSES
        self.weights = torch.FloatTensor(weights)
        
    def forward(self, inputs, targets):
        if self.device is None:
            self.device = inputs.device
            self.weights = self.weights.to(self.device)
            
        # Handle both one-hot encoded and class index targets
        if len(targets.shape) > 1 and targets.shape[1] == self.num_classes:
            # For mixup/cutmix (soft targets)
            if self.loss_type == 'focal':
                probs = F.softmax(inputs, dim=1)
                weighted_probs = probs * targets
                focal_weights = torch.pow(1 - weighted_probs.sum(dim=1), self.gamma)
                
                # Apply CB weights
                cb_loss = 0
                for c in range(self.num_classes):
                    target_c = targets[:, c]
                    pred_c = probs[:, c]
                    # CB weight for this class
                    cb_loss += -self.weights[c] * target_c * torch.log(pred_c + 1e-7) * focal_weights
                
                return cb_loss.mean()
            else:
                # Cross entropy for mixup
                weighted_targets = targets * self.weights.unsqueeze(0)
                weighted_targets = weighted_targets / weighted_targets.sum(dim=1, keepdim=True) * targets.sum(dim=1, keepdim=True)
                log_probs = F.log_softmax(inputs, dim=1)
                return -torch.sum(weighted_targets * log_probs) / inputs.size(0)
        else:
            # For standard class indices
            log_probs = F.log_softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, self.num_classes).float()
            
            if self.loss_type == 'focal':
                # Focal loss with CB weights
                probs = F.softmax(inputs, dim=1)
                focal_weights = torch.pow(1 - probs.gather(1, targets.unsqueeze(1)), self.gamma)
                cb_weights = self.weights[targets]
                sample_weights = focal_weights.squeeze() * cb_weights
                loss = -log_probs.gather(1, targets.unsqueeze(1))
                return (loss.squeeze() * sample_weights).mean()
            else:
                # CE with CB weights
                cb_weights = self.weights[targets]
                loss = -log_probs.gather(1, targets.unsqueeze(1))
                return (loss.squeeze() * cb_weights).mean()


class CombinedLoss(nn.Module):
    """
    Advanced combined loss function with:
    1. Class-balanced focal loss for handling imbalance
    2. Label smoothing for regularization
    3. KL divergence for soft targets
    4. Center loss component for better feature clustering
    
    Provides best performance for emotion recognition with imbalanced datasets.
    """
    def __init__(self, class_weights=None, samples_per_class=None, beta=0.9999, gamma=2.0, 
                 label_smoothing=0.1, kl_weight=0.1, center_weight=0.005):
        super(CombinedLoss, self).__init__()
        self.gamma = gamma  # Higher gamma means more focus on hard examples
        self.label_smoothing = label_smoothing
        self.kl_weight = kl_weight
        self.center_weight = center_weight
        self.device = None
        
        # Initialize focal loss with class weights
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
            
        # Initialize class-balanced loss if sample counts provided
        if samples_per_class is not None:
            self.cb_loss = ClassBalancedLoss(samples_per_class, beta=beta, gamma=gamma)
        else:
            self.cb_loss = None
            
    def get_device(self):
        if self.device is not None:
            return self.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, inputs, targets, features=None, centers=None):
        """
        Arguments:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            features: Optional feature embeddings for center loss
            centers: Optional class centers for center loss
        """
        if self.device is None:
            self.device = inputs.device
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(self.device)
                
        from config import NUM_CLASSES
        
        # If targets are one-hot encoded (for mixup/cutmix), use KL divergence
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            # KL divergence loss for soft targets
            log_probs = F.log_softmax(inputs, dim=1)
            kl_loss = -torch.sum(targets * log_probs, dim=1)
            
            # Apply class weights if provided
            if self.class_weights is not None:
                # Weight each sample based on class distribution in targets
                class_weights = self.class_weights.unsqueeze(0).expand(targets.size(0), -1)
                weight_per_sample = (targets * class_weights).sum(dim=1)
                kl_loss = kl_loss * weight_per_sample
                
            # If class-balanced loss is available, combine with it
            if self.cb_loss is not None:
                cb_loss_val = self.cb_loss(inputs, targets)
                return (kl_loss.mean() + cb_loss_val) / 2
            
            return kl_loss.mean()
            
        # For standard classification with integer labels
        else:
            # Generate smoothed one-hot targets
            smoothed_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smoothed_targets = smoothed_targets * (1.0 - self.label_smoothing) + \
                             self.label_smoothing / NUM_CLASSES
            
            # Cross-entropy loss with label smoothing
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -torch.sum(log_probs * smoothed_targets, dim=1)
            
            # Apply focal weighting
            probs = torch.gather(F.softmax(inputs, dim=1), 1, targets.unsqueeze(1)).squeeze()
            focal_weight = (1 - probs) ** self.gamma
            
            # Apply class weights if provided
            if self.class_weights is not None:
                weights = self.class_weights[targets]
                loss = loss * weights * focal_weight
            else:
                loss = loss * focal_weight
                
            # Add center loss component if features and centers provided
            center_loss = 0
            if features is not None and centers is not None:
                batch_size = features.size(0)
                features = features.view(batch_size, -1)
                target_centers = centers[targets]
                center_loss = self.center_weight * torch.mean(
                    torch.sum((features - target_centers) ** 2, dim=1)
                )
                
            # If class-balanced loss is available, combine with it
            if self.cb_loss is not None:
                cb_loss_val = self.cb_loss(inputs, targets)
                return loss.mean() * 0.7 + cb_loss_val * 0.3 + center_loss
                
            return loss.mean() + center_loss


class DistillationLoss(nn.Module):
    """Enhanced Knowledge Distillation Loss for model distillation.
    
    Enables learning from a teacher model's outputs with improved
    temperature scaling and attention transfer.
    
    Args:
        alpha: Balance between hard and soft targets (0-1)
        temperature: Temperature for softening probability distributions
        attention_beta: Weight for attention transfer (0-1)
    """
    def __init__(self, alpha=0.5, temperature=2.0, attention_beta=0.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.attention_beta = attention_beta
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels=None, criterion=None, 
                student_features=None, teacher_features=None):
        """Compute distillation loss between student and teacher outputs.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            labels: Optional ground truth labels for hard loss
            criterion: Optional criterion for hard loss
            student_features: Optional features from student for attention transfer
            teacher_features: Optional features from teacher for attention transfer
        """
        # Calculate soft target loss (KL divergence)
        # Apply temperature scaling to soften probability distributions
        soft_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_targets = soft_targets.detach()  # No gradient through teacher
        
        # Scale KL divergence by temperature squared for proper gradient scaling
        distill_loss = self.criterion_kl(soft_log_probs, soft_targets) * (self.temperature ** 2)
        
        # Add attention transfer component if features are provided
        attention_loss = 0.0
        if self.attention_beta > 0 and student_features is not None and teacher_features is not None:
            # Convert features to attention maps
            student_attention = self._get_attention(student_features)
            teacher_attention = self._get_attention(teacher_features).detach()
            
            # Calculate attention transfer loss
            attention_loss = self.attention_beta * self._attention_transfer_loss(
                student_attention, teacher_attention)
        
        # Combine with hard target loss if provided
        if labels is not None and criterion is not None:
            hard_loss = criterion(student_logits, labels)
            return self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss + attention_loss
        else:
            return distill_loss + attention_loss
    
    def _get_attention(self, features):
        """Convert feature maps to attention maps."""
        if isinstance(features, list):
            return [self._get_attention(f) for f in features]
            
        # For 4D features (B, C, H, W), convert to spatial attention
        if len(features.shape) == 4:
            # Sum across channels and normalize
            return F.normalize(features.pow(2).mean(1).view(features.size(0), -1), dim=1)
        return features
    
    def _attention_transfer_loss(self, student_attention, teacher_attention):
        """Calculate attention transfer loss between student and teacher attention maps."""
        if isinstance(student_attention, list):
            return sum(self._attention_transfer_loss(s, t) 
                      for s, t in zip(student_attention, teacher_attention))
        
        return F.mse_loss(student_attention, teacher_attention)