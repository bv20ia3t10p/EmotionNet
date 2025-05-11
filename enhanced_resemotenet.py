import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

# Feature Pyramid Network module
class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Lateral connections (1x1 convs to transform input feature maps to same channel dimension)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Top-down connections (upsampling + fusion)
        self.top_down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list) - 1)
        ])
        
    def forward(self, features):
        # Apply lateral convolutions to each input feature map
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        
        # Apply top-down pathway (from high-level to low-level features)
        results = [laterals[-1]]  # Start with the deepest feature map
        
        # Iteratively upsample and add features from coarser levels
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample current feature map
            upsampled = F.interpolate(results[0], size=laterals[i-1].shape[-2:], mode='bilinear', align_corners=False)
            # Add feature map from lateral connection
            added = upsampled + laterals[i-1]
            # Apply refinement conv
            refined = self.top_down_blocks[len(laterals) - 1 - i](added)
            # Add to results at the beginning (so results[0] is always the current level)
            results.insert(0, refined)
            
        return results

# Enhanced Multi-head Channel-Spatial Attention Module with residual connections
class CSAM(nn.Module):
    """Enhanced Channel-Spatial Attention Module with residual connections"""
    def __init__(self, channels):
        super(CSAM, self).__init__()
        
        # Channel attention with multi-scale pooling
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Use Group Normalization for more stable training
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.GroupNorm(min(channels // 16, 8), channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention with convolutional layers
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, 1),  # Group norm with 1 group is LayerNorm for a single channel
            nn.Sigmoid()
        )
        
        # Residual projection for when channel counts don't match
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(channels // 8, 32), channels)
        ) if channels % 8 != 0 else nn.Identity()
        
    def forward(self, x):
        # Save input for residual connection
        residual = x
        
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
        
        # Add residual connection
        out = x_spatial + residual
        
        return out

# Enhanced Emotion-Specific Feature Enhancement Module with residual connections
class EmotionFeatureEnhancer(nn.Module):
    """Enhanced module that enhances features specifically for emotion recognition with residual connections"""
    def __init__(self, in_channels, reduction=4):
        super(EmotionFeatureEnhancer, self).__init__()
        
        # Global context encoding
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.GroupNorm(min(in_channels // (reduction * 2), 32), in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Local feature enhancement with depthwise separable convolution
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise
            nn.GroupNorm(min(in_channels // 8, 32), in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # Pointwise
            nn.GroupNorm(min(in_channels // 8, 32), in_channels),
            nn.Sigmoid()
        )
        
        # Feature normalization for better gradient flow
        self.norm = nn.GroupNorm(min(in_channels // 8, 32), in_channels)
        
    def forward(self, x):
        # Save input for residual connection
        residual = x
        
        # Enhance with global context
        context = self.global_context(x)
        global_enhanced = x * context
        
        # Enhance with local features
        local_weights = self.local_enhance(global_enhanced)
        
        # Apply weighting and normalization
        enhanced = global_enhanced * local_weights
        enhanced = self.norm(enhanced)
        
        # Residual connection
        output = enhanced + residual
        
        return output

# Self-supervised Facial Landmark Detection Module
class FacialLandmarkBranch(nn.Module):
    """Auxiliary branch for facial landmark detection to improve feature learning"""
    def __init__(self, in_channels, num_landmarks=68):
        super(FacialLandmarkBranch, self).__init__()
        
        self.landmark_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.GroupNorm(min(in_channels // 16, 32), in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
            nn.GroupNorm(min(in_channels // 32, 16), in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, num_landmarks, kernel_size=1)
        )
        
    def forward(self, x):
        # Output heatmaps for each landmark point
        return self.landmark_branch(x)

# Feature-level MixUp augmentation
class FeatureMixup(nn.Module):
    """Applies mixup at the feature level for better generalization"""
    def __init__(self, alpha=0.2):
        super(FeatureMixup, self).__init__()
        self.alpha = alpha
    
    def forward(self, x, targets=None, training=True):
        if not training or self.alpha <= 0:
            return x, targets
        
        batch_size = x.size(0)
        indices = torch.randperm(batch_size, device=x.device)
        
        # Sample mixup parameter
        lam = torch.tensor(np.random.beta(self.alpha, self.alpha), device=x.device)
        
        # Apply mixup
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        if targets is not None:
            mixed_targets = lam * targets + (1 - lam) * targets[indices]
            return mixed_x, mixed_targets
        return mixed_x, None

# Contrastive Learning Module
class ContrastiveLearningModule(nn.Module):
    """Implements contrastive learning to better separate confusing emotions"""
    def __init__(self, feature_dim, temperature=0.07):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 128)
        )
        
    def forward(self, features, labels):
        # Project features to the contrastive embedding space
        embeddings = self.projection(features)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive pairs (same emotion)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask.fill_diagonal_(False)  # Remove self-similarity
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        
        # For each anchor, sum over all positive similarities
        pos_sim = exp_sim * mask
        pos_sim = torch.sum(pos_sim, dim=1)
        
        # For each anchor, sum over all similarities (excluding self)
        neg_mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        neg_sim = exp_sim * neg_mask
        neg_sim = torch.sum(neg_sim, dim=1)
        
        loss = -torch.log(pos_sim / neg_sim)
        loss = torch.mean(loss)
        
        return loss

# Enhanced Multi-Branch Emotion Recognition Network
class MultiPathEmotionNet(nn.Module):
    """Advanced emotion recognition model with multi-path architecture, FPN, and self-supervised learning"""
    def __init__(self, num_classes=7, dropout_rate=0.5, backbone='resnet18', use_fpn=True, use_landmarks=True, use_contrastive=True):
        super(MultiPathEmotionNet, self).__init__()
        
        self.use_fpn = use_fpn
        self.use_landmarks = use_landmarks
        self.use_contrastive = use_contrastive
        
        # Select backbone architecture
        if backbone == 'resnet18':
            base = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
            fpn_channels = [64, 128, 256, 512]  # Channels of each stage
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = 1280
            fpn_channels = [16, 24, 112, 1280]  # Approximate EfficientNet channels
        elif backbone == 'mobilenet_v3_small':
            base = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            feature_dim = 576
            fpn_channels = [16, 24, 48, 576]  # Approximate MobileNet channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create grayscale-compatible input layer
        if backbone == 'resnet18':
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(32, 64),  # Use GroupNorm instead of BatchNorm
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
                nn.GroupNorm(8, 32),  # Use GroupNorm instead of BatchNorm
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
                nn.GroupNorm(4, 16),  # Use GroupNorm instead of BatchNorm
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
        
        # Add FPN for multi-scale feature fusion
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(fpn_channels, 256)
            fpn_out_channels = 256
        else:
            fpn_out_channels = fpn_channels[-1]
        
        # Feature enhancement and attention modules
        self.enhance_s3 = EmotionFeatureEnhancer(256 if backbone == 'resnet18' else 112)
        
        # Use the correct channel count based on whether FPN is enabled
        if use_fpn:
            self.enhance_s4 = EmotionFeatureEnhancer(fpn_out_channels)
            self.attn_s4 = CSAM(fpn_out_channels)
        else:
            self.enhance_s4 = EmotionFeatureEnhancer(feature_dim)
            self.attn_s4 = CSAM(feature_dim)
        
        self.attn_s3 = CSAM(256 if backbone == 'resnet18' else 112)
        
        # Facial landmark detection branch
        if use_landmarks:
            self.landmark_branch = FacialLandmarkBranch(
                256 if backbone == 'resnet18' else 112, num_landmarks=68
            )
        
        # Feature-level mixup
        self.feature_mixup = FeatureMixup(alpha=0.2)
        
        # Global context pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization (with SpatialDropout)
        self.dropout = nn.Dropout(dropout_rate * 1.2)  # Increased dropout
        self.dropout2d = nn.Dropout2d(dropout_rate * 0.7)
        
        # Specialized pathways for different emotion groups
        # Use the correct input dimension based on whether FPN is enabled
        emotion_input_dim = fpn_out_channels if use_fpn else feature_dim
        
        # Negative emotions: angry, disgust, fear, sad
        self.negative_path = nn.Sequential(
            nn.Linear(emotion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Positive emotions: happy, surprise
        self.positive_path = nn.Sequential(
            nn.Linear(emotion_input_dim, 256), 
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Neutral emotions: neutral
        self.neutral_path = nn.Sequential(
            nn.Linear(emotion_input_dim, 256),
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Common feature extractor
        self.common_path = nn.Sequential(
            nn.Linear(emotion_input_dim, 256),
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
        
        # Enhanced class token attention with more heads and layer norm
        self.class_attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=16,  # Increased from 8 to 16
            dropout=dropout_rate * 0.6,
            batch_first=True
        )
        
        # Layer norm before and after attention
        self.pre_attention_norm = nn.LayerNorm(512)
        self.post_attention_norm = nn.LayerNorm(512)
        
        # Class tokens (learnable emotion prototypes with better initialization)
        self.class_tokens = nn.Parameter(torch.zeros(num_classes, 512))
        nn.init.xavier_uniform_(self.class_tokens)
        
        # Final classifiers with weight normalization
        self.emotion_classifier = nn.utils.weight_norm(
            nn.Linear(512, num_classes)
        )
        
        # Valence-arousal regression heads
        self.valence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()  # Bound between -1 and 1
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()  # Bound between -1 and 1
        )
        
        # Mid-level supervision
        mid_feat_dim = 256 if backbone == 'resnet18' else 112
        self.mid_classifier = nn.utils.weight_norm(
            nn.Linear(mid_feat_dim, num_classes)
        )
        self.mid_valence = nn.Linear(mid_feat_dim, 1)
        self.mid_arousal = nn.Linear(mid_feat_dim, 1)
        
        # Contrastive learning module
        if use_contrastive:
            self.contrastive_module = ContrastiveLearningModule(512)
        
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
            
        # Also freeze part of stage3 for more aggressive regularization
        if hasattr(self.stage3, 'layer'):  # ResNet-style
            for i in range(len(self.stage3) // 2):
                for param in self.stage3[i].parameters():
                    param.requires_grad = False
    
    def _initialize_weights(self):
        """Initialize new layers with better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, targets=None):
        # Initial feature extraction
        x = self.stem(x)
        
        # Feature extraction stages
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        
        # Mid-level features with enhancement and attention
        mid_features = x3
        mid_features = self.enhance_s3(mid_features)
        mid_features = self.attn_s3(mid_features)
        
        # Facial landmark detection (if enabled)
        landmark_outputs = None
        if self.use_landmarks and self.training:
            landmark_outputs = self.landmark_branch(mid_features)
        
        # Mid-level supervision
        mid_pool = self.global_pool(mid_features).view(mid_features.size(0), -1)
        mid_pool = self.dropout(mid_pool)
        
        aux_logits = self.mid_classifier(mid_pool)
        aux_valence = self.mid_valence(mid_pool)
        aux_arousal = self.mid_arousal(mid_pool)
        
        # Final features
        x4 = self.stage4(mid_features)
        
        # Apply FPN if enabled
        if self.use_fpn:
            fpn_features = self.fpn([x1, x2, x3, x4])
            # Use the highest resolution feature map from FPN
            x4 = fpn_features[-1]
        
        x4 = self.enhance_s4(x4)
        x4 = self.attn_s4(x4)
        
        # Apply 2D dropout for spatial regularization
        x4 = self.dropout2d(x4)
        
        # Feature-level mixup during training
        if self.training:
            x4, _ = self.feature_mixup(x4)
        
        # Global pooling
        x = self.global_pool(x4).view(x4.size(0), -1)
        
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
        
        # Enhanced attention mechanism with layer normalization
        batch_size = fused_features.size(0)
        # Layer norm before attention
        fused_features = self.pre_attention_norm(fused_features)
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
        
        # Apply post-attention normalization
        attn_output = self.post_attention_norm(attn_output)
        
        # Mean across class dimension to get final features
        final_features = attn_output.mean(dim=1)
        
        # Final layer dropout
        if self.training:
            final_features = F.dropout(final_features, p=0.3, training=True)
        
        # Classification and regression heads
        logits = self.emotion_classifier(final_features)
        valence = self.valence_head(final_features)
        arousal = self.arousal_head(final_features)
        
        # Contrastive loss computation if enabled and in training mode
        contrastive_loss = None
        if self.use_contrastive and self.training and targets is not None:
            contrastive_loss = self.contrastive_module(final_features, targets)
        
        # Return all outputs depending on training mode and available features
        if self.training:
            if contrastive_loss is not None:
                return logits, valence, arousal, aux_logits, aux_valence, aux_arousal, landmark_outputs, contrastive_loss
            elif landmark_outputs is not None:
                return logits, valence, arousal, aux_logits, aux_valence, aux_arousal, landmark_outputs
            else:
                return logits, valence, arousal, aux_logits, aux_valence, aux_arousal
        else:
            return logits, valence, arousal

# Class-Balanced Focal Loss for handling class imbalance
class ClassBalancedFocalLoss(nn.Module):
    """Focal Loss with class balancing for handling imbalanced datasets"""
    def __init__(self, num_classes=7, beta=0.99, gamma=2.5, samples_per_class=None, extra_angry_weight=3.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.samples_per_class = samples_per_class
        self.extra_angry_weight = extra_angry_weight  # Extra weight for angry class (0)
        
        # If samples_per_class is provided, precompute class weights
        if samples_per_class is not None:
            self.class_weights = self._compute_class_weights(samples_per_class)
        else:
            self.class_weights = None
            
    def _compute_class_weights(self, samples_per_class):
        """Compute class weights based on effective number of samples"""
        effective_num = 1.0 - torch.pow(self.beta, torch.tensor(samples_per_class))
        weights = (1.0 - self.beta) / effective_num
        
        # Add extra weight to angry class (index 0)
        if len(weights) > 0:
            weights[0] = weights[0] * self.extra_angry_weight
            
        # Normalize weights
        weights = weights / torch.sum(weights) * self.num_classes
        return weights
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [batch_size, num_classes]
            targets: Tensor of shape [batch_size]
        """
        # Convert targets to one-hot encoding
        one_hot_targets = F.one_hot(targets, self.num_classes).float()
        
        # Compute probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        # Get the probability for the target class
        pt = torch.sum(probs * one_hot_targets, dim=1)
        
        # Compute focal weight term with extra penalty for angry class misclassification
        focal_weight = (1 - pt) ** self.gamma
        
        # Extra penalty when target is angry class (class 0)
        angry_mask = (targets == 0)
        if angry_mask.any():
            focal_weight[angry_mask] = focal_weight[angry_mask] * self.extra_angry_weight
        
        # Apply class weights if available
        if self.class_weights is not None:
            # Get weights for each target class
            weights = self.class_weights[targets].to(logits.device)
            focal_weight = focal_weight * weights
        
        # Compute cross-entropy loss with class balancing
        loss = -focal_weight * torch.log(pt + 1e-10)
        
        return loss.mean()

# Enhanced with progressive learning rate warmup
class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts and warmup.
    This combines warmup period with cosine annealing and periodic restarts.
    """
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, 
                 warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize base learning rates
        self.base_lrs = []
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = max_lr
            self.base_lrs.append(min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
            
        # Warmup period
        if self.step_in_cycle < self.warmup_steps:
            lr_factor = self.step_in_cycle / self.warmup_steps
            return [(self.max_lr - base_lr) * lr_factor + base_lr for base_lr in self.base_lrs]
            
        # Cosine annealing with restarts
        else:
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            if progress >= 1.0:
                progress = 1.0
            
            return [base_lr + 0.5 * (self.max_lr - base_lr) * (1 + math.cos(math.pi * progress)) 
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            # Cycle completion
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.max_lr * self.gamma
        
        else:
            if epoch >= self.first_cycle_steps:
                # Calculate cycle and step_in_cycle in reverse
                n_cycles = 1
                cycle_steps = self.first_cycle_steps
                cumulative_steps = self.first_cycle_steps
                
                while epoch >= cumulative_steps + cycle_steps * self.cycle_mult:
                    cycle_steps = cycle_steps * self.cycle_mult
                    n_cycles += 1
                    cumulative_steps += cycle_steps
                
                self.cycle = n_cycles
                self.step_in_cycle = epoch - cumulative_steps
                self.cur_cycle_steps = cycle_steps
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
                
            else:
                self.step_in_cycle = epoch
                self.cur_cycle_steps = self.first_cycle_steps
                
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Maintain backward compatibility with the original class name
class EnhancedResEmoteNet(MultiPathEmotionNet):
    """Wrapper class for backward compatibility with enhanced architecture"""
    def __init__(self, num_classes=7, dropout_rate=0.65, backbone='resnet18', 
                 use_fpn=True, use_landmarks=True, use_contrastive=True):
        super(EnhancedResEmoteNet, self).__init__(
            num_classes=num_classes, 
            dropout_rate=dropout_rate,
            backbone=backbone,
            use_fpn=use_fpn,
            use_landmarks=use_landmarks,
            use_contrastive=use_contrastive
        ) 