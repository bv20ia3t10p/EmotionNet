"""
Expert model architecture for emotion recognition.
Implements a hierarchical architecture with specialized attention for facial emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict, Tuple, Optional, Union


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return torch.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class DecoupledHead(nn.Module):
    """Decoupled classification head with separate branches for localization and classification."""
    def __init__(self, in_features, embedding_size, num_classes):
        super(DecoupledHead, self).__init__()
        
        # Common feature extractor
        self.embedding = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(0.2)
        
        # Classification branch
        self.classifier = nn.Linear(embedding_size, num_classes)
        
        # Localization branch (for prominent facial features)
        self.localizer = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size // 2, 10)  # 5 key facial points (x,y)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        features = self.dropout(x)
        
        # Classification output
        logits = self.classifier(features)
        
        # Localization output (auxiliary task)
        facial_points = self.localizer(features)
        
        return logits, facial_points, features


class EmotionGroupClassifier(nn.Module):
    """Classifier for emotion groups in hierarchical architecture."""
    def __init__(self, in_features, num_groups):
        super(EmotionGroupClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_groups)
        
    def forward(self, x):
        return self.fc(x)


class SpecificEmotionClassifier(nn.Module):
    """Classifier for specific emotions within a group."""
    def __init__(self, in_features, num_emotions_in_group):
        super(SpecificEmotionClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_emotions_in_group)
        
    def forward(self, x):
        return self.fc(x)


class HierarchicalEmotionClassifier(nn.Module):
    """Hierarchical emotion classifier that first groups emotions then classifies specific emotions."""
    def __init__(self, in_features, emotion_groups, embedding_size=512):
        super(HierarchicalEmotionClassifier, self).__init__()
        
        self.emotion_groups = emotion_groups
        self.num_groups = len(emotion_groups)
        self.embedding_size = embedding_size
        
        # Shared feature extractor
        self.embedding = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(0.3)
        
        # Group classifier
        self.group_classifier = EmotionGroupClassifier(embedding_size, self.num_groups)
        
        # Specific emotion classifiers
        self.emotion_classifiers = nn.ModuleList([
            SpecificEmotionClassifier(embedding_size, len(group)) 
            for group in emotion_groups
        ])
        
    def forward(self, x):
        # Extract features
        x = self.embedding(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        features = self.dropout(x)
        
        # Group classification
        group_logits = self.group_classifier(features)
        
        # Specific emotion classification for each group
        emotion_logits_list = []
        for classifier in self.emotion_classifiers:
            emotion_logits_list.append(classifier(features))
            
        return group_logits, emotion_logits_list, features
    
    def inference(self, x):
        """Inference-time forward pass that combines predictions hierarchically."""
        group_logits, emotion_logits_list, features = self.forward(x)
        
        # Get the most likely group
        group_probs = F.softmax(group_logits, dim=1)
        
        # Initialize final logits for all emotions
        total_emotions = sum(len(group) for group in self.emotion_groups)
        final_logits = torch.zeros(x.size(0), total_emotions, device=x.device)
        
        # For each sample in the batch
        for i in range(x.size(0)):
            emotion_idx = 0
            for group_idx, group in enumerate(self.emotion_groups):
                group_prob = group_probs[i, group_idx]
                
                # Get specific emotion probabilities within this group
                emotion_probs = F.softmax(emotion_logits_list[group_idx][i], dim=0)
                
                # Weight the specific emotion probabilities by the group probability
                for j in range(len(group)):
                    final_logits[i, emotion_idx + j] = group_prob * emotion_probs[j]
                
                emotion_idx += len(group)
        
        return final_logits


class ExpertEmotionModel(nn.Module):
    """Expert-level emotion recognition model with hierarchical classification and CBAM attention."""
    def __init__(self, 
                 backbone_name: str = 'convnext_tiny',
                 num_classes: int = 7,
                 pretrained: bool = True,
                 embedding_size: int = 512,
                 emotion_groups_str: str = "sad-neutral-angry,happy-surprise,fear-disgust",
                 use_gem_pooling: bool = True,
                 use_decoupled_head: bool = True,
                 drop_path_rate: float = 0.2,
                 channels_last: bool = True):
        super(ExpertEmotionModel, self).__init__()
        
        # Parse emotion groups
        self.emotion_group_names = self._parse_emotion_groups(emotion_groups_str)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.use_decoupled_head = use_decoupled_head
        
        # List of supported backbone models
        supported_backbones = [
            'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1', 
            'efficientnet_b2', 'mobilenetv3_small', 'mobilenetv3_large'
        ]
        
        # Check if backbone is supported
        if backbone_name not in supported_backbones:
            supported_str = ", ".join(supported_backbones)
            raise ValueError(f"Unsupported backbone: {backbone_name}. Supported backbones are: {supported_str}")
        
        # Load backbone
        try:
            self.backbone = timm.create_model(
                backbone_name, 
                pretrained=pretrained,
                features_only=True,
                drop_path_rate=drop_path_rate
            )
        except Exception as e:
            raise ValueError(f"Error creating backbone {backbone_name}: {str(e)}. Try using one of: {', '.join(supported_backbones)}")
        
        # Get feature dimensions from backbone
        dummy_input = torch.zeros(1, 3, 224, 224)
        if channels_last:
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
            self.backbone = self.backbone.to(memory_format=torch.channels_last)
            
        features = self.backbone(dummy_input)
        
        # Add CBAM attention to each feature map
        self.cbam_modules = nn.ModuleList([
            CBAM(feature.shape[1]) for feature in features
        ])
        
        # Feature fusion
        self.feature_fusion = nn.ModuleList([
            nn.Conv2d(feature.shape[1], 256, kernel_size=1) 
            for feature in features
        ])
        
        # Pooling layers
        if use_gem_pooling:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            
        # Determine the size of flattened features after fusion
        self.fused_features_dim = 256 * len(features)
        
        # Define hierarchical classifier
        self.hierarchical_classifier = HierarchicalEmotionClassifier(
            self.fused_features_dim, 
            self.emotion_group_names,
            embedding_size
        )
        
        # Optional decoupled head for direct classification
        if use_decoupled_head:
            self.decoupled_head = DecoupledHead(
                self.fused_features_dim,
                embedding_size,
                num_classes
            )
        
        # Initialize weights
        self._initialize_weights()
        
    def _parse_emotion_groups(self, emotion_groups_str):
        """Parse the emotion groups string into a list of lists with emotion names."""
        groups = emotion_groups_str.split(',')
        return [group.split('-') for group in groups]
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
    
    def extract_features(self, x):
        """Extract features from the backbone and apply CBAM attention."""
        features = self.backbone(x)
        
        # Apply CBAM attention to each feature map
        attended_features = []
        for i, feature in enumerate(features):
            attended_feature = self.cbam_modules[i](feature)
            attended_features.append(attended_feature)
        
        # Apply feature fusion
        fused_features = []
        for i, feature in enumerate(attended_features):
            fused_feature = self.feature_fusion[i](feature)
            pooled_feature = self.pooling(fused_feature)
            fused_features.append(pooled_feature.flatten(1))
        
        # Concatenate all features
        x = torch.cat(fused_features, dim=1)
        return x
    
    def forward(self, x):
        """Forward pass through the model."""
        # Extract features with backbone and attention
        features = self.extract_features(x)
        
        # Get hierarchical classification results
        group_logits, emotion_logits_list, h_features = self.hierarchical_classifier(features)
        
        # Get direct classification results if using decoupled head
        if self.use_decoupled_head:
            direct_logits, facial_points, d_features = self.decoupled_head(features)
            return {
                'group_logits': group_logits,
                'emotion_logits_list': emotion_logits_list,
                'hierarchical_features': h_features,
                'direct_logits': direct_logits,
                'facial_points': facial_points,
                'decoupled_features': d_features
            }
        else:
            return {
                'group_logits': group_logits,
                'emotion_logits_list': emotion_logits_list,
                'hierarchical_features': h_features,
            }
    
    def inference(self, x):
        """Inference-time forward pass."""
        features = self.extract_features(x)
        
        # Get hierarchical predictions
        hierarchical_logits = self.hierarchical_classifier.inference(features)
        
        # Get direct predictions if using decoupled head
        if self.use_decoupled_head:
            direct_logits, _, _ = self.decoupled_head(features)
            
            # Combine predictions (weighted average)
            combined_logits = 0.6 * direct_logits + 0.4 * hierarchical_logits
            return combined_logits
        else:
            return hierarchical_logits


class HybridLoss(nn.Module):
    """
    Hybrid loss combining focal loss for classification with triplet loss for feature learning
    and consistency loss between hierarchical and direct predictions.
    """
    def __init__(self, 
                 num_classes=7,
                 focal_gamma=2.0,
                 label_smoothing=0.1,
                 triplet_margin=0.3,
                 weights=None):
        super(HybridLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.triplet_margin = triplet_margin
        self.weights = weights
        
    def focal_loss(self, logits, targets):
        """Compute focal loss."""
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weights, 
            label_smoothing=self.label_smoothing, reduction='none'
        )
        
        # Compute focal weights
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        return (focal_weight * ce_loss).mean()
    
    def triplet_loss(self, embeddings, targets):
        """Compute triplet loss with online mining."""
        batch_size = embeddings.size(0)
        if batch_size < 3:  # Need at least anchor, positive, negative
            return torch.tensor(0.0, device=embeddings.device)
            
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings)
        
        # Find hardest positive for each anchor
        mask_pos = targets.unsqueeze(1) == targets.unsqueeze(0)
        mask_pos = mask_pos.logical_xor(torch.eye(batch_size, device=embeddings.device).bool())
        hardest_pos_dist, _ = (dist_matrix * mask_pos.float()).max(dim=1)
        
        # Find hardest negative for each anchor
        mask_neg = targets.unsqueeze(1) != targets.unsqueeze(0)
        hardest_neg_dist, _ = (dist_matrix * mask_neg.float() + 
                               (1e6) * (~mask_neg).float()).min(dim=1)
        
        # Compute triplet loss with margin
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.triplet_margin)
        
        return loss.mean()
    
    def consistency_loss(self, direct_logits, hierarchical_logits):
        """Compute consistency loss between direct and hierarchical predictions."""
        return F.kl_div(
            F.log_softmax(direct_logits, dim=1),
            F.softmax(hierarchical_logits, dim=1),
            reduction='batchmean'
        )
    
    def forward(self, outputs, targets):
        """Forward pass of the hybrid loss."""
        # Hierarchical classification loss
        group_logits = outputs['group_logits']
        emotion_logits_list = outputs['emotion_logits_list']
        hierarchical_features = outputs['hierarchical_features']
        
        # TODO: Compute hierarchical loss based on group and emotion targets
        # For now, we'll just use the direct classification loss
        
        # Direct classification loss
        if 'direct_logits' in outputs:
            direct_logits = outputs['direct_logits']
            decoupled_features = outputs['decoupled_features']
            
            # Classification loss
            focal_loss = self.focal_loss(direct_logits, targets)
            
            # Feature embedding loss
            triplet_loss = self.triplet_loss(decoupled_features, targets)
            
            # Compute hierarchical logits for consistency
            hierarchical_logits = torch.zeros_like(direct_logits)
            # TODO: Fill hierarchical_logits from group_logits and emotion_logits_list
            
            # Consistency loss
            consistency = self.consistency_loss(direct_logits, hierarchical_logits)
            
            # Combine losses
            total_loss = focal_loss + 0.1 * triplet_loss + 0.1 * consistency
            
            return total_loss, {
                'focal': focal_loss.item(),
                'triplet': triplet_loss.item(),
                'consistency': consistency.item()
            }
        else:
            # Only hierarchical model is being used
            # TODO: Implement proper hierarchical loss
            return torch.tensor(0.0), {} 