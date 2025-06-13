import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, List

class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class AdaptiveGeometryModule(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        num_anchors: int = 10,
        num_heads: int = 8
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.feature_dim = feature_dim
        
        # Learnable anchors with positional encoding
        self.anchors = nn.Parameter(torch.randn(num_anchors, feature_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_anchors, feature_dim))
        
        # Multi-scale attention for feature refinement
        self.attention = MultiScaleAttention(feature_dim, num_heads)
        
        # Dynamic weighting
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.center_loss_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = features.size(0)
        
        # Add positional encoding to anchors
        anchors = self.anchors.unsqueeze(0) + self.pos_embedding  # [1, K, D]
        anchors = anchors.expand(B, -1, -1)  # [B, K, D]
        
        # Calculate distances between features [B, D] and anchors [B, K, D]
        # First reshape features to [B, 1, D] for broadcasting
        features_expanded = features.unsqueeze(1)  # [B, 1, D]
        
        # Calculate pairwise distances
        distances = torch.cdist(features_expanded, anchors)  # [B, 1, K]
        distances = distances.squeeze(1)  # [B, K]
        
        # Soft assignment with temperature scaling
        temperature = torch.norm(features, dim=-1, keepdim=True)
        attention = F.softmax(-distances / temperature, dim=-1)  # [B, K]
        
        # Dynamic weighting based on feature importance
        weights = self.weight_net(features)  # [B, 1]
        
        # Weighted feature aggregation
        balanced_features = torch.matmul(attention.unsqueeze(1), anchors).squeeze(1)  # [B, D]
        balanced_features = balanced_features * weights
        
        # Apply multi-scale attention
        refined_features = self.attention(balanced_features.unsqueeze(1)).squeeze(1)
        
        # Compute losses
        center_loss = torch.mean(torch.sum((features - refined_features) ** 2, dim=-1))
        diversity_loss = -torch.mean(torch.pdist(self.anchors))
        
        total_loss = self.center_loss_weight * center_loss + 0.1 * diversity_loss
        
        return refined_features, total_loss, attention  # [B, D], scalar, [B, K]

class HierarchicalReliabilityModule(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        num_classes: int,
        smoothing: float = 0.11,
        temperature: float = 1.0
    ):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature
        
        # Hierarchical feature projection
        self.global_proj = nn.Linear(feature_dim, feature_dim // 2)
        self.local_proj = nn.Linear(feature_dim, feature_dim // 2)
        
        # Classification heads
        self.global_head = nn.Linear(feature_dim // 2, num_classes)
        self.local_head = nn.Linear(feature_dim // 2, num_classes)
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        features: torch.Tensor,  # [B, D]
        attention_weights: torch.Tensor  # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Global features
        global_features = self.global_proj(features)  # [B, D/2]
        global_logits = self.global_head(global_features)  # [B, C]
        
        # Local features with attention
        attention_weights = attention_weights.unsqueeze(2)  # [B, K, 1]
        features_expanded = features.unsqueeze(1)  # [B, 1, D]
        features_expanded = features_expanded.expand(-1, attention_weights.size(1), -1)  # [B, K, D]
        
        # Compute weighted features
        local_features = attention_weights * features_expanded  # [B, K, D]
        local_features = local_features.sum(dim=1)  # [B, D]
        
        # Project and classify
        local_features = self.local_proj(local_features)  # [B, D/2]
        local_logits = self.local_head(local_features)  # [B, C]
        
        # Estimate confidence
        confidence = self.confidence_net(features)  # [B, 1]
        
        # Combine predictions with confidence weighting
        logits = confidence * global_logits + (1 - confidence) * local_logits  # [B, C]
        logits = logits / self.temperature  # Apply temperature scaling
        
        return logits, confidence

class GReFELPlusPlus(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        feature_dim: int = 768,
        num_anchors: int = 10,
        num_heads: int = 8,
        smoothing: float = 0.11,
        temperature: float = 1.0,
        backbone: str = 'vit_large_patch16_224'
    ):
        super().__init__()
        
        # Vision Transformer backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0
        )
        
        # Adaptive geometry module
        self.geometry_module = AdaptiveGeometryModule(
            feature_dim=feature_dim,
            num_anchors=num_anchors,
            num_heads=num_heads
        )
        
        # Hierarchical reliability module
        self.reliability_module = HierarchicalReliabilityModule(
            feature_dim=feature_dim,
            num_classes=num_classes,
            smoothing=smoothing,
            temperature=temperature
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Apply geometry-aware processing
        refined_features, geometry_loss, attention = self.geometry_module(features)
        
        # Apply reliability balancing
        probabilities, confidence = self.reliability_module(refined_features, attention)
        
        if return_features:
            return probabilities, refined_features, geometry_loss, confidence
        return probabilities, None, geometry_loss, confidence

class GReFELPlusPlusLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.4,
        beta: float = 0.2,
        gamma: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize class weights with sqrt scaling to reduce extreme values
        class_freq = torch.tensor([36.71, 26.26, 12.50, 12.36, 8.63, 0.68, 2.28, 0.58])
        class_weights = 1.0 / torch.sqrt(class_freq + 1e-6)  # sqrt to reduce extreme weights
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
        self.class_weights = nn.Parameter(class_weights, requires_grad=False)
        
        # Separate loss functions for better stability
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.ce_no_weight = nn.CrossEntropyLoss(reduction='none')
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        geometry_loss: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        # Weighted cross entropy loss
        ce_loss = self.ce(logits, targets)
        
        # Focal loss with original CE (without class weights to prevent double weighting)
        pt = torch.exp(-self.ce_no_weight(logits, targets))
        focal_weight = ((1 - pt) ** self.gamma)
        focal_loss = (focal_weight * self.ce_no_weight(logits, targets)).mean()
        
        # Scale geometry loss more conservatively
        scaled_geometry_loss = self.alpha * torch.clamp(geometry_loss, 0, 5)
        
        # Simpler confidence regularization
        conf_reg = self.beta * (1 - confidence.mean())
        
        # Combine losses with emphasis on CE and focal
        total_loss = 0.5 * (ce_loss + focal_loss) + scaled_geometry_loss + conf_reg
        return total_loss 