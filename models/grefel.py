import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple

class GeometryAwareModule(nn.Module):
    def __init__(self, feature_dim: int, num_anchors: int = 10):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchors = nn.Parameter(torch.randn(num_anchors, feature_dim))
        self.center_loss_weight = 0.1
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate distances to anchors
        distances = torch.cdist(features, self.anchors)
        
        # Soft assignment to anchors
        attention = F.softmax(-distances, dim=-1)
        
        # Weighted feature aggregation
        balanced_features = torch.matmul(attention, self.anchors)
        
        # Center loss computation
        center_loss = torch.mean(torch.sum((features - balanced_features) ** 2, dim=-1))
        
        return balanced_features, self.center_loss_weight * center_loss

class ReliabilityBalancingModule(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, smoothing: float = 0.11):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, num_classes)
        self.smoothing = smoothing
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.feature_projection(features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Apply label smoothing for reliability
        uniform = torch.ones_like(probabilities) / probabilities.size(-1)
        smooth_probabilities = (1 - self.smoothing) * probabilities + self.smoothing * uniform
        
        return smooth_probabilities

class GReFEL(nn.Module):
    def __init__(
        self, 
        num_classes: int = 8,
        pretrained: bool = True,
        feature_dim: int = 768,
        num_anchors: int = 10,
        smoothing: float = 0.11
    ):
        super().__init__()
        
        # Vision Transformer backbone
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Geometry-aware module
        self.geometry_module = GeometryAwareModule(
            feature_dim=feature_dim,
            num_anchors=num_anchors
        )
        
        # Reliability balancing module
        self.reliability_module = ReliabilityBalancingModule(
            feature_dim=feature_dim,
            num_classes=num_classes,
            smoothing=smoothing
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Extract features from ViT
        features = self.backbone(x)
        
        # Apply geometry-aware module
        balanced_features, center_loss = self.geometry_module(features)
        
        # Apply reliability balancing
        probabilities = self.reliability_module(balanced_features)
        
        if return_features:
            return probabilities, balanced_features, center_loss
        return probabilities, None, center_loss

class GReFELLoss(nn.Module):
    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        
    def forward(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        center_loss: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = self.ce(probabilities, targets)
        total_loss = ce_loss + center_loss
        return total_loss 