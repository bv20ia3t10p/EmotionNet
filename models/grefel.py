import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import timm

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
    def forward(self, x, cross=None):
        # x: [B, N, C], cross: [B, N, C] or None
        if cross is not None:
            x2, _ = self.attn(self.norm1(x), self.norm1(cross), self.norm1(cross))
        else:
            x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x = x + self.mlp(self.norm2(x))
        return x

class GReFEL(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        feature_dim: int = 256,
        num_anchors: int = 10,
        drop_rate: float = 0.1
    ):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            'resnet50', pretrained=True, features_only=True, out_indices=(1, 2, 3)
        )
        self.in_dims = [256, 512, 1024]  # Corrected ResNet50 channel dimensions
        self.scales = [56, 28, 14]  # Output spatial sizes for 224x224 input
        # Project to common dim
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ) for in_dim in self.in_dims
        ])
        # Transformer blocks for each scale
        self.blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads=4, mlp_ratio=2.0, drop=drop_rate)
            for _ in range(3)
        ])
        
        # Geometry-aware module
        self.num_anchors = num_anchors
        self.anchors = nn.Parameter(torch.randn(num_anchors, feature_dim*3))
        nn.init.xavier_uniform_(self.anchors)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim*3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim*3, feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)  # List of [B, C, H, W]
        # Project and flatten
        proj_feats = []
        for i, f in enumerate(feats):
            pf = self.proj[i](f)  # [B, C, H, W]
            pf = pf.flatten(2).transpose(1, 2)  # [B, N, C]
            proj_feats.append(pf)
        # Cross-attention: each scale attends to itself and the other two
        attn_feats = []
        for i in range(3):
            others = [proj_feats[j] for j in range(3) if j != i]
            x_self = self.blocks[i](proj_feats[i])
            x_cross1 = self.blocks[i](proj_feats[i], cross=others[0])
            x_cross2 = self.blocks[i](proj_feats[i], cross=others[1])
            # Fuse (sum) all attended features
            x_fused = x_self + x_cross1 + x_cross2
            # Global average pool
            x_fused = x_fused.mean(dim=1)  # [B, C]
            attn_feats.append(x_fused)
        # Concatenate all scales
        multi_scale_feat = torch.cat(attn_feats, dim=1)  # [B, C*3]
        # Geometry-aware
        geo_feat, geo_loss = self.geometry_learning(multi_scale_feat)
        # Classifier
        logits = self.classifier(geo_feat)
        confidence = self.confidence_head(geo_feat)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'geo_loss': geo_loss,
            'features': geo_feat
        }
    
    def geometry_learning(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = torch.cdist(features, self.anchors)
        attn = F.softmax(-distances, dim=1)
        enhanced = torch.matmul(attn, self.anchors)
        combined = features + enhanced
        anchor_dist = torch.pdist(self.anchors)
        diversity_loss = torch.exp(-anchor_dist.mean())
        assignment_entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1).mean()
        geo_loss = diversity_loss + 0.1 * assignment_entropy
        return combined, geo_loss

class SimplifiedGReFELLoss(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        # Main classification loss
        ce_loss = self.ce_loss(outputs['logits'], targets)
        
        # Geometry loss (already computed in model)
        geo_loss = outputs['geo_loss']
        
        # Total loss
        total_loss = ce_loss + 0.1 * geo_loss
        
        return total_loss 