import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, Swinv2Model
import math

class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        reduced_dim = dim // 4  # For ViT-Base: 768 // 4 = 192
        
        # Convolutions with correct dimensions
        self.conv1 = nn.Conv2d(768, reduced_dim, kernel_size=3, padding=1)  # Fixed input channels to 768
        self.conv3 = nn.Conv2d(768, reduced_dim, kernel_size=5, padding=2)  # Fixed input channels to 768
        
        self.norm = nn.LayerNorm(reduced_dim * 2)
        self.proj = nn.Linear(reduced_dim * 2, 768)  # Project back to 768
        
        # Initialize weights with smaller values
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv3.bias)
        
    def forward(self, x):
        B, N, C = x.shape
        # Remove CLS token and reshape
        x_cls = x[:, 0:1, :]  # Get CLS token
        x_patch = x[:, 1:, :]  # Remove CLS token
        H = W = int(math.sqrt(N - 1))  # -1 for CLS token
        x_patch = x_patch.transpose(1, 2).reshape(B, C, H, W)
        
        # Multi-scale feature extraction
        x1 = self.conv1(x_patch)
        x3 = self.conv3(x_patch)
        
        # Combine features
        x_combined = torch.cat([x1, x3], dim=1)  # B, 2*reduced_dim, H, W
        x_combined = x_combined.flatten(2).transpose(1, 2)  # B, HW, 2*reduced_dim
        x_combined = self.norm(x_combined)
        x_combined = self.proj(x_combined)  # B, HW, 768
        
        # Reattach CLS token
        x = torch.cat([x_cls, x_combined], dim=1)
        return x

class HierarchicalAttention(nn.Module):
    def __init__(self, dim=768):  # Default to ViT-Base dim
        super().__init__()
        num_heads = 12  # ViT-Base uses 12 heads
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Single efficient attention mechanism
        self.qkv = nn.Linear(768, 768 * 3)  # Fixed dimensions to 768
        self.proj = nn.Linear(768, 768)
        
        # Lightweight feature refinement
        self.refine = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Initialize weights with smaller values
        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        nn.init.zeros_(self.qkv.bias)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        nn.init.zeros_(self.proj.bias)
        
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Unified attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        # Simple refinement
        x = x + self.refine(x)
        return x

class AdaptiveReliabilityModule(nn.Module):
    def __init__(self, num_anchors=10, num_classes=8):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        dim = 768  # Fixed to ViT-Base dimension
        reduced_dim = dim // 2  # 384
        
        # Initialize anchors with smaller values
        self.register_buffer('temperature', torch.ones(1) * 0.1)  # Reduced temperature
        self.anchors = nn.Parameter(torch.randn(num_anchors, reduced_dim).div_(reduced_dim))  # Reduced initialization scale
        
        # Efficient confidence estimation
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize confidence head with smaller values
        for m in self.confidence_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Lightweight feature refinement
        self.feature_refine = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, 768)
        )
        
        # Initialize feature refinement with smaller values
        for m in self.feature_refine.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Projection for anchor comparison
        self.proj = nn.Linear(768, reduced_dim)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x, labels=None):
        # Refine features with residual connection
        x_refined = x + self.feature_refine(x)
        
        # Project to reduced dimension for anchor comparison
        x_proj = self.proj(x_refined)
        
        # Normalize features and anchors
        x_norm = F.normalize(x_proj, dim=-1, eps=1e-8)
        anchors_norm = F.normalize(self.anchors, dim=-1, eps=1e-8)
        
        # Compute anchor similarities with temperature scaling
        similarities = torch.matmul(x_norm, anchors_norm.t()) / torch.clamp(self.temperature, min=0.01)
        similarities = torch.clamp(similarities, min=-10, max=10)
        
        # Estimate confidence
        confidence = self.confidence_head(x_refined)
        confidence = torch.clamp(confidence, min=0.1, max=0.9)
        
        if self.training and labels is not None:
            # Simplified center loss computation
            center_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                mask = (labels == label)
                if mask.sum() > 0:
                    class_features = x_proj[mask]
                    class_center = class_features.mean(0, keepdim=True)
                    class_features = F.normalize(class_features, dim=-1, eps=1e-8)
                    class_center = F.normalize(class_center, dim=-1, eps=1e-8)
                    center_loss += (1 - (class_features * class_center).sum(dim=1)).mean()
            
            center_loss = torch.clamp(center_loss, min=0.0, max=10.0)
            weighted_similarities = similarities * confidence
            return weighted_similarities, center_loss
        
        return similarities, None

class EnhancedGReFEL(nn.Module):
    def __init__(self, num_classes=8, num_anchors=10):  # Removed embed_dim parameter
        super().__init__()
        
        # Use ViT-Base backbone
        self.backbone = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            add_pooling_layer=False,
            output_hidden_states=True
        )
        
        # Enable gradient checkpointing
        self.backbone.gradient_checkpointing_enable()
        
        # Feature processing
        self.multiscale = MultiScaleFeatureExtraction(dim=768)  # Fixed to ViT-Base dim
        
        # Single hierarchical attention layer
        self.attention = HierarchicalAttention()  # Uses default 768
        
        # Adaptive reliability module
        self.reliability_module = AdaptiveReliabilityModule(num_anchors, num_classes)
        
        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )
        
    def forward(self, x, labels=None):
        # Extract features from backbone
        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # Will be [B, 197, 768]
        
        # Process features
        x = self.multiscale(features)
        x = self.attention(x)
        
        # Get CLS token
        x_cls = x[:, 0]
        
        # Reliability balancing
        similarities, center_loss = self.reliability_module(x_cls, labels)
        
        # Classification
        logits = self.classifier(x_cls)
        
        if self.training and labels is not None:
            # Classification loss with label smoothing
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # Reliability loss
            rel_loss = -torch.log(similarities.max(dim=1)[0]).mean()
            
            # Total loss
            total_loss = cls_loss + 0.1 * rel_loss + 0.1 * center_loss
            
            return logits, total_loss
        
        return logits, None 