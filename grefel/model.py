import torch
import torch.nn as nn
import torch.nn.functional as F

class LightFaceNet(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Efficient channel progression
        self.features = nn.Sequential(
            # Initial conv with larger kernel for facial features
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            
            # Depthwise separable convs for efficiency
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # 7x7
        )
        
        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x

class GeometryAwareAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        # x shape: [B, L, D]
        B, L, D = x.shape
        
        # Compute Q, K, V
        q = self.query(x)  # [B, L, D]
        k = self.key(x)    # [B, L, D]
        v = self.value(x)  # [B, L, D]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, L]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = attn @ v  # [B, L, D]
        
        return out

class ReliabilityBalancingModule(nn.Module):
    def __init__(self, dim, num_anchors=10):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchors = nn.Parameter(torch.randn(num_anchors, dim))
        self.center_weight = 0.1
        
    def forward(self, x, labels=None):
        # Calculate distances to anchors
        x_norm = F.normalize(x, dim=-1)
        anchors_norm = F.normalize(self.anchors, dim=-1)
        
        # Compute similarity scores
        similarity = x_norm @ anchors_norm.T  # Range: [-1, 1]
        similarity = (similarity + 1) / 2      # Range: [0, 1]
        
        # If in training mode and labels are provided
        if self.training and labels is not None:
            # Center loss computation
            unique_labels = torch.unique(labels)
            centers = []
            for i in unique_labels:
                mask = (labels == i)
                if mask.sum() > 0:
                    centers.append(x[mask].mean(0))
            
            if len(centers) > 0:
                centers = torch.stack(centers)
                label_indices = torch.zeros_like(labels, dtype=torch.long, device=x.device)
                for idx, label in enumerate(unique_labels):
                    label_indices[labels == label] = idx
                center_loss = ((x - centers[label_indices]) ** 2).mean()
            else:
                center_loss = torch.tensor(0.0, device=x.device)
                
            return similarity, self.center_weight * center_loss
        return similarity, None

class GReFEL(nn.Module):
    def __init__(self, num_classes=8, num_anchors=10, embed_dim=512):
        super().__init__()
        
        # Lightweight CNN backbone
        self.backbone = LightFaceNet(embed_dim=embed_dim)
        
        # Geometry-aware attention with multi-scale processing
        self.geometry_attention = nn.ModuleList([
            GeometryAwareAttention(embed_dim),
            GeometryAwareAttention(embed_dim),
            GeometryAwareAttention(embed_dim)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Reliability balancing module
        self.reliability_module = ReliabilityBalancingModule(embed_dim, num_anchors=num_anchors)
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Loss weights
        self.lambda_cls = 1.0    # Classification loss weight
        self.lambda_a = 0.1      # Anchor loss weight
        self.lambda_c = 0.1      # Center loss weight
        
    def forward(self, x, labels=None):
        # Get CNN features
        x = self.backbone(x)  # [B, embed_dim]
        
        # Multi-scale geometry-aware attention
        B = x.shape[0]
        x_list = []
        x_unsqueeze = x.unsqueeze(1)  # [B, 1, embed_dim]
        
        for attn in self.geometry_attention:
            x_att = attn(x_unsqueeze)  # [B, 1, embed_dim]
            x_list.append(x_att.squeeze(1))  # [B, embed_dim]
        
        # Feature fusion
        x_cat = torch.cat(x_list, dim=-1)  # [B, embed_dim * 3]
        x = self.fusion(x_cat)  # [B, embed_dim]
        
        # Apply reliability balancing
        anchor_similarities, center_loss = self.reliability_module(x, labels)
        
        # Get classification logits
        logits = self.classifier(x)
        
        if self.training and labels is not None:
            # Calculate losses with label smoothing
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # Safer anchor loss calculation
            eps = 1e-7
            anchor_loss = -torch.log(anchor_similarities.max(dim=1)[0].clamp(min=eps)).mean()
            
            # Combine losses with proper scaling
            total_loss = (
                self.lambda_cls * cls_loss +
                self.lambda_a * anchor_loss +
                self.lambda_c * (center_loss if center_loss is not None else 0.0)
            )
            
            return logits, total_loss
        
        return logits 