"""FER-Transformer: State-of-the-art hybrid CNN-Transformer model for facial emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math

class MultiHeadSelfAttention(nn.Module):
    """Multi-head Self Attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Generate query, key, value for all heads in batch
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        # Convert scores to weights with softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        out = attn @ v  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        # Project back to embedding dimension
        out = self.proj(out)
        return out

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with pre-layer normalization."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        # Pre-norm attention block
        x = x + self.drop_path(self.attention(self.norm1(x)))
        # Pre-norm MLP block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        output = x.div(keep_prob) * random_tensor
        return output

class GEMPool(nn.Module):
    """Generalized Mean Pooling."""
    
    def __init__(self, p=3.0, dim=1):
        super().__init__()
        self.dim = dim
        self.p = nn.Parameter(torch.ones(1) * p)
        
    def forward(self, x):
        return self.gem(x, p=self.p, dim=self.dim)
    
    def gem(self, x, p, dim):
        return F.avg_pool2d(x.clamp(min=1e-6).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class FERTransformer(nn.Module):
    """Hybrid CNN-Transformer model for Facial Emotion Recognition."""
    
    def __init__(self, num_classes=7, embed_dim=768, depth=8, num_heads=8, 
                 mlp_ratio=4, dropout=0.1, drop_path_rate=0.1, pretrained=True,
                 use_gem_pooling=True, embedding_size=1024, attention_pool=True):
        super().__init__()
        
        # Load pretrained ResNet backbone
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # ResNet50 output features
        backbone_dim = 2048
        mid_backbone_dim = 512  # Dimension after layer2 of ResNet50
        
        # Spatial reduction and dimension projection
        self.reduction = nn.Sequential(
            nn.Conv2d(backbone_dim, embed_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, embed_dim))  # 7x7 feature map
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        # Norm after transformer
        self.norm = nn.LayerNorm(embed_dim)
        
        # Pooling options
        self.use_gem_pooling = use_gem_pooling
        self.attention_pool = attention_pool
        
        if use_gem_pooling:
            self.mid_pool = GEMPool(p=3.0)
            self.final_pool = GEMPool(p=3.0)
        else:
            self.mid_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.final_pool = nn.Identity()
        
        # Attention pooling for token sequence
        if attention_pool:
            self.attn_pool = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Final embedding and classification layers
        self.fc_embed = nn.Linear(embed_dim, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)
        
        # Auxiliary classifier from mid-level features
        # Note: The dimension of mid_features is 512 (after pool), not 2048
        self.aux_classifier = nn.Sequential(
            nn.Linear(mid_backbone_dim, embedding_size // 2),
            nn.BatchNorm1d(embedding_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_size // 2, num_classes)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize transformer weights
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.fc_embed.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
        # Initialize aux classifier
        for m in self.aux_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x):
        # Extract CNN features
        mid_features = self.backbone[:5](x)  # After layer2 (output dim is 512)
        x = self.backbone[5:](mid_features)  # Complete backbone (output dim is 2048)
        
        # Apply pooling to mid-level features for auxiliary classifier
        mid_pool = self.mid_pool(mid_features)
        mid_pool_flat = torch.flatten(mid_pool, 1)
        
        # Ensure correct dimensions for mid_pool_flat
        # mid_pool_flat shape should be [batch_size, 512] not [batch_size, 2048]
        
        # Reduce spatial dimensions and project to embedding dim
        x = self.reduction(x)
        
        # Reshape to sequence form
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Add class token
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Apply normalization
        x = self.norm(x)
        
        # Different pooling strategies
        if self.attention_pool:
            # Weighted average based on attention scores (excluding CLS token)
            attn_weights = self.attn_pool(x[:, 1:, :])  # [B, seq_len, 1]
            x = torch.sum(x[:, 1:, :] * attn_weights, dim=1)
        else:
            # Use CLS token representation
            x = x[:, 0]
        
        return x, mid_pool_flat
    
    def forward(self, x):
        # Get features from backbone and transformer
        features, mid_features = self.forward_features(x)
        
        # Apply final embedding
        embedding = self.fc_embed(features)
        embedding = F.gelu(embedding)
        embedding = self.dropout(embedding)
        
        # Main classifier
        logits = self.fc(embedding)
        
        # Auxiliary classifier
        aux_logits = self.aux_classifier(mid_features)
        
        return [logits, aux_logits, embedding]

def fer_transformer_base(num_classes=7, pretrained=True, embedding_size=1024, use_gem_pooling=True, drop_path_rate=0.1):
    """Create a base FER-Transformer model."""
    model = FERTransformer(
        num_classes=num_classes,
        embed_dim=768,
        depth=8,
        num_heads=8,
        dropout=0.1,
        drop_path_rate=drop_path_rate,
        pretrained=pretrained,
        use_gem_pooling=use_gem_pooling,
        embedding_size=embedding_size,
        attention_pool=True
    )
    return model

def fer_transformer_large(num_classes=7, pretrained=True, embedding_size=1024, use_gem_pooling=True, drop_path_rate=0.2):
    """Create a large FER-Transformer model."""
    model = FERTransformer(
        num_classes=num_classes,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        dropout=0.1,
        drop_path_rate=drop_path_rate,
        pretrained=pretrained,
        use_gem_pooling=use_gem_pooling,
        embedding_size=embedding_size,
        attention_pool=True
    )
    return model 