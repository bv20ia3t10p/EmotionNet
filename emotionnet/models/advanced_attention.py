"""
Super Advanced Attention Mechanisms for EmotionNet
Implementation of cutting-edge attention mechanisms designed to outperform PAtt-Lite

Key components:
1. Hybrid Vision Transformer Attention (HVTA) - combines CNN features with transformer-style attention
2. Cross-Scale Pyramidal Attention (CSPA) - enables multi-scale feature interactions
3. Efficient Local-Global Attention (ELGA) - captures both local and global dependencies efficiently
4. Dynamic Region-Aware Attention (DRAA) - focuses on emotionally salient facial regions
5. Lightweight Transformer Patch Attention (LTPA) - improved version of PAtt-Lite with transformer principles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LightweightTransformerPatchAttention(nn.Module):
    """
    Lightweight Transformer Patch Attention (LTPA)
    An enhanced version of PAtt-Lite that incorporates transformer principles
    while maintaining computational efficiency.
    
    Improvements over PAtt-Lite:
    1. Dynamically weighted patch interactions
    2. Feature modulation via local self-attention
    3. Efficient position encoding
    4. Cross-patch normalization
    """
    def __init__(self, channels, patch_size=4, num_heads=4, dropout=0.1):
        super(LightweightTransformerPatchAttention, self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        # Projection layers for queries, keys, values
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Relative positional encoding
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * patch_size - 1, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * patch_size - 1, self.head_dim))
        
        # Position encoding indices
        self.register_buffer("rel_pos_indices", self._get_rel_pos_indices(patch_size))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Initialize positional encodings
        nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
        nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
    
    def _get_rel_pos_indices(self, patch_size):
        """Create relative position encoding indices"""
        coords = torch.arange(patch_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flatten = coords.flatten(1)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel_coords = rel_coords + patch_size - 1  # Shift to [0, 2*patch_size-1]
        return rel_coords.permute(1, 2, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Make sure H and W are divisible by patch_size
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            B, C, H, W = x.shape
        
        # Calculate number of patches
        h_patches = H // P
        w_patches = W // P
        
        # Generate QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3)  # 3, B, num_heads, H, W, head_dim
        
        # Split into patches
        qkv = qkv.reshape(3, B, self.num_heads, h_patches, P, w_patches, P, self.head_dim)
        qkv = qkv.permute(0, 1, 2, 3, 5, 4, 6, 7)  # 3, B, num_heads, h_patches, w_patches, P, P, head_dim
        
        # Extract queries, keys, values
        q, k, v = qkv
        
        # Reshape for matrix multiplication
        q = q.reshape(B, self.num_heads, h_patches * w_patches, P * P, self.head_dim)
        k = k.reshape(B, self.num_heads, h_patches * w_patches, P * P, self.head_dim)
        v = v.reshape(B, self.num_heads, h_patches * w_patches, P * P, self.head_dim)
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative positional encoding
        rel_pos_h = self.rel_pos_h[self.rel_pos_indices[..., 0]]
        rel_pos_w = self.rel_pos_w[self.rel_pos_indices[..., 1]]
        rel_pos = rel_pos_h + rel_pos_w
        
        # Reshape positional encoding for addition to attention scores
        rel_pos = rel_pos.reshape(P*P, P*P)
        attn = attn + rel_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).reshape(B, self.num_heads, h_patches, w_patches, P, P, self.head_dim)
        
        # Reshape back to original format
        out = out.permute(0, 1, 2, 4, 3, 5, 6)
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        
        # Final projection
        out = self.proj(out)
        
        # Remove padding if needed
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H-pad_h, :W-pad_w]
            
        return out


class HybridVisionTransformerAttention(nn.Module):
    """
    Hybrid Vision Transformer Attention (HVTA)
    
    Combines the strengths of CNNs and transformers:
    - Uses CNN features as input
    - Applies transformer-style self-attention
    - Efficiently processes spatial relationships
    - Maintains global context while preserving local details
    """
    def __init__(self, channels, num_heads=8, dropout=0.1, qkv_bias=True):
        super(HybridVisionTransformerAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        # Local feature enhancement
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm1 = nn.BatchNorm2d(channels)
        
        # Multi-head attention components
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Local feature enhancement
        local_x = self.norm1(self.local_conv(x))
        residual = local_x
        
        # Generate q, k, v
        q = self.q(local_x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # B, num_heads, N, head_dim
        k = self.k(local_x).reshape(B, self.num_heads, self.head_dim, N)  # B, num_heads, head_dim, N
        v = self.v(local_x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # B, num_heads, N, head_dim
        
        # Calculate attention
        attn = (q @ k) * (self.head_dim ** -0.5)  # B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)  # B, C, H, W
        out = self.proj(out)
        
        # First residual connection
        out = out + residual
        
        # Feed-forward network with residual connection
        residual = out
        out = self.ffn(self.norm2(out)) + residual
        
        return out


class CrossScalePyramidalAttention(nn.Module):
    """
    Cross-Scale Pyramidal Attention (CSPA)
    
    Enables multi-scale feature interactions through:
    - Feature pyramid decomposition
    - Cross-scale attention mechanisms
    - Adaptive feature fusion
    - Multi-scale context integration
    """
    def __init__(self, channels, scales=[1, 2, 4, 8]):
        super(CrossScalePyramidalAttention, self).__init__()
        self.scales = scales
        num_scales = len(scales)
        
        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(channels, channels // num_scales, kernel_size=1, bias=False)
            for _ in range(num_scales)
        ])
        
        # Cross-scale attention
        self.cross_scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels // num_scales, channels // num_scales, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels // num_scales),
                nn.ReLU()
            ) for scale in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Scale-specific projections
        scale_features = [proj(x) for proj in self.scale_projs]
        
        # Cross-scale attention
        pyramid_features = []
        for i, (feature, attn) in enumerate(zip(scale_features, self.cross_scale_attn)):
            pooled = attn(feature)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # Concatenate pyramid features
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        # Generate attention weights
        attention = self.fusion(pyramid_concat)
        
        return x * attention


class EfficientLocalGlobalAttention(nn.Module):
    """
    Efficient Local-Global Attention (ELGA)
    
    Captures both local and global dependencies efficiently through:
    - Decomposed attention computation
    - Local neighborhood attention
    - Global sparse attention
    - Feature aggregation mechanism
    """
    def __init__(self, channels, local_window=7, global_points=16):
        super(EfficientLocalGlobalAttention, self).__init__()
        self.channels = channels
        self.local_window = local_window
        self.global_points = global_points
        
        # Local attention
        self.local_conv = nn.Conv2d(
            channels, channels, 
            kernel_size=local_window, 
            padding=local_window//2, 
            groups=channels
        )
        
        # Global attention through key points
        self.global_pool = nn.AdaptiveAvgPool2d(global_points)
        self.global_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Fusion layers
        self.local_weight = nn.Parameter(torch.ones(1))
        self.global_weight = nn.Parameter(torch.ones(1))
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Local attention
        local_out = self.local_conv(x)
        
        # Global attention
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_out = F.interpolate(
            global_feat, 
            size=(x.size(2), x.size(3)), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Dynamic weighting
        local_out = local_out * self.local_weight
        global_out = global_out * self.global_weight
        
        # Fusion
        concat = torch.cat([local_out, global_out], dim=1)
        attention = self.fusion(concat)
        
        return x * attention


class DynamicRegionAwareAttention(nn.Module):
    """
    Dynamic Region-Aware Attention (DRAA)
    
    Focuses on emotionally salient facial regions through:
    - Region proposal mechanism
    - Adaptive importance weighting
    - Emotional prior integration
    - Region-specific feature enhancement
    """
    def __init__(self, channels, num_regions=7):  # 7 for basic emotions
        super(DynamicRegionAwareAttention, self).__init__()
        self.channels = channels
        self.num_regions = num_regions
        
        # Region proposal network
        self.region_conv = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_regions, kernel_size=1)
        )
        
        # Region-specific feature enhancement
        self.region_enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // num_regions, kernel_size=1),
                nn.BatchNorm2d(channels // num_regions),
                nn.ReLU()
            ) for _ in range(num_regions)
        ])
        
        # Attention fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate region proposals
        region_maps = self.region_conv(x)  # B, num_regions, H, W
        region_weights = F.softmax(region_maps, dim=1)  # B, num_regions, H, W
        
        # Enhanced features per region
        enhanced_features = []
        for i in range(self.num_regions):
            # Extract region weight
            region_weight = region_weights[:, i:i+1]  # B, 1, H, W
            
            # Apply region-specific enhancement
            enhanced = self.region_enhancement[i](x)  # B, C//num_regions, H, W
            weighted_feature = enhanced * region_weight
            enhanced_features.append(weighted_feature)
        
        # Concatenate enhanced features
        concat_features = torch.cat(enhanced_features, dim=1)  # B, C, H, W
        
        # Generate attention map
        attention = self.fusion(concat_features)
        
        return x * attention


# Higher-level attention that combines multiple attention mechanisms
class HybridSuperAttention(nn.Module):
    """
    Hybrid Super Attention (HSA)
    
    Combines multiple advanced attention mechanisms for maximum effectiveness:
    - Leverages strengths of each attention type
    - Adaptively weights contribution of each mechanism
    - Optimized for facial emotion recognition
    - Designed to outperform PAtt-Lite
    """
    def __init__(self, channels, use_lightweight=True):
        super(HybridSuperAttention, self).__init__()
        self.channels = channels
        
        # Use LTPA as direct improvement over PAtt-Lite
        if use_lightweight:
            self.ltpa = LightweightTransformerPatchAttention(
                channels, patch_size=4, num_heads=4
            )
        
        # Hybrid Vision Transformer Attention
        self.hvta = HybridVisionTransformerAttention(
            channels, num_heads=4, dropout=0.1
        )
        
        # Cross-Scale Pyramidal Attention
        self.cspa = CrossScalePyramidalAttention(
            channels, scales=[1, 2, 4]
        )
        
        # Region-Aware Attention
        self.draa = DynamicRegionAwareAttention(
            channels, num_regions=7
        )
        
        # Attention weights (learnable)
        self.attention_weights = nn.Parameter(torch.ones(4 if use_lightweight else 3))
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.use_lightweight = use_lightweight
    
    def forward(self, x):
        attention_outputs = []
        
        # Apply the different attention mechanisms
        if self.use_lightweight:
            attention_outputs.append(self.ltpa(x))
        
        attention_outputs.append(self.hvta(x))
        attention_outputs.append(self.cspa(x))
        attention_outputs.append(self.draa(x))
        
        # Normalize attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        
        # Weighted sum of attention outputs
        weighted_sum = 0
        for i, output in enumerate(attention_outputs):
            weighted_sum += weights[i] * output
        
        # Final fusion
        result = self.fusion(weighted_sum)
        
        return result


# ========== Aggregators ==========

class AttentionAggregator(nn.Module):
    """
    Aggregates multiple attention mechanisms into a single module
    """
    def __init__(self, channels, mechanisms=None):
        super(AttentionAggregator, self).__init__()
        self.channels = channels
        
        # Default: use efficient mechanisms suitable for mobile applications
        if mechanisms is None:
            self.mechanisms = nn.ModuleList([
                LightweightTransformerPatchAttention(channels, patch_size=4, num_heads=4),
                EfficientLocalGlobalAttention(channels, local_window=7, global_points=8),
                CrossScalePyramidalAttention(channels, scales=[1, 2, 4])
            ])
        else:
            self.mechanisms = nn.ModuleList(mechanisms)
        
        # Learnable weights for each mechanism
        self.weights = nn.Parameter(torch.ones(len(self.mechanisms)))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        outputs = [mech(x) for mech in self.mechanisms]
        
        # Normalize weights
        norm_weights = F.softmax(self.weights, dim=0)
        
        # Weighted sum
        result = sum(w * out for w, out in zip(norm_weights, outputs))
        
        return self.fusion(result)


# Factory function to create specific attention mechanisms
def create_attention_mechanism(name, channels, **kwargs):
    """
    Factory function to create specific attention mechanisms
    
    Args:
        name: The name of the attention mechanism
        channels: Number of input channels
        **kwargs: Additional arguments for the specific attention mechanism
    
    Returns:
        An attention mechanism module
    """
    if name == 'ltpa':
        return LightweightTransformerPatchAttention(
            channels,
            patch_size=kwargs.get('patch_size', 4),
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif name == 'hvta':
        return HybridVisionTransformerAttention(
            channels,
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1),
            qkv_bias=kwargs.get('qkv_bias', True)
        )
    elif name == 'cspa':
        return CrossScalePyramidalAttention(
            channels,
            scales=kwargs.get('scales', [1, 2, 4, 8])
        )
    elif name == 'elga':
        return EfficientLocalGlobalAttention(
            channels,
            local_window=kwargs.get('local_window', 7),
            global_points=kwargs.get('global_points', 16)
        )
    elif name == 'draa':
        return DynamicRegionAwareAttention(
            channels,
            num_regions=kwargs.get('num_regions', 7)
        )
    elif name == 'hybrid':
        return HybridSuperAttention(
            channels,
            use_lightweight=kwargs.get('use_lightweight', True)
        )
    elif name == 'aggregator':
        return AttentionAggregator(
            channels,
            mechanisms=kwargs.get('mechanisms', None)
        )
    else:
        raise ValueError(f"Unknown attention mechanism: {name}")


# Test function
if __name__ == "__main__":
    # Test each attention mechanism with a sample input
    x = torch.randn(2, 64, 32, 32)
    
    # Test LTPA
    ltpa = LightweightTransformerPatchAttention(64, patch_size=4, num_heads=4)
    y_ltpa = ltpa(x)
    print(f"LTPA output shape: {y_ltpa.shape}")
    
    # Test HVTA
    hvta = HybridVisionTransformerAttention(64, num_heads=4)
    y_hvta = hvta(x)
    print(f"HVTA output shape: {y_hvta.shape}")
    
    # Test CSPA
    cspa = CrossScalePyramidalAttention(64)
    y_cspa = cspa(x)
    print(f"CSPA output shape: {y_cspa.shape}")
    
    # Test ELGA
    elga = EfficientLocalGlobalAttention(64)
    y_elga = elga(x)
    print(f"ELGA output shape: {y_elga.shape}")
    
    # Test DRAA
    draa = DynamicRegionAwareAttention(64)
    y_draa = draa(x)
    print(f"DRAA output shape: {y_draa.shape}")
    
    # Test Hybrid
    hybrid = HybridSuperAttention(64)
    y_hybrid = hybrid(x)
    print(f"Hybrid output shape: {y_hybrid.shape}") 