"""
Backbone architectures for EmotionNet models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from ..attention import SEBlock


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class EnhancedResidualBlock(nn.Module):
    """Enhanced Residual Block with Stochastic Depth and Squeeze-Excitation"""
    def __init__(self, in_ch, out_ch, stride=1, drop_path_rate=0.1):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Add SE block for channel attention
        self.se = SEBlock(out_ch, reduction=16)
        
        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE attention
        out = self.drop_path(out) + identity
        out = F.relu(out)
        return out


def create_efficientnet_backbone():
    """Create EfficientNet-B0 backbone."""
    backbone = create_model('efficientnet_b0', pretrained=True, in_chans=1, features_only=True)
    # Return the backbone and the number of output channels from the last feature layer
    # EfficientNet-B0 with features_only=True returns 320 channels in the final layer
    return backbone, 320


def create_custom_backbone():
    """Create custom ResNet-like backbone."""
    layers = []
    
    # Initial layers
    layers.extend([
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ])
    
    # Residual stages with increasing drop path rate
    channels = [64, 128, 256, 512]
    for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
        # Two blocks per stage
        drop_rate = 0.1 * (i + 1) / len(channels)
        layers.append(EnhancedResidualBlock(in_ch, out_ch, stride=2, drop_path_rate=drop_rate))
        layers.append(EnhancedResidualBlock(out_ch, out_ch, stride=1, drop_path_rate=drop_rate))
    
    return nn.Sequential(*layers), 512


def create_convnext_backbone():
    """Create ConvNeXt-Tiny backbone optimized for emotion recognition."""
    # Use ConvNeXt-Tiny which is suitable for smaller images and emotion tasks
    backbone = create_model('convnext_tiny', pretrained=True, in_chans=1, features_only=True)
    # ConvNeXt-Tiny with features_only=True returns 768 channels in the final layer
    return backbone, 768


def create_convnext_small_backbone():
    """Create ConvNeXt-Small backbone for higher capacity."""
    backbone = create_model('convnext_small', pretrained=True, in_chans=1, features_only=True)
    # ConvNeXt-Small with features_only=True returns 768 channels in the final layer
    return backbone, 768


def create_convnext_base_backbone():
    """Create ConvNeXt-Base backbone for maximum performance."""
    backbone = create_model('convnext_base', pretrained=True, in_chans=1, features_only=True)
    # ConvNeXt-Base with features_only=True returns 1024 channels in the final layer
    return backbone, 1024 