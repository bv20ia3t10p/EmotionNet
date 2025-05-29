"""
Advanced Attention Mechanisms for EmotionNet
Contains all attention modules used in the custom ResEmoteNet architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block with improved design"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling
        reduced_channels = max(in_channels // reduction, 8)  # Ensure minimum channels
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, reduced_channels, bias=False),  # *2 for avg+max
            nn.ReLU(),  # Remove inplace=True for SAM compatibility
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Combine average and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        y = self.fc(combined).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ECABlock(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CoordinateAttention(nn.Module):
    """Coordinate Attention for spatial awareness"""
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()  # Remove inplace=True for SAM compatibility
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention for feature relationships"""
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        return x_attn.transpose(1, 2).reshape(B, C, H, W) + x


class SGEAttention(nn.Module):
    """Spatial Group-wise Enhance attention for spatial feature enhancement"""
    def __init__(self, channels, groups=64):
        super(SGEAttention, self).__init__()
        self.groups = min(groups, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.contiguous()
        
        x_grouped = x.reshape(b * self.groups, -1, h, w)
        xn = x_grouped * self.avg_pool(x_grouped)
        xn = xn.sum(dim=1, keepdim=True)
        
        t = xn.reshape(b * self.groups, -1).contiguous()
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        
        t = t.reshape(b, self.groups, h, w)
        t = t.mean(dim=1, keepdim=True)
        t = t.reshape(b, 1, h, w)
        
        x = x * self.sig(t)
        return x


class ChannelShuffleAttention(nn.Module):
    """Channel Shuffle Attention for better feature mixing"""
    def __init__(self, channels, groups=8):
        super(ChannelShuffleAttention, self).__init__()
        self.groups = groups
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 4, bias=False),
            nn.ReLU(),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
        
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        
        return x
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Channel shuffle
        x_shuffled = self.channel_shuffle(x)
        
        # Attention
        avg_out = self.avg_pool(x_shuffled).view(b, c)
        max_out = self.max_pool(x_shuffled).view(b, c)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.fc(combined).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)


class PyramidPoolingAttention(nn.Module):
    """Pyramid Pooling Attention for multi-scale feature aggregation"""
    def __init__(self, channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingAttention, self).__init__()
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList()
        
        for pool_size in pool_sizes:
            self.convs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(channels, channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(channels // len(pool_sizes)),
                nn.ReLU()  # Remove inplace=True for SAM compatibility
            ))
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_features = []
        
        for conv in self.convs:
            pooled = conv(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        attention = self.fusion(pyramid_concat)
        
        return x * attention 