"""
Models package for EmotionNet.

Contains model architectures, attention mechanisms, and factory functions.
"""

from .factory import create_emotion_model
from .attention_emotion_net import AttentionEmotionNet, create_attention_emotion_net
from .attention import (
    SEBlock, 
    SpatialAttention, 
    CBAM, 
    ECABlock, 
    CoordinateAttention,
    MultiHeadSelfAttention,
    SGEAttention,
    PyramidPoolingAttention
)

__all__ = [
    "create_emotion_model",
    "AttentionEmotionNet",
    "create_attention_emotion_net",
    "SEBlock",
    "SpatialAttention", 
    "CBAM",
    "ECABlock",
    "CoordinateAttention",
    "MultiHeadSelfAttention",
    "SGEAttention",
    "PyramidPoolingAttention"
] 