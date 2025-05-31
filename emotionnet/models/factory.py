"""
Model factory functions for EmotionNet.
"""

import torch.nn as nn
from typing import Optional


def create_emotion_model(
    model_type: str = "attention_emotion_net",
    num_classes: int = 7,
    dropout_rate: float = 0.35,
    use_pretrained_backbone: bool = True,
    backbone_type: str = 'convnext',
    **kwargs
) -> nn.Module:
    """
    Create an emotion recognition model.
    
    Args:
        model_type: Type of model to create
        num_classes: Number of emotion classes
        dropout_rate: Dropout rate for regularization
        use_pretrained_backbone: Whether to use pretrained backbone
        backbone_type: Type of backbone to use
        **kwargs: Additional model-specific arguments
        
    Returns:
        PyTorch model instance
    """
    if model_type == "attention_emotion_net" or model_type == "ferplus":
        from .attention_emotion_net import create_attention_emotion_net
        return create_attention_emotion_net(
            num_classes=num_classes,
            dropout=dropout_rate,
            stochastic_depth=kwargs.get('stochastic_depth', True),
            training_mode=kwargs.get('training_mode', 'majority')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
    } 