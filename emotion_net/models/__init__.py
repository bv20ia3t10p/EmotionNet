"""Model definitions for emotion recognition."""

from .model import EmotionNet
from .ensemble import EnsembleModel
from .ema import EMA
from .model_parts import SimpleChannelAttention
from .expert_model import ExpertEmotionModel, HybridLoss

__all__ = ['EmotionNet', 'EnsembleModel', 'EMA', 'SimpleChannelAttention', 'ExpertEmotionModel', 'HybridLoss']

def create_model(model_name, num_classes, pretrained=True, **kwargs):
    """Create a model based on the model name."""
    if model_name == 'expert':
        return ExpertEmotionModel(
            backbone_name=kwargs.get('backbone_name', 'convnext_tiny'),
            num_classes=num_classes,
            pretrained=pretrained,
            embedding_size=kwargs.get('embedding_size', 512),
            emotion_groups_str=kwargs.get('emotion_groups', "sad-neutral-angry,happy-surprise,fear-disgust"),
            use_gem_pooling=kwargs.get('gem_pooling', True),
            use_decoupled_head=kwargs.get('decoupled_head', True),
            drop_path_rate=kwargs.get('drop_path_rate', 0.2),
            channels_last=kwargs.get('channels_last', True)
        )
    else:
        # For non-expert models, use the original EmotionNet
        return EmotionNet(
            backbone=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            drop_path_rate=kwargs.get('drop_path_rate', 0.0)
        )

def create_loss_fn(loss_type, num_classes, **kwargs):
    """Create a loss function based on the loss type."""
    import torch.nn as nn
    import torch.nn.functional as F
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.0))
    elif loss_type == 'focal':
        from .focal_loss import FocalLoss
        return FocalLoss(
            gamma=kwargs.get('focal_gamma', 2.0),
            alpha=kwargs.get('focal_alpha', None),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    elif loss_type == 'hybrid':
        # Create class weights if needed
        weights = None
        if kwargs.get('class_weights', False):
            weights = create_class_weights(kwargs.get('class_distribution', None))
        
        try:
            from .expert_model import HybridLoss
            return HybridLoss(
                num_classes=num_classes,
                focal_gamma=kwargs.get('focal_gamma', 2.0),
                label_smoothing=kwargs.get('label_smoothing', 0.1),
                triplet_margin=kwargs.get('triplet_margin', 0.3),
                weights=weights
            )
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import HybridLoss: {e}. Falling back to FocalLoss.")
            from .focal_loss import FocalLoss
            return FocalLoss(
                gamma=kwargs.get('focal_gamma', 2.0),
                alpha=kwargs.get('focal_alpha', None),
                label_smoothing=kwargs.get('label_smoothing', 0.0)
            )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def create_class_weights(class_distribution=None):
    """Create class weights based on the distribution."""
    import torch
    import numpy as np
    
    if class_distribution is None:
        # Use default FER2013 distribution if not provided
        class_distribution = {
            0: 4953,  # Angry
            1: 547,   # Disgust (rare)
            2: 5121,  # Fear
            3: 8989,  # Happy (common)
            4: 6077,  # Sad
            5: 4002,  # Surprise
            6: 6198   # Neutral
        }
    
    # Calculate inverse frequency weights
    total = sum(class_distribution.values())
    weights = torch.zeros(len(class_distribution))
    
    for cls, count in class_distribution.items():
        if count > 0:
            weights[cls] = total / (len(class_distribution) * count)
        else:
            weights[cls] = 1.0
    
    # Normalize weights
    weights = weights / weights.sum() * len(class_distribution)
    
    # Apply additional weight to sad class if specified
    sad_class_weight = 2.0  # Default value
    if sad_class_weight > 1.0:
        weights[4] *= sad_class_weight  # Sad class index is 4
        weights = weights / weights.sum() * len(class_distribution)
    
    return weights 