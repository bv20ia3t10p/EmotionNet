"""Model definitions for emotion recognition."""

import torch
import torch.nn as nn
from .pretrained_backbones import load_affectnet_pretrained_backbone, get_available_affectnet_backbones
from .sota_resemote import sota_resemote_small, sota_resemote_medium, sota_resemote_large, SOTAEmotionLoss

# Available AFFECTNET_BACKBONES are defined in pretrained_backbones.py
AFFECTNET_BACKBONES = get_available_affectnet_backbones()

__all__ = ['AFFECTNET_BACKBONES', 'sota_resemote_small', 'sota_resemote_medium', 'sota_resemote_large']

def create_model(model_name, num_classes=7, pretrained=True, backbone_name=None, embedding_size=512, 
               emotion_groups=None, gem_pooling=False, decoupled_head=False, channels_last=False,
               drop_path_rate=0.0, block_disgust=False, force_affectnet_backbone=True):
    """Create an emotion recognition model based on the name.
    
    Models:
        - sota_resemote_small: SOTAResEmote with ResNet18 backbone
        - sota_resemote_medium: SOTAResEmote with ResNet34 backbone
        - sota_resemote_large: SOTAResEmote with ResNet50 backbone
    """
    if model_name == 'sota_resemote_small':
        model = sota_resemote_small(num_classes=num_classes, pretrained=pretrained, 
                                   embedding_size=embedding_size, use_gem_pooling=gem_pooling,
                                   drop_path_rate=drop_path_rate, use_transformer=True)
    elif model_name == 'sota_resemote_medium':
        model = sota_resemote_medium(num_classes=num_classes, pretrained=pretrained, 
                                    embedding_size=embedding_size, use_gem_pooling=gem_pooling,
                                    drop_path_rate=drop_path_rate, use_transformer=True)
    elif model_name == 'sota_resemote_large':
        model = sota_resemote_large(num_classes=num_classes, pretrained=pretrained, 
                                   embedding_size=embedding_size, use_gem_pooling=gem_pooling,
                                   drop_path_rate=drop_path_rate, use_transformer=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: sota_resemote_small, sota_resemote_medium, sota_resemote_large")
    
    # Use channels_last memory format if requested (can improve performance on CUDA)
    if channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    return model

def create_loss_fn(loss_type, num_classes, **kwargs):
    """Create a loss function based on the loss type.
    
    Supported loss types:
    - cross_entropy: Standard cross-entropy loss with optional label smoothing
    - sota_emotion: SOTAEmotionLoss with auxiliary loss and label smoothing
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.0))
    elif loss_type == 'sota_emotion':
        return SOTAEmotionLoss(num_classes=num_classes, 
                               label_smoothing=kwargs.get('label_smoothing', 0.1),
                               aux_weight=kwargs.get('aux_weight', 0.4))
    else:
        raise ValueError(f"Loss type {loss_type} is not implemented. Available options: 'cross_entropy', 'sota_emotion'") 