"""Model definitions for emotion recognition."""

import torch
import torch.nn as nn
from .model import EmotionNet
from .advanced_emotion_model import AdvancedEmotionModel, AdvancedEmotionLoss
from .focal_loss import FocalLoss
from .expert_model import ExpertEmotionModel, HybridLoss
from .expert import ExpertModel
from .ensemble import EnsembleModel

__all__ = ['EmotionNet', 'AdvancedEmotionModel', 'FocalLoss', 'AdvancedEmotionLoss', 'ExpertEmotionModel', 'ExpertModel', 'HybridLoss', 'EnsembleModel']

def create_model(model_name, num_classes=7, pretrained=True, backbone_name=None, embedding_size=512, 
               emotion_groups=None, gem_pooling=False, decoupled_head=False, channels_last=False,
               drop_path_rate=0.0, block_disgust=False):
    """Create an emotion recognition model based on the name."""
    if model_name == 'affectnet' or model_name == 'advanced':
        return AdvancedEmotionModel(
            backbone_name=backbone_name or 'efficientnet_b0',
            num_classes=num_classes,
            pretrained=pretrained,
            embedding_size=embedding_size,
            drop_path_rate=drop_path_rate,
            drop_rate=0.3,
            attention_type='cbam',
            use_gem_pooling=gem_pooling
        )
    elif model_name == 'expert':
        # Use the ExpertModel implementation
        return ExpertModel(
            backbone_name=backbone_name or 'resnet50',
            num_classes=num_classes,
            pretrained=pretrained,
            embedding_size=embedding_size,
            emotion_groups=emotion_groups,
            gem_pooling=gem_pooling,
            decoupled_head=decoupled_head,
            drop_path_rate=drop_path_rate,
            channels_last=channels_last
        )
    else:
        # Use ensemble model (default) or fall back to the original EmotionNet
        if model_name == 'ensemble':
            return EnsembleModel(
                backbones=backbone_name if isinstance(backbone_name, list) else ['resnet50'],
                num_classes=num_classes,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate
            )
        else:
            return EmotionNet(
                backbone=model_name,
                num_classes=num_classes,
                pretrained=pretrained
            )
    
    # Wrap with disgust blocker if requested
    if block_disgust:
        print("Applying DisgustBlocker wrapper to completely block disgust predictions")
        model = DisgustBlocker(model, disgust_class_idx=1)  # disgust is index 1
    
    return model

def create_loss_fn(loss_type, num_classes, **kwargs):
    """Create a loss function based on the loss type."""
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.0))
    elif loss_type == 'focal':
        return FocalLoss(
            gamma=kwargs.get('focal_gamma', 2.0),
            alpha=kwargs.get('focal_alpha', None),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    elif loss_type == 'advanced':
        # Use our new advanced loss function
        return AdvancedEmotionLoss(
            num_classes=num_classes,
            gamma=kwargs.get('focal_gamma', 2.0),
            alpha=create_class_weights(kwargs.get('class_distribution', None)) if kwargs.get('class_weights', False) else None,
            label_smoothing=kwargs.get('label_smoothing', 0.1),
            confidence_penalty=kwargs.get('confidence_penalty', 0.1)
        )
    elif loss_type == 'hybrid':
        # For backward compatibility, use the AdvancedEmotionLoss
        weights = create_class_weights(kwargs.get('class_distribution', None)) if kwargs.get('class_weights', False) else None
        return AdvancedEmotionLoss(
            num_classes=num_classes,
            gamma=kwargs.get('focal_gamma', 2.0),
            alpha=weights,
            label_smoothing=kwargs.get('label_smoothing', 0.1),
            confidence_penalty=kwargs.get('confidence_penalty', 0.05)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def create_class_weights(class_distribution=None):
    """Create class weights based on the distribution."""
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

# Add a class to completely block the disgust class
class DisgustBlocker(nn.Module):
    """Wrapper module that blocks disgust predictions entirely."""
    
    def __init__(self, model, disgust_class_idx=1):
        """Initialize the wrapper module.
        
        Args:
            model: The base model to wrap
            disgust_class_idx: Index of the disgust class to block (default: 1)
        """
        super().__init__()
        self.model = model
        self.disgust_class_idx = disgust_class_idx
        print(f"DisgustBlocker initialized: Will completely block class index {disgust_class_idx}")
    
    def forward(self, x):
        """Forward pass with disgust blocking.
        
        During training and inference, this wrapper will:
        1. Get the original model's output
        2. Apply an extremely negative value (-100) to the disgust class logit
        3. Return the modified logits
        
        Args:
            x: Input tensor
            
        Returns:
            Modified logits with disgust class blocked
        """
        # Get original output
        outputs = self.model(x)
        
        # If it's a dictionary (for ensemble models), handle accordingly
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                # Apply extreme negative value to disgust class
                batch_size = outputs['logits'].shape[0]
                outputs['logits'][:, self.disgust_class_idx] = -100.0
                
                # Return modified outputs
                return outputs
        
        # For standard models returning tensor logits directly
        batch_size = outputs.shape[0]
        outputs[:, self.disgust_class_idx] = -100.0
        
        return outputs 