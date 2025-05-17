"""Models module for EmotionNet."""

from .ema import EMA
from .enhanced_resemote import EnhancedResEmote
from .expert_model import HierarchicalEmotionClassifier
from .fer_transformer import FERTransformer
from .focal_loss import FocalLoss
from .ensemble import EnsembleModel
from emotion_net.training.losses import (
    FocalLoss as StableFocalLoss,
    CompositeEmotionLoss,
    SOTAEmotionLoss,
    HybridLoss,
    AdaptiveEmotionLoss
)

__all__ = [
    'EMA',
    'EnhancedResEmote',
    'HierarchicalEmotionClassifier',
    'FERTransformer',
    'FocalLoss',
    'EnsembleModel',
    'create_model',
    'create_loss_fn'
]

import torch
import torch.nn as nn

# Model wrapper to fix list vs tensor compatibility issues
class ModelWrapper(nn.Module):
    """Wrapper for any model to ensure output format compatibility."""
    
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.use_sota_loss = False  # Flag to preserve tuple output for SOTA loss
        
    def forward(self, x):
        """Forward pass that ensures tensor output unless using SOTA loss."""
        outputs = self.model(x)
        
        # If using SOTA loss, preserve the tuple format (logits, aux_logits, embeddings)
        if self.use_sota_loss and isinstance(outputs, tuple) and len(outputs) >= 3:
            return outputs
        
        # If outputs is a list, return only the first element (main logits)
        if isinstance(outputs, list):
            return outputs[0]
        
        # If outputs is a tuple, return only the first element
        elif isinstance(outputs, tuple):
            return outputs[0]
        
        # If outputs is a dict, return main logits
        elif isinstance(outputs, dict):
            if 'direct_logits' in outputs:
                return outputs['direct_logits']
            elif 'logits' in outputs:
                return outputs['logits']
            else:
                # Return first value in dict
                return list(outputs.values())[0]
        
        # Otherwise just return outputs as is
        return outputs
        
    # Forward any attributes/methods to wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

def create_model(model_name, num_classes=7, pretrained=True, backbone_name=None, embedding_size=512, 
               emotion_groups=None, gem_pooling=False, decoupled_head=False, channels_last=False,
               drop_path_rate=0.0, block_disgust=False, return_features=True):
    """Create an emotion recognition model based on the name.
    
    Models:
        - enhanced_resemote: Enhanced ResEmote model with improved architecture
        - fer_transformer: FER Transformer model
        - hierarchical: Hierarchical Emotion Classifier
    """
    if model_name == 'enhanced_resemote':
        from .enhanced_resemote import enhanced_resemote
        model = enhanced_resemote(
            num_classes=num_classes, 
            pretrained=pretrained,
            embedding_size=embedding_size, 
            use_gem_pooling=gem_pooling,
            drop_path_rate=drop_path_rate,
            backbone_name=backbone_name or 'resnext50_32x4d',
            return_features=return_features
        )
    elif model_name == 'fer_transformer':
        model = FERTransformer(num_classes=num_classes, pretrained=pretrained,
                            embedding_size=embedding_size, use_gem_pooling=gem_pooling,
                            drop_path_rate=drop_path_rate)
    elif model_name == 'hierarchical':
        model = HierarchicalEmotionClassifier(embedding_size, emotion_groups or [])
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: enhanced_resemote, fer_transformer, hierarchical")
    
    # Use channels_last memory format if requested (can improve performance on CUDA)
    if channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    # Wrap model to ensure output format compatibility
    wrapper = ModelWrapper(model)
    
    return wrapper

def create_loss_fn(loss_type, num_classes, **kwargs):
    """Create a loss function based on the loss type.
    
    Supported loss types:
    - cross_entropy: Standard cross-entropy loss with optional label smoothing
    - focal: Focal loss for handling class imbalance
    - sota, sota_emotion: State-of-the-art custom emotion loss function
    - hybrid: Hybrid loss combining multiple losses
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.0))
    elif loss_type == 'focal':
        return StableFocalLoss(
            num_classes=num_classes, 
            gamma=kwargs.get('focal_gamma', 2.0),
            alpha=kwargs.get('focal_alpha', None),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    elif loss_type in ['sota', 'sota_emotion']:
        # Import the SOTA loss here to prevent circular imports
        from emotion_net.training.losses import SOTAEmotionLoss
        print("Using SOTA Emotion Loss for training")
        return SOTAEmotionLoss(
            num_classes=num_classes,
            embedding_size=kwargs.get('embedding_size', 1024),
            label_smoothing=kwargs.get('label_smoothing', 0.1),
            aux_weight=kwargs.get('aux_weight', 0.4)
        )
    elif loss_type == 'hybrid':
        from emotion_net.training.losses import HybridLoss
        return HybridLoss(
            alpha=kwargs.get('focal_alpha', None),
            gamma=kwargs.get('focal_gamma', 2.0),
            label_smoothing=kwargs.get('label_smoothing', 0.1)
        )
    elif loss_type == 'adaptive':
        return AdaptiveEmotionLoss(
            num_classes=num_classes,
            embedding_size=kwargs.get('embedding_size', 1024),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    else:
        print(f"Warning: Unknown loss type '{loss_type}', falling back to cross_entropy")
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.0)) 