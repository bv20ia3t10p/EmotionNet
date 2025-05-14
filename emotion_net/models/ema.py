"""Exponential Moving Average (EMA) implementation for model weights.

EMA is a common technique to improve model stability during training.
"""

import torch


class EMA:
    """Exponential Moving Average for model weights.
    
    This maintains a moving average of model parameters during training,
    which can produce better results and more stable convergence.
    """
    
    def __init__(self, model, decay=0.999):
        """Initialize EMA with a model.
        
        Args:
            model: The PyTorch model
            decay: The EMA decay rate (higher = slower moving average)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update the EMA parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply the EMA weights to the model for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore the original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {} 