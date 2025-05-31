"""
Training modules for EmotionNet.

Provides trainer, optimizers, losses, and training utilities.
"""

from .trainer import Trainer
from .optimizer import create_optimizer, create_scheduler
from .losses import LabelSmoothingCrossEntropy, FocalLoss, ClassBalancedLoss
from .checkpoint_manager import CheckpointManager
from .sam_optimizer import SAM
from .augmentation_handler import TrainingAugmentationHandler
from .forward_pass import ForwardPassHandler


def create_trainer(trainer_type='standard', **kwargs):
    """
    Create a trainer instance based on type and configuration.
    
    Args:
        trainer_type: Type of trainer ('standard')
        **kwargs: Additional configuration parameters
        
    Returns:
        Trainer instance
    """
    if trainer_type == 'standard':
        return Trainer(
            model=kwargs.get('model'),
            device=kwargs.get('device'),
            config=kwargs.get('config')
        )
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


__all__ = [
    'Trainer',
    'create_optimizer',
    'create_scheduler', 
    'LabelSmoothingCrossEntropy',
    'FocalLoss',
    'ClassBalancedLoss',
    'CheckpointManager',
    'SAM',
    'TrainingAugmentationHandler',
    'ForwardPassHandler',
    'create_trainer'
]