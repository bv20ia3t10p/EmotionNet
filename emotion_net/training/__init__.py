"""Training module for emotion recognition."""

from .trainer import EmotionTrainer
from .model_manager import setup_training, calculate_class_weights, create_criterion, save_model
from .training_loops import train_epoch, validate
from .metrics import calculate_metrics, save_training_history, plot_training_history

__all__ = [
    'EmotionTrainer',
    'setup_training',
    'calculate_class_weights',
    'create_criterion',
    'save_model',
    'train_epoch',
    'validate',
    'calculate_metrics',
    'save_training_history',
    'plot_training_history'
] 