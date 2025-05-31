"""
EmotionNet: Facial Emotion Recognition

This package provides a solution for facial emotion recognition including:
- Attention-based neural network architectures
- Effective training techniques
- Comprehensive data handling for FER2013 and FERPlus
- Configuration management
"""

# Core model components
from .models import create_emotion_model, AttentionEmotionNet

# Data handling
from .data import get_data_loaders, FER2013Dataset, get_ferplus_data_loaders, FERPlusDataset

# Training components
from .training import Trainer

# Configuration
from .config import Config, create_config, create_ferplus_config

# Utilities
from .utils import set_seed, get_device

__version__ = "2.0.0"
__author__ = "EmotionNet Team"

__all__ = [
    # Models
    'create_emotion_model',
    'AttentionEmotionNet',
    
    # Data
    'get_data_loaders', 
    'FER2013Dataset',
    'get_ferplus_data_loaders',
    'FERPlusDataset',
    
    # Training
    'Trainer',
    
    # Configuration
    'Config',
    'create_config',
    'create_ferplus_config',
    
    # Utilities
    'set_seed',
    'get_device'
] 