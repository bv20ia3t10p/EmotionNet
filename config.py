"""
Clean SOTA Configuration for EmotionNet FER2013 Training
Simplified configuration focused only on what's needed for optimal performance.
"""

import argparse
from typing import Dict, Any

# Core constants
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# SOTA Class Weights (fixed for optimal performance)
CLASS_WEIGHTS = {
    'angry': 1.2,
    'disgust': 3.0,  # Reduced from 8.0 to prevent model collapse
    'fear': 1.1,
    'happy': 0.9,
    'sad': 1.0,
    'surprise': 1.5,
    'neutral': 1.1
}

# Default paths
PATHS = {
    'train_csv': 'dataset/fer2013/fer2013.csv',
    'val_csv': 'dataset/fer2013/fer2013.csv',
    'test_csv': 'dataset/fer2013/fer2013.csv',
    'img_dir': 'dataset/fer2013',
    'checkpoint_dir': 'checkpoints'
}

def get_sota_config() -> Dict[str, Any]:
    """Get SOTA optimized configuration for FER2013."""
    return {
        # Model settings
        'num_classes': NUM_CLASSES,
        'backbone': 'sota_emotionnet',
        'img_size': 48,
        'dropout_rate': 0.3,  # Increased from 0.25 to combat overfitting
        
        # Training settings - UPDATED FOR PLATEAU BREAKING
        'batch_size': 16,  # REDUCED from 32 to handle enhanced model complexity
        'epochs': 150,
        'lr': 0.003,  # INCREASED from 0.001 to break plateau
        'weight_decay': 0.0005,  # INCREASED regularization
        'focal_gamma': 1.5,  # INCREASED from 1.0 for harder examples
        'grad_clip': 0.5,  # REDUCED for more aggressive updates
        
        # Scheduler settings - SWITCHED TO COSINE ANNEALING
        'lr_scheduler': 'cosine_warm_restarts',
        'T_0': 15,  # Restart every 15 epochs
        'T_mult': 2,  # Double restart period each time
        'eta_min': 0.0001,  # Minimum learning rate
        'step_size': 30,  # Keep for backward compatibility
        'gamma': 0.5,
        
        # Data settings
        'num_workers': 4,
        'pin_memory': True,
        'use_weighted_sampler': True,
        
        # Augmentation - ENHANCED for difficult classes
        'aug_prob': 0.6,  # INCREASED from 0.3
        'mixup_alpha': 0.2,  # ENABLED MixUp
        'cutmix_alpha': 0.3,  # ENABLED CutMix
        
        # Paths
        **PATHS,
        
        # Class weights - REBALANCED based on confusion patterns
        'class_weights': {
            'angry': 1.5,     # INCREASED - confused with neutral
            'disgust': 2.0,   # REDUCED - performing too well
            'fear': 2.5,      # INCREASED - worst performer
            'happy': 0.8,     # REDUCED - already performing well
            'sad': 2.2,       # INCREASED - second worst performer
            'surprise': 1.2,  # REDUCED - performing well
            'neutral': 1.8    # INCREASED - poor precision
        },
        
        # Training stability
        'use_amp': True,  # ENABLED AMP for faster training
        'early_stopping_patience': 25,  # INCREASED patience
        'save_best_only': True,
    }

def get_parser() -> argparse.ArgumentParser:
    """Get command line argument parser."""
    parser = argparse.ArgumentParser(description='Train SOTA EmotionNet on FER2013')
    
    # Essential arguments only
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    return parser

def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    if hasattr(args, 'batch_size'):
        config['batch_size'] = args.batch_size
    if hasattr(args, 'epochs'):
        config['epochs'] = args.epochs
    if hasattr(args, 'lr'):
        config['lr'] = args.lr
    if hasattr(args, 'checkpoint_dir'):
        config['checkpoint_dir'] = args.checkpoint_dir
        
    return config 