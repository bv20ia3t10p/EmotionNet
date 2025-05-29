"""
Enhanced SOTA Configuration for EmotionNet - 79%+ Target
Optimized configuration with all advanced features for beating SOTA accuracy.
"""

import argparse
from typing import Dict, Any

# Core constants
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Default paths
PATHS = {
    'train_csv': 'dataset/fer2013/fer2013.csv',
    'val_csv': 'dataset/fer2013/fer2013.csv',
    'test_csv': 'dataset/fer2013/fer2013.csv',
    'img_dir': 'dataset/fer2013',
    'checkpoint_dir': 'checkpoints'
}

def get_config() -> Dict[str, Any]:
    """Get the optimized SOTA configuration for 79%+ accuracy."""
    return {
        # Model settings
        'num_classes': NUM_CLASSES,
        'backbone': 'enhanced_emotionnet',
        'img_size': 48,
        'dropout_rate': 0.5,  # Increased from 0.35 for better regularization
        'use_enhanced_model': True,
        
        # Training settings
        'batch_size': 64,  # Increased for more stable gradients
        'gradient_accumulation_steps': 1,  # Removed accumulation for faster updates
        'epochs': 300,
        'lr': 0.001,  # Increased from 0.005 for faster initial learning
        'max_lr': 0.01,  # Higher peak learning rate
        'weight_decay': 0.0001,  # Reduced weight decay
        'grad_clip': 1.0,  # Increased gradient clipping threshold
        
        # Advanced optimizer and scheduler
        'use_sam': True,
        'sam_rho': 0.05,
        'lr_scheduler': 'one_cycle',
        'pct_start': 0.1,  # Faster warmup
        'anneal_strategy': 'cos',
        'div_factor': 10,  # Less aggressive initial LR reduction
        'final_div_factor': 1000,  # Less aggressive final LR
        
        # Loss and regularization
        'label_smoothing': 0.1,
        'focal_gamma': 2.0,
        
        # Data settings
        'num_workers': 4,
        'pin_memory': True,
        'use_weighted_sampler': True,
        
        # Advanced augmentation
        'mixup_alpha': 0.4,
        'cutmix_alpha': 0.5,
        'randaugment_n': 2,
        'randaugment_m': 9,
        
        # Test Time Augmentation
        'use_tta': False,  # Disabled for faster training and debugging
        'tta_transforms': 5,
        
        # Optimized class weights for FER2013
        'class_weights': {
            'angry': 3.0,      # Increased from 1.8 - very poor performance
            'disgust': 2.5,    # Increased from 1.5
            'fear': 2.0,       # Reduced from 3.0 - already decent recall
            'happy': 0.8,      # Slightly increased from 0.7
            'sad': 2.0,        # Reduced from 2.5 - decent performance
            'surprise': 1.2,   # Increased from 1.0
            'neutral': 4.0     # Doubled from 2.0 - worst performer
        },
        
        # Paths
        **PATHS,
    }

def get_parser() -> argparse.ArgumentParser:
    """Get command line argument parser."""
    parser = argparse.ArgumentParser(description='Enhanced SOTA EmotionNet Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--quick', action='store_true', help='Quick start mode')
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