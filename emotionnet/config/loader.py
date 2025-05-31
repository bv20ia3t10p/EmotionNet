"""
Configuration loader and utilities for EmotionNet.
"""

import os
import json
import datetime
import shutil
import argparse
from typing import Dict, Any, Optional

from .base import Config, DEFAULT_CONFIG


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or command line arguments.
    
    Args:
        config_path: Optional path to a config file
        
    Returns:
        Config instance
    """
    # Start with default configuration
    config = DEFAULT_CONFIG
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    
    # Override with command line arguments
    parser = create_parser()
    args = parser.parse_args()
    config = update_config_from_args(config, args)
    
    return config


def save_config(config: Config, checkpoint_dir: Optional[str] = None) -> str:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration to save
        checkpoint_dir: Directory to save config in
        
    Returns:
        Path to saved config file
    """
    if checkpoint_dir is None:
        checkpoint_dir = config.data.checkpoint_dir
        
    config_dir = os.path.join(checkpoint_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    # Clean epoch_stats folder at the start of each run
    epoch_stats_dir = os.path.join(checkpoint_dir, 'epoch_stats')
    if os.path.exists(epoch_stats_dir):
        shutil.rmtree(epoch_stats_dir)
    os.makedirs(epoch_stats_dir, exist_ok=True)
    
    # Create timestamped config file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = os.path.join(config_dir, f'emotionnet_config_{timestamp}.json')
    
    # Save configuration as dictionary
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    return config_file


def print_config_summary(config: Config) -> None:
    """Print a comprehensive summary of the configuration."""
    print(f"\n🚀 EmotionNet Configuration:")
    print(f"- Architecture: {config.model.backbone}")
    print(f"- Input: Grayscale {config.model.img_size}x{config.model.img_size}")
    print(f"- Classes: {config.model.num_classes}")
    print(f"- Dropout Rate: {config.model.dropout_rate}")
    
    # Training settings
    if hasattr(config.training, 'gradient_accumulation_steps'):
        effective_batch = config.training.batch_size * config.training.gradient_accumulation_steps
        print(f"- Batch Size: {config.training.batch_size} (Effective: {effective_batch})")
    else:
        print(f"- Batch Size: {config.training.batch_size}")
        
    if hasattr(config.training, 'max_lr'):
        print(f"- Learning Rate: {config.training.lr} -> {config.training.max_lr}")
    else:
        print(f"- Learning Rate: {config.training.lr}")
        
    print(f"- Epochs: {config.training.epochs}")
    
    if hasattr(config.training, 'weight_decay'):
        print(f"- Weight Decay: {config.training.weight_decay}")
    
    # Advanced features
    if hasattr(config, 'loss') and hasattr(config.loss, 'label_smoothing'):
        print(f"- Label Smoothing: {config.loss.label_smoothing}")
    
    # Augmentation
    if hasattr(config, 'augmentation'):
        aug_features = []
        if hasattr(config.augmentation, 'mixup_alpha') and getattr(config.augmentation, 'use_mixup', False):
            aug_features.append(f"MixUp(α={config.augmentation.mixup_alpha})")
        if hasattr(config.augmentation, 'cutmix_alpha') and getattr(config.augmentation, 'use_cutmix', False):
            aug_features.append(f"CutMix(α={config.augmentation.cutmix_alpha})")
        if hasattr(config.augmentation, 'randaugment_n') and getattr(config.augmentation, 'use_randaugment', False):
            aug_features.append(f"RandAugment({config.augmentation.randaugment_n},{config.augmentation.randaugment_m})")
        if hasattr(config.augmentation, 'tta_transforms') and getattr(config.augmentation, 'use_tta', False):
            aug_features.append(f"TTA({config.augmentation.tta_transforms})")
        
        if aug_features:
            print(f"- Augmentation: {', '.join(aug_features)}")
    
    print("="*70)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description='EmotionNet Training')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=48, help='Input image size')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum learning rate')
    
    # Data arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='fer2013', help='Data directory')
    
    # Mode arguments
    parser.add_argument('--quick', action='store_true', help='Quick start mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    return parser


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration with command line arguments."""
    # Model config updates
    if hasattr(args, 'img_size') and args.img_size is not None:
        config.model.img_size = args.img_size
    if hasattr(args, 'dropout_rate') and args.dropout_rate is not None:
        config.model.dropout_rate = args.dropout_rate
    
    # Training config updates  
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.training.epochs = args.epochs
    if hasattr(args, 'lr') and args.lr is not None:
        config.training.lr = args.lr
    if hasattr(args, 'max_lr') and args.max_lr is not None:
        config.training.max_lr = args.max_lr
    
    # Data config updates
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir is not None:
        config.data.checkpoint_dir = args.checkpoint_dir
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        config.data.img_dir = args.data_dir
        config.data.train_csv = os.path.join(args.data_dir, 'fer2013.csv')
        config.data.val_csv = os.path.join(args.data_dir, 'fer2013.csv') 
        config.data.test_csv = os.path.join(args.data_dir, 'fer2013.csv')
    
    return config 