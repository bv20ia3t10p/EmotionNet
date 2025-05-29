"""
SOTA Configuration Loader for EmotionNet
Simplified loader for the optimized enhanced configuration.
"""

import os
import json
import datetime
import shutil
from typing import Dict, Any
from config import get_config, get_parser, update_config_from_args


def load_config() -> Dict[str, Any]:
    """Load the optimized SOTA configuration."""
    # Get command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Get enhanced SOTA configuration
    config = get_config()
    
    # Update with command line arguments
    config = update_config_from_args(config, args)
    
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to JSON file."""
    config_dir = os.path.join(config['checkpoint_dir'], 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    # Clean epoch_stats folder at the start of each run
    epoch_stats_dir = os.path.join(config['checkpoint_dir'], 'epoch_stats')
    if os.path.exists(epoch_stats_dir):
        shutil.rmtree(epoch_stats_dir)
    os.makedirs(epoch_stats_dir, exist_ok=True)
    
    # Create a timestamp for the config file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = os.path.join(config_dir, f'enhanced_sota_config_{timestamp}.json')
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the enhanced SOTA configuration."""
    print(f"\n🚀 Enhanced SOTA EmotionNet Configuration:")
    print(f"- Architecture: Enhanced SOTA EmotionNet (79%+ Target)")
    print(f"- Input: Grayscale {config['img_size']}x{config['img_size']}")
    print(f"- Dropout Rate: {config['dropout_rate']}")
    
    # Training settings
    effective_batch = config['batch_size'] * config.get('gradient_accumulation_steps', 1)
    print(f"- Batch Size: {config['batch_size']} (Effective: {effective_batch})")
    print(f"- Learning Rate: {config['lr']} -> {config['max_lr']}")
    print(f"- Epochs: {config['epochs']}")
    print(f"- Weight Decay: {config['weight_decay']}")
    
    # Advanced features
    print(f"- Optimizer: SAM (ρ={config['sam_rho']})")
    print(f"- Scheduler: OneCycleLR")
    print(f"- Label Smoothing: {config['label_smoothing']}")
    
    # Augmentation
    aug_features = [
        f"MixUp(α={config['mixup_alpha']})",
        f"CutMix(α={config['cutmix_alpha']})",
        f"RandAugment({config['randaugment_n']},{config['randaugment_m']})",
        f"TTA({config['tta_transforms']})"
    ]
    print(f"- Augmentation: {', '.join(aug_features)}")
    
    print(f"- Attention: Multi-Scale (SE + CBAM + Coordinate + Self)")
    print(f"- Class Weights: Optimized for FER2013")
    print("="*70) 