"""
Clean Configuration Loader for EmotionNet
Simplified configuration loading with minimal dependencies.
"""

import os
import json
import datetime
import shutil
from typing import Dict, Any
from config import get_sota_config, get_parser, update_config_from_args


def load_config() -> Dict[str, Any]:
    """Load SOTA optimized configuration."""
    # Get command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Get SOTA configuration
    config = get_sota_config()
    
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
    config_file = os.path.join(config_dir, f'sota_config_{timestamp}.json')
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the SOTA configuration."""
    print(f"\n🚀 SOTA EmotionNet Configuration:")
    print(f"- Architecture: SOTA EmotionNet (Simplified ResEmoteNet)")
    print(f"- Input: Grayscale 48x48")
    print(f"- Batch Size: {config['batch_size']}")
    print(f"- Learning Rate: {config['lr']}")
    print(f"- Epochs: {config['epochs']}")
    print(f"- Focal Gamma: {config['focal_gamma']}")
    print(f"- Dropout Rate: {config['dropout_rate']}")
    print(f"- Attention: SE + CBAM")
    print(f"- Class Weights: Optimized for FER2013")
    print("="*60) 