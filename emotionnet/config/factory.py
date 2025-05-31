"""
Configuration factory functions for EmotionNet.
"""

import os
from typing import Dict, Any, Optional
from .base import Config
from .loader import load_config as load_config_file
from .ferplus_config import create_ferplus_config as _create_ferplus_config


def create_config(
    config_type: str = "default",
    config_path: Optional[str] = None,
    **overrides
) -> Config:
    """
    Create a configuration with predefined settings.
    
    Args:
        config_type: Type of configuration to create
        config_path: Optional path to config file to load
        **overrides: Any configuration values to override
        
    Returns:
        Config instance
    """
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        config = load_config_file(config_path)
    elif config_type == "default":
        config = create_default_config()
    elif config_type == "quick":
        config = create_quick_config()
    elif config_type == "ferplus":
        config = _create_ferplus_config(**overrides)
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # Apply overrides
    _apply_overrides(config, overrides)
    
    return config


def create_default_config(**overrides) -> Config:
    """Create the default configuration."""
    config = Config()
    
    # Apply overrides
    _apply_overrides(config, overrides)
    
    return config


def create_quick_config(**overrides) -> Config:
    """Create a quick training configuration with faster settings."""
    config = Config()
    
    # Reduce training time for quick experimentation
    config.training.epochs = 100
    config.training.batch_size = 32
    config.augmentation.use_tta = False
    
    # Apply overrides
    _apply_overrides(config, overrides)
    
    return config


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """Create configuration from dictionary (legacy support)."""
    return Config.from_dict(config_dict)


def _apply_overrides(config: Config, overrides: Dict[str, Any]) -> None:
    """Apply override values to configuration."""
    for key, value in overrides.items():
        # Handle nested keys like 'training.lr' or 'model.dropout_rate'
        if '.' in key:
            section, attr = key.split('.', 1)
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, attr):
                    setattr(section_obj, attr, value)
        else:
            # Try to find the attribute in any section
            for section_name in ['model', 'training', 'loss', 'data', 'augmentation']:
                section_obj = getattr(config, section_name)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    break 