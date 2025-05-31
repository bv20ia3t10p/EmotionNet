"""
Base configuration classes and default settings for EmotionNet.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os

# Core constants
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    num_classes: int = NUM_CLASSES
    backbone: str = 'convnext'
    img_size: int = 64
    dropout_rate: float = 0.5
    use_enhanced_model: bool = True
    use_pretrained_backbone: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    epochs: int = 300
    lr: float = 0.001
    max_lr: float = 0.01
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    
    # Optimizer settings
    use_sam: bool = True
    sam_rho: float = 0.05
    
    # Scheduler settings
    lr_scheduler: str = 'one_cycle'
    pct_start: float = 0.1
    anneal_strategy: str = 'cos'
    div_factor: float = 10
    final_div_factor: float = 1000


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = 'sam'  # 'sam', 'adamw', 'sgd'
    momentum: float = 0.9
    nesterov: bool = False
    eps: float = 1e-8
    betas: tuple = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = 'one_cycle'  # 'one_cycle', 'plateau', 'cosine'
    factor: float = 0.1
    patience: int = 10
    min_lr: float = 1e-6
    warmup_epochs: int = 5


@dataclass 
class LossConfig:
    """Loss function configuration."""
    name: str = 'cross_entropy'  # 'cross_entropy', 'focal'
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    aux_weight: float = 0.4  # Weight for auxiliary loss


@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    train_csv: str = 'fer2013/fer2013.csv'
    val_csv: str = 'fer2013/fer2013.csv'
    test_csv: str = 'fer2013/fer2013.csv'
    img_dir: str = 'fer2013'
    
    # Balanced dataset option
    use_balanced_dataset: bool = False
    balanced_dir: str = 'fer2013_balanced'
    
    checkpoint_dir: str = 'checkpoints'
    
    num_workers: int = 4
    pin_memory: bool = True
    use_weighted_sampler: bool = False  # Disabled since balanced dataset doesn't need it


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Basic augmentations
    use_random_horizontal_flip: bool = True
    use_rotation: bool = True
    rotation_range: int = 15
    
    # Advanced augmentations
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 0.5
    use_mixup: bool = True
    use_cutmix: bool = True
    
    # RandAugment
    randaugment_n: int = 2
    randaugment_m: int = 9
    
    # Test Time Augmentation
    use_tta: bool = False
    tta_transforms: int = 5


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for backward compatibility."""
        result = {}
        
        # Flatten the nested dataclasses
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result.update(field_value.__dict__)
            else:
                result[field_name] = field_value
                
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary."""
        # This is a simplified version - in practice you'd want more robust parsing
        config = cls()
        
        # Update nested configs based on keys
        for key, value in config_dict.items():
            # Try to find which sub-config this belongs to
            for field_name in ['model', 'training', 'optimizer', 'scheduler', 'loss', 'data', 'augmentation']:
                field_obj = getattr(config, field_name)
                if hasattr(field_obj, key):
                    setattr(field_obj, key, value)
                    break
        
        return config


# Default configuration instance
DEFAULT_CONFIG = Config() 