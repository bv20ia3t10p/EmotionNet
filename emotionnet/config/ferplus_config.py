"""
FERPlus Configuration

Configuration for the FERPlus dataset with 8 emotion classes:
neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
"""

from emotionnet.config.base import Config


def create_ferplus_config(**overrides):
    """
    Create configuration for FERPlus dataset.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Config object
    """
    config = Config()
    
    # Model settings
    config.model.img_size = 48
    config.model.num_classes = 8  # FERPlus has 8 classes
    config.model.backbone = 'attention_emotion_net'
    config.model.dropout_rate = 0.4
    
    # Training settings
    config.training.lr = 0.001
    config.training.max_lr = 0.01
    config.training.epochs = 100
    config.training.batch_size = 64
    config.training.optimizer = 'adam'
    config.training.scheduler = 'onecycle'
    config.training.weight_decay = 1e-5
    config.training.gradient_accumulation_steps = 1
    
    # Loss settings
    config.loss = Config()
    config.loss.label_smoothing = 0.1
    config.loss.use_focal_loss = True
    config.loss.gamma = 2.0
    
    # Augmentation settings
    config.augmentation = Config()
    config.augmentation.use_mixup = True
    config.augmentation.mixup_alpha = 0.2
    config.augmentation.use_cutmix = True
    config.augmentation.cutmix_alpha = 1.0
    config.augmentation.use_randaugment = True
    config.augmentation.use_test_time_augmentation = True
    
    # FERPlus specific settings
    config.ferplus = Config()
    config.ferplus.mode = 'majority'  # 'majority', 'probability', or 'multi_target'
    
    # Data settings
    config.data.img_dir = 'FERPlus-master'
    config.data.train_csv = None  # Not used for FERPlus
    config.data.val_csv = None    # Not used for FERPlus
    config.data.test_csv = None   # Not used for FERPlus
    config.data.num_workers = 4
    config.data.use_weighted_sampler = False
    
    # Apply overrides
    for key, value in overrides.items():
        keys = key.split('.')
        obj = config
        for k in keys[:-1]:
            if not hasattr(obj, k):
                setattr(obj, k, Config())
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)
    
    return config 