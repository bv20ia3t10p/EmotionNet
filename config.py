import os

# Refactored configuration constants to avoid redundancy and improve clarity
ROOT = os.getenv("ROOT", "./extracted")
DATASET_PATH = os.getenv("DATASET_PATH", f"{ROOT}/emotion")
MODEL_PATH = os.getenv("MODEL_PATH", f"{ROOT}/model.pth")
LOG_CSV_PATH = os.getenv("LOG_CSV_PATH", f"{ROOT}/training_log.csv")
RESUME_EPOCH = int(os.getenv("RESUME_EPOCH", 0))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 24))  # Larger batch size for more stable gradients
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))  # Reduced learning rate for stability
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 150))  # Increased max epochs

# Derived paths
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Model architecture settings
MODEL_TYPE = os.getenv("MODEL_TYPE", "AdvancedEmoteNet")
BACKBONE = os.getenv("BACKBONE", "efficientnet_b2")  # Using EfficientNet-B2
PRETRAINED = int(os.getenv("PRETRAINED", 1)) == 1  # Use pretrained weights by default
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))  # Standard image size for pretrained models

# Training optimization settings
ACCUMULATION_STEPS = int(os.getenv("ACCUMULATION_STEPS", 4))  # Use gradient accumulation for effective batch size
USE_AMP = int(os.getenv("USE_AMP", 1)) == 1  # Use Automatic Mixed Precision for faster training
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", 3))  # Shorter warmup phase
FREEZE_BACKBONE_EPOCHS = int(os.getenv("FREEZE_BACKBONE_EPOCHS", 2))  # Freeze backbone initially
ENABLE_LR_FINDER = int(os.getenv("ENABLE_LR_FINDER", 0)) == 1
USE_LOOKAHEAD = int(os.getenv("LOOKAHEAD_ENABLED", 1)) == 1  # Lookahead optimizer for better convergence
LOOKAHEAD_ALPHA = float(os.getenv("LOOKAHEAD_ALPHA", 0.5))
LOOKAHEAD_K = int(os.getenv("LOOKAHEAD_K", 5))

# For very large batch training (> 256 effective batch size)
LARGE_BATCH_BN = int(os.getenv("LARGE_BATCH_BN", 0)) == 1  # Special batch norm handling

# Augmentation settings - more gradual approach
USE_MIXUP = int(os.getenv("USE_MIXUP", 1)) == 1
USE_CUTMIX = int(os.getenv("USE_CUTMIX", 1)) == 1
MIXUP_PROB = float(os.getenv("MIXUP_PROB", 0.2))  # Lower mixup probability for early stability
CUTMIX_PROB = float(os.getenv("CUTMIX_PROB", 0.1))  # Lower cutmix probability for early stability
MIXUP_ALPHA = float(os.getenv("MIXUP_ALPHA", 0.3))  # Reduced strength
CUTMIX_ALPHA = float(os.getenv("CUTMIX_ALPHA", 0.8))  # Reduced strength
USE_ADVANCED_AUGMENTATION = int(os.getenv("USE_ADVANCED_AUGMENTATION", 1)) == 1

# Loss function parameters
USE_FOCAL_LOSS = int(os.getenv("USE_FOCAL_LOSS", 1)) == 1
FOCAL_ALPHA = float(os.getenv("FOCAL_ALPHA", 1.0))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", 1.5))  # Gentler gamma value for better early training
USE_LABEL_SMOOTHING = int(os.getenv("USE_LABEL_SMOOTHING", 1)) == 1
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", 0.05))  # Reduced label smoothing for early stability
KL_WEIGHT = float(os.getenv("KL_WEIGHT", 0.05))  # Reduced KL divergence weight for better early training

# Regularization parameters
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 0.0001))  # Increased weight decay
HEAD_DROPOUT = float(os.environ.get('HEAD_DROPOUT', 0.3))  # Increased dropout for classification head
FEATURE_DROPOUT = float(os.environ.get('FEATURE_DROPOUT', 0.2))  # Increased dropout for feature extractor

# Early stopping & gradient clipping
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 25))  # More patience
GRAD_CLIP_VALUE = float(os.getenv("GRAD_CLIP_VALUE", 1.0))  # Clip gradients to stabilize training

# Scheduler parameters
PATIENCE = int(os.environ.get('PATIENCE', 8))  # Reduced patience for faster LR reduction
FACTOR = float(os.environ.get('FACTOR', 0.5))  # More aggressive LR reduction
LR_SCHEDULER_STEP_SIZE = int(os.environ.get('LR_SCHEDULER_STEP_SIZE', 8))  # Shorter interval between LR adjustments
LR_SCHEDULER_GAMMA = float(os.environ.get('LR_SCHEDULER_GAMMA', 0.7))  # More aggressive gamma for LR scheduler

# Data augmentation parameters (increased intensity)
DEGREES = int(os.environ.get('DEGREES', 15))  # Increased rotation angle
TRANSLATE = (float(os.environ.get('TRANSLATE_X', 0.15)), float(os.environ.get('TRANSLATE_Y', 0.15)))  # Increased translation
SCALE = (float(os.environ.get('SCALE_MIN', 0.8)), float(os.environ.get('SCALE_MAX', 1.2)))  # Increased scaling
SHEAR = int(os.environ.get('SHEAR', 10))  # Increased shear angle
PERSPECTIVE_DISTORTION = float(os.environ.get('PERSPECTIVE_DISTORTION', 0.3))  # Increased perspective distortion

# Color augmentation parameters (increased intensity)
BRIGHTNESS = float(os.environ.get('BRIGHTNESS', 0.2))  # Increased brightness adjustment
CONTRAST = float(os.environ.get('CONTRAST', 0.2))  # Increased contrast adjustment
SATURATION = float(os.environ.get('SATURATION', 0.2))  # Increased saturation adjustment
HUE = float(os.environ.get('HUE', 0.1))  # Increased hue adjustment

# More advanced augmentation options
USE_RANDOM_ERASING = int(os.environ.get('USE_RANDOM_ERASING', 1)) == 1
ERASING_PROB = float(os.environ.get('ERASING_PROB', 0.2))  # Increased erasing probability
USE_COLOR_JITTER = int(os.environ.get('USE_COLOR_JITTER', 1)) == 1

# Test time augmentation (TTA) parameters
TTA_ENABLED = int(os.getenv("TTA_ENABLED", 1)) == 1
TTA_NUM_AUGMENTS = int(os.getenv("TTA_NUM_AUGMENTS", 5))  # Reduced number of TTA augmentations

# Ensemble parameters
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", 5))

# Number of classes in the dataset
NUM_CLASSES = 7  # Standard emotion classes

# Balance dataset parameters
BALANCE_DATASET = int(os.getenv("BALANCE_DATASET", 1)) == 1  # Whether to balance the dataset
TARGET_SAMPLES_PER_CLASS = int(os.getenv("TARGET_SAMPLES_PER_CLASS", 7500))  # Target number of samples per class