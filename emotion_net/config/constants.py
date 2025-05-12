"""Constants and configuration values for EmotionNet."""

# Image preprocessing constants
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Emotion classes
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Training constants
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_EPOCHS = 100
DEFAULT_PATIENCE = 10

# Model constants
DEFAULT_BACKBONE = 'efficientnet_b0'  # Single backbone for non-ensemble models
DEFAULT_BACKBONES = ['efficientnet_b0', 'resnet18']  # Multiple backbones for ensemble
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_NUM_CLASSES = len(EMOTIONS)

# Data augmentation constants
AUGMENTATION_PROBABILITY = 0.5
ROTATION_RANGE = (-5, 5)
SCALE_RANGE = (0.98, 1.02)
TRANSLATE_RANGE = (-0.02, 0.02)
BRIGHTNESS_RANGE = (-0.1, 0.1)
CONTRAST_RANGE = (-0.1, 0.1)

# Paths
CHECKPOINT_DIR = 'checkpoints'
METRICS_DIR = 'checkpoints/metrics'
MODEL_SAVE_PATH = 'checkpoints/best_model.pth' 