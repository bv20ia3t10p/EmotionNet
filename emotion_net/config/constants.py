"""Constants and configuration settings for the emotion recognition model."""

# Emotion class mappings
EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Image processing constants
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32

# Training constants
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_CLASS_WEIGHTS_BOOST = 1.5

# Model constants
DEFAULT_BACKBONES = ["efficientnet_b0", "resnet18"]

# Image normalization constants
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225] 