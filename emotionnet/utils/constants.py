"""
Constants for EmotionNet.

Contains all constants used throughout the project.
"""

# Core constants
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion mapping for FER2013 dataset
EMOTION_MAP = {
    0: "Angry",
    1: "Disgust", 
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Reverse mapping
EMOTION_TO_ID = {v: k for k, v in EMOTION_MAP.items()}

# Default paths
DEFAULT_PATHS = {
    'train_csv': 'dataset/fer2013/fer2013.csv',
    'val_csv': 'dataset/fer2013/fer2013.csv', 
    'test_csv': 'dataset/fer2013/fer2013.csv',
    'img_dir': 'dataset/fer2013',
    'checkpoint_dir': 'checkpoints'
}

# Model constants
DEFAULT_IMG_SIZE = 48
DEFAULT_DROPOUT_RATE = 0.5

# Training constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001 