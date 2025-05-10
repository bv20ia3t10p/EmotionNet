import os

# ==============================================================================
# BASIC PATHS AND SETTINGS
# ==============================================================================
ROOT = os.getenv("ROOT", "./extracted")
DATASET_PATH = os.getenv("DATASET_PATH", f"{ROOT}/emotion")
MODEL_PATH = os.getenv("MODEL_PATH", f"{ROOT}/model.pth")
LOG_CSV_PATH = os.getenv("LOG_CSV_PATH", f"{ROOT}/training_log.csv")

# Derived paths
# Override TRAIN_PATH directly if provided in environment
TRAIN_PATH = os.getenv("TRAIN_PATH", os.path.join(DATASET_PATH, "train"))
TEST_PATH = os.getenv("TEST_PATH", os.path.join(DATASET_PATH, "test"))

# ==============================================================================
# TRAINING PARAMETERS
# ==============================================================================
RESUME_EPOCH = int(os.getenv("RESUME_EPOCH", 0))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 192))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0005))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 200))
NUM_CLASSES = 7  # Standard emotion classes

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
# Only support EmotionViT and ConvNeXtEmoteNet models
MODEL_TYPE = os.getenv("MODEL_TYPE", "ConvNeXtEmoteNet")

# Set default backbone based on model type
if MODEL_TYPE == "EmotionViT":
    DEFAULT_BACKBONE = "vit_base_patch16_224"
else:  # ConvNeXtEmoteNet
    DEFAULT_BACKBONE = "convnext_base"

BACKBONE = os.getenv("BACKBONE", DEFAULT_BACKBONE)
PRETRAINED = int(os.getenv("PRETRAINED", 1)) == 1
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))

# ==============================================================================
# TRAINING OPTIMIZATION
# ==============================================================================
# Gradient accumulation for effective larger batch sizes
ACCUMULATION_STEPS = int(os.getenv("ACCUMULATION_STEPS", 1))

# Mixed precision training
USE_AMP = int(os.getenv("USE_AMP", 1)) == 1

# Progressive training strategies
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", 5))
FREEZE_BACKBONE_EPOCHS = int(os.getenv("FREEZE_BACKBONE_EPOCHS", 3))

# Optimizer enhancements
USE_LOOKAHEAD = int(os.getenv("LOOKAHEAD_ENABLED", 1)) == 1
LOOKAHEAD_ALPHA = float(os.getenv("LOOKAHEAD_ALPHA", 0.5))
LOOKAHEAD_K = int(os.getenv("LOOKAHEAD_K", 5))

# Learning rate finder
ENABLE_LR_FINDER = int(os.getenv("ENABLE_LR_FINDER", 0)) == 1

# Batch normalization for large batches
LARGE_BATCH_BN = int(os.getenv("LARGE_BATCH_BN", 1)) == 1

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================
# Advanced augmentation strategies
USE_MIXUP = int(os.getenv("USE_MIXUP", 1)) == 1
USE_CUTMIX = int(os.getenv("USE_CUTMIX", 1)) == 1
MIXUP_PROB = float(os.getenv("MIXUP_PROB", 0.7))
CUTMIX_PROB = float(os.getenv("CUTMIX_PROB", 0.6))
MIXUP_ALPHA = float(os.getenv("MIXUP_ALPHA", 1.2))
CUTMIX_ALPHA = float(os.getenv("CUTMIX_ALPHA", 1.4))

# Progressive augmentation phases
PROGRESSIVE_AUGMENTATION = int(os.getenv("PROGRESSIVE_AUGMENTATION", 1)) == 1
PHASE_1_EPOCHS = int(os.getenv("PHASE_1_EPOCHS", 5))
PHASE_2_EPOCHS = int(os.getenv("PHASE_2_EPOCHS", 10))
PHASE_3_EPOCHS = int(os.getenv("PHASE_3_EPOCHS", 15))
PHASE_4_EPOCHS = int(os.getenv("PHASE_4_EPOCHS", 25))
PHASE_5_EPOCHS = int(os.getenv("PHASE_5_EPOCHS", 40))

# Color and geometrical transformations
USE_COLOR_JITTER = int(os.getenv("USE_COLOR_JITTER", 1)) == 1
BRIGHTNESS = float(os.getenv("BRIGHTNESS", 0.2))
CONTRAST = float(os.getenv("CONTRAST", 0.2))
SATURATION = float(os.getenv("SATURATION", 0.2))
HUE = float(os.getenv("HUE", 0.05))
DEGREES = int(os.getenv("ROTATION_DEGREES", 15))
TRANSLATE = (float(os.getenv("TRANSLATE_X", 0.1)),
             float(os.getenv("TRANSLATE_Y", 0.1)))
SCALE = (float(os.getenv("SCALE_MIN", 0.8)),
         float(os.getenv("SCALE_MAX", 1.2)))
USE_RANDOM_ERASING = int(os.getenv("USE_RANDOM_ERASING", 1)) == 1
ERASING_PROB = float(os.getenv("ERASING_PROB", 0.3))

# Dataset balancing
# Allow direct control via MODEL_BALANCE_DATASET if set, otherwise fallback to BALANCE_DATASET
MODEL_BALANCE_DATASET = os.getenv("MODEL_BALANCE_DATASET")
if MODEL_BALANCE_DATASET is not None:
    BALANCE_DATASET = int(MODEL_BALANCE_DATASET) == 1
else:
    BALANCE_DATASET = int(os.getenv("BALANCE_DATASET", 1)) == 1
TARGET_SAMPLES_PER_CLASS = int(os.getenv("TARGET_SAMPLES_PER_CLASS", 7500))

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
USE_FOCAL_LOSS = int(os.getenv("USE_FOCAL_LOSS", 1)) == 1
FOCAL_ALPHA = float(os.getenv("FOCAL_ALPHA", 1.0))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", 2.0))
USE_LABEL_SMOOTHING = int(os.getenv("USE_LABEL_SMOOTHING", 1)) == 1
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", 0.1))
KL_WEIGHT = float(os.getenv("KL_WEIGHT", 0.1))

# ==============================================================================
# REGULARIZATION
# ==============================================================================
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 0.001))
HEAD_DROPOUT = float(os.environ.get('HEAD_DROPOUT', 0.6))
FEATURE_DROPOUT = float(os.environ.get('FEATURE_DROPOUT', 0.4))

# Gradient clipping for stability
GRAD_CLIP_VALUE = float(os.getenv("GRAD_CLIP_VALUE", 1.0))

# ==============================================================================
# LEARNING RATE SCHEDULING
# ==============================================================================
SCHEDULER_TYPE = os.getenv("SCHEDULER_TYPE", "cosine_restart")
PATIENCE = int(os.environ.get('PATIENCE', 8))
FACTOR = float(os.environ.get('FACTOR', 0.7))

# ==============================================================================
# ADVANCED TRAINING TECHNIQUES
# ==============================================================================
# Stochastic Weight Averaging
SWA_ENABLED = int(os.getenv("SWA_ENABLED", 1)) == 1
SWA_START_EPOCH = int(os.getenv("SWA_START_EPOCH", 40))
SWA_FREQ = int(os.getenv("SWA_FREQ", 1))

# Knowledge distillation
SELF_DISTILLATION_ENABLED = int(os.getenv("SELF_DISTILLATION_ENABLED", 1)) == 1
SELF_DISTILLATION_START = int(os.getenv("SELF_DISTILLATION_START", 30))
SELF_DISTILLATION_TEMP = float(os.getenv("SELF_DISTILLATION_TEMP", 3.0))
SELF_DISTILLATION_ALPHA = float(os.getenv("SELF_DISTILLATION_ALPHA", 0.3))

# Early stopping
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 35))

# ==============================================================================
# INFERENCE SETTINGS
# ==============================================================================
# Test-time augmentation for robust predictions
TTA_ENABLED = int(os.getenv("TTA_ENABLED", 1)) == 1
TTA_NUM_AUGMENTS = int(os.getenv("TTA_NUM_AUGMENTS", 16))
# Ensemble model evaluation settings
# Number of best checkpoints to use for ensemble
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", 5))
