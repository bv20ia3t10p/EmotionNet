#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training..."

# Default parameters
DATA_DIR="./extracted/emotion/train"
TEST_DIR="./extracted/emotion/test"
MODEL_DIR="./models"
BATCH_SIZE=128
EPOCHS=128
LEARNING_RATE=0.0001
IMAGE_SIZE=224
BACKBONES=("efficientnet_b0" "efficientnet_b1")  # Array of backbones
LOSS_TYPE="focal"  # Can be "cross_entropy" or "focal"
FOCAL_GAMMA=2.0    # Gamma for Focal Loss
LABEL_SMOOTHING=0.1 # Label smoothing for CrossEntropyLoss
MIXUP_ALPHA=0.2     # Alpha for Mixup (0.0 to disable)
DROP_PATH_RATE=0.2  # Increased Drop path rate
SCHEDULER_TYPE="one_cycle" # Scheduler type: one_cycle, cosine_annealing, none

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Run training with error handling
echo "Running training script..."
if ! python3 -m emotion_net.train \
    --data_dir "$DATA_DIR" \
    --test_dir "$TEST_DIR" \
    --model_dir "$MODEL_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones "${BACKBONES[@]}" \
    --loss_type "$LOSS_TYPE" \
    --focal_gamma $FOCAL_GAMMA \
    --label_smoothing $LABEL_SMOOTHING \
    --mixup_alpha $MIXUP_ALPHA \
    --drop_path_rate $DROP_PATH_RATE \
    --scheduler_type "$SCHEDULER_TYPE"; then
    echo "Error: Training failed!"
    exit 1
fi

echo "Training completed successfully!" 