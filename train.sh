#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training..."

# Default parameters
DATA_DIR="./extracted/emotion/train"
TEST_DIR="./extracted/emotion/test"
MODEL_DIR="./models"
BATCH_SIZE=80
EPOCHS=80
LEARNING_RATE=0.001
IMAGE_SIZE=224
BACKBONES=("efficientnet_b0" "efficientnet_b1")  # Array of backbones

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
    --backbones "${BACKBONES[@]}"; then
    echo "Error: Training failed!"
    exit 1
fi

echo "Training completed successfully!" 