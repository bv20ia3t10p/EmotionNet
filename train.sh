#!/bin/bash
# High-accuracy emotion recognition model training script for Unix/Linux

echo "Training high-accuracy emotion recognition model..."

# Default parameters (can be customized)
DATA_DIR="./extracted/emotion/train"
TEST_DIR="./extracted/emotion/test"
MODEL_DIR="./models"
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.0001
IMAGE_SIZE=224
BACKBONES="efficientnet_b0 resnet18"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Run the training script using the new module structure
python3 -m emotion_net.train \
    --data_dir "$DATA_DIR" \
    --test_dir "$TEST_DIR" \
    --model_dir "$MODEL_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones $BACKBONES

echo "Training completed! The model is saved in the $MODEL_DIR directory." 