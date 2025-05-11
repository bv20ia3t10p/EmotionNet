#!/bin/bash
# Run balanced training with optimal settings
# Usage: ./run_balanced_training.sh [data_dir] [model_dir]

# Set default paths if not provided
TRAIN_DIR=${1:-"./extracted/emotion/train"}
TEST_DIR=${2:-"./extracted/emotion/test"}
MODEL_DIR=${3:-"./models"}

echo "Starting class-balanced training with these settings:"
echo "Training data directory: $TRAIN_DIR"
echo "Test data directory: $TEST_DIR"
echo "Model directory: $MODEL_DIR"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Step 1: Run pre-training phase specifically for underrepresented classes
echo
echo "Step 1: Pre-training on underrepresented classes..."
python3 balanced_training.py --data_dir "$TRAIN_DIR" --model_dir "$MODEL_DIR" --pretrain --augmentation_level 3 --epochs 15 --learning_rate 0.0002 --backbone resnet18

# Step 2: Run main balanced training phase
echo
echo "Step 2: Running main balanced training..."
python3 balanced_training.py --data_dir "$TRAIN_DIR" --model_dir "$MODEL_DIR" --backbone resnet18 --epochs 50 --augmentation_level 2 --learning_rate 0.0001 --patience 10

echo
echo "Training complete!"
echo "Final balanced model saved to: $MODEL_DIR/final_balanced_model.pth" 