#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training for RAF-DB..."

# --- Configuration ---
# Path to the RAF-DB dataset in this repo
DATA_DIR="./dataset/rafdb"  # Updated path to RAF-DB dataset root directory
MODEL_DIR="./models/rafdb_ensemble" # Directory to save models for this dataset
# TEST_DIR is not typically used here, RAF-DB test split is used for validation.

# Training Hyperparameters - Adjusted for RAF-DB learning issues (REVISED)
BATCH_SIZE=32           # Smaller batch size for better gradient updates
EPOCHS=150              # Keep longer training schedule
LEARNING_RATE=0.0001    # Back to standard learning rate, previous was too low
IMAGE_SIZE=224          # Keep same image size
BACKBONES=("efficientnet_b0" "efficientnet_b1")
LOSS_TYPE="focal"       # Keep focal loss which is good for imbalanced data
FOCAL_GAMMA=2.0         # Back to standard gamma, 3.0 may have been too aggressive
LABEL_SMOOTHING=0.1     # Standard label smoothing (0.2 might cause underfitting)
MIXUP_ALPHA=0.2         # Less aggressive mixup
DROP_PATH_RATE=0.1      # Reduced dropout to prevent underfitting
SCHEDULER_TYPE="one_cycle" # Back to one_cycle which often works better initially
NUM_WORKERS=4
PATIENCE=20             # Keep increased patience
# VAL_SPLIT_RATIO is not used for RAF-DB

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Use python3 directly
PYTHON_CMD="python3"

echo "Using Python interpreter: $PYTHON_CMD"

# --- Run Training ---
echo "Running train.py..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

$PYTHON_CMD emotion_net/train.py \
    --dataset_name "rafdb" \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones "${BACKBONES[@]}" \
    --loss_type "$LOSS_TYPE" \
    --focal_gamma $FOCAL_GAMMA \
    --label_smoothing $LABEL_SMOOTHING \
    --mixup_alpha $MIXUP_ALPHA \
    --drop_path_rate $DROP_PATH_RATE \
    --scheduler_type "$SCHEDULER_TYPE" \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE 
    # No --test_dir or --val_split_ratio needed for RAF-DB standard setup

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "RAF-DB training completed successfully!"
echo "Checkpoints saved in $MODEL_DIR" 