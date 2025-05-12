#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training for FER2013..."

# --- Configuration ---
# Updated paths to use the FER2013 dataset in the repository
DATA_DIR="./dataset/fer2013"       # Path to FER2013 data directory with CSV files
MODEL_DIR="./models/fer2013_ensemble" # Directory to save models for this dataset

# Training Hyperparameters (adjust as needed)
BATCH_SIZE=128
EPOCHS=128
LEARNING_RATE=0.0001  # Max LR for OneCycleLR
IMAGE_SIZE=224
BACKBONES=("efficientnet_b0" "efficientnet_b1")
LOSS_TYPE="focal"      # focal or cross_entropy
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1   # Only used if LOSS_TYPE="cross_entropy"
MIXUP_ALPHA=0.2       # 0.0 to disable
DROP_PATH_RATE=0.2    # 0.0 to disable
SCHEDULER_TYPE="one_cycle" # one_cycle, cosine_annealing, none
NUM_WORKERS=4
PATIENCE=15           # Early stopping patience
VAL_SPLIT_RATIO=0.1   # Ratio of training data to use for validation

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Use python3 directly since that's what's available on the user's machine
PYTHON_CMD="python3"

echo "Using Python interpreter: $PYTHON_CMD"

# --- Run Training ---
echo "Running train.py..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

$PYTHON_CMD emotion_net/train.py \
    --dataset_name "fer2013" \
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
    --patience $PATIENCE \
    --val_split_ratio $VAL_SPLIT_RATIO

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "FER2013 training completed successfully!"
echo "Checkpoints saved in $MODEL_DIR" 