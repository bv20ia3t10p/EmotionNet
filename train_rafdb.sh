#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training for RAF-DB (using maximum accuracy settings)..."

# --- Configuration ---
# Path to the RAF-DB dataset in this repo
DATA_DIR="./dataset/rafdb"  # Updated path to RAF-DB dataset root directory
MODEL_DIR="./models/rafdb_maximum" # New model directory with maximum accuracy settings

# Accuracy-focused hyperparameters (matched with FER2013)
BATCH_SIZE=32                      # Smaller batch size for better gradient updates
EPOCHS=128                         # Keep longer training to see improvements
LEARNING_RATE=0.0001               # Conservative learning rate
IMAGE_SIZE=224
BACKBONES=("efficientnet_b1")      # Slightly better than b0, still fast enough
LOSS_TYPE="focal"                  # Keep focal loss for class imbalance
FOCAL_GAMMA=2.0                    # Standard gamma (not too aggressive)
LABEL_SMOOTHING=0.1
MIXUP_ALPHA=0.2                    # Moderate mixup
CUTMIX_ALPHA=0.0                   # Disabled cutmix initially
DROP_PATH_RATE=0.1                 # Moderate dropout
SCHEDULER_TYPE="one_cycle"         # Better for consistent learning
NUM_WORKERS=4
PATIENCE=20                        # Higher patience for better convergence
WEIGHT_DECAY=0.0001                # Light regularization
OPTIMIZER="adam"                   # Standard Adam optimizer
GRAYSCALE_INPUT=false              # RGB input (3 channels)
USE_AMP=true                       # Mixed precision for speed
CLASS_BALANCED_LOSS=true           # Handle class imbalance
WARMUP_EPOCHS=5                    # Warmup period
GRADIENT_CLIP=1.0                  # Prevent exploding gradients
CLASS_WEIGHTS=true                 # Use class weights based on frequency
PRETRAINED=true                    # Use pretrained model weights

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Use python3 directly
PYTHON_CMD="python3"

echo "Using Python interpreter: $PYTHON_CMD"

# --- Run Training ---
echo "Running train.py with accuracy-focused hyperparameters..."
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
    --cutmix_alpha $CUTMIX_ALPHA \
    --drop_path_rate $DROP_PATH_RATE \
    --scheduler_type "$SCHEDULER_TYPE" \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --grayscale_input $GRAYSCALE_INPUT \
    --use_amp $USE_AMP \
    --class_balanced_loss $CLASS_BALANCED_LOSS \
    --warmup_epochs $WARMUP_EPOCHS \
    --gradient_clip $GRADIENT_CLIP \
    --class_weights $CLASS_WEIGHTS \
    --pretrained $PRETRAINED
    # No --val_split_ratio needed for RAF-DB standard setup

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "RAF-DB training (with maximum accuracy settings) completed successfully!"
echo "Checkpoints saved in $MODEL_DIR" 