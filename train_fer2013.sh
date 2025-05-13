#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training for FER2013 with focal loss (no bias correction)..."

# --- Configuration ---
# Updated paths to use the FER2013 dataset in the repository
DATA_DIR="./dataset/fer2013"       # Path to FER2013 data directory with CSV files
MODEL_DIR="./models/fer2013_focal_loss" # New model directory for focal loss approach

# Advanced model hyperparameters optimized for focal loss approach
BATCH_SIZE=32                      # Lower batch size for better learning with focal loss
EPOCHS=100                         # Standard number of epochs 
LEARNING_RATE=0.0002               # Standard learning rate
IMAGE_SIZE=224                     # Standard image size 
BACKBONE="efficientnet_b0.ra_in1k" # Using proven, available backbone
LOSS_TYPE="focal"                  # Use focal loss directly instead of cross entropy
FOCAL_GAMMA=3.0                    # Higher gamma focuses more on hard examples
LABEL_SMOOTHING=0.1                # Standard label smoothing
MIXUP_ALPHA=0.2                    # Added mixup for better generalization
CUTMIX_ALPHA=0.1                   # Light cutmix for augmentation
DROP_PATH_RATE=0.1                 # Added regularization
SCHEDULER_TYPE="cosine_annealing"  # Better exploration of parameter space
NUM_WORKERS=4
PATIENCE=15                        # Increased patience for focal loss training
VAL_SPLIT_RATIO=0.1                # Standard validation split
WEIGHT_DECAY=0.0005                # Standard weight decay
OPTIMIZER="adamw"                  # Better optimizer with weight decay
WARMUP_EPOCHS=5                    # Increased warmup for focal loss stability
GRADIENT_CLIP=1.0                  # Standard gradient clipping
ARCHITECTURE="expert"              # Using expert model
EMBEDDING_SIZE=512                 # Standard embedding size
ATTENTION_TYPE="cbam"              # Use CBAM attention mechanism

# Install additional dependencies if not already installed
if ! pip list | grep -q "timm"; then
    echo "Installing additional dependencies..."
    pip install timm>=0.6.0 transformers>=4.20.0 requests>=2.27.0
fi

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check which python command is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither python3 nor python found. Please install Python."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_CMD"

# --- Run Training with Focal Loss ---
echo "Running training with focal loss (no bias correction)..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Print debug info
echo "DEBUG: Using BATCH_SIZE=$BATCH_SIZE and EPOCHS=$EPOCHS"

$PYTHON_CMD emotion_net/train.py \
    --dataset_name "fer2013" \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones $BACKBONE \
    --loss_type "$LOSS_TYPE" \
    --focal_gamma $FOCAL_GAMMA \
    --label_smoothing $LABEL_SMOOTHING \
    --mixup_alpha $MIXUP_ALPHA \
    --cutmix_alpha $CUTMIX_ALPHA \
    --drop_path_rate $DROP_PATH_RATE \
    --scheduler_type "$SCHEDULER_TYPE" \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --val_split_ratio $VAL_SPLIT_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --warmup_epochs $WARMUP_EPOCHS \
    --gradient_clip $GRADIENT_CLIP \
    --architecture $ARCHITECTURE \
    --embedding_size $EMBEDDING_SIZE \
    --attention_type "$ATTENTION_TYPE" \
    --class_weights \
    --gem_pooling \
    --feature_fusion \
    --pretrained \
    --seed 42

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training with focal loss failed!"
    exit 1
fi

echo "Training completed successfully!"
echo "Model saved in $MODEL_DIR" 