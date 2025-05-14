#!/bin/bash

# Exit on any error
set -e

echo "Starting improved EmotionNet training for RAF-DB using SOTA ResEmote model..."

# --- Configuration ---
DATA_DIR="./dataset/raf-db"      # Path to RAF-DB data directory
MODEL_DIR="./models/rafdb_sota"  # Model directory

# Training configuration
BATCH_SIZE=32                    # Batch size for RAF-DB
EPOCHS=60                        # Number of epochs
LEARNING_RATE=0.0001             # Learning rate
IMAGE_SIZE=256                   # Image size
BACKBONE="resnet34"              # Backbone architecture
LOSS_TYPE="sota_emotion"         # Use custom SOTA loss with auxiliary components
LABEL_SMOOTHING=0.1              # Label smoothing factor
MIXUP_ALPHA=0.2                  # Mixup alpha
CUTMIX_ALPHA=0.2                 # CutMix alpha
DROP_PATH_RATE=0.1               # Drop path rate
SCHEDULER_TYPE="cosine_annealing" # Scheduler type
NUM_WORKERS=4                    # Number of workers
PATIENCE=15                      # Early stopping patience
WEIGHT_DECAY=0.0001              # Weight decay
OPTIMIZER="adamw"                # Optimizer
WARMUP_EPOCHS=5                  # Warmup epochs
GRADIENT_CLIP=1.0                # Gradient clipping
ARCHITECTURE="sota_resemote_medium" # Use our state-of-the-art model
EMBEDDING_SIZE=512               # Embedding size (512 for ResNet34)
USE_AMP="--use_amp"              # Use Automatic Mixed Precision

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

# --- Run Training ---
echo "Running training with SOTA ResEmote model (architecture: $ARCHITECTURE)..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Print debug info
echo "DEBUG: Using BATCH_SIZE=$BATCH_SIZE, BACKBONE=$BACKBONE and EPOCHS=$EPOCHS"
echo "DEBUG: Using architecture=$ARCHITECTURE with embedding_size=$EMBEDDING_SIZE"

$PYTHON_CMD emotion_net/train.py \
    --dataset_name "rafdb" \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones $BACKBONE \
    --loss_type "$LOSS_TYPE" \
    --label_smoothing $LABEL_SMOOTHING \
    --mixup_alpha $MIXUP_ALPHA \
    --cutmix_alpha $CUTMIX_ALPHA \
    --drop_path_rate $DROP_PATH_RATE \
    --scheduler_type "$SCHEDULER_TYPE" \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --warmup_epochs $WARMUP_EPOCHS \
    --gradient_clip $GRADIENT_CLIP \
    --architecture $ARCHITECTURE \
    --embedding_size $EMBEDDING_SIZE \
    --class_weights \
    --gem_pooling \
    --pretrained \
    $USE_AMP \
    --seed 42

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training phase failed!"
    exit 1
fi

echo "Training completed successfully!"
echo "Model saved in $MODEL_DIR"

# --- Run evaluation on the trained model ---
echo "Running evaluation on the trained model..."
$PYTHON_CMD emotion_net/evaluate.py \
    --dataset_name "rafdb" \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_DIR/best_model.pth" \
    --architecture $ARCHITECTURE \
    --image_size $IMAGE_SIZE \
    --batch_size 32

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Evaluation phase failed!"
    exit 1
fi

echo "Evaluation completed successfully!" 