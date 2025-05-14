#!/bin/bash

# Exit on any error
set -e

echo "Starting improved EmotionNet training for FER2013 using SOTA ResEmote model..."

# --- Configuration ---
# Updated paths to use the FER2013 dataset in the repository
DATA_DIR="./dataset/fer2013"       # Path to FER2013 data directory with CSV files
MODEL_DIR="./models/fer2013_sota"  # New model directory for SOTA model

# Training configuration
BATCH_SIZE=16                      # Smaller batch size for better generalization
EPOCHS=80                          # More epochs for initial training
LEARNING_RATE=0.00005              # Lower learning rate for fine-tuning on FER2013
IMAGE_SIZE=256                     # Larger image size for more details 
BACKBONE="resnet34"                # Use ResNet34 backbone for sota_resemote_medium
LABEL_SMOOTHING=0.2                # Increased label smoothing for better generalization
MIXUP_ALPHA=0.4                    # Stronger mixup augmentation
CUTMIX_ALPHA=0.3                   # Stronger cutmix augmentation
DROP_PATH_RATE=0.2                 # Increased regularization
SCHEDULER_TYPE="cosine_annealing"  # Better exploration of parameter space
NUM_WORKERS=4                      # Adjust based on CPU cores available
PATIENCE=20                        # Increased patience for better convergence
VAL_SPLIT_RATIO=0.15               # Increased validation set for better evaluation
WEIGHT_DECAY=0.01                  # Reduced weight decay for fine-tuning
OPTIMIZER="adamw"                  # Better optimizer with weight decay
WARMUP_EPOCHS=5                    # Shorter warmup for fine-tuning
GRADIENT_CLIP=1.0                  # Standard gradient clipping
ARCHITECTURE="sota_resemote_medium" # Using the state-of-the-art ResEmoteNet model
EMBEDDING_SIZE=512                 # Match embedding size to ResNet34 feature dim

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
    --dataset_name "fer2013" \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones $BACKBONE \
    --loss_type "sota_emotion" \
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
    --attention_type "cbam" \
    --class_weights \
    --gem_pooling \
    --feature_fusion \
    --pretrained \
    --random_erase 0.2 \
    --multi_crop_inference \
    --use_ema \
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
    --dataset_name "fer2013" \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_DIR/best_model.pth" \
    --architecture $ARCHITECTURE \
    --image_size $IMAGE_SIZE \
    --backbone $BACKBONE \
    --embedding_size $EMBEDDING_SIZE \
    --batch_size 32

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Evaluation phase failed!"
    exit 1
fi

echo "Evaluation completed successfully!" 