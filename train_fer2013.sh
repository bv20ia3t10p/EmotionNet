#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training for FER2013 with expert architecture..."

# --- Configuration ---
# Updated paths to use the FER2013 dataset in the repository
DATA_DIR="./dataset/fer2013"       # Path to FER2013 data directory with CSV files
MODEL_DIR="./models/fer2013_expert" # New model directory for the expert architecture

# Expert architecture hyperparameters
BATCH_SIZE=16                      # Smaller batch size for larger model
EPOCHS=200                         # Extended training time for complex model
LEARNING_RATE=0.00005              # Very small learning rate for stability
IMAGE_SIZE=224
BACKBONE="resnet50"                # Using a supported vision backbone
LOSS_TYPE="focal"                  # Using focal loss which is supported
FOCAL_GAMMA=2.0                    # Standard gamma setting
LABEL_SMOOTHING=0.15               # Moderate label smoothing
MIXUP_ALPHA=0.0                    # Disable mixup (using better techniques)
CUTMIX_ALPHA=0.0                   # Disable cutmix (using better techniques)
DROP_PATH_RATE=0.2                 # Moderate dropout
SCHEDULER_TYPE="cosine_annealing"  # Better for long training runs
NUM_WORKERS=4
PATIENCE=30                        # Higher patience for complex model
VAL_SPLIT_RATIO=0.1
WEIGHT_DECAY=0.0002                # Moderate regularization
OPTIMIZER="adamw"                  # AdamW with proper weight decay
WARMUP_EPOCHS=15                   # Extended warmup period
GRADIENT_CLIP=1.0                  # Prevent exploding gradients
ARCHITECTURE="expert"              # Use expert model architecture
ATTENTION_TYPE="cbam"              # Use CBAM attention mechanism
SAD_CLASS_WEIGHT=2.0               # Higher weight for sad class
EMOTION_GROUPS="sad-neutral-angry,happy-surprise,fear-disgust" # Group similar emotions
EMBEDDING_SIZE=512                 # Embedding size for features
STOCHASTIC_DEPTH=0.2               # Add stochastic depth for regularization
FREEZE_BACKBONE_EPOCHS=5           # Freeze backbone for initial epochs

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
echo "Running train.py with expert model architecture..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

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
    --attention_type $ATTENTION_TYPE \
    --sad_class_weight $SAD_CLASS_WEIGHT \
    --embedding_size $EMBEDDING_SIZE \
    --emotion_groups "$EMOTION_GROUPS" \
    --class_weights \
    --pretrained \
    --gem_pooling \
    --decoupled_head \
    --feature_fusion \
    --stochastic_depth $STOCHASTIC_DEPTH \
    --multi_crop_inference \
    --use_ema \
    --consistency_loss \
    --freeze_backbone_epochs $FREEZE_BACKBONE_EPOCHS \
    --channels_last \
    --use_amp \
    --class_balanced_loss

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "FER2013 training with expert model completed successfully!"
echo "Checkpoints saved in $MODEL_DIR" 