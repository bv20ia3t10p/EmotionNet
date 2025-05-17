#!/bin/bash

# Exit on any error
set -e

echo "Starting FER2013 high-accuracy training (80%+ target)..."

# --- Configuration for 80%+ Accuracy ---
DATA_DIR="./dataset/fer2013"          # Path to original FER2013 data directory with CSV files
MODEL_DIR="./models/fer2013_80plus"    # Model directory for high-accuracy model

# Hyper-optimized parameters for 80%+ accuracy
BATCH_SIZE=32                         # Optimal batch size for stable gradients
EPOCHS=300                            # More training time for better convergence
LEARNING_RATE=0.00005                 # Well-tuned learning rate
IMAGE_SIZE=224                        # Standard resolution
BACKBONE="swin_tiny_patch4_window7_224" # Transformer backbone works better for faces
LABEL_SMOOTHING=0.1                  # Prevent overconfidence
SCHEDULER_TYPE="cosine_annealing"     # Cosine annealing scheduler
NUM_WORKERS=4                        # Parallel data loading
PATIENCE=40                          # Longer patience for finding global minimum
VAL_SPLIT_RATIO=0.15                 # Balanced validation split
WEIGHT_DECAY=0.02                    # Optimal weight decay for transformer models
OPTIMIZER="adamw"                    # Best optimizer for transformer
WARMUP_EPOCHS=10                     # Warmup helps stabilize transformer training
GRADIENT_CLIP=1.0                    # Higher gradient clip for stability
ARCHITECTURE="enhanced_resemote"     # Our custom enhanced architecture
EMBEDDING_SIZE=768                   # Optimal embedding size for transformers
CLASS_WEIGHTS=1                      # Use class balancing
GEM_POOLING=1                        # Use GEM pooling
LOSS_TYPE="adaptive"                 # Use adaptive emotion loss for 80%+ accuracy
USE_EMA=1                            # Use exponential moving average
EMA_DECAY=0.998                      # EMA decay parameter
CENTER_LOSS_WEIGHT=0.005             # Center loss weight for better feature separation
TRIPLET_LOSS_WEIGHT=0.01             # Low triplet loss weight
AMP_SCALE=128                        # AMP scale for mixed precision
USE_TTA=1                            # Use test-time augmentation
DROP_PATH=0.1                        # Drop path rate for regularization
FOCAL_GAMMA=2.0                      # Focal loss gamma parameter
AUGMENTATION_STRENGTH="strong"       # Use strong augmentations for better generalization
SAVE_FREQUENCY=5                     # Save every 5 epochs

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check which python command is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither python3 nor python found. Please install Python."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_CMD"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist."
    exit 1
fi

# Check CSV files exist
if [ ! -f "$DATA_DIR/train.csv" ] && [ ! -f "$DATA_DIR/icml_face_data.csv" ]; then
    echo "Error: Neither train.csv nor icml_face_data.csv found in $DATA_DIR."
    exit 1
fi

# --- Run Training ---
echo "Running high-performance training with optimized hyperparameters for 80%+ accuracy..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Print debug info
echo "DEBUG: Using ARCHITECTURE=$ARCHITECTURE with BACKBONE=$BACKBONE"
echo "DEBUG: Using EMBEDDING_SIZE=$EMBEDDING_SIZE"
echo "DEBUG: Using LEARNING_RATE=$LEARNING_RATE with WEIGHT_DECAY=$WEIGHT_DECAY"
echo "DEBUG: Using LOSS_TYPE=$LOSS_TYPE with optimized parameters"
echo "DEBUG: Using AUGMENTATION_STRENGTH=$AUGMENTATION_STRENGTH"

# Redirect output to log file for debugging
LOG_FILE="$MODEL_DIR/training_log.txt"
echo "Logging output to $LOG_FILE"

# Execute with output redirected to log file and stderr
$PYTHON_CMD -m emotion_net.train \
    --dataset_name "fer2013" \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --backbones $BACKBONE \
    --loss_type "$LOSS_TYPE" \
    --label_smoothing $LABEL_SMOOTHING \
    --focal_gamma $FOCAL_GAMMA \
    --drop_path_rate $DROP_PATH \
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
    --pretrained \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --use_amp \
    --center_loss_weight $CENTER_LOSS_WEIGHT \
    --triplet_loss_weight $TRIPLET_LOSS_WEIGHT \
    --amp_scale $AMP_SCALE \
    --use_tta $USE_TTA \
    --stabilize_norm 1 \
    --channels_last \
    --augmentation_strength "$AUGMENTATION_STRENGTH" \
    --save_ckpt_freq $SAVE_FREQUENCY \
    2>&1 | tee "$LOG_FILE"

# Check exit status of training
TRAIN_STATUS=${PIPESTATUS[0]}
if [ $TRAIN_STATUS -ne 0 ]; then
    echo "Error: Training phase failed with exit code $TRAIN_STATUS. See $LOG_FILE for details."
    exit 1
fi

echo "Training completed successfully!"
echo "Model saved in $MODEL_DIR"
echo "Log saved in $LOG_FILE"

# --- Run evaluation with test-time augmentation ---
echo "Running evaluation with test-time augmentation on the trained model..."
EVAL_LOG_FILE="$MODEL_DIR/evaluation_log.txt"

$PYTHON_CMD -m emotion_net.evaluate \
    --dataset_name "fer2013" \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_DIR/best_model.pth" \
    --architecture $ARCHITECTURE \
    --image_size $IMAGE_SIZE \
    --backbone $BACKBONE \
    --embedding_size $EMBEDDING_SIZE \
    --use_tta \
    2>&1 | tee "$EVAL_LOG_FILE"

# Check exit status of evaluation
EVAL_STATUS=${PIPESTATUS[0]}
if [ $EVAL_STATUS -ne 0 ]; then
    echo "Error: Evaluation phase failed with exit code $EVAL_STATUS. See $EVAL_LOG_FILE for details."
    exit 1
fi

echo "Evaluation completed successfully!"
echo "Log saved in $EVAL_LOG_FILE"

# Ensemble last 5 checkpoints for even better performance
echo "Ensembling last 5 checkpoints for optimal performance..."
ENSEMBLE_LOG_FILE="$MODEL_DIR/ensemble_log.txt"

$PYTHON_CMD -m emotion_net.ensemble_eval \
    --dataset_name "fer2013" \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --checkpoint_pattern "ckpt_epoch_*.pth" \
    --top_k 5 \
    --architecture $ARCHITECTURE \
    --image_size $IMAGE_SIZE \
    --backbone $BACKBONE \
    --embedding_size $EMBEDDING_SIZE \
    2>&1 | tee "$ENSEMBLE_LOG_FILE"

# Print a timestamp to confirm completion
echo "Script completed at $(date)" 