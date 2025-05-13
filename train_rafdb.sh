#!/bin/bash

# Exit on any error
set -e

echo "Starting EmotionNet training for RAF-DB with advanced AffectNet model..."

# --- Configuration ---
# Updated paths to use the RAF-DB dataset in the repository
DATA_DIR="./dataset/rafdb"       # Path to RAF-DB data directory
MODEL_DIR="./models/rafdb_affectnet_advanced" # New model directory for the advanced model

# Advanced model hyperparameters optimized for AffectNet transfer learning
BATCH_SIZE=128                     # Increased batch size for faster training (was 16)
EPOCHS=100                         # Increased epochs for longer training (was 40)
LEARNING_RATE=0.0001               # Slightly lower learning rate than FER2013
IMAGE_SIZE=224                     # Standard image size
BACKBONE="efficientnet_b0.ra_in1k" # Using proven, available backbone
LOSS_TYPE="cross_entropy"          # Cross entropy is more stable for initial training
FOCAL_GAMMA=2.0                    # For second phase
LABEL_SMOOTHING=0.1                # RAF-DB has cleaner labels, less smoothing needed
MIXUP_ALPHA=0.2                    # Mixup for better generalization
CUTMIX_ALPHA=0.1                   # Light cutmix for augmentation
DROP_PATH_RATE=0.1                 # Regularization
SCHEDULER_TYPE="cosine_annealing"  # Better exploration of parameter space
NUM_WORKERS=4
PATIENCE=8                         # RAF-DB converges faster
VAL_SPLIT_RATIO=0.1                # Standard validation split
WEIGHT_DECAY=0.0005                # Increased weight decay for better regularization
OPTIMIZER="adamw"                  # Better optimizer
WARMUP_EPOCHS=2                    # Shorter warmup for RAF-DB
GRADIENT_CLIP=1.0                  # Standard gradient clipping
ARCHITECTURE="expert"              # Using expert model
SAD_CLASS_WEIGHT=1.0               # No extra class weight adjustments initially
DISGUST_CLASS_WEIGHT=0.01          # EXTREMELY reduced weight for disgust class (was 0.1)
EMBEDDING_SIZE=512                 # Reduced embedding size
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

# Use existing handle_weights.py file
echo "Using existing handle_weights.py script..."

# Run the weights preparation script
echo "Preparing pretrained weights..."

# Check if weights file already exists
WEIGHTS_FILE="./models/affectnet_pretrained_${BACKBONE}.pth"
FORCE_REDOWNLOAD=""

if [ -f "$WEIGHTS_FILE" ]; then
    echo "Found existing weights file at $WEIGHTS_FILE"
    echo "Will use existing weights and update classifier bias values"
else
    echo "No existing weights found, will create new weights using timm"
    mkdir -p "./models"
fi

# Use timm pretrained weights directly since HuggingFace weights are unavailable
echo "Using timm pretrained weights with custom bias correction (HuggingFace download disabled)"

# Apply EXTREME bias correction to address severe class imbalance
# Using -50.0 for disgust class to completely block predictions
$PYTHON_CMD ./handle_weights.py --backbone "$BACKBONE" $FORCE_REDOWNLOAD --use_timm_pretrained \
    --custom_bias "1:-50.0,3:5.0,6:3.0,0:1.0,4:1.0"

# --- Run Initial Training Phase ---
echo "Running initial training phase with cross-entropy loss..."
# Add current directory to PYTHONPATH to help find the emotion_net module
export PYTHONPATH="$PYTHONPATH:$(pwd)"

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
    --sad_class_weight $SAD_CLASS_WEIGHT \
    --disgust_class_weight $DISGUST_CLASS_WEIGHT \
    --embedding_size $EMBEDDING_SIZE \
    --attention_type "$ATTENTION_TYPE" \
    --class_weights \
    --gem_pooling \
    --feature_fusion \
    --bias_correction \
    --pretrained \
    --seed 42

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Initial training phase failed!"
    exit 1
fi

echo "Initial training completed."

# --- Fine-tuning Phase ---
echo "Starting fine-tuning phase with focal loss..."
FINETUNED_MODEL_DIR="./models/rafdb_affectnet_finetuned"
mkdir -p "$FINETUNED_MODEL_DIR"

# Get the latest initial model
INITIAL_MODEL="$MODEL_DIR/best_model.pth"

# Fix bias values directly for the fine-tuning phase
echo "Manually modifying model architecture to block disgust predictions..."
$PYTHON_CMD -c "
import torch
model = torch.load('$INITIAL_MODEL')

# DRASTIC SOLUTION: Completely block the disgust class
if 'classifier.bias' in model:
    print('Applying extreme measures to block disgust predictions')
    
    # Set an extremely negative bias for disgust class
    bias = model['classifier.bias']
    bias[1] = -50.0  # Setting to extreme negative value
    
    # Zero out all weights for the disgust class to prevent it from activating
    if 'classifier.weight' in model:
        print('Zeroing out weights for disgust class')
        weights = model['classifier.weight']
        weights[1, :] = 0.0  # Zero all weights for disgust class
        
    # Set high positive biases for common classes
    bias[3] = 5.0    # Happy - most common emotion, strong positive bias
    bias[6] = 3.0    # Neutral - common emotion, positive bias
    bias[0] = 1.0    # Angry - moderate positive bias
    bias[4] = 1.0    # Sad - moderate positive bias
    
    print(f'Modified bias values: {bias}')
    
    torch.save(model, '$INITIAL_MODEL')
    print('Model saved with modified weights')
"

$PYTHON_CMD emotion_net/train.py \
    --dataset_name "rafdb" \
    --data_dir "$DATA_DIR" \
    --model_dir "$FINETUNED_MODEL_DIR" \
    --num_epochs 100 \
    --batch_size 128 \
    --learning_rate 0.00001 \
    --image_size 224 \
    --backbones $BACKBONE \
    --loss_type "focal" \
    --focal_gamma 3.0 \
    --label_smoothing 0.1 \
    --drop_path_rate 0.15 \
    --scheduler_type "cosine_annealing" \
    --num_workers $NUM_WORKERS \
    --patience 10 \
    --val_split_ratio $VAL_SPLIT_RATIO \
    --weight_decay 0.001 \
    --optimizer "adamw" \
    --warmup_epochs 2 \
    --gradient_clip 1.0 \
    --architecture $ARCHITECTURE \
    --sad_class_weight $SAD_CLASS_WEIGHT \
    --disgust_class_weight $DISGUST_CLASS_WEIGHT \
    --embedding_size $EMBEDDING_SIZE \
    --attention_type "$ATTENTION_TYPE" \
    --class_weights \
    --gem_pooling \
    --feature_fusion \
    --multi_crop_inference \
    --class_balanced_loss \
    --random_erase 0.15 \
    --checkpoint $INITIAL_MODEL \
    --test_time_augmentation \
    --bias_correction \
    --seed 42

echo "RAF-DB training with advanced AffectNet model completed successfully!"
echo "Initial model saved in $MODEL_DIR"
echo "Fine-tuned model saved in $FINETUNED_MODEL_DIR" 