#!/bin/bash
#==============================================================================
# ConvNeXtEmoteNet Training Script
# 
# This script sets up and runs training for the ConvNeXtEmoteNet model,
# a ConvNeXt-based model with advanced attention mechanisms for emotion recognition.
#==============================================================================
echo "Setting up ConvNeXtEmoteNet training environment..."
#==============================================================================
# DIRECTORIES
#==============================================================================
# Create required directories
mkdir -p ./extracted/emotion
#==============================================================================
# BASIC CONFIGURATION
#==============================================================================
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/emotion_convnext_model.pth"
export MODEL_TYPE="ConvNeXtEmoteNet"
export BACKBONE="convnext_xlarge"
export IMAGE_SIZE="224"
# Default paths - using original dataset
export TRAIN_PATH="${DATASET_PATH}/train"
export TEST_PATH="${DATASET_PATH}/test"
export FORCE_ORIGINAL_DATASET="1"
export MODEL_BALANCE_DATASET="0"
#==============================================================================
# TRAINING PARAMETERS
#==============================================================================
export BATCH_SIZE="128"                # Increased from 64 to 128
export LEARNING_RATE="0.0005"          # Optimal learning rate based on LR finder plot
export NUM_EPOCHS="400"                # Maintain total training epochs
#==============================================================================
# OPTIMIZATION SETTINGS
#==============================================================================
export ACCUMULATION_STEPS="1"          # Reset to 1 since batch size is already large
export USE_AMP="1"                     # Automatic Mixed Precision for faster training
export LARGE_BATCH_BN="1"              # Optimized batch normalization for large batches
export WARMUP_EPOCHS="15"              # Extended warmup period from 10 to 15 for stable initial training
export FREEZE_BACKBONE_EPOCHS="3"      # Increased from 2 to 3 for more stable early training
export DISABLE_CHECKPOINTS="0"         # Set to "1" to disable checkpoint saving (speeds up training)
export FAST_EVALUATION="0"             # Set to "1" for faster evaluation (fewer augmentations)
export ULTRA_FAST_EVAL="0"             # Set to "1" for ultra-fast evaluation (subset of validation data)
export SKIP_TTA="0"                    # Set to "1" to skip test-time augmentation entirely (fastest validation)
# Enable learning rate cycling (reset every 50 epochs)
export SCHEDULER_TYPE="cosine_restart"
export COSINE_CYCLES="6"               # Increased from 4 to 6 for more learning rate resets
#==============================================================================
# SCHEDULER & ADVANCED TRAINING
#==============================================================================
# Stochastic Weight Averaging for better generalization
export SWA_ENABLED="1"
export SWA_START_EPOCH="30"            # Start SWA earlier (was 40)
export SWA_FREQ="1"                    # More frequent model averaging
export FAST_SWA_FINALIZE="0"           # Set to 1 for faster SWA finalization (less accurate but much quicker)
# Knowledge Distillation settings
export SELF_DISTILLATION_ENABLED="1"
export SELF_DISTILLATION_START="20"    # Start self-distillation earlier (was 30)
export SELF_DISTILLATION_TEMP="2.5"    # Adjusted temperature (was 3.0)
export SELF_DISTILLATION_ALPHA="0.4"   # Increased from 0.3 for stronger distillation
#==============================================================================
# CLASS BALANCING AND LOSS FUNCTION
#==============================================================================
# Focus more on difficult/rare classes
export USE_FOCAL_LOSS="1"              # Enabled focal loss (was 0)
export FOCAL_GAMMA="2.0"               # Increased gamma to focus more on hard examples
export LABEL_SMOOTHING="0.2"           # Reduced from 0.25 for better performance
export KL_WEIGHT="0.35"                # Increased from 0.3 for better knowledge distillation
# Disable class balancing since we're using focal loss
export BALANCE_DATASET="0"
export TARGET_SAMPLES_PER_CLASS="8000" # Not used with balancing disabled
#==============================================================================
# REGULARIZATION
#==============================================================================
export WEIGHT_DECAY="0.0005"           # Reduced from 0.001 to avoid too much regularization
export HEAD_DROPOUT="0.5"              # Reduced from 0.6 to avoid over-regularization
export FEATURE_DROPOUT="0.3"           # Reduced from 0.4 to avoid over-regularization
export GRAD_CLIP_VALUE="1.5"           # Increased from 1.0 for slightly more gradient flow
#==============================================================================
# DATA AUGMENTATION
#==============================================================================
export MIXUP_PROB="0.6"                # Reduced from 0.7 for less aggressive augmentation
export CUTMIX_PROB="0.5"               # Reduced from 0.6 for better stability
export MIXUP_ALPHA="1.0"               # Reduced from 1.2 for less aggressive mixing
export CUTMIX_ALPHA="1.2"              # Reduced from 1.4 for less aggressive cutting
# Enable progressive augmentation
export PROGRESSIVE_AUGMENTATION="1"
export PHASE_1_EPOCHS="8"              # Extended phase 1 (was 5)
export PHASE_2_EPOCHS="15"             # Extended phase 2 (was 10)
export PHASE_3_EPOCHS="25"             # Extended phase 3 (was 15)
export PHASE_4_EPOCHS="35"             # Extended phase 4 (was 25) 
export PHASE_5_EPOCHS="50"             # Extended phase 5 (was 40)
#=============================================================================
# EARLY STOPPING
#==============================================================================
export EARLY_STOPPING_PATIENCE="50"    # Increased from 35 to allow more exploration
#==============================================================================
# CLEANUP ANY EXISTING BALANCED DATASETS
#==============================================================================
echo "[INFO] FORCE_ORIGINAL_DATASET is set. Cleaning up any existing balanced datasets..."
# Find and remove any balanced directories
EMOTION_DIR="./extracted/emotion"
if [ -d "$EMOTION_DIR" ]; then
    for dir in "$EMOTION_DIR"/balanced_*; do
        if [ -d "$dir" ]; then
            echo "[INFO] Removing existing balanced directory: $dir"
            rm -rf "$dir"
            echo "[DONE] Removed: $dir"
        fi
    done
fi

# Make sure we use the original dataset
export TRAIN_PATH="./extracted/emotion/train"
export TEST_PATH="./extracted/emotion/test"

# Create environment variable file for python
cat > env_vars.py << EOF
import os
import sys

# Set fixed paths directly
os.environ['TRAIN_PATH'] = '$(realpath "$TRAIN_PATH")'
os.environ['TEST_PATH'] = '$(realpath "$TEST_PATH")'
os.environ['MODEL_BALANCE_DATASET'] = '0'

if __name__ == "__main__":
    print(f"🔹 Using original dataset path:")
    print(f"   TRAIN_PATH: {os.environ['TRAIN_PATH']}")
    print(f"   MODEL_BALANCE_DATASET: {os.environ['MODEL_BALANCE_DATASET']}")
EOF
#==============================================================================
# SUMMARY
#==============================================================================
echo "=== ConvNeXtEmoteNet Training Configuration ==="
echo "Model: ConvNeXtEmoteNet with ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Training Path: ${TRAIN_PATH}"
echo "Dataset Balancing: Disabled (using original unbalanced dataset)"
echo "Class Imbalance Handling: Focal Loss (gamma=${FOCAL_GAMMA})"
echo "Advanced Features: SWA, Self-Distillation, Mixed Precision"
echo "=========================================="
#==============================================================================
# START TRAINING
#==============================================================================
echo "Starting ConvNeXtEmoteNet training..."
echo "[INFO] Using original dataset path: ${TRAIN_PATH}"
echo "[INFO] FORCE_ORIGINAL_DATASET=${FORCE_ORIGINAL_DATASET}"

echo "[INFO] Final paths for training:"
echo "[INFO] TRAIN_PATH: ${TRAIN_PATH}"
echo "[INFO] TEST_PATH: ${TEST_PATH}" 
echo "[INFO] MODEL_BALANCE_DATASET: ${MODEL_BALANCE_DATASET}"

# Create required directories before starting training
mkdir -p "${ROOT}"
python3 main.py --mode train 

# Final evaluation with ensemble and test-time augmentation
export TTA_ENABLED=1
export TTA_NUM_AUGMENTS=16
export ENSEMBLE_SIZE=5
python3 test_model.py --tta --ensemble 