#!/bin/bash

# Clear CUDA cache to ensure clean start
python -c "import torch; torch.cuda.empty_cache()"

# Set environment variables for model configuration
export MODEL_TYPE="ConvNeXtEmoteNet"
export BACKBONE="convnext_xlarge"
export BATCH_SIZE=128
export LEARNING_RATE=0.0003
export NUM_EPOCHS=400
export IMAGE_SIZE=256
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/emotion_convnext_model.pth"

# Set environment variables for advanced training features
export SWA_ENABLED=1
export SWA_START_EPOCH=30
export SWA_FREQ=1

export SELF_DISTILLATION_ENABLED=1
export SELF_DISTILLATION_START=20
export SELF_DISTILLATION_TEMP=2.5
export SELF_DISTILLATION_ALPHA=0.4
export ATTENTION_BETA=0.1

export PROGRESSIVE_AUGMENTATION=1
export PHASE_1_EPOCHS=8
export PHASE_2_EPOCHS=15
export PHASE_3_EPOCHS=25
export PHASE_4_EPOCHS=40
export PHASE_5_EPOCHS=60

# Loss function parameters
export FOCAL_GAMMA=2.5
export FOCAL_ALPHA=1.0
export LABEL_SMOOTHING=0.15
export USE_CLASS_BALANCED_LOSS=1
export CB_BETA=0.9999
export CENTER_WEIGHT=0.005
export SCALE_POS_WEIGHT=1.2

# Optimizer settings
export WEIGHT_DECAY=0.0005
export HEAD_DROPOUT=0.5
export FEATURE_DROPOUT=0.3
export GRAD_CLIP_VALUE=1.5
export HEAD_LR_MULTIPLIER=10.0

# Advanced batch handling
export LOOKAHEAD_ENABLED=1
export LOOKAHEAD_K=6
export LOOKAHEAD_ALPHA=0.5
export LARGE_BATCH_BN=1

# Learning rate scheduler
export SCHEDULER_TYPE="cosine_restart"
export COSINE_CYCLES=6
export PATIENCE=10
export FACTOR=0.75

# Early stopping to prevent overfitting
export EARLY_STOPPING_PATIENCE=50

# Test time augmentation
export TTA_ENABLED=1
export TTA_NUM_AUGMENTS=24
export ENSEMBLE_SIZE=7

# Data augmentation settings
export USE_MIXUP=1
export USE_CUTMIX=1
export MIXUP_PROB=0.6
export CUTMIX_PROB=0.5
export MIXUP_ALPHA=1.0
export CUTMIX_ALPHA=1.2
export USE_COLOR_JITTER=1
export BRIGHTNESS=0.15
export CONTRAST=0.15
export SATURATION=0.15
export HUE=0.05
export ROTATION_DEGREES=12
export TRANSLATE_X=0.08
export TRANSLATE_Y=0.08
export SCALE_MIN=0.85
export SCALE_MAX=1.15
export USE_RANDOM_ERASING=1
export ERASING_PROB=0.2

# Dataset settings
export TRAIN_PATH="${DATASET_PATH}/train"
export TEST_PATH="${DATASET_PATH}/test"
export MODEL_BALANCE_DATASET=0  # Use original dataset for training

# Ensure extracted directory exists
mkdir -p ${ROOT}

# Print configuration
echo "=== ConvNeXtEmoteNet Enhanced Training Configuration ==="
echo "Model: ${MODEL_TYPE} with ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE} (Head: ${HEAD_LR_MULTIPLIER}x)"
echo "Epochs: ${NUM_EPOCHS}"
echo "Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Training Path: ${TRAIN_PATH}"
echo "Dataset Balancing: Disabled (using original unbalanced dataset)"
echo "Class Imbalance Handling: Class-Balanced Focal Loss (gamma=${FOCAL_GAMMA})"
echo "Advanced Features: SWA, Self-Distillation, Mixed Precision"
echo "==========================================\n"

# Run the training script
echo "Starting Enhanced ConvNeXtEmoteNet training..."
python main.py --mode train 