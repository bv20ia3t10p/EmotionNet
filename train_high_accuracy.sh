#!/bin/bash

# Set environment variables for high accuracy training (targeting 85%+)
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/model_high_accuracy.pth"
export MODEL_TYPE="AdvancedEmoteNet"     # Better architecture for emotion recognition
export BACKBONE="efficientnet_b2"        # Our updated backbone
export BATCH_SIZE="24"                   # Slightly larger batch size for more stable gradients
export ACCUMULATION_STEPS="4"            # Reduced accumulation steps
export LEARNING_RATE="0.0001"            # Lower learning rate for more stable start
export NUM_EPOCHS="150"                  # More epochs for convergence
export WARMUP_EPOCHS="3"                 # Shorter warmup for faster adaptation
export FREEZE_BACKBONE_EPOCHS="2"        # Unfreeze backbone earlier
export USE_AMP="1"                       # Use mixed precision for memory efficiency
export PRETRAINED="1"                    # Use pretrained weights
export IMAGE_SIZE="224"                  # Standard image size to match pretrained weights exactly
export ENABLE_LR_FINDER="0"              # Disable LR finder - we know the optimal rate
export RESUME_EPOCH="0"                  # Start from the beginning

# Enhanced augmentation settings - reduced for better early learning
export MIXUP_PROB="0.2"                  # Reduced mixup probability
export CUTMIX_PROB="0.1"                 # Reduced cutmix probability
export MIXUP_ALPHA="0.3"                 # Gentler mixup alpha
export CUTMIX_ALPHA="0.8"                # Gentler cutmix alpha
export LABEL_SMOOTHING="0.05"            # Reduced label smoothing factor
export USE_MIXUP="1"                     # Enable mixup
export USE_CUTMIX="1"                    # Enable cutmix
export USE_ADVANCED_AUGMENTATION="1"     # Enable advanced augmentations

# New dropout settings - reduced for early training
export HEAD_DROPOUT="0.2"                # Reduced dropout rate for classification head
export FEATURE_DROPOUT="0.1"             # Reduced dropout in feature extraction

# Advanced augmentations - reduced strength
export USE_RANDOM_ERASING="1"            # Enable random erasing
export ERASING_PROB="0.1"                # Reduced probability for random erasing
export USE_COLOR_JITTER="1"              # Enable color jitter

# Optimization settings
export WEIGHT_DECAY="0.00005"            # Reduced weight decay
export FOCAL_ALPHA="1.0"                 # Focal loss alpha parameter
export FOCAL_GAMMA="1.5"                 # Reduced gamma for easier learning
export GRAD_CLIP_VALUE="1.0"             # Clip gradients to stabilize training
export TTA_NUM_AUGMENTS="5"              # Reduced TTA augmentations during training
export TTA_ENABLED="1"                   # Enable test-time augmentation
export ENSEMBLE_SIZE="5"                 # Use ensemble for final evaluation
export USE_FOCAL_LOSS="1"                # Use focal loss for handling class imbalance
export USE_LABEL_SMOOTHING="1"           # Enable label smoothing

# Scheduler settings
export PATIENCE="10"                     # Increased LR scheduler patience
export FACTOR="0.7"                      # Gentler LR reduction factor
export LR_SCHEDULER_STEP_SIZE="12"       # Increased step size for scheduler
export LR_SCHEDULER_GAMMA="0.8"          # Gentler gamma for scheduler
export EARLY_STOPPING_PATIENCE="25"      # Patient early stopping

# Large batch specific settings
export LOOKAHEAD_ENABLED="1"             # Enable Lookahead optimizer
export LARGE_BATCH_BN="0"                # Disable special batch norm handling for small batches

# Run the training script with high accuracy configuration
echo "🔹 Starting high-accuracy training with FIXED parameters..."
echo "🔹 Target: 85%+ accuracy with EfficientNet-B2 backbone"
echo "🔹 Image size: ${IMAGE_SIZE}px, Batch size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "🔹 IMPORTANT: Reduced augmentation strength and learning rate for better early training"

python3 main.py --mode train