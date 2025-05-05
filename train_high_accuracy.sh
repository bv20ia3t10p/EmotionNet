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
export NUM_EPOCHS="180"                  # Extended training for full convergence
export WARMUP_EPOCHS="3"                 # Shorter warmup for faster adaptation
export FREEZE_BACKBONE_EPOCHS="2"        # Unfreeze backbone earlier
export USE_AMP="1"                       # Use mixed precision for memory efficiency
export PRETRAINED="1"                    # Use pretrained weights
export IMAGE_SIZE="224"                  # Standard image size to match pretrained weights exactly
export ENABLE_LR_FINDER="0"              # Disable LR finder - we know the optimal rate
export RESUME_EPOCH="0"                  # Start from the beginning

# Progressive augmentation strategy (phase-based training)
export PROGRESSIVE_AUGMENTATION="1"      # Enable phase-based augmentation strategy
export PHASE_1_EPOCHS="15"               # Phase 1: Simple augmentations only
export PHASE_2_EPOCHS="40"               # Phase 2: Introduce intermediate augmentations
export PHASE_3_EPOCHS="70"               # Phase 3: Full augmentation strength

# Enhanced augmentation settings - progressive approach
export MIXUP_PROB="0.3"                  # Increased mixup probability for later phases
export CUTMIX_PROB="0.2"                 # Increased cutmix probability for later phases
export MIXUP_ALPHA="0.4"                 # Stronger mixup alpha for later phases
export CUTMIX_ALPHA="0.8"                # Stronger cutmix alpha for later phases
export LABEL_SMOOTHING="0.05"            # Reduced label smoothing factor
export USE_MIXUP="1"                     # Enable mixup
export USE_CUTMIX="1"                    # Enable cutmix
export USE_ADVANCED_AUGMENTATION="1"     # Enable advanced augmentations

# Adaptive dropout settings
export HEAD_DROPOUT="0.2"                # Reduced dropout rate for classification head
export FEATURE_DROPOUT="0.1"             # Reduced dropout in feature extraction

# Advanced augmentations - adaptive strength
export USE_RANDOM_ERASING="1"            # Enable random erasing
export ERASING_PROB="0.15"               # Moderate probability for random erasing
export USE_COLOR_JITTER="1"              # Enable color jitter

# Optimization settings
export WEIGHT_DECAY="0.00005"            # Reduced weight decay
export FOCAL_ALPHA="1.0"                 # Focal loss alpha parameter
export FOCAL_GAMMA="1.5"                 # Reduced gamma for easier learning
export GRAD_CLIP_VALUE="1.0"             # Clip gradients to stabilize training
export TTA_NUM_AUGMENTS="8"              # Increased TTA augmentations during testing
export TTA_ENABLED="1"                   # Enable test-time augmentation
export ENSEMBLE_SIZE="5"                 # Use ensemble for final evaluation
export USE_FOCAL_LOSS="1"                # Use focal loss for handling class imbalance
export USE_LABEL_SMOOTHING="1"           # Enable label smoothing

# Improved scheduler settings
export SCHEDULER_TYPE="cosine"           # Use cosine annealing scheduler
export PATIENCE="12"                     # Increased LR scheduler patience
export FACTOR="0.7"                      # Gentler LR reduction factor
export LR_SCHEDULER_STEP_SIZE="15"       # Increased step size for scheduler
export LR_SCHEDULER_GAMMA="0.85"         # Gentler gamma for scheduler
export EARLY_STOPPING_PATIENCE="30"      # Patient early stopping

# Large batch specific settings
export LOOKAHEAD_ENABLED="1"             # Enable Lookahead optimizer
export LOOKAHEAD_K="5"                   # Lookahead synchronization period
export LOOKAHEAD_ALPHA="0.5"             # Lookahead alpha parameter
export LARGE_BATCH_BN="0"                # Disable special batch norm handling for small batches

# Feature consistency regularization
export FEATURE_CONSISTENCY_LAMBDA="0.1"  # Weight for feature consistency loss
export FEATURE_CONSISTENCY_ENABLED="1"   # Enable feature consistency regularization

# Run the training script with high accuracy configuration
echo "🔹 Starting high-accuracy training with ENHANCED parameters..."
echo "🔹 Target: 85%+ accuracy with EfficientNet-B2 backbone"
echo "🔹 Image size: ${IMAGE_SIZE}px, Batch size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "🔹 IMPROVED: Progressive augmentation strategy and cosine annealing scheduler"

python3 main.py --mode train