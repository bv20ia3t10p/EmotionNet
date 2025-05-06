#!/bin/bash

# Set environment variables for ultra-high accuracy training (targeting 90%+)
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/model_ultra_accuracy.pth"
export MODEL_TYPE="ConvNeXtEmoteNet"  # Switch to ConvNeXt architecture
export BACKBONE="convnext_small"      # Use convnext_small backbone
export BATCH_SIZE="192"               # Large batch size for better training
export ACCUMULATION_STEPS="1"         # No accumulation needed with large batch
export LEARNING_RATE="0.0001"         # Adjusted for large batch size
export NUM_EPOCHS="300"               # Extended training for full convergence
export WARMUP_EPOCHS="8"              # Extended warmup for better initialization
export FREEZE_BACKBONE_EPOCHS="2"     # Freeze backbone initially
export USE_AMP="1"                    # Use mixed precision
export PRETRAINED="1"                 # Use pretrained weights
export IMAGE_SIZE="224"               # Standard ConvNeXt size
export ENABLE_LR_FINDER="0"           # Disable LR finder
export RESUME_EPOCH="0"               # Start from beginning

# Large batch specific settings
export LARGE_BATCH_BN="1"             # Enable special batch norm handling for large batches
export LOOKAHEAD_ENABLED="1"          # Enable Lookahead optimizer for large batch training
export LOOKAHEAD_K="5"                # Lookahead synchronization period
export LOOKAHEAD_ALPHA="0.5"          # Lookahead alpha parameter

# Extremely aggressive progressive augmentation
export PROGRESSIVE_AUGMENTATION="1"   # Enable phase-based augmentation
export PHASE_1_EPOCHS="10"            # Phase 1: Simple augmentations (shortened)
export PHASE_2_EPOCHS="25"            # Phase 2: Moderate augmentations (shortened)
export PHASE_3_EPOCHS="50"            # Phase 3: Full augmentations (earlier)
export PHASE_4_EPOCHS="100"           # Phase 4: Advanced augmentations (earlier)

# Enhanced augmentation settings
export MIXUP_PROB="0.7"               # High mixup probability
export CUTMIX_PROB="0.5"              # High cutmix probability
export MIXUP_ALPHA="0.8"              # Strong mixup effect
export CUTMIX_ALPHA="1.0"             # Strong cutmix effect
export LABEL_SMOOTHING="0.15"         # Significant label smoothing
export USE_MIXUP="1"                  # Enable mixup
export USE_CUTMIX="1"                 # Enable cutmix
export USE_ADVANCED_AUGMENTATION="1"  # Enable advanced augmentations

# Adaptive dropout settings
export HEAD_DROPOUT="0.4"             # High dropout in classification head
export FEATURE_DROPOUT="0.25"         # Moderate dropout in feature extraction

# Advanced augmentations
export USE_RANDOM_ERASING="1"         # Enable random erasing
export ERASING_PROB="0.35"            # High probability for random erasing
export USE_COLOR_JITTER="1"           # Enable color jitter
export BRIGHTNESS="0.3"               # Strong brightness adjustment
export CONTRAST="0.3"                 # Strong contrast adjustment
export SATURATION="0.3"               # Strong saturation adjustment
export HUE="0.15"                     # Strong hue adjustment

# Optimization settings
export WEIGHT_DECAY="0.0005"          # Significantly increased weight decay
export FOCAL_ALPHA="1.0"              # Balanced focal loss
export FOCAL_GAMMA="2.5"              # Stronger focus on hard examples
export GRAD_CLIP_VALUE="0.5"          # Reduced gradient clipping for stability
export TTA_NUM_AUGMENTS="16"          # Extensive TTA augmentations
export TTA_ENABLED="1"                # Enable test-time augmentation
export ENSEMBLE_SIZE="7"              # Use ensemble for final evaluation
export USE_FOCAL_LOSS="1"             # Use focal loss for handling class imbalance
export USE_LABEL_SMOOTHING="1"        # Enable label smoothing

# Cosine scheduler with restarts for better convergence
export SCHEDULER_TYPE="cosine_restart" # Use cosine annealing with warm restarts
export COSINE_CYCLES="5"              # 5 cycles during training
export PATIENCE="8"                   # Reduced patience for faster LR reduction
export FACTOR="0.5"                   # Stronger LR reduction
export EARLY_STOPPING_PATIENCE="30"   # Patient early stopping

# Advanced training techniques
export SWA_ENABLED="1"                # Enable Stochastic Weight Averaging
export SWA_START_EPOCH="50"           # Start SWA much earlier
export SWA_FREQ="3"                   # Update SWA more frequently

# Enable knowledge distillation
export SELF_DISTILLATION_ENABLED="1"   # Enable self-distillation
export SELF_DISTILLATION_START="40"    # Start distillation very early
export SELF_DISTILLATION_TEMP="3.0"    # Higher temperature for softer probabilities
export SELF_DISTILLATION_ALPHA="0.5"   # Higher weight for distillation loss

# Feature consistency and attention mechanisms
export FEATURE_CONSISTENCY_LAMBDA="0.2" # Stronger feature consistency regularization
export FEATURE_CONSISTENCY_ENABLED="1"  # Enable feature consistency regularization

# Run the training script with ultra-high accuracy configuration
echo "🔹 Starting ULTRA-high-accuracy training with ConvNeXt..."
echo "🔹 Target: 90%+ accuracy with ConvNeXt backbone"
echo "🔹 Image size: ${IMAGE_SIZE}px, Batch size: ${BATCH_SIZE}"
echo "🔹 ADVANCED: Large batch training with Lookahead, Early SWA, and extreme augmentation"

python3 main.py --mode train 