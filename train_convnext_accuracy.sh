#!/bin/bash

# Set environment variables for ultra-high accuracy training with ConvNeXt (targeting 90%+)
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/model_convnext_accuracy.pth"
export MODEL_TYPE="ConvNeXtEmoteNet"  # Using ConvNeXt-based architecture
export BACKBONE="convnext_base"       # ConvNeXt base backbone
export BATCH_SIZE="12"                # Smaller batch size for better generalization
export ACCUMULATION_STEPS="8"         # Increased accumulation steps for effective batch size
export LEARNING_RATE="0.00003"        # Lower learning rate for precision fine-tuning
export NUM_EPOCHS="250"               # Extended training for full convergence
export WARMUP_EPOCHS="5"              # Longer warmup for stable initialization
export FREEZE_BACKBONE_EPOCHS="3"     # Longer freeze period for better adaptation
export USE_AMP="1"                    # Use mixed precision for memory efficiency
export PRETRAINED="1"                 # Use pretrained weights
export IMAGE_SIZE="224"               # Standard image size compatible with ConvNeXt
export ENABLE_LR_FINDER="0"           # Disable LR finder
export RESUME_EPOCH="0"               # Start from the beginning

# Progressive augmentation strategy with more phases
export PROGRESSIVE_AUGMENTATION="1"   # Enable phase-based augmentation strategy
export PHASE_1_EPOCHS="20"            # Phase 1: Simple augmentations only
export PHASE_2_EPOCHS="50"            # Phase 2: Introduce intermediate augmentations
export PHASE_3_EPOCHS="100"           # Phase 3: Full augmentation strength
export PHASE_4_EPOCHS="180"           # Phase 4: Advanced augmentation + knowledge distillation

# Enhanced augmentation settings - more aggressive approach
export MIXUP_PROB="0.5"               # Increased mixup probability
export CUTMIX_PROB="0.4"              # Increased cutmix probability
export MIXUP_ALPHA="0.6"              # Stronger mixup alpha
export CUTMIX_ALPHA="1.0"             # Stronger cutmix alpha
export LABEL_SMOOTHING="0.1"          # Increased label smoothing factor
export USE_MIXUP="1"                  # Enable mixup
export USE_CUTMIX="1"                 # Enable cutmix
export USE_ADVANCED_AUGMENTATION="1"  # Enable advanced augmentations

# Dropout settings optimized for ConvNeXt
export HEAD_DROPOUT="0.2"             # Optimal dropout rate for ConvNeXt
export FEATURE_DROPOUT="0.1"          # Optimal feature dropout for ConvNeXt

# Advanced augmentations - higher intensity
export USE_RANDOM_ERASING="1"         # Enable random erasing
export ERASING_PROB="0.3"             # Higher probability for random erasing
export USE_COLOR_JITTER="1"           # Enable color jitter

# Optimization settings - precision-focused
export WEIGHT_DECAY="0.00002"         # Finely tuned weight decay for ConvNeXt
export FOCAL_ALPHA="1.0"              # Focal loss alpha parameter
export FOCAL_GAMMA="2.0"              # Increased gamma for harder examples
export GRAD_CLIP_VALUE="1.0"          # Higher gradient clipping for ConvNeXt
export TTA_NUM_AUGMENTS="12"          # Significantly increased TTA augmentations
export TTA_ENABLED="1"                # Enable test-time augmentation
export ENSEMBLE_SIZE="7"              # Larger ensemble for final evaluation
export USE_FOCAL_LOSS="1"             # Use focal loss for handling class imbalance
export USE_LABEL_SMOOTHING="1"        # Enable label smoothing

# Advanced optimization - cosine with restarts and warmup
export SCHEDULER_TYPE="cosine"        # Use simple cosine annealing (better for ConvNeXt)
export PATIENCE="15"                  # Increased LR scheduler patience
export FACTOR="0.75"                  # Gentler LR reduction factor
export LR_SCHEDULER_STEP_SIZE="20"    # Increased step size for scheduler
export LR_SCHEDULER_GAMMA="0.9"       # Gentler gamma for scheduler
export EARLY_STOPPING_PATIENCE="40"   # More patient early stopping

# Advanced training techniques
export LOOKAHEAD_ENABLED="1"          # Enable Lookahead optimizer
export LOOKAHEAD_K="5"                # Lookahead synchronization period
export LOOKAHEAD_ALPHA="0.5"          # Lookahead alpha parameter
export LARGE_BATCH_BN="1"             # Enable special batch norm handling

# Stochastic Weight Averaging for final convergence
export SWA_ENABLED="1"                # Enable Stochastic Weight Averaging
export SWA_START_EPOCH="150"          # Start SWA after this epoch
export SWA_FREQ="5"                   # SWA model update frequency (epochs)
export SWA_LR="0.00001"               # SWA learning rate

# Feature consistency and self-distillation
export FEATURE_CONSISTENCY_LAMBDA="0.15"           # Weight for feature consistency loss
export FEATURE_CONSISTENCY_ENABLED="1"             # Enable feature consistency regularization
export SELF_DISTILLATION_ENABLED="1"               # Enable self-distillation from earlier checkpoints
export SELF_DISTILLATION_TEMP="2.0"                # Temperature for knowledge distillation
export SELF_DISTILLATION_ALPHA="0.3"               # Weight for distillation loss
export SELF_DISTILLATION_START="100"               # Start epoch for self-distillation
export SELF_DISTILLATION_MODEL_PATH="${MODEL_PATH}_epoch_50.pth"  # Path to teacher model (epoch 50 checkpoint)

# Class balancing (important for small dataset)
export BALANCE_DATASET="1"                         # Enable dataset balancing
export TARGET_SAMPLES_PER_CLASS="7500"             # Target samples per class

# Run the training script with ConvNeXt configuration
echo "🔹 Starting ULTRA-high-accuracy training with ConvNeXt..."
echo "🔹 Target: 90%+ accuracy with ConvNeXt-Base backbone"
echo "🔹 Image size: ${IMAGE_SIZE}px, Batch size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "🔹 ADVANCED: SWA, Self-distillation, and aggressive augmentation"

python3 main.py --mode train 