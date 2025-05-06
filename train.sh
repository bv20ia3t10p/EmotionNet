#!/bin/bash

# Set environment variables for optimal training configuration
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/model.pth"
export MODEL_TYPE="AdvancedEmoteNet"
export BACKBONE="efficientnet_b2"  # Back to b2 for better regularization
export BATCH_SIZE="32"             # Smaller batch size for better generalization
export ACCUMULATION_STEPS="8"      # Increased accumulation for effective batch size
export LEARNING_RATE="0.0003"      # Lower learning rate for better stability
export NUM_EPOCHS="150"            # More epochs to reach higher accuracy
export WARMUP_EPOCHS="5"           # Warmup for better weight initialization
export FREEZE_BACKBONE_EPOCHS="3"  # Freeze backbone initially
export USE_AMP="1"                 # Use mixed precision for faster training
export PRETRAINED="1"              # Use pretrained weights
export IMAGE_SIZE="224"            # Standard image size
export ENABLE_LR_FINDER="0"        # Disable LR finder
export RESUME_EPOCH="0"            # Start from beginning

# Enhanced augmentation settings
export MIXUP_PROB="0.5"            # Moderate mixup probability
export CUTMIX_PROB="0.3"           # Moderate cutmix probability
export MIXUP_ALPHA="0.3"           # Moderate mixup effect
export CUTMIX_ALPHA="0.3"          # Moderate cutmix effect
export LABEL_SMOOTHING="0.1"       # Label smoothing for better generalization

# Progressive augmentation strategy
export PROGRESSIVE_AUGMENTATION="1"  # Enable phase-based augmentation
export PHASE_1_EPOCHS="20"          # Phase 1: Simple augmentations
export PHASE_2_EPOCHS="50"          # Phase 2: Intermediate augmentations
export PHASE_3_EPOCHS="100"         # Phase 3: Full augmentations

# Optimization settings
export WEIGHT_DECAY="0.0001"        # Increased weight decay
export FOCAL_ALPHA="1.0"            # Balanced focal loss
export FOCAL_GAMMA="2.0"            # Focus on hard examples
export GRAD_CLIP_VALUE="1.0"        # Gradient clipping
export HEAD_DROPOUT="0.3"           # Increased dropout
export FEATURE_DROPOUT="0.2"        # Increased dropout

# Advanced training techniques
export SWA_ENABLED="1"              # Enable Stochastic Weight Averaging
export SWA_START_EPOCH="100"        # Start SWA after 100 epochs
export SWA_FREQ="5"                 # Update SWA every 5 epochs
export SELF_DISTILLATION_ENABLED="1" # Enable self-distillation
export SELF_DISTILLATION_START="80"  # Start distillation after 80 epochs
export SELF_DISTILLATION_TEMP="2.0"  # Temperature for distillation
export SELF_DISTILLATION_ALPHA="0.3" # Weight for distillation loss

# Run the training script
python3 main.py --mode train 