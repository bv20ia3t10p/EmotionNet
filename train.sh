#!/bin/bash

# Set environment variables for optimal training configuration
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/model.pth"
export MODEL_TYPE="AdvancedEmoteNet"
export BACKBONE="efficientnet_b4"  # Upgrade from b2 to b4 for better feature extraction
export BATCH_SIZE="150"            # Reduced batch size for stability
export ACCUMULATION_STEPS="6"      # Adjusted gradient accumulation for smaller batch size
export LEARNING_RATE="0.0006"      # Adjusted learning rate for smaller batch (sqrt scaling rule)
export NUM_EPOCHS="150"            # More epochs to reach higher accuracy
export WARMUP_EPOCHS="5"           # Warmup for better weight initialization with large batches
export FREEZE_BACKBONE_EPOCHS="3"  # Freeze backbone initially
export USE_AMP="1"                 # Use mixed precision for faster training and less memory
export PRETRAINED="1"              # Use pretrained weights
export IMAGE_SIZE="320"            # Slightly smaller images for memory efficiency
export ENABLE_LR_FINDER="0"        # Disable LR finder to start training directly
export RESUME_EPOCH="0"           # Resume training from the latest epoch

# Enhanced augmentation settings
export MIXUP_PROB="0.6"            # Increase mixup probability
export CUTMIX_PROB="0.4"           # Enable cutmix augmentation
export MIXUP_ALPHA="0.4"           # Stronger mixup effect
export CUTMIX_ALPHA="0.4"          # Stronger cutmix effect
export LABEL_SMOOTHING="0.1"       # Use label smoothing for better generalization

# Optimization settings for large batch training
export WEIGHT_DECAY="0.02"         # Increased weight decay for large batch training
export FOCAL_ALPHA="1.0"           # Balanced focal loss
export FOCAL_GAMMA="2.0"           # Focus more on hard examples
export TTA_ENABLED="1"             # Enable test-time augmentation
export ENSEMBLE_SIZE="5"           # Use ensemble of best models
export GRAD_CLIP_VALUE="1.0"       # Add gradient clipping to improve stability

# Large batch specific settings
export LOOKAHEAD_ENABLED="1"       # Enable Lookahead optimizer wrapper
export LARGE_BATCH_BN="1"          # Use special batch norm handling for large batches

# Run the training script
python3 main.py --mode train 