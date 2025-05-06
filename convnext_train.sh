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
export BACKBONE="convnext_base"
export IMAGE_SIZE="224"

#==============================================================================
# TRAINING PARAMETERS
#==============================================================================
export BATCH_SIZE="192"                # Large batch size for efficient training
export LEARNING_RATE="0.0004"          # Adjusted learning rate for large batch size
export NUM_EPOCHS="200"

#==============================================================================
# OPTIMIZATION SETTINGS
#==============================================================================
export ACCUMULATION_STEPS="1"          # No accumulation needed with large batch size
export USE_AMP="1"                     # Automatic Mixed Precision for faster training
export LARGE_BATCH_BN="1"              # Optimized batch normalization for large batches
export WARMUP_EPOCHS="5"               # Gradual warmup of learning rate
export FREEZE_BACKBONE_EPOCHS="3"      # Initially freeze backbone for stable training

#==============================================================================
# SCHEDULER & ADVANCED TRAINING
#==============================================================================
export SCHEDULER_TYPE="cosine"         # Cosine annealing scheduler

# Stochastic Weight Averaging for better generalization
export SWA_ENABLED="1"
export SWA_START_EPOCH="120"           # Start SWA later for ConvNeXt
export SWA_FREQ="5"

# Knowledge Distillation settings
export SELF_DISTILLATION_ENABLED="1"
export SELF_DISTILLATION_START="80"    # Start self-distillation later for ConvNeXt
export SELF_DISTILLATION_TEMP="3.0"    # Higher temperature for ConvNeXt
export SELF_DISTILLATION_ALPHA="0.5"   # Stronger distillation weight

#==============================================================================
# REGULARIZATION
#==============================================================================
export WEIGHT_DECAY="0.0003"           # Increased weight decay for ConvNeXt
export HEAD_DROPOUT="0.4"              # Increased dropout for classification head
export FEATURE_DROPOUT="0.2"           # Increased feature dropout

#==============================================================================
# LOSS FUNCTION
#==============================================================================
export USE_FOCAL_LOSS="1"
export FOCAL_GAMMA="2.0"
export LABEL_SMOOTHING="0.15"          # Increased label smoothing
export KL_WEIGHT="0.15"                # Increased KL divergence weight

#==============================================================================
# DATA AUGMENTATION
#==============================================================================
export MIXUP_PROB="0.5"                # More aggressive augmentation
export CUTMIX_PROB="0.4"
export MIXUP_ALPHA="0.8"
export CUTMIX_ALPHA="1.0"

#==============================================================================
# EARLY STOPPING
#==============================================================================
export EARLY_STOPPING_PATIENCE="35"    # More patience for ConvNeXt

#==============================================================================
# SUMMARY
#==============================================================================
echo "=== ConvNeXtEmoteNet Training Configuration ==="
echo "Model: ConvNeXtEmoteNet with ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Advanced Features: SWA, Self-Distillation, Mixed Precision"
echo "========================================"

#==============================================================================
# START TRAINING
#==============================================================================
echo "Starting ConvNeXtEmoteNet training..."
python3 main.py --mode train 