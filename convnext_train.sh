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
export LEARNING_RATE="0.0003"          # Slightly reduced learning rate to combat overfitting
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
export SWA_START_EPOCH="50"            # Start SWA earlier to capture more models before overfitting
export SWA_FREQ="3"                    # More frequent model averaging

# Knowledge Distillation settings
export SELF_DISTILLATION_ENABLED="1"
export SELF_DISTILLATION_START="40"    # Start self-distillation earlier to prevent overfitting
export SELF_DISTILLATION_TEMP="4.0"    # Higher temperature for smoother probabilities
export SELF_DISTILLATION_ALPHA="0.6"   # Stronger distillation weight

#==============================================================================
# REGULARIZATION
#==============================================================================
export WEIGHT_DECAY="0.0008"           # Increased weight decay to combat overfitting
export HEAD_DROPOUT="0.5"              # Increased dropout for classification head
export FEATURE_DROPOUT="0.3"           # Increased feature dropout

#==============================================================================
# LOSS FUNCTION
#==============================================================================
export USE_FOCAL_LOSS="1"
export FOCAL_GAMMA="2.0"
export LABEL_SMOOTHING="0.2"           # Increased label smoothing for regularization
export KL_WEIGHT="0.2"                 # Increased KL divergence weight

#==============================================================================
# DATA AUGMENTATION
#==============================================================================
export MIXUP_PROB="0.65"               # More aggressive augmentation to combat overfitting
export CUTMIX_PROB="0.55"              # Increased CutMix probability
export MIXUP_ALPHA="1.0"               # Stronger mixup interpolation
export CUTMIX_ALPHA="1.2"              # Stronger cutmix effect

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