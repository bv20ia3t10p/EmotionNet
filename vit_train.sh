#!/bin/bash
#==============================================================================
# EmotionViT Training Script
# 
# This script sets up and runs training for the EmotionViT model,
# a Vision Transformer-based model optimized for emotion recognition.
#==============================================================================

echo "Setting up EmotionViT training environment..."

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
export MODEL_PATH="${ROOT}/emotion_vit_model.pth"
export MODEL_TYPE="EmotionViT"
export BACKBONE="vit_base_patch16_224"
export IMAGE_SIZE="224"

#==============================================================================
# TRAINING PARAMETERS
#==============================================================================
export BATCH_SIZE="192"
export LEARNING_RATE="0.0005"
export NUM_EPOCHS="200"

#==============================================================================
# OPTIMIZATION SETTINGS
#==============================================================================
export ACCUMULATION_STEPS="1"         # No accumulation needed with large batch size
export USE_AMP="1"                    # Automatic Mixed Precision for faster training
export LARGE_BATCH_BN="1"             # Optimized batch normalization for large batches
export WARMUP_EPOCHS="5"              # Gradual warmup of learning rate
export FREEZE_BACKBONE_EPOCHS="3"     # Initially freeze backbone for stable training

#==============================================================================
# SCHEDULER & ADVANCED TRAINING
#==============================================================================
export SCHEDULER_TYPE="cosine"        # Cosine annealing scheduler

# Stochastic Weight Averaging for better generalization
export SWA_ENABLED="1"
export SWA_START_EPOCH="100"
export SWA_FREQ="5"

# Knowledge Distillation settings
export SELF_DISTILLATION_ENABLED="1"
export SELF_DISTILLATION_START="60"
export SELF_DISTILLATION_TEMP="2.5"
export SELF_DISTILLATION_ALPHA="0.4"

#==============================================================================
# REGULARIZATION
#==============================================================================
export WEIGHT_DECAY="0.00025"
export HEAD_DROPOUT="0.35"
export FEATURE_DROPOUT="0.15"

#==============================================================================
# LOSS FUNCTION
#==============================================================================
export USE_FOCAL_LOSS="1"
export FOCAL_GAMMA="2.0"
export LABEL_SMOOTHING="0.1"
export KL_WEIGHT="0.1"

#==============================================================================
# DATA AUGMENTATION
#==============================================================================
export MIXUP_PROB="0.4"
export CUTMIX_PROB="0.3"
export MIXUP_ALPHA="0.5"
export CUTMIX_ALPHA="1.0"

#==============================================================================
# EARLY STOPPING
#==============================================================================
export EARLY_STOPPING_PATIENCE="30"

#==============================================================================
# SUMMARY
#==============================================================================
echo "=== EmotionViT Training Configuration ==="
echo "Model: EmotionViT with ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Advanced Features: SWA, Self-Distillation, Mixed Precision"
echo "========================================"

#==============================================================================
# START TRAINING
#==============================================================================
echo "Starting EmotionViT training..."
python3 main.py --mode train 