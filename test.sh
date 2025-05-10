#!/bin/bash
#==============================================================================
# Emotion Recognition Model Testing Script
# 
# This script runs inference using trained emotion recognition models.
# It automatically detects whether to use EmotionViT or ConvNeXtEmoteNet models
# based on the model file name.
#==============================================================================

echo "Setting up emotion recognition model testing environment..."

#==============================================================================
# BASIC CONFIGURATION
#==============================================================================
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export IMAGE_SIZE="224"                # Standard image size for both models
export BATCH_SIZE="192"                # Use same large batch size as training for efficient evaluation
export TTA_NUM_AUGMENTS="10"           # Test-time augmentation count for robust predictions

#==============================================================================
# MODEL SELECTION
#==============================================================================
# Get model path from argument or use default
MODEL_PATH=${1:-"${ROOT}/model.pth"}
export MODEL_PATH=$MODEL_PATH

# Infer model type from filename
if [[ $MODEL_PATH == *"convnext"* ]]; then
    export MODEL_TYPE="ConvNeXtEmoteNet"
    export BACKBONE="convnext_base"
    MODEL_NAME="ConvNeXtEmoteNet"
elif [[ $MODEL_PATH == *"vit"* ]]; then
    export MODEL_TYPE="EmotionViT"
    export BACKBONE="vit_base_patch16_224"
    MODEL_NAME="EmotionViT"
else
    # Default to ConvNeXtEmoteNet if model type can't be inferred
    echo "⚠️ Could not determine model type from filename, defaulting to ConvNeXtEmoteNet"
    export MODEL_TYPE="ConvNeXtEmoteNet"
    export BACKBONE="convnext_base" 
    MODEL_NAME="ConvNeXtEmoteNet"
fi

#==============================================================================
# TESTING SETTINGS
#==============================================================================
export TTA_ENABLED="1"                 # Enable test-time augmentation for better results

#==============================================================================
# SUMMARY
#==============================================================================
echo "=== Emotion Recognition Model Test Configuration ==="
echo "Model: ${MODEL_NAME} with ${BACKBONE}"
echo "Model Path: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}/test"
echo "Test-Time Augmentation: Enabled (${TTA_NUM_AUGMENTS} augmentations)"
echo "=================================================="

#==============================================================================
# START TESTING
#==============================================================================
echo "Starting evaluation of ${MODEL_NAME}..."

# Set the train and test paths explicitly
export TRAIN_PATH="${DATASET_PATH}/train"
export TEST_PATH="${DATASET_PATH}/test"
export MODEL_BALANCE_DATASET="0"

python3 main.py --mode test 