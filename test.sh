#!/bin/bash

# Set environment variables for testing
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export TTA_NUM_AUGMENTS="8"     # Increased TTA augmentations for better accuracy
export USE_RANDOM_ERASING="0"   # Disable erasing for testing
export IMAGE_SIZE="260"         # Match the training image size

# Get model path from argument or use default
MODEL_PATH=${1:-"${ROOT}/model.pth"}
export MODEL_PATH=$MODEL_PATH

# Infer model type from filename
if [[ $MODEL_PATH == *"high_accuracy"* ]]; then
    export MODEL_TYPE="AdvancedEmoteNet"
    export BACKBONE="efficientnet_b2"
    export IMAGE_SIZE="260"
elif [[ $MODEL_PATH == *"vit"* ]]; then
    export MODEL_TYPE="EmotionViT"
    export BACKBONE="vit_base_patch16_224"
    export IMAGE_SIZE="224"
else
    # Default model configuration
    export MODEL_TYPE="AdvancedEmoteNet"
    export BACKBONE="efficientnet_b2"
    export IMAGE_SIZE="260"
fi

echo "Testing model: $MODEL_PATH"
echo "Model type: $MODEL_TYPE with $BACKBONE backbone"
echo "Image size: ${IMAGE_SIZE}px, TTA augmentations: $TTA_NUM_AUGMENTS"

# Run the test script
python3 main.py --mode test --model_path $MODEL_PATH 