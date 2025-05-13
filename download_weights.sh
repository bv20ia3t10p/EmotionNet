#!/bin/bash

# Exit on any error
set -e

echo "Downloading AffectNet pretrained weights for emotion recognition models..."

# Check which python command is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither python3 nor python found. Please install Python."
    exit 1
fi

# Check if required packages are installed
if ! $PYTHON_CMD -c "import requests" 2>/dev/null; then
    echo "Installing required packages..."
    pip install requests tqdm
fi

# Create models directory if it doesn't exist
mkdir -p models

# Define supported backbones
BACKBONES=("efficientnet_b0" "resnet50" "swin_v2_b" "vit_base_patch16_224")

# Parse command-line arguments
if [ "$#" -ge 1 ]; then
    # Use the specified backbone
    if [[ " ${BACKBONES[@]} " =~ " $1 " ]] || [ "$1" == "all" ]; then
        BACKBONE="$1"
    else
        echo "Invalid backbone specified. Supported options: all, ${BACKBONES[*]}"
        exit 1
    fi
else
    # Download all by default
    BACKBONE="all"
fi

# Run the download script
echo "Running download for backbone: $BACKBONE"
$PYTHON_CMD download_affectnet_weights.py --backbone "$BACKBONE" --output_dir models

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Download failed!"
    exit 1
fi

echo "AffectNet pretrained weights download completed successfully!"
echo "Weights saved in the 'models' directory." 