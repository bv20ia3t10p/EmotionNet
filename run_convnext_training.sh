#!/bin/bash

# Check if required libraries are installed
pip_list=$(pip list)

# Function to check and install a package
check_install() {
    package_name=$1
    if ! echo "$pip_list" | grep -q "$package_name"; then
        echo "Installing $package_name..."
        pip install $package_name
    fi
}

# Check for required packages
check_install "timm"
check_install "albumentations"
check_install "opencv-python"
check_install "scikit-learn"
check_install "torch_optimizer" # For advanced optimizers

# Clean up previous log file
if [ -f nohup.out ]; then
    echo "Removing existing nohup.out file..."
    rm nohup.out
fi

# Create the extracted directory if it doesn't exist
mkdir -p ./extracted

# Make the training script executable
chmod +x train_convnext_accuracy.sh

# Set permissions for the scripts
chmod +x train_convnext_accuracy.sh

# Run ConvNeXt high accuracy training with nohup to keep it running in background
echo "Starting CONVNEXT ultra-high accuracy training (targeting 90%+)..."
nohup bash train_convnext_accuracy.sh &

# Print monitoring instructions
echo "Training started in background. Monitor progress with:"
echo "  tail -f nohup.out"
echo ""
echo "Expected training time: 36-60 hours depending on your hardware."
echo "Target accuracy: 90%+ on validation set using ConvNeXt architecture."
echo ""
echo "This training script uses ConvNeXt which should provide better results for emotion recognition tasks."
echo "ConvNeXt has shown excellent performance on facial analysis tasks in recent research." 