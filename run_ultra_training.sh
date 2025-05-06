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
check_install "torch_optimizer"

# Clean up previous log file
if [ -f nohup.out ]; then
    echo "Removing existing nohup.out file..."
    rm nohup.out
fi

# Create the extracted directory if it doesn't exist
mkdir -p ./extracted

# Make the training script executable
chmod +x train_ultra_accuracy.sh

# Run ultra-high accuracy training with nohup to keep it running in background
echo "Starting ULTRA-high accuracy training (targeting 90%+)..."
nohup bash train_ultra_accuracy.sh &

# Print monitoring instructions
echo "Training started in background. Monitor progress with:"
echo "  tail -f nohup.out"
echo ""
echo "Expected training time: 48-72 hours depending on your hardware."
echo "Target accuracy: 90%+ on validation set using ConvNeXt architecture."
echo ""
echo "This training script uses:"
echo "1. ConvNeXt Small backbone (superior to EfficientNet for facial recognition)"
echo "2. Very early SWA (starting at epoch 50)"
echo "3. Knowledge distillation (starting at epoch 40)"
echo "4. Extremely aggressive data augmentation"
echo "5. Aggressive regularization (dropout, weight decay)"
echo ""
echo "Estimated validation accuracy progression:"
echo "- 75% after ~30 epochs"
echo "- 80% after ~60 epochs"
echo "- 85% after ~120 epochs"
echo "- 90% after ~180-250 epochs" 