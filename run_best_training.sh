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
check_install "torch>=1.7.0"

# Make the training scripts executable
chmod +x train_ultra_accuracy.sh
chmod +x train_convnext_accuracy.sh

# Clear screen
clear

echo "==================================================================="
echo "🚀 EmotionNet High-Accuracy Training Helper"
echo "==================================================================="
echo ""
echo "Choose a training approach to achieve 90%+ validation accuracy:"
echo ""
echo "1) 🔹 Vision Transformer (ViT) Based Training"
echo "   - Self-attention mechanism for global feature understanding"
echo "   - Good with limited data after proper augmentation"
echo "   - Optimized for 224x224 input size"
echo ""
echo "2) 🔹 ConvNeXt Based Training (Recommended)"
echo "   - Better performance on facial analysis tasks"
echo "   - More parameter-efficient than ViT"
echo "   - Stronger feature extraction capabilities"
echo "   - Better handling of imbalanced classes"
echo ""
echo "3) 🔹 Exit without starting training"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Starting ViT-based training (expected time: 24-48 hours)..."
        # Clean up previous log file
        if [ -f nohup.out ]; then
            echo "Removing existing nohup.out file..."
            rm nohup.out
        fi
        
        # Create the extracted directory if it doesn't exist
        mkdir -p ./extracted
        mkdir -p ./extracted/emotion
        
        # Run ViT training with nohup to keep it running in background
        nohup bash train_ultra_accuracy.sh &
        
        echo "Training started in background. Monitor progress with:"
        echo "  tail -f nohup.out"
        ;;
    2)
        echo "Starting ConvNeXt-based training (expected time: 24-48 hours)..."
        # Clean up previous log file
        if [ -f nohup.out ]; then
            echo "Removing existing nohup.out file..."
            rm nohup.out
        fi
        
        # Create the extracted directory if it doesn't exist
        mkdir -p ./extracted
        mkdir -p ./extracted/emotion
        
        # Run ConvNeXt training with nohup to keep it running in background
        nohup bash train_convnext_accuracy.sh &
        
        echo "Training started in background. Monitor progress with:"
        echo "  tail -f nohup.out"
        ;;
    3)
        echo "Exiting without starting training."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "==================================================================="
echo "📊 Expected Results:"
echo "==================================================================="
echo "- Target accuracy: 90%+ on validation set"
echo "- Progressive augmentation phases will handle class imbalance"
echo "- SWA will activate after epoch 150 for better generalization"
echo "- Self-distillation will begin at epoch 100"
echo ""
echo "💡 Note: You can stop training at any time with 'pkill -f python3'"
echo "===================================================================" 