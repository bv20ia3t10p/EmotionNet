#!/bin/bash

# Complete pipeline for achieving 80%+ accuracy on FER2013
# This script runs all the required steps in sequence

# Exit on error
set -e

# Determine Python command
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither python3 nor python found. Please install Python."
    exit 1
fi

# Directory definitions
FER_DATA_DIR="./dataset/fer2013"
MODEL_DIR="./models/fer2013_80plus"

echo "=== STEP 1: Installing required Python packages ==="
# Install required packages
pip install numpy
pip install torch torchvision
pip install opencv-python albumentations timm
pip install pandas scikit-learn matplotlib
pip install tqdm pillow

echo "=== STEP 2: Running the high-accuracy training script ==="
# Run the high-accuracy training script
bash train_fer2013_80plus.sh

echo "=== STEP 3: Evaluating model performance with comprehensive metrics ==="
# Run a detailed evaluation with advanced metrics
$PYTHON_CMD -m emotion_net.evaluate \
    --dataset_name "fer2013" \
    --data_dir "$FER_DATA_DIR" \
    --model_path "$MODEL_DIR/best_model.pth" \
    --architecture "enhanced_resemote" \
    --image_size 224 \
    --backbone "swin_tiny_patch4_window7_224" \
    --embedding_size 768 \
    --use_tta \
    --detailed_metrics \
    --confusion_matrix \
    --save_misclassified

echo "=== STEP 4: Creating model ensemble for optimal performance ==="
# Create an ensemble of the best checkpoints
$PYTHON_CMD -m emotion_net.ensemble_eval \
    --dataset_name "fer2013" \
    --data_dir "$FER_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --checkpoint_pattern "ckpt_epoch_*.pth" \
    --top_k 5 \
    --architecture "enhanced_resemote" \
    --image_size 224 \
    --backbone "swin_tiny_patch4_window7_224" \
    --embedding_size 768 \
    --save_ensemble

echo "=== STEP 5: Running inference with model ensemble ==="
# Generate predictions with the ensemble model
$PYTHON_CMD -m emotion_net.inference \
    --dataset_name "fer2013" \
    --data_dir "$FER_DATA_DIR" \
    --model_path "$MODEL_DIR/ensemble_model.pth" \
    --architecture "enhanced_resemote" \
    --image_size 224 \
    --backbone "swin_tiny_patch4_window7_224" \
    --embedding_size 768 \
    --use_tta \
    --output_csv "$MODEL_DIR/predictions.csv" \
    --batch_size 64

echo "=== COMPLETE: 80%+ accuracy pipeline finished ==="
echo "Model saved in $MODEL_DIR"
echo "Ensemble model saved in $MODEL_DIR/ensemble_model.pth"
echo "Performance metrics saved in evaluation logs" 