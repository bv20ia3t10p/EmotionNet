#!/bin/bash
# Script to run balanced training and F1 score validation

# Make script executable
chmod +x balanced_training.py
chmod +x f1_report.py

# Default settings
DATA_DIR="./extracted/emotion/train"
TEST_DIR="./extracted/emotion/test"
MODEL_DIR="./models"
BACKBONE="resnet18"
BATCH_SIZE=32
EPOCHS=50
AUGMENTATION_LEVEL=2
PATIENCE=7
LEARNING_RATE=0.0001
COMPARE_AUG=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --test_dir)
      TEST_DIR="$2"
      shift 2
      ;;
    --model_dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --backbone)
      BACKBONE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --augmentation_level)
      AUGMENTATION_LEVEL="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --pretrain)
      PRETRAIN="--pretrain"
      shift
      ;;
    --compare_aug)
      COMPARE_AUG=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Step 1: Train the model with balanced training
echo "=========================================="
echo "Starting balanced training..."
echo "=========================================="
python3 balanced_training.py \
  --data_dir $DATA_DIR \
  --test_dir $TEST_DIR \
  --model_dir $MODEL_DIR \
  --backbone $BACKBONE \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --augmentation_level $AUGMENTATION_LEVEL \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  $PRETRAIN

# Check if training completed successfully
if [ $? -ne 0 ]; then
  echo "Training failed. Exiting."
  exit 1
fi

# Step 2: Generate detailed F1 report on the test dataset
FINAL_MODEL_PATH="$MODEL_DIR/final_balanced_model.pth"

echo ""
echo "=========================================="
echo "Generating detailed F1 score report..."
echo "=========================================="

# Add --compare_aug flag if requested
if [ "$COMPARE_AUG" = true ]; then
  COMPARE_AUG_FLAG="--compare_aug"
  echo "Will compare augmentation impact on model performance"
else
  COMPARE_AUG_FLAG=""
fi

python3 f1_report.py \
  --model_path $FINAL_MODEL_PATH \
  --test_dir $TEST_DIR \
  --backbone $BACKBONE \
  --batch_size $BATCH_SIZE \
  $COMPARE_AUG_FLAG

echo ""
echo "=========================================="
echo "Process completed!"
if [ "$COMPARE_AUG" = true ]; then
  echo "Check detailed_confusion_matrix.png, f1_scores.png, and augmentation_comparison.png for visualizations"
else
  echo "Check detailed_confusion_matrix.png and f1_scores.png for visualizations"
fi
echo "===========================================" 