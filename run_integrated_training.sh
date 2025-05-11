#!/bin/bash
# Script to run integrated bias correction training for facial emotion recognition

# Make executable
chmod +x integrate_bias_correction.py

# Default settings
DATA_DIR="./extracted/emotion/train"
TEST_DIR="./extracted/emotion/test"
MODEL_DIR="./models"
MODEL_PATH=""
SAVE_PATH="integrated_model.pth"
BACKBONE="resnet18"
BATCH_SIZE=32
EPOCHS=30
PATIENCE=7
LEARNING_RATE=0.0001
CORRECTION_STRENGTH=3.0
CORRECTION_FREQUENCY=1

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
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --save_path)
      SAVE_PATH="$2"
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
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --correction_strength)
      CORRECTION_STRENGTH="$2"
      shift 2
      ;;
    --correction_frequency)
      CORRECTION_FREQUENCY="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

echo "=========================================="
echo "Starting Integrated Bias Correction Training"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Test directory: $TEST_DIR"
echo "Using backbone: $BACKBONE"
echo "Bias correction strength: $CORRECTION_STRENGTH"
echo "Correction update frequency: Every $CORRECTION_FREQUENCY epoch(s)"
echo "=========================================="

# Run integrated training
if [ -n "$MODEL_PATH" ]; then
  echo "Continuing training from: $MODEL_PATH"
  python3 integrate_bias_correction.py \
    --data_dir $DATA_DIR \
    --test_dir $TEST_DIR \
    --model_dir $MODEL_DIR \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --correction_strength $CORRECTION_STRENGTH \
    --correction_frequency $CORRECTION_FREQUENCY
else
  echo "Training from scratch"
  python3 integrate_bias_correction.py \
    --data_dir $DATA_DIR \
    --test_dir $TEST_DIR \
    --model_dir $MODEL_DIR \
    --save_path $SAVE_PATH \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --correction_strength $CORRECTION_STRENGTH \
    --correction_frequency $CORRECTION_FREQUENCY
fi

# Check if training completed successfully
if [ $? -ne 0 ]; then
  echo "Training failed. Exiting."
  exit 1
fi

echo ""
echo "=========================================="
echo "Integrated bias correction training completed!"
echo "Model saved to: $MODEL_DIR/$SAVE_PATH"
echo "Check bias_correction.png and integrated_training_curves.png for visualizations"
echo "==========================================="

# Run evaluation with the F1 report script if available
if [ -f "f1_report.py" ]; then
  echo ""
  echo "=========================================="
  echo "Generating F1 score report for the integrated model..."
  echo "=========================================="
  
  python3 f1_report.py \
    --model_path "$MODEL_DIR/$SAVE_PATH" \
    --test_dir $TEST_DIR \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --compare_aug
fi 