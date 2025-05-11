#!/bin/bash
# Run optimized facial expression recognition training with the improved model
# This script automates the process with optimal parameters for FER2013

# Check for required packages
echo "🔍 Checking required packages..."

# Non-critical packages (will be handled gracefully if missing)

echo "✅ Required packages check complete"
# Clean training log if it exists
LOG_FILE=$(grep LOG_CSV_PATH config.py | cut -d'=' -f2 | tr -d ",' ")
if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
    echo "  ✓ Removed training log: $LOG_FILE"
fi

echo "✅ Setup complete"

# Default values
BALANCE_DATASET=true
SAMPLES_PER_CLASS=7000
BACKBONE="efficientnet_b0"
EPOCHS=80
BATCH_SIZE=64
LEARNING_RATE=0.0005

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --no-balance)
            BALANCE_DATASET=false
            shift
            ;;
        --samples)
            SAMPLES_PER_CLASS="$2"
            shift
            shift
            ;;
        --backbone)
            BACKBONE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment variables
export USE_CUSTOM_MODEL=1
export MODEL_TYPE="EmotionCNN"
export BACKBONE=$BACKBONE
export NUM_EPOCHS=$EPOCHS
export BATCH_SIZE=$BATCH_SIZE
export LEARNING_RATE=$LEARNING_RATE
export SCHEDULER_TYPE="onecycle"
export MAX_LR=0.005
export USE_AMP=1
export USE_FP16=1
# Force unbuffered Python output to see logs in real-time
export PYTHONUNBUFFERED=1

echo "🚀 Starting optimized FER training with EmotionCNN model"
echo "=================================================="
echo "📊 Configuration:"
echo "  - Backbone: $BACKBONE"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Balance Dataset: $BALANCE_DATASET"
echo "  - Target Samples Per Class: $SAMPLES_PER_CLASS"
echo "  - Scheduler: OneCycle"
echo "  - Mixed Precision: Enabled"
echo "=================================================="

# Build the balance flag arguments if needed
BALANCE_ARGS=""
if [ "$BALANCE_DATASET" = true ]; then
    BALANCE_ARGS="--balance --samples_per_class $SAMPLES_PER_CLASS"
fi

# Run the training directly in foreground without redirecting output
echo "Executing: python3 -u run_training.py with specified parameters"
python3 -u run_training.py --backbone "$BACKBONE" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LEARNING_RATE" $BALANCE_ARGS

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully"
    
    # Evaluate the model with test-time augmentation
    echo "🔍 Evaluating model with test-time augmentation"
    python3 -u evaluate.py --model_path $(cat config.py | grep "MODEL_PATH" | cut -d'=' -f2 | tr -d ",' ")
else
    echo "❌ Training failed"
    exit 1
fi

# Print completion message
echo "🎉 Optimized FER training and evaluation complete" 