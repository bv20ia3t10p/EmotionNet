#!/bin/bash
#==============================================================================
# ConvNeXtEmoteNet Training Script
# 
# This script sets up and runs training for the ConvNeXtEmoteNet model,
# a ConvNeXt-based model with advanced attention mechanisms for emotion recognition.
#==============================================================================

echo "Setting up ConvNeXtEmoteNet training environment..."

#==============================================================================
# DIRECTORIES
#==============================================================================
# Create required directories
mkdir -p ./extracted/emotion

#==============================================================================
# CLEAN UP PREVIOUS BALANCED DATASETS
#==============================================================================
# Clean up function to remove any existing balanced directories
cleanup_balanced_datasets() {
    echo "🔹 Cleaning up any existing balanced datasets..."
    
    # Check for balanced directories in the dataset path
    PARENT_DIR=$(dirname "$TRAIN_PATH")
    if [ -d "$PARENT_DIR" ]; then
        for dir in "$PARENT_DIR"/balanced_*; do
            if [ -d "$dir" ]; then
                echo "🔹 Removing existing balanced directory: $dir"
                rm -rf "$dir"
                echo "✅ Removed: $dir"
            fi
        done
    fi
    
    echo "✅ Balanced dataset cleanup complete"
}

# Run cleanup
cleanup_balanced_datasets

#==============================================================================
# BASIC CONFIGURATION
#==============================================================================
export ROOT="./extracted"
export DATASET_PATH="${ROOT}/emotion"
export MODEL_PATH="${ROOT}/emotion_convnext_model.pth"
export MODEL_TYPE="ConvNeXtEmoteNet"
export BACKBONE="convnext_base"
export IMAGE_SIZE="224"

# Default paths (may be overridden by balancing)
export TRAIN_PATH="${DATASET_PATH}/train"
export TEST_PATH="${DATASET_PATH}/test"

#==============================================================================
# TRAINING PARAMETERS
#==============================================================================
export BATCH_SIZE="96"                # Large batch size for efficient training
export LEARNING_RATE="0.00025"         # Further reduced learning rate for better generalization
export NUM_EPOCHS="300"                # Longer training to accommodate cyclic learning rate

#==============================================================================
# OPTIMIZATION SETTINGS
#==============================================================================
export ACCUMULATION_STEPS="1"          # No accumulation needed with large batch size
export USE_AMP="1"                     # Automatic Mixed Precision for faster training
export LARGE_BATCH_BN="1"              # Optimized batch normalization for large batches
export WARMUP_EPOCHS="10"              # Extended warmup period for stable initial training
export FREEZE_BACKBONE_EPOCHS="3"      # Initially freeze backbone for stable training
export DISABLE_CHECKPOINTS="0"         # Set to "1" to disable checkpoint saving (speeds up training)
export FAST_EVALUATION="0"             # Set to "1" for faster evaluation (fewer augmentations)
export ULTRA_FAST_EVAL="0"             # Set to "1" for ultra-fast evaluation (subset of validation data)
export SKIP_TTA="0"                    # Set to "1" to skip test-time augmentation entirely (fastest validation)

# Enable learning rate cycling (reset every 50 epochs)
export SCHEDULER_TYPE="cosine_restart"
export COSINE_CYCLES="6"               # For ~300 epochs, reset every ~50 epochs

#==============================================================================
# SCHEDULER & ADVANCED TRAINING
#==============================================================================
# Stochastic Weight Averaging for better generalization
export SWA_ENABLED="1"
export SWA_START_EPOCH="40"            # Start SWA even earlier
export SWA_FREQ="2"                    # More frequent model averaging
export FAST_SWA_FINALIZE="0"           # Set to 1 for faster SWA finalization (less accurate but much quicker)

# Knowledge Distillation settings
export SELF_DISTILLATION_ENABLED="1"
export SELF_DISTILLATION_START="30"    # Start self-distillation earlier
export SELF_DISTILLATION_TEMP="5.0"    # Even higher temperature for smoother probabilities
export SELF_DISTILLATION_ALPHA="0.7"   # Stronger distillation weight

#==============================================================================
# CLASS BALANCING AND LOSS FUNCTION
#==============================================================================
# Focus more on difficult/rare classes
export USE_FOCAL_LOSS="0"
export FOCAL_GAMMA="2.5"               # Increased gamma to focus more on hard examples
export LABEL_SMOOTHING="0.25"          # Increased label smoothing for regularization
export KL_WEIGHT="0.3"                 # Increased KL divergence weight

# Enable class balancing
export BALANCE_DATASET="1"
export TARGET_SAMPLES_PER_CLASS="5000" # Target for oversampling minority classes

#==============================================================================
# REGULARIZATION
#==============================================================================
export WEIGHT_DECAY="0.001"            # Further increased weight decay to combat overfitting
export HEAD_DROPOUT="0.6"              # Increased dropout for classification head
export FEATURE_DROPOUT="0.4"           # Increased feature dropout
export GRAD_CLIP_VALUE="0.5"           # Explicit gradient clipping value

#==============================================================================
# DATA AUGMENTATION
#==============================================================================
export MIXUP_PROB="0.7"                # Even more aggressive augmentation
export CUTMIX_PROB="0.6"               # Increased CutMix probability
export MIXUP_ALPHA="1.2"               # Stronger mixup interpolation
export CUTMIX_ALPHA="1.4"              # Stronger cutmix effect

# Enable progressive augmentation
export PROGRESSIVE_AUGMENTATION="1"
export PHASE_1_EPOCHS="5"
export PHASE_2_EPOCHS="10"
export PHASE_3_EPOCHS="15"
export PHASE_4_EPOCHS="20"

#==============================================================================
# EARLY STOPPING
#==============================================================================
export EARLY_STOPPING_PATIENCE="35"    # More patience for ConvNeXt

#==============================================================================
# APPLY DATASET BALANCING
#==============================================================================
if [ "$BALANCE_DATASET" == "1" ]; then
    echo "🔹 Creating balanced dataset with advanced techniques..."
    
    # Check if balance_dataset_advanced.py exists
    if [ ! -f "balance_dataset_advanced.py" ]; then
        echo "❌ Error: balance_dataset_advanced.py not found. This script is required."
        exit 1
    fi
    
    echo "🔹 Using standalone balancing script (balance_dataset_advanced.py)..."
    chmod +x balance_dataset_advanced.py
    
    # Run the script without capturing its output
    python3 balance_dataset_advanced.py "$TRAIN_PATH" "$TARGET_SAMPLES_PER_CLASS"
    STANDALONE_RESULT=$?
    
    if [ $STANDALONE_RESULT -eq 0 ]; then
        # Script ran successfully, load the environment variables it created
        echo "✅ Dataset balancing completed successfully"
        
        # Source the environment variables
        if [ -f "env_vars.py" ]; then
            echo "🔹 Loading environment variables from env_vars.py..."
            python3 -c "import env_vars" || {
                echo "❌ Error loading environment variables. Exiting."
                exit 1
            }
        else
            echo "❌ Error: env_vars.py not found after balancing. Exiting."
            exit 1
        fi
    else
        echo "❌ Error in dataset balancing. Exiting."
        exit 1
    fi
else
    # No balancing requested
    export MODEL_BALANCE_DATASET="0"
    
    # Create environment variable file for python
    cat > env_vars.py << EOF
import os

# Dataset path settings
os.environ['TRAIN_PATH'] = """${TRAIN_PATH}"""
os.environ['MODEL_BALANCE_DATASET'] = "0"

print(f"🔹 Loaded environment variables:")
print(f"   TRAIN_PATH: {os.environ['TRAIN_PATH']}")
print(f"   MODEL_BALANCE_DATASET: {os.environ['MODEL_BALANCE_DATASET']}")
EOF
fi

#==============================================================================
# SUMMARY
#==============================================================================
echo "=== ConvNeXtEmoteNet Training Configuration ==="
echo "Model: ConvNeXtEmoteNet with ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Training Path: ${TRAIN_PATH}"
echo "Dataset Balancing: $([ "$BALANCE_DATASET" == "1" ] && echo "Enabled (target: $TARGET_SAMPLES_PER_CLASS samples/class)" || echo "Disabled")"
echo "Balancing Method: $([ "$MODEL_BALANCE_DATASET" == "1" ] && echo "DataLoader Oversampling" || [ "$MODEL_BALANCE_DATASET" == "0" ] && [ "$BALANCE_DATASET" == "1" ] && echo "Pre-generated Balanced Dataset" || echo "None")"
echo "Advanced Features: SWA, Self-Distillation, Mixed Precision"
echo "========================================"

#==============================================================================
# START TRAINING
#==============================================================================
echo "Starting ConvNeXtEmoteNet training..."

# Make sure we include env_vars.py in the main Python script execution
if [ -f "env_vars.py" ]; then
    echo "🔹 Using environment variables from env_vars.py for training"
    # Extract TRAIN_PATH from env_vars.py to shell environment
    EXTRACTED_TRAIN_PATH=$(python3 env_vars.py --path-only)
    if [ -n "$EXTRACTED_TRAIN_PATH" ]; then
        export TRAIN_PATH="$EXTRACTED_TRAIN_PATH"
        export MODEL_BALANCE_DATASET="0"  # Explicitly disable balancing since we're using a balanced dataset
        echo "🔹 Using balanced dataset at: $TRAIN_PATH"
        echo "🔹 MODEL_BALANCE_DATASET set to: $MODEL_BALANCE_DATASET"
    fi
fi

python3 main.py --mode train 