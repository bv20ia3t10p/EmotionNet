#!/bin/bash

# ResEmoteNet Training Pipeline with Synthetic Data
# This script automates the process of:
# 1. Installing dependencies
# 2. Generating synthetic data (optional)
# 3. Merging synthetic data with real training data
# 4. Training the emotion recognition model

# Default parameters
NUM_SYNTHETIC_IMAGES=500
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
BASE_DIR="$(dirname "$(realpath "$0")")"
SYNTHETIC_DIR="./extracted/emotion/synthetic"
TRAIN_DIR="./extracted/emotion/train"
TEST_DIR="./extracted/emotion/test"
MODELS_DIR="./models"
RESULTS_DIR="./results"
MODEL_TYPE="enhanced"  # Default to enhanced model
USE_FOCAL_LOSS=true
FOCAL_GAMMA=2.0

# Style for console output
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set default to skip synthetic data generation due to diffusers module issues
SKIP_SYNTHETIC=true

# Usage information
function show_usage {
    echo -e "${BOLD}Usage:${NC} $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -s, --synthetic-images NUM    Number of synthetic images per emotion (default: $NUM_SYNTHETIC_IMAGES)"
    echo "  -e, --epochs NUM              Number of training epochs (default: $EPOCHS)"
    echo "  -b, --batch-size NUM          Batch size for training (default: $BATCH_SIZE)"
    echo "  -l, --learning-rate RATE      Learning rate (default: $LEARNING_RATE)"
    echo "  -w, --weight-decay RATE       Weight decay for regularization (default: $WEIGHT_DECAY)"
    echo "  -t, --train-dir DIR           Training data directory (default: $TRAIN_DIR)"
    echo "  -v, --test-dir DIR            Test data directory (default: $TEST_DIR)"
    echo "  -o, --output-dir DIR          Models output directory (default: $MODELS_DIR)"
    echo "  -m, --model-type TYPE         Model type: 'original' or 'enhanced' (default: $MODEL_TYPE)"
    echo "  --skip-install                Skip dependency installation"
    echo "  --use-synthetic               Enable synthetic data generation (default: disabled)"
    echo "  --cuda                        Use CUDA for training (if available)"
    echo "  --no-focal-loss               Disable focal loss (enabled by default)"
    echo "  --focal-gamma NUM             Set gamma parameter for focal loss (default: $FOCAL_GAMMA)"
    echo "  --compare                     Train and compare both model types"
    echo ""
    echo "Example:"
    echo "  $0 --synthetic-images 1000 --epochs 100 --batch-size 128 --model-type enhanced --cuda"
    exit 1
}

# Error handling function
function error_exit {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

# Check if directory exists, if not create it
function ensure_dir {
    if [ ! -d "$1" ]; then
        mkdir -p "$1" || error_exit "Failed to create directory: $1"
        echo -e "${GREEN}[INFO]${NC} Created directory: $1"
    fi
}

# Function to check if a python3 module is installed
function check_module {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Parse command line arguments
SKIP_INSTALL=true
USE_CUDA=true
COMPARE_MODELS=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_usage
            ;;
        -s|--synthetic-images)
            NUM_SYNTHETIC_IMAGES="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -w|--weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        -t|--train-dir)
            TRAIN_DIR="$2"
            shift 2
            ;;
        -v|--test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -m|--model-type)
            MODEL_TYPE="$2"
            if [[ "$MODEL_TYPE" != "original" && "$MODEL_TYPE" != "enhanced" ]]; then
                error_exit "Invalid model type. Must be 'original' or 'enhanced'."
            fi
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --use-synthetic)
            SKIP_SYNTHETIC=false
            shift
            ;;
        --cuda)
            USE_CUDA=true
            shift
            ;;
        --no-focal-loss)
            USE_FOCAL_LOSS=false
            shift
            ;;
        --focal-gamma)
            FOCAL_GAMMA="$2"
            shift 2
            ;;
        --compare)
            COMPARE_MODELS=true
            shift
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Make sure all paths are absolute
if [[ ! "$TRAIN_DIR" = /* ]]; then
    TRAIN_DIR="$BASE_DIR/$TRAIN_DIR"
fi

if [[ ! "$TEST_DIR" = /* ]]; then
    TEST_DIR="$BASE_DIR/$TEST_DIR"
fi

if [[ ! "$SYNTHETIC_DIR" = /* ]]; then
    SYNTHETIC_DIR="$BASE_DIR/$SYNTHETIC_DIR"
fi

if [[ ! "$MODELS_DIR" = /* ]]; then
    MODELS_DIR="$BASE_DIR/$MODELS_DIR"
fi

if [[ ! "$RESULTS_DIR" = /* ]]; then
    RESULTS_DIR="$BASE_DIR/$RESULTS_DIR"
fi

# Create required directories
ensure_dir "$MODELS_DIR"
ensure_dir "$RESULTS_DIR"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    error_exit "python3 is not installed. Please install python3 to continue."
fi

# Setup python3 commands
python3_CMD="python3"
PIP_CMD="python3 -m pip"

# Print configuration
echo -e "${BOLD}ResEmoteNet Training Pipeline${NC}"
echo "=================================="
echo -e "Base directory: ${YELLOW}$BASE_DIR${NC}"
echo -e "Training data: ${YELLOW}$TRAIN_DIR${NC}"
echo -e "Test data: ${YELLOW}$TEST_DIR${NC}"
echo -e "Models output: ${YELLOW}$MODELS_DIR${NC}"
echo -e "Model type: ${YELLOW}$MODEL_TYPE${NC}"
echo -e "Compare models: ${YELLOW}$COMPARE_MODELS${NC}"
echo -e "Using synthetic data: ${YELLOW}$([ "$SKIP_SYNTHETIC" = false ] && echo "Yes" || echo "No")${NC}"
echo -e "Synthetic images per emotion: ${YELLOW}$NUM_SYNTHETIC_IMAGES${NC}"
echo -e "Training epochs: ${YELLOW}$EPOCHS${NC}"
echo -e "Batch size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "Learning rate: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "Weight decay: ${YELLOW}$WEIGHT_DECAY${NC}"
echo -e "Use CUDA: ${YELLOW}$USE_CUDA${NC}"
echo -e "Use focal loss: ${YELLOW}$([ "$USE_FOCAL_LOSS" = true ] && echo "Yes" || echo "No")${NC}"
if [ "$USE_FOCAL_LOSS" = true ]; then
    echo -e "Focal loss gamma: ${YELLOW}$FOCAL_GAMMA${NC}"
fi
echo "=================================="

# Step 1: Install dependencies
if [ "$SKIP_INSTALL" = false ]; then
    echo -e "\n${BOLD}Step 1: Installing dependencies${NC}"
    echo "--------------------------------"
    if [ -f "$BASE_DIR/requirements.txt" ]; then
        echo -e "${GREEN}[INFO]${NC} Installing packages from requirements.txt (excluding diffusers)"
        
        # Install all requirements except diffusers which is causing issues
        grep -v "diffusers" "$BASE_DIR/requirements.txt" | xargs -n 1 $PIP_CMD install || echo -e "${YELLOW}[WARNING]${NC} Some packages may not have installed correctly"
        
        # Explicitly ensure critical packages are installed
        echo -e "${GREEN}[INFO]${NC} Ensuring critical packages are installed"
        $PIP_CMD install pandas numpy torch torchvision scikit-learn pillow opencv-python tqdm matplotlib diffusers || error_exit "Failed to install critical dependencies"
        
        echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed"
    else
        error_exit "requirements.txt not found in $BASE_DIR"
    fi
else
    echo -e "\n${YELLOW}[INFO]${NC} Skipping dependency installation"
    
    # Even if skipping general installation, verify critical packages
    echo -e "${GREEN}[INFO]${NC} Checking critical dependencies"
    for pkg in "pandas" "numpy" "torch" "torchvision" "sklearn" "PIL" "cv2" "tqdm" "matplotlib" "diffusers"; do
        if ! check_module $pkg; then
            echo -e "${YELLOW}[WARNING]${NC} Module '$pkg' is not installed. Installing now..."
            $PIP_CMD install $pkg || error_exit "Failed to install $pkg"
        fi
    done
fi

# Step 2: Generate synthetic data
if [ "$SKIP_SYNTHETIC" = false ]; then
    echo -e "\n${BOLD}Step 2: Generating synthetic data${NC}"
    echo "--------------------------------"
    
    # Create synthetic data directory
    ensure_dir "$SYNTHETIC_DIR"
    
    # Check if synthetic data already exists
    declare -a emotions=("angry" "disgust" "fear" "happy" "sad" "surprise" "neutral")
    EXISTING_DATA=false
    DATA_COUNT=0
    
    for emotion in "${emotions[@]}"; do
        if [ -d "$SYNTHETIC_DIR/$emotion" ] && [ "$(ls -A "$SYNTHETIC_DIR/$emotion" 2>/dev/null)" ]; then
            DATA_COUNT=$((DATA_COUNT + $(find "$SYNTHETIC_DIR/$emotion" -type f -name "*.png" -o -name "*.jpg" | wc -l)))
        fi
    done
    
    if [ $DATA_COUNT -gt 0 ]; then
        EXISTING_DATA=true
        echo -e "${YELLOW}[INFO]${NC} Found $DATA_COUNT existing synthetic images in $SYNTHETIC_DIR"
        echo -e "${YELLOW}[INFO]${NC} Skipping synthetic data generation"
    fi
    
    # Check if synthetic data generation script exists
    if [ ! -f "$BASE_DIR/generate_synthetic_data.py" ]; then
        error_exit "Synthetic data generation script not found: $BASE_DIR/generate_synthetic_data.py"
    fi
    
    # Only generate data if none exists
    if [ "$EXISTING_DATA" = false ]; then
        echo -e "${GREEN}[INFO]${NC} Generating synthetic data with $NUM_SYNTHETIC_IMAGES images per emotion"
        $python3_CMD "$BASE_DIR/generate_synthetic_data.py" --output_dir "$SYNTHETIC_DIR" --num_images "$NUM_SYNTHETIC_IMAGES" || error_exit "Failed to generate synthetic data"
        echo -e "${GREEN}[SUCCESS]${NC} Synthetic data generated at $SYNTHETIC_DIR"
    fi
    
    # Merge synthetic data with training data
    echo -e "${GREEN}[INFO]${NC} Merging synthetic data with training data"
    
    # Ensure training directories exist
    for emotion in "${emotions[@]}"; do
        ensure_dir "$TRAIN_DIR/$emotion"
        
        # Copy synthetic data to training directory
        if [ -d "$SYNTHETIC_DIR/$emotion" ]; then
            cp -r "$SYNTHETIC_DIR/$emotion"/* "$TRAIN_DIR/$emotion/" || error_exit "Failed to copy synthetic $emotion data"
            echo -e "${GREEN}[INFO]${NC} Copied synthetic $emotion data to training directory"
        else
            echo -e "${YELLOW}[WARNING]${NC} Synthetic data directory not found: $SYNTHETIC_DIR/$emotion"
        fi
    done
    
    echo -e "${GREEN}[SUCCESS]${NC} Synthetic data merged with training data"
else
    echo -e "\n${YELLOW}[INFO]${NC} Skipping synthetic data generation (use --use-synthetic to enable)"
fi

# Function to train a specific model type
function train_model {
    local model_type=$1
    echo -e "\n${BOLD}Step 3: Training the ${model_type} model${NC}"
    echo "--------------------------------"
    
    # Check if training script exists
    if [ ! -f "$BASE_DIR/train_emotion_recognition.py" ]; then
        error_exit "Training script not found: $BASE_DIR/train_emotion_recognition.py"
    fi
    
    # Verify dependencies before training
    echo -e "${GREEN}[INFO]${NC} Verifying dependencies for training"
    for pkg in "pandas" "numpy" "torch" "torchvision" "PIL" "matplotlib" "tqdm" "cv2" "sklearn"; do
        if ! check_module $pkg; then
            echo -e "${YELLOW}[WARNING]${NC} Required module '$pkg' is not installed. Installing it now..."
            $PIP_CMD install $pkg || error_exit "Failed to install $pkg, which is required for training"
        fi
    done
    
    # Build training command
    local train_cmd="$python3_CMD \"$BASE_DIR/train_emotion_recognition.py\" --train_dir \"$TRAIN_DIR\" --test_dir \"$TEST_DIR\" --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LEARNING_RATE --weight_decay $WEIGHT_DECAY --save_dir \"$MODELS_DIR\" --use_mixup --model $model_type"
    
    # Add focal loss if enabled
    if [ "$USE_FOCAL_LOSS" = true ]; then
        train_cmd="$train_cmd --focal_loss --focal_gamma $FOCAL_GAMMA"
    fi
    
    # Add CUDA flag if requested
    if [ "$USE_CUDA" = false ]; then
        train_cmd="$train_cmd --no_cuda"
    fi
    
    # Execute training
    echo -e "${GREEN}[INFO]${NC} Starting $model_type model training"
    echo -e "${YELLOW}[COMMAND]${NC} $train_cmd"
    eval $train_cmd || error_exit "Training failed for $model_type model"
    
    echo -e "\n${GREEN}[SUCCESS]${NC} $model_type model training completed!"
    echo -e "Model saved to ${YELLOW}$MODELS_DIR/best_${model_type}_model.pth${NC}"
}

# Step 3: Train model(s)
if [ "$COMPARE_MODELS" = true ]; then
    # Train both models for comparison
    train_model "original"
    train_model "enhanced"
    
    # Compare the models
    echo -e "\n${BOLD}Step 4: Comparing model results${NC}"
    echo "--------------------------------"
    
    # Check if prediction script exists
    if [ ! -f "$BASE_DIR/predict_emotion.py" ]; then
        error_exit "Prediction script not found: $BASE_DIR/predict_emotion.py"
    fi
    
    # Evaluate original model
    echo -e "${GREEN}[INFO]${NC} Evaluating original model"
    orig_cmd="$python3_CMD \"$BASE_DIR/predict_emotion.py\" --model_path \"$MODELS_DIR/best_original_model.pth\" --test_dir \"$TEST_DIR\" --output_dir \"$RESULTS_DIR/original\""
    if [ "$USE_CUDA" = false ]; then
        orig_cmd="$orig_cmd --no_cuda"
    fi
    echo -e "${YELLOW}[COMMAND]${NC} $orig_cmd"
    eval $orig_cmd || error_exit "Original model evaluation failed"
    
    # Evaluate enhanced model
    echo -e "${GREEN}[INFO]${NC} Evaluating enhanced model"
    enh_cmd="$python3_CMD \"$BASE_DIR/predict_emotion.py\" --model_path \"$MODELS_DIR/best_enhanced_model.pth\" --test_dir \"$TEST_DIR\" --output_dir \"$RESULTS_DIR/enhanced\""
    if [ "$USE_CUDA" = false ]; then
        enh_cmd="$enh_cmd --no_cuda"
    fi
    echo -e "${YELLOW}[COMMAND]${NC} $enh_cmd"
    eval $enh_cmd || error_exit "Enhanced model evaluation failed"
    
    echo -e "\n${GREEN}[SUCCESS]${NC} Model comparison completed!"
    echo -e "Results saved to ${YELLOW}$RESULTS_DIR${NC}"
else
    # Train the selected model type
    train_model "$MODEL_TYPE"
fi

echo -e "\n${BOLD}${GREEN}ResEmoteNet training pipeline completed successfully!${NC}"
echo -e "Model is ready for use with predict_emotion.py" 