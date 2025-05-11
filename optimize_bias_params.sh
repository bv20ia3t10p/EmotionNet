#!/bin/bash
# Optimize bias parameters for facial emotion recognition
# Usage: ./optimize_bias_params.sh [test_dir] [model_path] [method]

# Set default paths if not provided
TEST_DIR=${1:-"./extracted/emotion/test"}
MODEL_PATH=${2:-"./models/hybrid_model.pth"}
METHOD=${3:-"grid_search"}

OUTPUT_DIR="bias_optimization_results"

echo "Starting bias parameter optimization with these settings:"
echo "Test images directory: $TEST_DIR"
echo "Model path: $MODEL_PATH"
echo "Optimization method: $METHOD"
echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run optimization
echo
echo "Running bias parameter optimization..."
python3 advanced_bias_tuning.py --image_dir "$TEST_DIR" --model "$MODEL_PATH" --output_dir "$OUTPUT_DIR" --method "$METHOD" --jobs 4

# Find a test image if available
TEST_IMAGE=""
for img in "$TEST_DIR"/*/*.jpg "$TEST_DIR"/*/*.png; do
    if [ -f "$img" ]; then
        TEST_IMAGE="$img"
        break
    fi
done

# Test with optimized parameters if a test image is found
if [ -n "$TEST_IMAGE" ]; then
    echo
    echo "Testing with optimized parameters on image: $TEST_IMAGE"
    python3 advanced_bias_tuning.py --image_dir "$TEST_DIR" --model "$MODEL_PATH" --output_dir "$OUTPUT_DIR" --test "$TEST_IMAGE" --create_ground_truth
else
    echo
    echo "No test images found. To test optimized parameters, run:"
    echo "python3 advanced_bias_tuning.py --image_dir <image_dir> --model $MODEL_PATH --output_dir $OUTPUT_DIR --test <image_path>"
fi

echo
echo "Optimization complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Best parameters saved to: $OUTPUT_DIR/best_bias_params.json" 