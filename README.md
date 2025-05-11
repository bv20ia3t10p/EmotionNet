# Facial Emotion Recognition - Class Balancing and Bias Optimization

This repository contains advanced tools for addressing class imbalance in facial emotion recognition models. The tools help overcome the common problem where models struggle to identify emotions like "angry" and "neutral" due to class distribution issues.

## Problem Overview

Our facial emotion recognition model faced severe class imbalance issues, resulting in:
- 0% accuracy for "angry" and "neutral" classes in validation
- Uneven predictions heavily skewed toward overrepresented classes
- Misrepresentation of facial expressions in real-world testing

## Solutions

We've created two main solutions:

1. **Balanced Training Framework**: A comprehensive training pipeline with advanced oversampling and targeted augmentation
2. **Bias Optimization Framework**: A system to find optimal bias parameters for post-processing predictions

## Setup and Requirements

### Dependencies

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.5
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
albumentations>=1.0.0
matplotlib>=3.4.0
optuna>=2.10.0
scipy>=1.7.0
tqdm>=4.62.0
```

### Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Make sure you have the facial emotion dataset (e.g., FER2013) organized by emotion class
4. Make scripts executable: `chmod +x *.sh`

## 1. Balanced Training Framework

The `balanced_training.py` script provides a complete solution for training with balanced classes:

### Key Features

- **Class-weighted sampling**: Dynamically balances class representation during training
- **Targeted augmentation**: Applies more aggressive augmentation to underrepresented classes
- **Two-phase training**: Specializes on underrepresented classes first
- **Layer-specific learning rates**: Preserves feature extraction capabilities
- **Advanced metrics**: Monitors per-class accuracy and weighted validation metrics

### Usage

The easiest way to run the balanced training is using the provided shell script:

```
./run_balanced_training.sh [data_dir] [model_dir]
```

Alternatively, you can run it manually:

```
# Pre-training phase focused on underrepresented classes
python3 balanced_training.py --data_dir /path/to/dataset --model_dir ./models --pretrain --augmentation_level 3 --epochs 15

# Main balanced training phase
python3 balanced_training.py --data_dir /path/to/dataset --model_dir ./models --backbone resnet18 --epochs 50 --augmentation_level 2
```

### Parameters

- `--data_dir`: Path to the dataset (required)
- `--model_dir`: Directory to save models
- `--backbone`: CNN backbone (resnet18, efficientnet_b0, mobilenet_v3_small)
- `--pretrain`: Perform specialized pre-training on underrepresented classes
- `--augmentation_level`: Intensity of data augmentation (0-3)
- `--epochs`: Number of training epochs
- `--learning_rate`: Base learning rate
- `--batch_size`: Training batch size
- `--patience`: Early stopping patience

## 2. Bias Optimization Framework

The `advanced_bias_tuning.py` script provides a way to find optimal bias parameters for the model predictions:

### Key Features

- **Multiple optimization methods**: Grid search, Bayesian optimization, and evolutionary algorithms
- **Target distribution optimization**: Balances predictions toward desired class ratios
- **Ground truth validation**: Can optimize using labeled validation data
- **Parallel processing**: Efficiently explores parameter space
- **Comprehensive visualizations**: Generates heatmaps and distribution comparisons

### Usage

The easiest way to run the bias optimization is using the provided shell script:

```
./optimize_bias_params.sh [image_dir] [model_path] [method]
```

Alternatively, you can run it manually:

```
# Grid search optimization
python3 advanced_bias_tuning.py --image_dir /path/to/test_images --model ./models/hybrid_model.pth --method grid_search

# Bayesian optimization with Optuna
python3 advanced_bias_tuning.py --image_dir /path/to/test_images --model ./models/hybrid_model.pth --method optuna --trials 100

# Test with optimized parameters
python3 advanced_bias_tuning.py --image_dir /path/to/test_images --model ./models/hybrid_model.pth --test /path/to/image.jpg
```

### Parameters

- `--image_dir`: Directory with test images (required)
- `--model`: Path to the model weights
- `--output_dir`: Directory to save optimization results
- `--method`: Optimization method (grid_search, optuna, evolutionary)
- `--trials`: Number of trials for Bayesian or evolutionary optimization
- `--jobs`: Number of parallel jobs
- `--use_ground_truth`: Use ground truth for optimization
- `--ground_truth_file`: Path to ground truth JSON file
- `--create_ground_truth`: Create ground truth file from directory structure
- `--test`: Run a test with optimal parameters on a specific image

## Prediction with Bias Parameters

Once you've found optimal bias parameters, you can use them for prediction:

```
python3 predict_emotion.py --image /path/to/image.jpg --model ./models/hybrid_model.pth --angry_bias 4.5 --neutral_bias 2.8
```

## Results and Visualization

The tools generate various visualizations to help you understand the results:

- **Training plots**: Loss curves, accuracy per class, and weighted metrics
- **Confusion matrix**: Class-wise prediction patterns
- **Parameter space exploration**: Visualizations of the bias parameter search
- **Distribution comparison**: Target vs. achieved class distribution

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This work builds upon the original enhanced emotion recognition model to solve the specific problem of class imbalance and bias in emotion recognition. 