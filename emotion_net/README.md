# EmotionNet

A high-performance emotion recognition model that achieves over 80% accuracy on facial emotion recognition tasks.

## Features

- Ensemble model combining multiple backbone architectures (EfficientNet, ResNet, etc.)
- Advanced data augmentation techniques optimized for facial images
- Attention mechanism for dynamic model weighting
- Exponential Moving Average (EMA) for stable training
- Automatic Mixed Precision (AMP) for faster training
- Comprehensive evaluation metrics and visualizations
- Easy-to-use prediction interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion_net.git
cd emotion_net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
emotion_net/
├── config/
│   ├── __init__.py
│   └── constants.py         # Configuration constants
├── data/
│   ├── __init__.py
│   └── dataset.py          # Dataset and data loading utilities
├── models/
│   ├── __init__.py
│   ├── ensemble.py         # Ensemble model definition
│   └── ema.py             # Exponential Moving Average implementation
├── training/
│   ├── __init__.py
│   └── trainer.py         # Training utilities
├── utils/
│   ├── __init__.py
│   └── visualization.py   # Visualization utilities
├── __init__.py
├── train.py               # Main training script
├── predict.py            # Prediction script
└── README.md
```

## Usage

### Training

To train the model:

```bash
python -m emotion_net.train \
    --data_dir ./extracted/emotion/train \
    --test_dir ./extracted/emotion/test \
    --model_dir ./models \
    --backbones efficientnet_b0 resnet18 \
    --batch_size 32 \
    --image_size 224 \
    --epochs 50 \
    --patience 10 \
    --learning_rate 0.0001
```

Optional arguments:
- `--no_amp`: Disable automatic mixed precision
- `--no_ema`: Disable exponential moving average
- `--model_path`: Path to pre-trained model to resume from

### Prediction

To make predictions on new images:

```bash
python -m emotion_net.predict \
    --model_path ./models/high_accuracy_model.pth \
    --image_path path/to/your/image.jpg \
    --backbones efficientnet_b0 resnet18 \
    --image_size 224
```

## Model Architecture

The model uses an ensemble approach with the following components:

1. Multiple backbone networks (default: EfficientNet-B0 and ResNet-18)
2. Feature extraction and dimensionality reduction for each backbone
3. Attention mechanism to dynamically weight the models
4. Final classifier with batch normalization and dropout

## Training Features

- **Data Augmentation**: Carefully tuned augmentations that preserve facial features
- **Class Weighting**: Automatic class weight calculation with boost for underrepresented emotions
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Mixed Precision**: Automatic mixed precision training for faster computation
- **Model Averaging**: Exponential Moving Average for more stable results
- **Early Stopping**: Patience-based early stopping to prevent overfitting

## Evaluation

The model generates several evaluation metrics:
- Overall accuracy
- Per-class accuracy
- Confusion matrix visualization
- Detailed classification report (precision, recall, F1-score)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The model architecture is inspired by state-of-the-art ensemble approaches
- Data augmentation techniques are optimized based on facial recognition research
- Uses pre-trained models from the `timm` library 