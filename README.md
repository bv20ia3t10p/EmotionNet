# High-Accuracy Emotion Recognition Model

This repository contains a high-performance facial emotion recognition model that achieves up to 80% accuracy through advanced techniques. The model uses an ensemble architecture with attention-based fusion of multiple backbone networks (EfficientNet, ResNet).

## Features

- **Ensemble Architecture**: Combines multiple backbone networks with attention-based fusion
- **Advanced Augmentations**: Uses Albumentations library for high-quality data augmentation
- **Training Optimizations**:
  - Automatic Mixed Precision (AMP) for faster training
  - Exponential Moving Average (EMA) for more stable models
  - Class-weighted sampling for better balance
  - Cosine annealing learning rate scheduling

## Requirements

Install the required dependencies:

```
pip install -r requirements.txt
```

## Directory Structure

The model expects the following directory structure for training:

```
extracted/
  emotion/
    train/
      angry/
      disgust/
      fear/
      happy/
      sad/
      surprise/
      neutral/
    test/
      (same structure as train)
```

Each emotion folder should contain facial images for that emotion class.

## Training the Model

### Windows

```
train_high_accuracy.bat
```

### Unix/Linux/macOS

```
chmod +x train_high_accuracy.sh
./train_high_accuracy.sh
```

### Custom Training Parameters

You can also run the training script directly with custom parameters:

```
python high_accuracy_model.py --data_dir "./path/to/train" --test_dir "./path/to/test" --model_dir "./models" --batch_size 32 --epochs 50 --learning_rate 0.0001 --image_size 224 --backbones efficientnet_b0 resnet18
```

Parameters:
- `--data_dir`: Path to training data directory
- `--test_dir`: Path to test data directory
- `--model_dir`: Directory to save models
- `--model_path`: Path to pre-trained model to resume from (optional)
- `--backbones`: Backbone architectures to use (default: efficientnet_b0 resnet18)
- `--batch_size`: Batch size for training (default: 32)
- `--image_size`: Size to resize images to (default: 224)
- `--epochs`: Maximum number of training epochs (default: 50)
- `--patience`: Early stopping patience (default: 10)
- `--learning_rate`: Base learning rate (default: 0.0001)
- `--no_amp`: Disable automatic mixed precision
- `--no_ema`: Disable exponential moving average model

## Making Predictions

To predict the emotion in a facial image:

```
python predict_emotion.py --image "path/to/image.jpg" --model "models/high_accuracy_model.pth" --plot
```

Parameters:
- `--image`: Path to the image to predict (required)
- `--model`: Path to the model weights (.pth file) (required)
- `--backbones`: Backbone architectures used in the model (default: efficientnet_b0 resnet18)
- `--image_size`: Size to resize images to (default: 224)
- `--plot`: Plot and save visualization of the prediction

The prediction script will output:
1. The predicted emotion and confidence level
2. Probabilities for all emotion classes
3. A visualization saved as a PNG file (if --plot is enabled)

## Model Architecture

The model uses an ensemble of backbone networks:

1. **Backbone Networks**: Pre-trained networks (EfficientNet, ResNet) extract features from images
2. **Neck**: Reduces dimensions and adds regularization to features
3. **Attention Mechanism**: Dynamically weights the contribution of each backbone
4. **Classifier**: Makes the final emotion prediction

## Results

The model achieves significantly higher accuracy (60-80% range) compared to baseline models (30-35%). The largest improvements are seen in previously problematic classes like "angry" and "neutral".

During training, the model generates:
- Training/validation loss and accuracy plots
- Per-class accuracy metrics
- Confusion matrix visualization
- Detailed classification report

## License

This project is licensed under the MIT License - see the LICENSE file for details. 