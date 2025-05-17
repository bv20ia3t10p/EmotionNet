# State-of-the-Art Facial Emotion Recognition

This repository contains a modular implementation of a state-of-the-art facial emotion recognition model for the FER2013 dataset. The model uses ensemble learning with attention mechanisms to achieve high accuracy in emotion classification.

## Architecture

The model combines multiple state-of-the-art approaches:

- **Ensemble Architecture**: Uses EfficientNetB0, Xception, and a custom CNN to extract complementary features
- **Attention Mechanisms**: Incorporates Convolutional Block Attention Module (CBAM) for focusing on relevant facial features
- **Test-Time Augmentation**: Averages predictions across multiple augmented versions of each test image
- **Advanced Training Techniques**: Implements class weighting, learning rate scheduling, and early stopping

## Organization

The codebase follows a modular structure with SOLID design principles:

```
src/
  ├── config/         # Configuration modules
  │   ├── model_config.py
  │   ├── training_config.py
  │   ├── augmentation_config.py
  │   └── data_config.py
  ├── data/           # Data processing modules
  │   ├── data_processor.py
  │   └── fer2013_processor.py
  ├── models/         # Model architecture modules
  │   ├── attention_module.py
  │   ├── cbam_attention.py
  │   ├── base_model_factory.py
  │   ├── efficientnet_factory.py
  │   ├── xception_factory.py
  │   ├── custom_cnn_factory.py
  │   └── model_builder.py
  ├── training/       # Training and evaluation modules
  │   ├── model_trainer.py
  │   ├── test_time_augmentation.py
  │   └── model_evaluator.py
  └── main.py         # Main application script
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train and evaluate the model:

```
python -m src.main
```

## Results

The model achieves state-of-the-art accuracy on the FER2013 dataset through:

- Ensemble learning with multiple backbone networks
- Attention mechanisms to focus on discriminative features
- Test-time augmentation to improve robustness
- Addressing class imbalance with appropriate weighting

## License

MIT License 