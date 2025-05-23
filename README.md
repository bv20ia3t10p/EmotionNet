# EmotionNet: State-of-the-Art Facial Emotion Recognition

This repository contains a state-of-the-art implementation for facial emotion recognition on the FER2013 dataset using PyTorch.

## Model Architecture

The EmotionNet model uses a hybrid architecture that combines several advanced techniques to achieve state-of-the-art performance:

1. **EfficientNet-B0 Backbone**: Pretrained on ImageNet for efficient feature extraction
2. **Self-Attention Mechanism**: Helps the model focus on important facial regions
3. **Channel Attention**: Emphasizes important feature channels
4. **Transformer Encoder**: Captures global contextual relationships
5. **Advanced Classifier Head**: Multi-layer classifier with dropout and batch normalization

## Dataset

The FER2013 dataset contains 35,887 grayscale 48x48 pixel facial images of emotions categorized into 7 classes:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## Key Features

- **Advanced Data Augmentation**: Includes random horizontal flipping, rotation, color jitter, and affine transformations
- **Mixed Precision Training**: For faster training and reduced memory usage
- **Cosine Annealing Learning Rate Scheduler**: With warm restarts for better convergence
- **AdamW Optimizer**: With weight decay for improved generalization
- **Batch Size of 8**: As specified in requirements

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.1
pillow>=8.2.0
```

## Usage

### Training

```bash
python train.py --csv_file dataset/fer2013/fer2013.csv --batch_size 8 --num_epochs 30 --mixed_precision
```

### Key Parameters

- `--data_dir`: Directory containing the dataset (default: 'dataset/fer2013')
- `--csv_file`: Path to the CSV file with data (default: 'dataset/fer2013/fer2013.csv')
- `--batch_size`: Batch size for training (default: 8)
- `--num_epochs`: Number of epochs (default: 30)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--output_dir`: Output directory for saving models and results (default: 'outputs')
- `--mixed_precision`: Use mixed precision training (flag)
- `--seed`: Random seed for reproducibility (default: 42)

## Expected Performance

This implementation is designed to achieve competitive accuracy on the FER2013 dataset, aiming for:

- Validation accuracy: ~70-75%
- Test accuracy: ~70-75%

Which is comparable to the state-of-the-art results on this challenging dataset.

## Model Architecture Details

The EmotionNet architecture consists of:

1. **Backbone**: EfficientNet-B0 for efficient feature extraction
2. **Feature Enhancement**:
   - Channel Attention Module to emphasize important feature channels
   - Self-Attention Module to focus on relevant spatial regions
3. **Context Modeling**: Transformer Encoder to capture global dependencies
4. **Classification Head**: Multi-layer classifier with dropout for regularization

## References

- FER2013 Dataset: [Challenges in Representation Learning: A report on three machine learning contests](https://arxiv.org/abs/1307.0414)
- EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- Self-Attention: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 