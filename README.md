# EmotionNet for Facial Emotion Recognition

A facial emotion recognition model with attention mechanisms for both FER2013 and FERPlus datasets.

## Key Features

- **Attention-Based Architecture**: Effective attention mechanisms including:
  - Squeeze-and-Excitation (SE) blocks
  - Residual connections
  - Stochastic depth for regularization

- **Training Features**:
  - Data augmentation techniques
  - Learning rate scheduling
  - Regularization methods
  - Support for both FER2013 (7 emotions) and FERPlus (8 emotions) datasets

## Datasets

The model supports two datasets:

### FER2013
- 48x48 grayscale facial images
- Seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### FERPlus
- Extension of FER2013 with 8 emotions
- Adds "Contempt" as the 8th emotion
- Multiple annotation modes: majority voting, probability distribution, or multi-target

## Model Architecture

AttentionEmotionNet uses an effective architecture with:

1. **Initial Feature Extraction**: Convolutional stem for feature capture
2. **Attention Mechanisms**: Channel attention via SE blocks
3. **Progressive Feature Extraction**: Carefully designed convolutional blocks
4. **Regularization**: Stochastic depth and dropout for preventing overfitting
5. **Configurable Classifier**: Adaptable to different numbers of emotion classes

## Usage

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Training the Model

To train on the FER2013 dataset (7 emotions):

```bash
python train.py --mode default --stochastic-depth
```

To train on the FERPlus dataset (8 emotions):

```bash
python train.py --mode ferplus --ferplus-dir path/to/ferplus --ferplus-mode majority
```

Key command-line arguments:

- `--mode`: Training mode (default, quick, ferplus)
- `--model-type`: Model architecture type (default: attention_emotion_net)
- `--dropout-rate`: Dropout rate (default: 0.4)
- `--stochastic-depth`: Enable stochastic depth
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Initial learning rate
- `--max-lr`: Maximum learning rate for OneCycleLR
- `--data-dir`: Data directory for FER2013
- `--ferplus-dir`: Directory for FERPlus dataset
- `--ferplus-mode`: FERPlus label mode (majority, probability, multi_target)

For the full list of options:

```bash
python train.py --help
```

### Evaluation

The model will be automatically evaluated on the validation set during training.

## Customization

To modify the model architecture:

1. Edit `emotionnet/models/attention_emotion_net.py`
2. Adjust hyperparameters in `train.py`

## Citation

If you use this model in your research, please cite:

```
@misc{EmotionNet,
  author = {Your Name},
  title = {EmotionNet: Attention-Based Facial Emotion Recognition},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/EmotionNet}}
}
``` 