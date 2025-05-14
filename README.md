# Advanced Emotion Recognition for FER2013

This repository contains a state-of-the-art implementation for facial emotion recognition on the FER2013 dataset, achieving close to 80% accuracy using advanced deep learning techniques.

## Model Architecture

Our implementation (`SOTAResEmote`) combines multiple state-of-the-art techniques from top-performing models:

1. **Multi-stage Feature Fusion**: Combines features from different network depths for more comprehensive emotion understanding
2. **Advanced Attention Mechanisms**:
   - CBAM (Convolutional Block Attention Module) for feature refinement
   - Facial Region Attention for focusing on important areas like eyes, mouth, etc.
   - Anchor Attention for landmark-based feature enhancement
3. **Transformer-based Refinement**: Vision transformer layers to capture global dependencies
4. **GeM Pooling**: Generalized Mean Pooling for better feature aggregation
5. **Auxiliary Loss**: Multi-stage supervision for better gradient flow
6. **Robust Training**: Label smoothing, stochastic depth, and other regularization techniques

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- timm
- sklearn
- matplotlib
- seaborn

Install requirements:
```bash
pip install -r requirements.txt
```

## Dataset

Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and extract it to a directory of your choice.

## Training

To train the model on FER2013, run:

```bash
./train_fer2013.sh
```

This script will:
1. Set up appropriate hyperparameters
2. Train a SOTAResEmote model with ResNet34 backbone
3. Use advanced data augmentation techniques
4. Automatically evaluate the model after training

### Model Variants

We provide three model variants:
- `sota_resemote_small`: Based on ResNet18 (faster, but less accurate)
- `sota_resemote_medium`: Based on ResNet34 (balanced performance)
- `sota_resemote_large`: Based on ResNet50 (highest accuracy, but slower)

To choose a specific variant, modify the `ARCHITECTURE` variable in the training script.

## Evaluation

To evaluate a trained model:

```bash
python emotion_net/evaluate.py \
    --dataset_name "fer2013" \
    --data_dir "/path/to/fer2013" \
    --model_path "/path/to/model/best_model.pth" \
    --architecture "sota_resemote_medium" \
    --image_size 256 \
    --batch_size 32
```

This will compute accuracy, F1 score, precision, recall, and generate a confusion matrix.

## Results

Our SOTAResEmote model achieves close to 80% accuracy on the FER2013 test set, comparable to the top-performing model (ResEmoteNet) on the [Papers with Code leaderboard](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013).

## Acknowledgements

This implementation includes techniques inspired by several state-of-the-art models:
- ResEmoteNet
- VGG-Segmentation
- LHC-Net
- ResMaskingNet

## License

MIT 