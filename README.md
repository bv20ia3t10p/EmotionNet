# EmotionNet - Advanced Facial Emotion Recognition

EmotionNet is a state-of-the-art facial emotion recognition system designed to accurately classify emotions from facial expressions. This implementation includes enhanced models, training pipelines, and stability improvements to outperform ResEmoteNet on the FER2013 dataset.

## Features

- **Advanced Model Architecture**: State-of-the-art ResEmoteNet architecture with CBAM attention, feature fusion, and GEM pooling.
- **Robust Training**: Stabilized training loop with dynamic gradient clipping, NaN detection, and error handling.
- **Multiple Backbones**: Support for various backbone networks including ResNet, ConvNeXt, and EfficientNet.
- **Cross-Dataset Compatibility**: Works with FER2013 and RAFDB emotion datasets.
- **Performance Optimizations**: Mixed precision training, memory-efficient backpropagation, and EMA model averaging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EmotionNet.git
   cd EmotionNet
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm scikit-learn pandas opencv-python pillow matplotlib seaborn
   ```

## Dataset Preparation

### FER2013 Dataset

The FER2013 dataset should be placed in the `dataset/fer2013/` directory. You can use either:

1. CSV format (recommended):
   - `train.csv` and `test.csv` files
   - Or a single `icml_face_data.csv` file

2. Extracted image directories:
   - Organize images in class subdirectories: angry, disgust, fear, happy, neutral, sad, surprise

## Training

### Training on Windows

Run the PowerShell script:

```powershell
.\train_fer2013.ps1
```

### Training on Linux/Mac

Run the bash script:

```bash
chmod +x train_fer2013.sh
./train_fer2013.sh
```

### Custom Training

You can also run training directly with custom parameters:

```bash
python -m emotion_net.train \
    --dataset_name "fer2013" \
    --data_dir "./dataset/fer2013" \
    --model_dir "./models/fer2013_custom" \
    --architecture "sota_resemote_large" \
    --backbones "convnext_tiny" \
    --embedding_size 768 \
    --learning_rate 0.000005 \
    --batch_size 8 \
    --num_epochs 100 \
    --attention_type "cbam" \
    --class_weights \
    --sad_class_weight 1.5 \
    --warmup_epochs 15
```

## Training Configuration

The improved model uses these key settings for optimal training:

- Learning rate: 0.000005 (ultra-low for stability)
- Batch size: 8 (reduced for better gradient flow)
- Optimizer: AdamW with 0.02 weight decay
- Backbone: ConvNeXt Tiny (better performance than ResNet)
- Architecture: sota_resemote_large with 768 embedding dimension
- Loss: StableFocalLoss with 0.15 label smoothing
- Regularization: 0.1 gradient clipping, 0.1 stochastic depth, 0.2 drop path
- Scheduler: Cosine annealing with 15 warmup epochs

## Model Evaluation

Evaluate a trained model using:

```bash
python -m emotion_net.evaluate \
    --dataset_name "fer2013" \
    --data_dir "./dataset/fer2013" \
    --model_path "./models/fer2013_improved/best_model.pth" \
    --architecture "sota_resemote_large" \
    --multi_crop_inference
```

## Troubleshooting

### Common Issues

1. **Training fails with "unable to open/read file" errors**:
   - Make sure your dataset structure is correct
   - Try the CSV-based loading instead of image directory loading
   - Verify file permissions

2. **NaN values in loss or gradients**:
   - Reduce learning rate further (try 0.000001)
   - Increase gradient clipping (try 0.05)
   - Disable mixup/cutmix augmentations
   - Check for corrupted images in the dataset

3. **Out of memory errors**:
   - Reduce batch size
   - Use a smaller model (sota_resemote_medium or sota_resemote_small)
   - Enable gradient checkpointing (enabled by default)
   - Reduce image_size to 224 or 192

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FER2013 dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- RAFDB dataset: http://www.whdeng.cn/raf/model1.html 