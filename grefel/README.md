# GReFEL: Geometry-Aware Reliable Facial Expression Learning

This is an implementation of the GReFEL paper for the FERPlus dataset. The implementation includes geometry-aware attention and reliability balancing modules as described in the paper.

## Features

- Vision Transformer backbone with geometry-aware attention
- Reliability balancing module with learnable anchors
- Center loss and anchor loss for better feature learning
- Mixed precision training for faster training
- Wandb integration for experiment tracking
- Data augmentation using Albumentations

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The FERPlus dataset should be organized with the following structure:
```
FERPlus-master/
├── data/
│   ├── train.csv
│   └── val.csv
```

Each CSV file should contain the following columns:
- `pixels`: Space-separated pixel values of the 48x48 grayscale image
- `emotion`: Emotion label (0-7)

## Training

To train the model:

```bash
python train.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --output_dir outputs \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --num_anchors 10
```

### Arguments

- `--train_csv`: Path to training CSV file
- `--val_csv`: Path to validation CSV file
- `--output_dir`: Directory to save outputs (default: 'outputs')
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-2)
- `--num_classes`: Number of emotion classes (default: 8)
- `--num_anchors`: Number of anchors for reliability balancing (default: 10)

## Model Architecture

The model consists of three main components:

1. **Vision Transformer Backbone**: Pretrained ViT-Base model for feature extraction
2. **Geometry-Aware Attention**: Custom attention mechanism that considers facial geometry
3. **Reliability Balancing Module**: Uses learnable anchors to improve prediction reliability

## Monitoring

Training progress can be monitored using Weights & Biases (wandb). The following metrics are tracked:
- Training loss and accuracy
- Validation loss and accuracy
- Learning rate
- Confusion matrices

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{wasi2024grefel,
  title={GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution},
  author={Wasi, Azmine Toushik and Rafi, Taki Hasan and Islam, Raima and Šerbetar, Karlo and Chae, Dong-Kyu},
  journal={arXiv preprint arXiv:2410.15927},
  year={2024}
}
``` 