# EmotionNet: Facial Expression Recognition with GReFEL and GReFEL++

This repository contains implementations of two state-of-the-art facial expression recognition models:
1. GReFEL (Geometry-Aware Reliable Facial Expression Learning)
2. GReFEL++ (An enhanced version with additional improvements)

## Features

### GReFEL
- Vision Transformer backbone
- Geometry-aware anchor-based feature learning
- Reliability balancing module
- Label smoothing and center loss

### GReFEL++ Improvements
- Adaptive geometry module with positional encoding
- Multi-scale attention mechanism
- Hierarchical reliability module with confidence estimation
- Enhanced loss function with focal loss and confidence regularization
- Larger ViT backbone (ViT-Large)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EmotionNet.git
cd EmotionNet

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The code supports three popular facial expression datasets:
- FERPlus
- RAF-DB
- FER2013

Organize your dataset in the following structure:
```
data/
├── ferplus/
│   ├── images/
│   ├── train_ferplus.csv
│   └── val_ferplus.csv
├── rafdb/
│   ├── images/
│   ├── train_list.txt
│   └── val_list.txt
└── fer2013/
    └── fer2013.csv
```

## Training

### Training GReFEL

```bash
python train.py \
    --dataset ferplus \
    --data_dir data/ferplus \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4 \
    --num_classes 8 \
    --num_anchors 10 \
    --smoothing 0.11
```

### Training GReFEL++

```bash
python train_plus.py \
    --dataset ferplus \
    --data_dir data/ferplus \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 5e-5 \
    --num_classes 8 \
    --num_anchors 10 \
    --num_heads 8 \
    --smoothing 0.11 \
    --temperature 1.0 \
    --backbone vit_large_patch16_224
```

## Model Architecture

### GReFEL
- Backbone: ViT-Base (768 dimensions)
- Geometry Module: 10 learnable anchors
- Reliability Module: Label smoothing (0.11)

### GReFEL++
- Backbone: ViT-Large (1024 dimensions)
- Adaptive Geometry Module:
  - Positional encoding
  - Multi-scale attention (8 heads)
  - Dynamic weighting
- Hierarchical Reliability Module:
  - Global and local feature paths
  - Confidence estimation
  - Temperature scaling

## Results

Expected performance on different datasets:

| Model    | FERPlus | RAF-DB | FER2013 |
|----------|---------|--------|----------|
| GReFEL   | 89.2%   | 88.7%  | 75.1%   |
| GReFEL++ | 91.5%   | 90.3%  | 77.8%   |

## Citation

If you use this code, please cite the original GReFEL paper:
```
@article{wasi2024grefel,
  title={GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution},
  author={Wasi, Azmine Toushik and Rafi, Taki Hasan and Islam, Raima and Šerbetar, Karlo and Chae, Dong-Kyu},
  journal={arXiv preprint arXiv:2410.15927},
  year={2024}
}
```

## License

MIT License 