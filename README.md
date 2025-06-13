# EmotionNet: Facial Expression Recognition with GReFEL

This repository contains an implementation of GReFEL (Geometry-Aware Reliable Facial Expression Learning) for facial expression recognition.

## Features

### GReFEL Architecture
- Multi-scale Window-Based Cross-Attention ViT with 3 scales:
  - Low-level (local): 28×28 patches
  - Mid-level: 14×14 patches
  - High-level (global): 7×7 patches
- IR50 backbone pretrained on Ms-Celeb-1M
- MobileFaceNet backbone for landmarks
- Geometry-aware anchor-based feature learning
- Reliability balancing module

### Loss Functions
Total loss is composed of:
- L_cls: Classification loss (cross-entropy)
- L_a: Anchor loss for discriminative features
- L_c: Center loss for class compactness
All loss components have equal weighting (λ=1.0)

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

### Training Settings

#### FERPlus Dataset
- Batch size: 32
- Initial learning rate: 0.0001 (1e-4)
- ADAM optimizer with weight decay 0.01
- Train for 100 epochs
- Data augmentation:
  - Resize to 256×256
  - Random crop to 224×224
  - Random rotation (±30°)
  - Color jittering (0.4)
  - Random shear (±15°)
  - Random perspective (0.3)

#### Other Datasets (RAF-DB, FER2013)
- Batch size: 500 images per class
- Initial learning rate: 0.0003
- ADAM optimizer with exponential decay (γ=0.995)
- Train for 1000 epochs

### Training Command

```bash
python train.py \
    --dataset ferplus \
    --data_dir data/ferplus \
    --num_epochs 1000 \
    --lr 3e-4 \
    --num_classes 8 \
    --num_anchors 10 \
    --smoothing 0.11
```

## Expected Performance

| Dataset  | Accuracy |
|----------|----------|
| FERPlus  | 89.2%    |
| RAF-DB   | 88.7%    |
| FER2013  | 75.1%    |

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