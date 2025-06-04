# GReFEL: Geometry-Aware Reliable Facial Expression Learning

[![arXiv](https://img.shields.io/badge/arXiv-2410.15927-b31b1b.svg)](https://arxiv.org/abs/2410.15927)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **GReFEL** (Geometry-Aware Reliable Facial Expression Learning), as described in our paper:

> **GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution**  
> Azmine Toushik Wasi*, Taki Hasan Rafi*, Raima Islam, Karlo Šerbetar, Dong-Kyu Chae  
> *Co-first authors

## Overview

GReFEL is a novel facial expression recognition framework that addresses key challenges in real-world scenarios:

- **Geometry-Aware Processing**: Leverages facial landmark geometry for enhanced feature understanding
- **Reliability Balancing**: Combines anchor-based and attention-based corrections for robust predictions
- **Multi-Scale Feature Learning**: Uses Vision Transformers with window-based cross-attention
- **Bias Mitigation**: Handles class imbalance and dataset bias through advanced label correction

### Key Features

🎯 **State-of-the-art Performance**: Achieves superior results on multiple benchmark datasets  
🔄 **Reliability Balancing**: Novel approach combining geometric and attentive corrections  
🌐 **Multi-Scale Learning**: Hierarchical feature extraction at different spatial resolutions  
⚡ **Efficient Architecture**: Optimized for both accuracy and computational efficiency  

## Architecture

![GReFEL Architecture](assets/grefel_architecture.png)

The GReFEL framework consists of three main components:

1. **Multi-Scale Feature Extractor**: Window-based cross-attention Vision Transformer
2. **Geometry-Aware Anchors**: Trainable anchor points in embedding space
3. **Reliability Balancing**: Combines anchor-based and attention-based corrections

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- CMake (for dlib installation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/grefel-implementation.git
cd grefel-implementation
```

2. Create a virtual environment:
```bash
python -m venv grefel_env
source grefel_env/bin/activate  # On Windows: grefel_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Additional Setup for Face Recognition

For facial landmark extraction, you may need to install additional dependencies:

```bash
# For dlib (if installation fails)
conda install -c conda-forge dlib

# Alternative: use pre-compiled wheels
pip install dlib-binary
```

## Dataset Preparation

### FER2013/FERPlus Dataset

1. Download the FER2013+ dataset from [Microsoft FERPlus](https://github.com/Microsoft/FERPlus)
2. Place the `fer2013new.csv` file in the `FERPlus-master/` directory
3. The dataset will be automatically split into training and validation sets

### Custom Dataset

For custom datasets, ensure your CSV file has the following format:
```csv
emotion,pixels,Usage
0,"129 128 126 ...",Training
1,"132 135 133 ...",PublicTest
...
```

Or for image files:
```csv
emotion,image,Usage
0,image1.jpg,Training
1,image2.jpg,PublicTest
...
```

## Usage

### Training

To train the GReFEL model:

```bash
python train_grefel.py
```

**Configuration options** (modify in `train_grefel.py`):
```python
config = {
    'num_classes': 8,           # Number of emotion classes
    'batch_size': 32,           # Training batch size
    'num_epochs': 100,          # Training epochs
    'embed_dim': 512,           # Feature embedding dimension
    'num_heads': 8,             # Number of attention heads
    'depth': 6,                 # Transformer depth
    'num_anchors_per_class': 10, # Anchors per emotion class
    'device': 'cuda'            # Device (cuda/cpu)
}
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate_grefel.py
```

### Inference on Single Image

```python
from evaluate_grefel import GReFELEvaluator

# Load trained model
evaluator = GReFELEvaluator('grefel_best_model.pth')

# Predict emotion for a single image
result = evaluator.predict_single_image('path/to/image.jpg')
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Quick Start Example

```python
import torch
from grefel_implementation import GReFELModel, create_dummy_data

# Create model
model = GReFELModel(num_classes=8, embed_dim=512)

# Generate dummy data for testing
images, landmarks, labels = create_dummy_data(batch_size=4)

# Forward pass
outputs = model(images, landmarks)
predictions = torch.argmax(outputs['final_probs'], dim=-1)

print(f"Predictions: {predictions}")
print(f"Probabilities: {outputs['final_probs']}")
```

## Model Components

### 1. Window-Based Cross-Attention ViT

```python
class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction with Vision Transformer"""
    
    def __init__(self, img_size=224, patch_sizes=[28, 14, 7], 
                 embed_dim=512, num_heads=8, depth=6):
        # Implementation details...
```

### 2. Geometry-Aware Anchors

```python
class GeometryAwareAnchors(nn.Module):
    """Geometry-aware anchor points for reliability balancing"""
    
    def __init__(self, num_classes, num_anchors_per_class, embed_dim):
        # Implementation details...
```

### 3. Reliability Balancing Module

```python
class ReliabilityBalancing(nn.Module):
    """Reliability balancing with anchor and attention correction"""
    
    def __init__(self, num_classes, embed_dim, num_anchors_per_class=10):
        # Implementation details...
```

## Performance

### Benchmark Results

| Dataset | GReFEL | POSTER++ | LA-Net | TransFER |
|---------|--------|----------|--------|----------|
| AffectNet | **68.02%** | 63.76% | 64.54% | 66.23% |
| AffWild2 | **72.48%** | 69.18% | 66.76% | 68.92% |
| RAF-DB | **92.47%** | 92.21% | 91.56% | 90.91% |
| FER+ | **93.09%** | 92.28% | 91.78% | - |

### Key Improvements

- ✅ **+4.26%** improvement over POSTER++ on AffectNet
- ✅ **+3.30%** improvement over POSTER++ on AffWild2  
- ✅ **+0.26%** improvement over POSTER++ on RAF-DB
- ✅ **+0.81%** improvement over POSTER++ on FER+

## Advanced Features

### Reliability Balancing Analysis

The framework provides detailed analysis of the reliability balancing mechanism:

```python
# Analyze the effect of reliability balancing
reliability_analysis = evaluator.analyze_reliability_balancing(probabilities_data)
print(f"Predictions changed by reliability balancing: {reliability_analysis['change_rate']:.2%}")
```

### Confidence Calibration

GReFEL includes confidence estimation and calibration analysis:

```python
# Analyze confidence distribution
evaluator.analyze_confidence_distribution(confidences, predictions, labels)
```

### Visualization Tools

Built-in visualization for:
- Confusion matrices
- Class-wise performance metrics
- Training history plots
- Confidence distribution analysis

## File Structure

```
grefel-implementation/
├── grefel_implementation.py    # Core GReFEL model implementation
├── train_grefel.py            # Training script and utilities
├── evaluate_grefel.py         # Evaluation and inference tools
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── FERPlus-master/           # Dataset directory
│   └── fer2013new.csv       # FER2013+ dataset
└── assets/                   # Documentation assets
    └── grefel_architecture.png
```

## Configuration

### Model Hyperparameters

The model supports extensive customization:

```python
# Architecture parameters
embed_dim = 512              # Feature embedding dimension
num_heads = 8               # Multi-head attention heads
depth = 6                   # Transformer encoder depth
patch_sizes = [28, 14, 7]   # Multi-scale patch sizes
num_anchors_per_class = 10  # Geometry-aware anchors per class

# Training parameters
lambda_cls = 1.0            # Classification loss weight
lambda_anchor = 1.0         # Anchor loss weight  
lambda_center = 1.0         # Center loss weight
learning_rate = 3e-4        # AdamW learning rate
weight_decay = 0.01         # L2 regularization
```

### Data Augmentation

Heavy augmentation pipeline for robust training:

```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{wasi2024grefel,
  title={GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution},
  author={Wasi, Azmine Toushik and Rafi, Taki Hasan and Islam, Raima and Šerbetar, Karlo and Chae, Dong-Kyu},
  journal={arXiv preprint arXiv:2410.15927},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Microsoft FERPlus team for the enhanced FER2013 dataset
- The authors of POSTER and POSTER++ for baseline comparisons
- PyTorch team for the excellent deep learning framework

## Contact

For questions or issues, please contact:
- Azmine Toushik Wasi: [azmine.wasi@example.com]
- Taki Hasan Rafi: [taki.rafi@example.com]
- Dong-Kyu Chae: [dongkyu@hanyang.ac.kr]

## Updates

- **2024-10**: Initial release of GReFEL implementation
- **2024-10**: Added comprehensive evaluation tools
- **2024-10**: Improved documentation and examples

---

**Keywords**: Facial Expression Recognition, Vision Transformer, Reliability Balancing, Geometry-Aware Learning, Multi-Scale Features, Bias Mitigation 