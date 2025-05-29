# Enhanced SOTA EmotionNet - 79%+ Target

🚀 **Optimized emotion recognition model designed to beat ResEmoteNet's 79% SOTA accuracy on FER2013**

## Features

### ✅ Advanced Architecture
- **Enhanced SOTA EmotionNet**: Multi-scale attention architecture with EfficientNet backbone
- **Attention Mechanisms**: SE, CBAM, Coordinate, and Multi-Head Self-Attention
- **Multi-scale Feature Fusion**: Pyramid pooling and cross-scale connections
- **Emotion-specific Classification**: Multi-task learning with auxiliary outputs

### ✅ Advanced Training Techniques
- **SAM Optimizer**: Sharpness Aware Minimization for better generalization
- **OneCycleLR Scheduler**: Optimal learning rate scheduling
- **Label Smoothing**: 0.1 smoothing to prevent overconfidence
- **Gradient Accumulation**: Effective batch size of 64
- **Extended Training**: 300 epochs for maximum performance

### ✅ Advanced Augmentation
- **MixUp & CutMix**: α=0.4, α=0.5 for better regularization
- **RandAugment**: Automatic augmentation policy optimization
- **Class-Specific Augmentation**: Fear (95%), Sad (90%) for class balance
- **Test Time Augmentation**: 5 transforms for enhanced inference

### ✅ Optimized Configuration
- **Pre-tuned Hyperparameters**: All settings optimized for FER2013
- **Class Weights**: Balanced for optimal emotion recognition
- **Clean Architecture**: Simplified, maintainable codebase

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Quick start with all optimizations
python run_enhanced_training.py

# Or run directly
python main.py
```

## Project Structure

```
EmotionNet/
├── main.py                    # Main training script
├── run_enhanced_training.py   # Quick start script
├── config.py                  # Optimized configuration
├── config_loader.py           # Configuration loader
├── trainer.py                 # Enhanced SOTA trainer
├── emotion_model.py           # Model factory
├── enhanced_emotion_model.py  # Enhanced model architecture
├── sam_optimizer.py           # SAM optimizer implementation
├── dataset.py                 # Data loading and augmentation
├── attention_modules.py       # Attention mechanisms
├── loss_functions.py          # Loss functions
├── metrics.py                 # Evaluation metrics
├── checkpoint_manager.py      # Model checkpointing
├── utils.py                   # Utility functions
└── requirements.txt           # Dependencies
```

## Configuration

The configuration is optimized for 79%+ accuracy with these key settings:

- **Model**: Enhanced EmotionNet with EfficientNet backbone
- **Input Size**: 48x48 grayscale images
- **Batch Size**: 32 (Effective: 64 with gradient accumulation)
- **Learning Rate**: 0.005 → 0.01 (OneCycleLR)
- **Epochs**: 300
- **Optimizer**: SAM (ρ=0.05)
- **Loss**: Label Smoothing (0.1)
- **Augmentation**: MixUp, CutMix, RandAugment, Class-specific

## Results Target

- **Goal**: Beat ResEmoteNet's 79% SOTA accuracy on FER2013
- **Features**: All state-of-the-art techniques for emotion recognition
- **Architecture**: Multi-scale attention with advanced optimization

## Training Progress

The training will automatically:
1. Clean previous artifacts
2. Load optimized configuration
3. Create enhanced model with attention mechanisms
4. Apply advanced training techniques (SAM, OneCycleLR, etc.)
5. Use sophisticated augmentation strategies
6. Save best checkpoints with TTA validation
7. Report progress toward 79%+ target

## Key Improvements

1. **Simplified Architecture**: Removed redundant code, kept only the best configuration
2. **Optimized Training**: SAM optimizer + OneCycleLR for superior convergence
3. **Advanced Augmentation**: Class-specific strategies for better balance
4. **Clean Codebase**: Maintainable, well-documented structure
5. **SOTA Target**: All features aimed at beating 79% accuracy

## Usage Examples

### Basic Training
```python
from main import main
best_metrics = main()
print(f"Best accuracy: {best_metrics['val_acc']:.2f}%")
```

### Custom Configuration
```python
from config import get_config
from trainer import Trainer
from emotion_model import create_emotion_model

config = get_config()
config['epochs'] = 500  # Extend training
model = create_emotion_model()
trainer = Trainer(model, device, config)
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- timm (for EfficientNet backbone)
- Other dependencies in requirements.txt

## License

This project is optimized for academic and research use in emotion recognition. 