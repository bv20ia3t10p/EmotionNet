# ConvNeXtEmoteNet Model Training Optimizations

## Overview of Changes

The training script has been optimized with the following objectives:
1. Increase batch size from 48 to 64 (with an effective batch size of 128)
2. Improve model generalization and prevent early validation accuracy plateau
3. Balance regularization to prevent overfitting without limiting model capacity
4. Optimize learning dynamics with better learning rate scheduling

## Key Modifications

### Training Parameters
- **Batch Size**: Increased from 48 to 64
- **Learning Rate**: Reduced from 0.0001 to 0.00005 for stability with larger batch
- **Gradient Accumulation**: Added 2 steps accumulation for effective batch size of 128

### Optimization Strategy
- **Extended Warmup**: Increased from 10 to 15 epochs
- **Backbone Freezing**: Extended from 2 to 3 epochs
- **Learning Rate Cycles**: Increased from 4 to 6 for more adaptive learning
- **SWA (Stochastic Weight Averaging)**: Start earlier at epoch 30 (was 40)
- **Self-Distillation**: Start earlier at epoch 20 (was 30)

### Loss Function & Learning Objectives
- **Focal Loss**: Enabled to focus on harder examples
- **Label Smoothing**: Reduced from 0.25 to 0.2 
- **KL Divergence Weight**: Increased from 0.3 to 0.35
- **Self-Distillation Parameters**: Adjusted temperature and alpha for better knowledge transfer

### Regularization Balance
- **Weight Decay**: Reduced from 0.001 to 0.0005
- **Dropout Rates**: Reduced head dropout from 0.6 to 0.5 and feature dropout from 0.4 to 0.3
- **Gradient Clipping**: Increased from 1.0 to 1.5 for better gradient flow

### Data Augmentation
- **MixUp/CutMix**: Slightly reduced probabilities and alpha values for better stability
- **Progressive Augmentation**: Extended all phase durations to allow more training with each augmentation level

### Training Duration
- **Early Stopping Patience**: Increased from 35 to 50 epochs to allow more exploration

## Expected Improvements

These changes should lead to:
1. More stable training with larger batch size and accumulation
2. Better generalization due to optimized regularization and augmentation
3. Continued improvement in validation accuracy beyond epoch 12
4. Faster convergence to higher accuracy due to better learning dynamics 