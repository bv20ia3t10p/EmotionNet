#!/usr/bin/env python3
"""
Enhanced SOTA EmotionNet Training - 79%+ Target
Unified training script with quick start and full functionality
"""

import torch
import warnings
import os
import sys
import shutil
warnings.filterwarnings('ignore')

from config_loader import load_config, save_config, print_config_summary
from trainer import Trainer
from emotion_model import create_emotion_model
from dataset import get_data_loaders
from utils import set_seed, get_device, print_model_summary


def cleanup_previous_runs():
    """Clean up previous training artifacts."""
    items_cleaned = []
    
    # Clean epoch stats
    epoch_stats_dir = "checkpoints/epoch_stats"
    if os.path.exists(epoch_stats_dir):
        shutil.rmtree(epoch_stats_dir)
        items_cleaned.append("epoch stats")
    
    # Clean epoch checkpoints
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        for file in os.listdir(checkpoints_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
                os.remove(os.path.join(checkpoints_dir, file))
                if "epoch checkpoints" not in items_cleaned:
                    items_cleaned.append("epoch checkpoints")
        
        # Clean best model checkpoint for fresh start
        best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            items_cleaned.append("best model")
        
        # Clean config files from previous runs
        configs_dir = os.path.join(checkpoints_dir, 'configs')
        if os.path.exists(configs_dir):
            shutil.rmtree(configs_dir)
            items_cleaned.append("config files")
    
    if items_cleaned:
        print(f"🗑️  Cleared: {', '.join(items_cleaned)}")
    else:
        print("🗑️  No previous artifacts to clean")


def print_startup_banner():
    """Print startup banner."""
    print("🚀 Enhanced SOTA EmotionNet Training - 79%+ Target")
    print("="*70)
    print("SOTA FEATURES ACTIVE:")
    print("✅ SAM Optimizer: Sharpness Aware Minimization")
    print("✅ OneCycleLR Scheduler: Optimal learning rate scheduling")
    print("✅ Label Smoothing: 0.1 smoothing to prevent overconfidence")
    print("✅ Gradient Accumulation: Effective batch size of 64")
    print("✅ Extended Training: 300 epochs for maximum performance")
    print("✅ MixUp & CutMix: α=0.4, α=0.5 for better regularization")
    print("✅ RandAugment: Automatic augmentation policy")
    print("✅ Test Time Augmentation: 5 transforms for enhanced inference")
    print("✅ Class-Specific Augmentation: Fear (95%), Sad (90%)")
    print("✅ Enhanced Model Architecture: Multi-scale attention & fusion")
    print("="*70)


def print_training_targets():
    """Print training targets."""
    print("\n🎯 Training Targets:")
    print("- Target Validation Accuracy: >79% (beat ResEmoteNet SOTA)")
    print("- Stable gradient flow with SAM optimizer")
    print("- Optimal learning rate scheduling with OneCycleLR")
    print("- Robust generalization with advanced augmentation")
    print("- Class balance with targeted augmentation")
    print("- Extended training for maximum convergence")


def quick_start():
    """Quick start function for simple usage."""
    print("🚀 Enhanced SOTA EmotionNet Training - Quick Start")
    print("="*60)
    print("Optimized Features: SAM, OneCycleLR, Label Smoothing, TTA")
    print("Target: Beat ResEmoteNet's 79% SOTA accuracy on FER2013")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
    else:
        print("💻 Using CPU (training will be slower)")
    
    print("\n🏁 Starting Optimized SOTA Training...")
    return main()


def main():
    """Main training function with enhanced SOTA features."""
    # Check if this is a quick start call
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--quick"):
        # Print startup information
        print_startup_banner()
    
    # Clean up previous runs
    cleanup_previous_runs()
    print_training_targets()
    
    try:
        # Load configuration
        config = load_config()
        print_config_summary(config)
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Get device
        device = get_device()
        
        # Save configuration
        save_config(config)
        
        # Create data loaders
        print("\n📊 Loading FER2013 dataset with enhanced augmentation...")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_csv=config['train_csv'],
            val_csv=config['val_csv'],
            test_csv=config['test_csv'],
            img_dir=config['img_dir'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers'],
            use_weighted_sampler=config['use_weighted_sampler']
        )
        
        print(f"   - Training samples: {len(train_loader.dataset):,}")
        print(f"   - Validation samples: {len(val_loader.dataset):,}")
        print(f"   - Effective batch size: {config['batch_size'] * config.get('gradient_accumulation_steps', 1)}")
        
        # Create model
        print("\n🤖 Creating Enhanced SOTA EmotionNet model...")
        model = create_emotion_model(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
        
        # Print model summary
        print_model_summary(model, input_size=(1, 1, config['img_size'], config['img_size']))
        
        # Create trainer
        trainer = Trainer(model, device, config)
        
        # Start training
        print(f"\n🚀 Starting Enhanced Training for {config['epochs']} epochs...")
        best_metrics = trainer.train(train_loader, val_loader)
        
        # Print final results
        print(f"\n🎉 Enhanced Training completed successfully!")
        print(f"🎯 Final SOTA Results:")
        print(f"   - Best Validation Accuracy: {best_metrics['val_acc']:.2f}%")
        print(f"   - Best Validation F1-Score: {best_metrics['val_f1']:.4f}")
        print(f"   - Achieved at Epoch: {best_metrics['epoch']}")
        
        # Check if we beat SOTA
        if best_metrics['val_acc'] > 79.0:
            print(f"🏆 CONGRATULATIONS! Beat ResEmoteNet SOTA (79%) with {best_metrics['val_acc']:.2f}%!")
        else:
            print(f"📈 Current best: {best_metrics['val_acc']:.2f}% | SOTA target: 79.0%")
            remaining = 79.0 - best_metrics['val_acc']
            print(f"💪 Need {remaining:.2f}% more to beat SOTA")
        
        return best_metrics
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("Check the error logs above for details")
        raise


if __name__ == "__main__":
    # Support both quick start and normal usage
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        best_metrics = quick_start()
    else:
        best_metrics = main()
    
    # Summary for quick start users
    if best_metrics and len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print(f"\n🎯 Quick Start Summary:")
        print(f"Best Accuracy: {best_metrics['val_acc']:.2f}%")
        
        if best_metrics['val_acc'] > 79.0:
            print(f"🏆 SUCCESS! Beat SOTA by {best_metrics['val_acc'] - 79.0:.2f}%")
        else:
            remaining = 79.0 - best_metrics['val_acc']
            print(f"📈 Close! Need {remaining:.2f}% more to beat SOTA") 