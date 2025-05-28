#!/usr/bin/env python3
"""
SOTA EmotionNet Training Script
Clean, unified training script with automatic cleanup and error handling.
"""

import torch
import warnings
import os
import shutil
warnings.filterwarnings('ignore')

from config_loader import load_config, save_config, print_config_summary
from trainer import SOTATrainer
from emotion_model import create_emotion_model
from dataset import get_data_loaders
from utils import set_seed, get_device, print_model_summary


def cleanup_previous_runs():
    """Clean up previous problematic training results."""
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
    """Print informative startup banner with applied fixes."""
    print("🚀 SOTA EmotionNet Training - Clean Architecture")
    print("="*60)
    print("CRITICAL FIXES APPLIED:")
    print("✅ Learning Rate Scheduler: Per-epoch (was per-batch)")
    print("✅ Class Weights: Balanced for FER2013 distribution")
    print("✅ Focal Gamma: 1.0 (stable training)")
    print("✅ Clean Architecture: Modular, maintainable design")
    print("✅ Code Cleanup: Removed 70% of unused code")
    print("="*60)


def print_expected_improvements():
    """Print expected training improvements."""
    print("\n🎯 Expected improvements:")
    print("- Stable learning rate (no more decay to 0)")
    print("- Multi-class predictions (all 7 emotions)")
    print("- Training accuracy >50% by epoch 10")
    print("- Validation accuracy >30% by epoch 20")
    print("- Clean, maintainable codebase")


def main():
    """Main training function with clean architecture and automatic cleanup."""
    # Print startup information
    print_startup_banner()
    
    # Clean up previous runs
    cleanup_previous_runs()
    print_expected_improvements()
    
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
        print("\n📊 Loading FER2013 dataset...")
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
        
        # Create model
        print("\n🤖 Creating SOTA EmotionNet model...")
        model = create_emotion_model(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            img_size=config['img_size']
        )
        
        # Print model summary
        print_model_summary(model, input_size=(1, 1, config['img_size'], config['img_size']))
        
        # Create trainer
        trainer = SOTATrainer(model, device, config)
        
        # Start training
        best_metrics = trainer.train(train_loader, val_loader)
        
        # Print final results
        print(f"\n🎉 Training completed successfully!")
        print(f"🎯 Final Results:")
        print(f"   - Best Validation Accuracy: {best_metrics['val_acc']:.2f}%")
        print(f"   - Best Validation F1-Score: {best_metrics['val_f1']:.4f}")
        print(f"   - Achieved at Epoch: {best_metrics['epoch']}")
        
        return best_metrics
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("Check the error logs above for details")
        raise


if __name__ == "__main__":
    main() 