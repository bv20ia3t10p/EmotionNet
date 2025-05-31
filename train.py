#!/usr/bin/env python3
"""
EmotionNet Training Script

A unified training script for facial emotion recognition models.
"""

import os
import argparse
import torch

# Disable albumentations update check warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from emotionnet import Config, create_config, create_emotion_model, Trainer, get_data_loaders, set_seed, get_device
from emotionnet.config import create_ferplus_config
from emotionnet.data import get_ferplus_data_loaders


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EmotionNet Training')
    
    # Mode selection
    parser.add_argument('--mode', choices=['default', 'quick', 'ferplus'], default='default',
                       help='Training mode: default, quick (fast), ferplus (8 emotion classes)')
    
    # Model arguments
    parser.add_argument('--model-type', default='attention_emotion_net', help='Model architecture type')
    parser.add_argument('--dropout-rate', type=float, default=0.4, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--small-batch', action='store_true', help='Use smaller batch size (16) for better learning')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--max-lr', type=float, help='Maximum learning rate for OneCycleLR')
    
    # Data arguments
    parser.add_argument('--data-dir', default='fer2013', help='Data directory')
    parser.add_argument('--use-balanced', action='store_true', help='Use balanced dataset')
    parser.add_argument('--balanced-dir', default='fer2013_balanced', help='Balanced dataset directory')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    
    # FERPlus specific arguments
    parser.add_argument('--ferplus-dir', default='FERPlus-master', help='FERPlus directory')
    parser.add_argument('--ferplus-mode', choices=['majority', 'probability', 'multi_target'], 
                       default='majority', help='FERPlus label processing mode')
    
    # Loss function arguments
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss for better handling of class imbalance')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config-file', help='Path to configuration file')
    parser.add_argument('--stochastic-depth', action='store_true', help='Use stochastic depth for regularization')
    
    return parser.parse_args()


def print_banner():
    """Print startup banner."""
    print("EmotionNet Training")
    print("="*70)
    print("Features:")
    print("- Unified training for FER2013 and FERPlus datasets")
    print("- Attention-based architectures")
    print("- Advanced data augmentation")
    print("- Learning rate scheduling")
    print("- Regularization techniques")
    print("- Class imbalance handling")
    print("="*70)


def main():
    """Main training function."""
    args = parse_arguments()
    
    print_banner()
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Get device
    device = get_device()
    
    # Create configuration
    print(f"\nCreating {args.mode} configuration...")
    
    config_overrides = {}
    if args.epochs:
        config_overrides['training.epochs'] = args.epochs
    # Use smaller batch size if requested
    if args.small_batch:
        config_overrides['training.batch_size'] = 16  # Reduced batch size for better learning
        print(f"Using smaller batch size (16) for better gradient estimates")
    elif args.batch_size:
        config_overrides['training.batch_size'] = args.batch_size
    if args.lr:
        config_overrides['training.lr'] = args.lr
    if args.max_lr:
        config_overrides['training.max_lr'] = args.max_lr
    if args.dropout_rate != 0.4:
        config_overrides['model.dropout_rate'] = args.dropout_rate
    if args.data_dir != 'fer2013':
        config_overrides['data.img_dir'] = args.data_dir
    if args.use_balanced:
        config_overrides['data.use_balanced_dataset'] = True
        config_overrides['data.balanced_dir'] = args.balanced_dir
    if args.checkpoint_dir != 'checkpoints':
        config_overrides['data.checkpoint_dir'] = args.checkpoint_dir
    
    # Set up loss function
    if args.focal_loss:
        config_overrides['loss.use_focal_loss'] = True
        config_overrides['loss.gamma'] = args.focal_gamma
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
        
    # Create configuration based on mode
    if args.mode == 'ferplus':
        config_overrides['ferplus.mode'] = args.ferplus_mode
        config_overrides['data.img_dir'] = args.ferplus_dir
        config = create_ferplus_config(**config_overrides)
        print(f"FERPlus Training - 8 emotion classes")
        print(f"Label mode: {args.ferplus_mode}")
    else:
        config = create_config(
            config_type=args.mode,
            config_path=args.config_file,
            **config_overrides
        )
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"   - Mode: {args.mode}")
    print(f"   - Architecture: {config.model.backbone}")
    print(f"   - Image Size: {config.model.img_size}x{config.model.img_size}")
    print(f"   - Epochs: {config.training.epochs}")
    print(f"   - Batch Size: {config.training.batch_size}")
    if hasattr(config.training, 'max_lr'):
        print(f"   - Learning Rate: {config.training.lr} -> {config.training.max_lr}")
    else:
        print(f"   - Learning Rate: {config.training.lr}")
    if hasattr(config.model, 'dropout_rate'):
        print(f"   - Dropout Rate: {config.model.dropout_rate}")
    if hasattr(config, 'loss') and hasattr(config.loss, 'label_smoothing'):
        print(f"   - Label Smoothing: {config.loss.label_smoothing}")
    
    # Create data loaders
    if args.mode == 'ferplus':
        print(f"\nLoading FERPlus dataset...")
        train_loader, val_loader, test_loader, class_distribution = get_ferplus_data_loaders(
            data_dir=args.ferplus_dir,
            batch_size=config.training.batch_size,
            img_size=config.model.img_size,
            num_workers=config.data.num_workers,
            mode=args.ferplus_mode
        )
        # Store class distribution in config for loss function
        config.data.class_counts = class_distribution['class_counts']
        print(f"   Class distribution: {class_distribution['class_counts']}")
        
    elif hasattr(config.data, 'use_balanced_dataset') and config.data.use_balanced_dataset:
        print(f"\nLoading Balanced FER2013 dataset...")
        print(f"   Using HYBRID approach:")
        print(f"   - Training: Balanced dataset from {config.data.balanced_dir}")
        print(f"   - Validation/Test: Original distribution from fer2013")
        from emotionnet.data.balanced_loaders import get_balanced_data_loaders
        train_loader, val_loader, test_loader = get_balanced_data_loaders(
            balanced_dir=config.data.balanced_dir,
            batch_size=config.training.batch_size,
            img_size=config.model.img_size,
            num_workers=config.data.num_workers
        )
    else:
        print(f"\nLoading FER2013 dataset...")
        print(f"   Using original dataset from: {config.data.img_dir}")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_csv=config.data.train_csv,
            val_csv=config.data.val_csv,
            test_csv=config.data.test_csv,
            img_dir=config.data.img_dir,
            batch_size=config.training.batch_size,
            img_size=config.model.img_size,
            num_workers=config.data.num_workers,
            use_weighted_sampler=config.data.use_weighted_sampler
        )
    
    # Swap validation and test sets as requested
    print(f"   Swapping validation and test sets...")
    val_loader, test_loader = test_loader, val_loader
    
    print(f"   - Training samples: {len(train_loader.dataset):,}")
    print(f"   - Validation samples: {len(val_loader.dataset):,}")
    print(f"   - Test samples: {len(test_loader.dataset):,}")
    if hasattr(config.training, 'gradient_accumulation_steps'):
        effective_batch = config.training.batch_size * config.training.gradient_accumulation_steps
        print(f"   - Effective batch size: {effective_batch}")
    else:
        print(f"   - Batch size: {config.training.batch_size}")
    
    # Create model
    print(f"\nCreating emotion recognition model...")
    
    model = create_emotion_model(
        model_type=args.model_type,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        use_pretrained_backbone=getattr(config.model, 'use_pretrained_backbone', False),
        backbone_type=getattr(config.model, 'backbone', 'attention_emotion_net'),
        input_size=config.model.img_size,
        stochastic_depth=args.stochastic_depth,
        training_mode=getattr(config.ferplus, 'mode', 'majority') if args.mode == 'ferplus' else None
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: {total_params * 4 / (1024*1024):.2f} MB")
    
    # Create trainer
    print(f"\nInitializing Trainer...")
    trainer = Trainer(model, device, config)
    print(f"\nStarting training for {config.training.epochs} epochs...")
    print("-" * 70)
    
    try:
        # Standard trainer
        best_metrics = trainer.train(train_loader, val_loader)
        
        # Print final results
        print(f"\nTraining completed successfully!")
        print(f"\nFinal Results:")
        print(f"   - Best Validation Accuracy: {best_metrics['val_acc']:.2f}%")
        print(f"   - Best Validation F1-Score: {best_metrics['val_f1']:.4f}")
        print(f"   - Achieved at Epoch: {best_metrics['epoch']}")
        
        return best_metrics
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise


if __name__ == "__main__":
    best_metrics = main()
    
    if best_metrics:
        print(f"\n✅ Training Summary:")
        print(f"   Best Accuracy: {best_metrics['val_acc']:.2f}%")
    
    print("\nThank you for using EmotionNet! 🚀") 