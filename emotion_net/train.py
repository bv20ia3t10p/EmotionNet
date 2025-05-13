"""Main training script for emotion recognition model."""

import os
import torch
import argparse
import numpy as np
from emotion_net.training import EmotionTrainer
from emotion_net.data import BaseEmotionDataset, FER2013DataManager, RAFDBDataManager
from emotion_net.models.ensemble import EnsembleModel
from emotion_net.models import create_model, create_loss_fn
from emotion_net.config.constants import (
    DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_BACKBONES, CHECKPOINT_DIR, DEFAULT_IMAGE_SIZE
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train EmotionNet model')
    
    # Add Dataset Name Argument
    parser.add_argument('--dataset_name', type=str, required=True,
                      choices=['fer2013', 'rafdb'],
                      help='Name of the dataset to use (fer2013 or rafdb)')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    
    # Optional arguments with defaults
    parser.add_argument('--num_epochs', '--epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--backbones', type=str, nargs='+', default=DEFAULT_BACKBONES,
                      help='List of backbone architectures for the ensemble model or backbone for expert model')
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                      help='Size to resize images to')
    parser.add_argument('--test_dir', type=str, default=None,
                      help='Path to test data directory (optional)')
    parser.add_argument('--model_dir', type=str, default=CHECKPOINT_DIR,
                      help='Directory to save models')
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                      help='Validation split ratio for FER2013 (default: 0.1)')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                      choices=['cross_entropy', 'focal', 'hybrid'],
                      help="Type of loss function to use (default: cross_entropy)")
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help="Gamma parameter for Focal Loss (default: 2.0)")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                      help="Label smoothing factor for CrossEntropyLoss (default: 0.1)")
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                      help="Alpha parameter for Mixup (default: 0.0 to disable)")
    parser.add_argument('--cutmix_alpha', type=float, default=0.0,
                      help="Alpha parameter for CutMix (default: 0.0 to disable)")
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                      help="Drop path rate (Stochastic Depth) for backbones (default: 0.0)")
    parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                      choices=['one_cycle', 'cosine_annealing', 'none'],
                      help="Type of LR scheduler (default: one_cycle)")
    parser.add_argument('--num_workers', type=int, default=4,
                      help="Number of data loading workers (default: 4)")
    parser.add_argument('--architecture', type=str, default='ensemble',
                      choices=['ensemble', 'hierarchical', 'expert'],
                      help="Model architecture type (default: ensemble)")
    parser.add_argument('--attention_type', type=str, default=None,
                      choices=[None, 'self', 'cbam'],
                      help="Attention mechanism to use (default: None)")
    parser.add_argument('--sad_class_weight', type=float, default=1.0,
                      help="Additional weight for the 'sad' class (default: 1.0)")
    parser.add_argument('--class_weights', action='store_true',
                      help="Use class weights based on frequency (default: False)")
    parser.add_argument('--feature_fusion', action='store_true',
                      help="Use feature fusion from different layers (default: False)")
    parser.add_argument('--stochastic_depth', type=float, default=0.0,
                      help="Stochastic depth probability (default: 0.0)")
    parser.add_argument('--multi_crop_inference', action='store_true',
                      help="Use multi-crop inference for test time (default: False)")
    parser.add_argument('--use_ema', action='store_true',
                      help="Use exponential moving average of weights (default: False)")
    parser.add_argument('--triplet_margin', type=float, default=0.3,
                      help="Margin for triplet loss (default: 0.3)")
    parser.add_argument('--emotion_groups', type=str, 
                      default="sad-neutral-angry,happy-surprise,fear-disgust",
                      help="Groups of similar emotions (default: sad-neutral-angry,happy-surprise,fear-disgust)")
    parser.add_argument('--gem_pooling', action='store_true',
                      help="Use Generalized Mean Pooling (default: False)")
    parser.add_argument('--decoupled_head', action='store_true',
                      help="Use decoupled classification head (default: False)")
    parser.add_argument('--embedding_size', type=int, default=512,
                      help="Embedding size for features (default: 512)")
    parser.add_argument('--consistency_loss', action='store_true',
                      help="Add consistency regularization (default: False)")
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                      help="Freeze backbone for initial epochs (default: 0)")
    parser.add_argument('--channels_last', action='store_true',
                      help="Use channels_last memory format for performance (default: False)")
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help="Weight decay for optimizer (default: 0.0001)")
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'adamw', 'sgd'],
                      help="Optimizer to use (default: adam)")
    parser.add_argument('--grayscale_input', action='store_true',
                      help="Use grayscale input (default: False)")
    parser.add_argument('--use_amp', action='store_true',
                      help="Use automatic mixed precision (default: False)")
    parser.add_argument('--class_balanced_loss', action='store_true',
                      help="Use class balanced loss (default: False)")
    parser.add_argument('--warmup_epochs', type=int, default=0,
                      help="Number of warmup epochs (default: 0)")
    parser.add_argument('--gradient_clip', type=float, default=None,
                      help="Gradient clipping value (default: None)")
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help="Use pretrained model weights (default: True)")
    
    # Parse known args to handle both old and new formats
    args, unknown = parser.parse_known_args()
    
    return args

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Debug print
    print(f"DEBUG: Received arguments: num_epochs={args.num_epochs}, batch_size={args.batch_size}")
    
    # Set logging level for PIL to show more info
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Variable to track the created data manager for cleanup
    data_manager = None
    
    try:
        # Create model directory
        os.makedirs(args.model_dir, exist_ok=True)
        
        # --- Instantiate the appropriate Data Manager ---
        if args.dataset_name == 'fer2013':
            data_manager = FER2013DataManager(
                data_dir=args.data_dir,
                test_dir=args.test_dir,
                image_size=args.image_size,
                val_split_ratio=args.val_split_ratio
            )
        elif args.dataset_name == 'rafdb':
            data_manager = RAFDBDataManager(
                data_dir=args.data_dir,
                image_size=args.image_size
                # test_dir and val_split_ratio are not used by RAFDBDataManager
            )
        else:
            # Should be caught by argparse choices, but good practice
            raise ValueError(f"Invalid dataset_name: {args.dataset_name}") 
            
        # --- Load data using the manager ---
        print(f"\nLoading data using {type(data_manager).__name__}...")
        train_dataset, val_dataset, test_dataset, train_labels = data_manager.get_datasets()
        
        if not train_dataset:
            print("CRITICAL: train_dataset is None. Exiting.")
            return 1
        
        # Get number of classes dynamically
        num_classes = len(train_dataset.classes)
        print(f"Number of classes detected: {num_classes}")
        print(f"Class names: {train_dataset.classes}")
        
        # Print dataset sizes
        print(f"\nDataset sizes:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
        
        # Update data paths if temp directory was created
        if hasattr(data_manager, 'temp_dir') and data_manager.temp_dir:
            print(f"\nUpdating data paths to use temporary directory: {data_manager.temp_dir}")
            # If using FER2013 with CSV data, update the paths
            if args.dataset_name == 'fer2013' and args.test_dir is None:
                args.test_dir = os.path.join(data_manager.temp_dir, 'test')
                print(f"Updated test_dir to: {args.test_dir}")
                # Do not update args.data_dir as it's still needed for reference
        
        # Create training configuration
        config = {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'train_labels': train_labels,
            'patience': args.patience,
            'label_smoothing': getattr(args, 'label_smoothing', 0.1),
            'loss_type': getattr(args, 'loss_type', 'cross_entropy'),
            'focal_gamma': getattr(args, 'focal_gamma', 2.0),
            'scheduler_type': getattr(args, 'scheduler_type', 'one_cycle'),
            'dataset_name': args.dataset_name,
            'use_ema': getattr(args, 'use_ema', False),
            'model_dir': args.model_dir,
            'num_classes': num_classes,
            'num_workers': getattr(args, 'num_workers', 4),
            'mixup_alpha': getattr(args, 'mixup_alpha', 0.0),
            'cutmix_alpha': getattr(args, 'cutmix_alpha', 0.0),
            'ema_decay': getattr(args, 'ema_decay', 0.999),
            'drop_path_rate': getattr(args, 'drop_path_rate', 0.0),
            'weight_decay': getattr(args, 'weight_decay', 0.0001),
            'optimizer': getattr(args, 'optimizer', 'adam'),
            'warmup_epochs': getattr(args, 'warmup_epochs', 0),
            'gradient_clip': getattr(args, 'gradient_clip', None),
            'class_weights': getattr(args, 'class_weights', False),
            'sad_class_weight': getattr(args, 'sad_class_weight', 1.0),
            'triplet_margin': getattr(args, 'triplet_margin', 0.3),
        }
        
        # Initialize model based on architecture type
        if args.architecture == 'expert':
            print("\nCreating expert model...")
            model = create_model(
                model_name='expert',
                num_classes=num_classes,
                pretrained=args.pretrained,
                backbone_name=args.backbones[0] if args.backbones else 'resnet50',
                embedding_size=args.embedding_size,
                emotion_groups=args.emotion_groups,
                gem_pooling=args.gem_pooling,
                decoupled_head=args.decoupled_head,
                drop_path_rate=args.drop_path_rate,
                channels_last=args.channels_last,
                block_disgust=True  # Always block the disgust class
            ).to(device)
            
            # Create loss function
            loss_fn = create_loss_fn(
                loss_type=args.loss_type,
                num_classes=num_classes,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                triplet_margin=args.triplet_margin,
                class_weights=args.class_weights,
                sad_class_weight=args.sad_class_weight
            )
            
            config['custom_loss_fn'] = loss_fn
            architecture_name = f"ExpertModel(backbone={args.backbones[0] if args.backbones else 'resnet50'})"
        else:
            # Use the original ensemble model
            model = EnsembleModel(
                backbones=args.backbones,
                num_classes=num_classes,
                pretrained=True,
                drop_path_rate=getattr(args, 'drop_path_rate', 0.0)
            ).to(device)
            architecture_name = f"EnsembleModel with {args.backbones}"
            config['custom_loss_fn'] = None
        
        # Print info to logs
        print("\nCreating trainer...")
        print(f"Model architecture: {architecture_name}")
        print(f"Loss type: {args.loss_type}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Patience for early stopping: {args.patience}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Model directory: {args.model_dir}")
        
        # Create trainer and run training
        trainer = EmotionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=config,
            device=device,
            data_manager=data_manager  # Pass data_manager to the trainer
        )
        
        # Start training
        print("\nStarting training...")
        best_val_f1 = trainer.train()
        
        print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")
        
        # Test on test set if provided
        if args.test_dir:
            print("\nEvaluating on test set...")
            test_metrics = trainer.evaluate(test_dataset)
            print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return 1
    finally:
        # Ensure temp directories are cleaned up even if an error occurs
        if data_manager and args.dataset_name == 'fer2013' and hasattr(data_manager, 'cleanup_temp_dir'):
            print("\nCleaning up temporary directories...")
            data_manager.cleanup_temp_dir()

if __name__ == '__main__':
    exit(main()) 