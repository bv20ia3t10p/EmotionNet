"""Main training script for emotion recognition model."""

import os
import torch
import argparse
import numpy as np
from emotion_net.training import EmotionTrainer
from emotion_net.data import BaseEmotionDataset, FER2013DataManager, RAFDBDataManager
from emotion_net.models.ensemble import EnsembleModel
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
                      help='List of backbone architectures for the ensemble model')
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
                      choices=['cross_entropy', 'focal'],
                      help="Type of loss function to use (default: cross_entropy)")
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help="Gamma parameter for Focal Loss (default: 2.0)")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                      help="Label smoothing factor for CrossEntropyLoss (default: 0.1)")
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                      help="Alpha parameter for Mixup (default: 0.0 to disable)")
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                      help="Drop path rate (Stochastic Depth) for backbones (default: 0.0)")
    parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                      choices=['one_cycle', 'cosine_annealing', 'none'],
                      help="Type of LR scheduler (default: one_cycle)")
    parser.add_argument('--num_workers', type=int, default=4,
                      help="Number of data loading workers (default: 4)")
    
    # Parse known args to handle both old and new formats
    args, unknown = parser.parse_known_args()
    
    return args

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
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
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create model
        print(f"\nCreating EnsembleModel with backbones: {args.backbones}")
        config = {
            'drop_path_rate': args.drop_path_rate
        }
        model = EnsembleModel(
            num_classes=num_classes,
            backbones=args.backbones,
            drop_path_rate=config.get('drop_path_rate', 0.0),
            pretrained=True  # Use pretrained weights for better initialization
        ).to(device)
        
        # Set model to fixed averaging mode for initial training
        model.use_fixed_averaging = True
        
        # Training configuration
        trainer_config = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'patience': args.patience,
            'model_dir': args.model_dir,
            'use_ema': True, 
            'ema_decay': 0.999,
            'num_workers': args.num_workers,
            'loss_type': args.loss_type,
            'focal_gamma': args.focal_gamma,
            'label_smoothing': args.label_smoothing,
            'mixup_alpha': args.mixup_alpha,
            'drop_path_rate': args.drop_path_rate,
            'scheduler_type': args.scheduler_type,
            'train_labels': train_labels,
            'dataset_name': args.dataset_name
        }
        
        # Create trainer
        trainer = EmotionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=trainer_config
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

if __name__ == '__main__':
    exit(main()) 