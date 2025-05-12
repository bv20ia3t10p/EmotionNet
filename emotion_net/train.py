"""Main training script for emotion recognition model."""

import os
import torch
import argparse
import numpy as np
from emotion_net.training import EmotionTrainer
from emotion_net.data.dataset import AdvancedEmotionDataset, load_data
from emotion_net.models.ensemble import EnsembleModel
from emotion_net.config.constants import (
    DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_BACKBONES, EMOTIONS, CHECKPOINT_DIR, DEFAULT_IMAGE_SIZE
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train EmotionNet model')
    
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
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Validation split ratio (default: 0.1)')
    
    # Parse known args to handle both old and new formats
    args, unknown = parser.parse_known_args()
    
    return args

def split_data(paths, labels, val_ratio=0.1, seed=42):
    """Split data into train and validation sets."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get indices for splitting
    indices = np.arange(len(paths))
    np.random.shuffle(indices)
    
    # Calculate split point
    split = int(len(indices) * (1 - val_ratio))
    
    # Split indices
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Split data
    train_paths = [paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_paths, train_labels, val_paths, val_labels

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
        
        # Load data
        print("\nLoading data...")
        if args.test_dir:
            # If test directory is provided, use it for validation
            train_paths, train_labels = load_data(args.data_dir, EMOTIONS)
            val_paths, val_labels = load_data(args.test_dir, EMOTIONS)
        else:
            # Otherwise, split training data
            all_paths, all_labels = load_data(args.data_dir, EMOTIONS)
            train_paths, train_labels, val_paths, val_labels = split_data(
                all_paths, all_labels, args.val_split
            )
        
        # Print dataset sizes
        print(f"\nDataset sizes:")
        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        
        # Create datasets
        train_dataset = AdvancedEmotionDataset(
            train_paths, train_labels,
            mode='train',
            image_size=args.image_size
        )
        val_dataset = AdvancedEmotionDataset(
            val_paths, val_labels,
            mode='val',
            image_size=args.image_size
        )
        
        # Create model
        print(f"\nCreating EnsembleModel with backbones: {args.backbones}")
        model = EnsembleModel(
            num_classes=len(EMOTIONS),
            backbones=args.backbones,
        ).to(device)
        
        # Training configuration
        config = {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'patience': args.patience,
            'image_size': args.image_size
        }
        
        # Create trainer
        trainer = EmotionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=config
        )
        
        # Start training
        print("\nStarting training...")
        best_val_f1 = trainer.train()
        
        print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")
        
        # Test on test set if provided
        if args.test_dir:
            print("\nEvaluating on test set...")
            test_paths, test_labels = load_data(args.test_dir, EMOTIONS)
            test_dataset = AdvancedEmotionDataset(
                test_paths, test_labels,
                mode='test',
                image_size=args.image_size
            )
            test_metrics = trainer.evaluate(test_dataset)
            print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main()) 