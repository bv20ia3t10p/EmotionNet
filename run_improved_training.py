import torch
import os
import sys

# Import the advanced trainer
from train_advanced import AdvancedTrainer, main as train_main
from models import create_model
from dataset import get_data_loaders

def run_improved_training():
    """Run training with progressively advanced configurations"""
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Configuration optimized for FER2013 and your hardware
    config = {
        # Model architecture
        'backbone': 'convnext_large',  # Start with large instead of xlarge
        'batch_size': 32 if torch.cuda.is_available() else 16,
        'img_size': 224,
        
        # Training parameters
        'epochs': 40,
        'lr': 1e-3,  # Higher initial learning rate
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        
        # Scheduler
        'scheduler': 'cosine_warm_restarts',
        'T_0': 5,  # Restart every 5 epochs
        'T_mult': 2,
        'eta_min': 1e-6,
        
        # Loss and augmentation
        'loss_type': 'focal',  # Focal loss for class imbalance
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.3,  # Slightly stronger mixup
        'cutmix_alpha': 1.0,
        
        # Training efficiency
        'use_amp': torch.cuda.is_available(),  # Mixed precision only if GPU
        'gradient_accumulation_steps': 2 if not torch.cuda.is_available() else 1,
        'clip_grad': 1.0,
        
        # Regularization
        'dropout_rate': 0.6,  # Higher dropout for FER2013
        'use_tta': True,  # Test-time augmentation
        
        # Early stopping
        'patience': 12,
        
        # Class weights for focal loss (based on your data distribution)
        'class_weights': [1.0, 5.0, 2.0, 0.8, 1.2, 0.9, 1.0]  # Boost Disgust and Fear
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type='single',
        backbone_name=config['backbone'],
        num_classes=7,
        dropout_rate=config['dropout_rate'],
        use_attention=True,
        pretrained=True
    )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv', 
        test_csv='data/test.csv',
        img_dir='data',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=2 if os.name == 'nt' else 4,  # Fewer workers on Windows
        use_weighted_sampler=True,
        use_mixup=config['use_mixup'],
        use_cutmix=config['use_cutmix']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    trainer = AdvancedTrainer(model, device, config)
    
    # Train model
    print("\nStarting training...")
    print("=" * 50)
    
    try:
        history = trainer.train(train_loader, val_loader)
        
        # Save training history
        import json
        with open('outputs/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Load best model for evaluation
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        print("\n" + "=" * 50)
        print("Evaluating on test set...")
        test_loss, test_acc, test_f1, test_preds, test_labels = trainer.validate(test_loader)
        
        print(f'\nFinal Results:')
        print(f'Best Validation Accuracy: {trainer.best_val_acc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test F1 Score: {test_f1:.4f}')
        
        # Detailed classification report
        from sklearn.metrics import classification_report
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=emotion_names))
        
        # Save final results
        results = {
            'best_val_acc': float(trainer.best_val_acc),
            'best_val_f1': float(trainer.best_val_f1),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'config': config,
            'classification_report': classification_report(test_labels, test_preds, 
                                                        target_names=emotion_names, 
                                                        output_dict=True)
        }
        
        with open('outputs/final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nTraining completed successfully!")
        print(f"Results saved to outputs/")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    run_improved_training() 