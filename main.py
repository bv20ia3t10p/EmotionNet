import torch
import os
from torchvision import transforms, datasets
import argparse
from dynamic_training import ImprovedDynamicTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    parser.add_argument('--data_dir', type=str, default='data/emotion_dataset', 
                        help='Path to emotion dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                        help='Directory to save outputs')
    parser.add_argument('--backbone', type=str, default='convnext_xlarge', 
                        choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                                'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
                                'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Image size')
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='Dropout rate')
    parser.add_argument('--no_class_weighting', action='store_true', 
                        help='Disable class weighting')
    parser.add_argument('--no_weighted_sampler', action='store_true', 
                        help='Disable weighted sampler')
    parser.add_argument('--no_mixup', action='store_true', 
                        help='Disable mixup augmentation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Define class names
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Create trainer
    trainer = ImprovedDynamicTrainer(
        num_classes=7,
        class_names=class_names,
        backbone_name=args.backbone,
        img_size=args.img_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        dropout_rate=args.dropout,
        focal_gamma=2.0,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_mixup=not args.no_mixup,
        use_class_weighting=not args.no_class_weighting,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=args.output_dir
    )
    
    print(f"Training with backbone: {args.backbone}")
    print(f"Using weighted sampler: {not args.no_weighted_sampler}")
    print(f"Using mixup: {not args.no_mixup}")
    print(f"Using class weighting: {not args.no_class_weighting}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Learning rate: {args.lr}")
    
    # Data augmentation transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Dataset directory {args.data_dir} not found.")
        return
    
    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train'])
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['val'])
        
        print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
        
        # Check class distribution in training set
        class_counts = {}
        for _, label in train_dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("Class distribution in training set:")
        for i in range(len(class_names)):
            count = class_counts.get(i, 0)
            percentage = (count / len(train_dataset)) * 100
            print(f"  {class_names[i]}: {count} images ({percentage:.2f}%)")
        
        # Train model
        history = trainer.train(train_dataset, val_dataset, num_epochs=args.epochs)
        
        # Save final model
        trainer.save_model(os.path.join(args.output_dir, 'models', 'final_model.pt'))
        
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 