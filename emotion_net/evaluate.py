"""Evaluation script for the emotion recognition model."""

import os
import torch
import argparse
import numpy as np
from emotion_net.data import FER2013DataManager, RAFDBDataManager
from emotion_net.models import create_model
from emotion_net.training.metrics import calculate_metrics
from emotion_net.config.constants import (
    DEFAULT_IMAGE_SIZE, DEFAULT_BACKBONES, EMOTIONS
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate EmotionNet model')
    
    # Add Dataset Name Argument
    parser.add_argument('--dataset_name', type=str, required=True,
                      choices=['fer2013', 'rafdb'],
                      help='Name of the dataset to use (fer2013 or rafdb)')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model weights')
    
    # Optional arguments
    parser.add_argument('--architecture', type=str, default='sota_resemote_medium',
                      choices=['ensemble', 'hierarchical', 'expert', 'multihead',
                              'sota_resemote_small', 'sota_resemote_medium', 'sota_resemote_large'],
                      help='Model architecture type')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                      help='Backbone architecture to use')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                      help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--embedding_size', type=int, default=512,
                      help='Embedding size for the model')
    parser.add_argument('--force_affectnet_backbone', action='store_true',
                      help='Force using AffectNet pretrained backbone')
    
    args = parser.parse_args()
    return args

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data manager based on dataset name
    if args.dataset_name == 'fer2013':
        data_manager = FER2013DataManager(data_dir=args.data_dir)
    elif args.dataset_name == 'rafdb':
        data_manager = RAFDBDataManager(data_dir=args.data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    # Create datasets and data loaders
    print(f"Loading {args.dataset_name} dataset...")
    train_dataset, val_dataset, _ = data_manager.create_datasets(image_size=args.image_size)
    _, val_loader = data_manager.create_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"Creating {args.architecture} model...")
    model = create_model(
        model_name=args.architecture,
        num_classes=num_classes,
        pretrained=False,  # Don't need pretrained for evaluation
        backbone_name=args.backbone if args.architecture not in ['sota_resemote_small', 'sota_resemote_medium', 'sota_resemote_large'] else None,
        embedding_size=args.embedding_size,
        force_affectnet_backbone=args.force_affectnet_backbone if args.architecture not in ['sota_resemote_small', 'sota_resemote_medium', 'sota_resemote_large'] else False
    ).to(device)
    
    # Load model weights
    print(f"Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate model
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle different model output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # For ensemble models
            elif isinstance(outputs, dict):
                if 'direct_logits' in outputs:
                    outputs = outputs['direct_logits']
                elif 'logits' in outputs:
                    outputs = outputs['logits']
            # For SOTA ResEmote models with auxiliary outputs
            elif isinstance(outputs, list) and len(outputs) > 1:
                outputs = outputs[0]  # Take main outputs
            
            # Get predicted class
            _, preds = torch.max(outputs, 1)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        val_dataset.classes
    )
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    classes = [EMOTIONS[i] for i in range(num_classes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(args.model_path), 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write("\nClass-wise F1 Scores:\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls}: {metrics['class_f1'][i]:.4f}\n")
    
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.txt')}")
    
    # Show class-wise F1 scores
    print("\nClass-wise F1 Scores:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {metrics['class_f1'][i]:.4f}")

if __name__ == "__main__":
    main() 