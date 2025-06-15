import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.grefel import GReFEL
from datasets.fer_datasets import FERPlusDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GReFEL on FERPlus test set')
    parser.add_argument('--data_dir', type=str, default='./FERPlus', help='Path to FERPlus dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save evaluation results')
    parser.add_argument('--use_soft_labels', action='store_true', default=True, help='Use probability distribution from votes')
    parser.add_argument('--use_hard_labels', dest='use_soft_labels', action='store_false', help='Use majority vote')
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate(model, loader, args):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(images)
            preds = outputs['logits'].argmax(dim=1)
            
            # Handle both soft and hard labels
            if labels.dim() > 1 and labels.size(1) > 1:
                # Soft labels - compare with argmax
                true_labels = labels.argmax(dim=1)
            else:
                # Hard labels
                true_labels = labels
            
            correct += (preds == true_labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    # Per-class accuracy
    class_acc = cm.diagonal()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=emotion_names,
                                 digits=4)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save confusion matrix plot
    plot_confusion_matrix(cm, emotion_names, 
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save metrics to text file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f'Overall Accuracy: {accuracy*100:.2f}%\n\n')
        f.write('Per-class Accuracy:\n')
        for name, acc in zip(emotion_names, class_acc):
            f.write(f'{name}: {acc*100:.2f}%\n')
        f.write('\nDetailed Classification Report:\n')
        f.write(report)
    
    return accuracy, class_acc

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model with fixed parameters
    model = GReFEL(
        num_classes=8,  # FERPlus has 8 emotion classes
        feature_dim=768,  # ViT-Base default
        num_anchors=10,
        drop_rate=0.1
    ).to(args.device)
    
    # Load checkpoint with proper error handling
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        # Try loading with weights_only=False first (legacy mode)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        print("Successfully loaded checkpoint in legacy mode")
    except Exception as e1:
        print(f"Legacy loading failed, trying weights_only mode: {e1}")
        try:
            # Add Namespace to safe globals and try weights_only=True
            torch.serialization.add_safe_globals(['argparse.Namespace'])
            checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            print("Successfully loaded checkpoint in weights_only mode")
        except Exception as e2:
            print(f"Both loading attempts failed. Final error: {e2}")
            raise
    
    # Load model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state from checkpoint dict")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state directly")
    
    model.eval()
    
    # Create test dataset and loader
    test_dataset = FERPlusDataset(
        root_dir=args.data_dir,
        split='test',
        use_soft_labels=args.use_soft_labels
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    print(f"Using {'soft' if args.use_soft_labels else 'hard'} labels")
    
    # Evaluate
    accuracy, class_acc = evaluate(model, test_loader, args)
    
    # Print results
    print(f"\nResults saved to {args.output_dir}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("\nPer-class Accuracy:")
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    for name, acc in zip(emotion_names, class_acc):
        print(f"{name}: {acc*100:.2f}%")

if __name__ == '__main__':
    main() 