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
from datasets.fer_datasets import RAFDBDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GReFEL on RAF-DB test set')
    parser.add_argument('--data_dir', type=str, default='./rafdb', help='Path to RAF-DB dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='eval_results_rafdb', help='Directory to save evaluation results')
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
    
    emotion_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(images)
            preds = outputs['logits'].argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    
    # Create model
    model = GReFEL(
        num_classes=7,  # RAF-DB has 7 emotion classes
        feature_dim=768,  # ViT-Base default
        num_anchors=10,
        drop_rate=0.1
    ).to(args.device)
    
    # Load checkpoint with proper settings for PyTorch 2.6+
    try:
        # First try loading with weights_only=True and safe globals
        torch.serialization.add_safe_globals(['argparse.Namespace'])
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=True, attempting legacy loading: {e}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create test dataset and loader
    test_dataset = RAFDBDataset(
        root_dir=args.data_dir,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    
    # Evaluate
    accuracy, class_acc = evaluate(model, test_loader, args)
    
    # Print results
    print(f"\nResults saved to {args.output_dir}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("\nPer-class Accuracy:")
    emotion_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    for name, acc in zip(emotion_names, class_acc):
        print(f"{name}: {acc*100:.2f}%")

if __name__ == '__main__':
    main() 