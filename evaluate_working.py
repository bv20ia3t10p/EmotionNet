#!/usr/bin/env python3
"""
Evaluation script for SimpleConvNet model
Compatible with train_working_fix.py architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import argparse
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_score, 
                           recall_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the SimpleConvNet model from train_working_fix.py
import sys
sys.path.append('.')

try:
    from train_working_fix import SimpleConvNet, FERPlusDataset
    print("✅ Successfully imported SimpleConvNet model")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    print("Make sure train_working_fix.py is in the current directory")
    sys.exit(1)

def load_working_model(checkpoint_path, device):
    """Load SimpleConvNet model from checkpoint"""
    print(f"Loading SimpleConvNet model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same parameters as training
    model = SimpleConvNet(num_classes=8).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Training val_acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model state dict")
    
    model.eval()
    return model

def create_test_transforms():
    """Create transforms for testing (same as validation)"""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate_model(model, dataloader, device, dataset_name="Test"):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_samples = 0
    correct = 0
    
    print(f"Evaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for images, hard_labels, soft_labels in tqdm(dataloader, desc=f'{dataset_name} Evaluation'):
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get probabilities and predictions
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Accumulate results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Accuracy calculation
            total_samples += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = 100. * correct / total_samples
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'total_samples': total_samples
    }

def calculate_detailed_metrics(results, emotion_names):
    """Calculate detailed classification metrics"""
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*60)
    
    # Basic accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # F1 scores
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    print(f"F1-Score (Macro):    {f1_macro:.4f}")
    print(f"F1-Score (Micro):    {f1_micro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Precision and Recall
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')
    
    print(f"Precision (Macro):   {precision_macro:.4f}")
    print(f"Recall (Macro):      {recall_macro:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print("-" * 60)
    class_report = classification_report(labels, predictions, 
                                       target_names=emotion_names, 
                                       output_dict=True)
    
    for i, emotion in enumerate(emotion_names):
        if emotion in class_report:
            metrics = class_report[emotion]
            support = int(metrics['support'])
            print(f"{emotion:12s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                  f"Support={support}")
    
    # ROC AUC (multiclass)
    try:
        roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        print(f"\nROC AUC (Macro):     {roc_auc:.4f}")
    except Exception as e:
        print(f"\nROC AUC calculation failed: {e}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'classification_report': class_report
    }

def plot_confusion_matrix(results, emotion_names, save_path="confusion_matrix_working.png"):
    """Plot and save confusion matrix"""
    predictions = results['predictions']
    labels = results['labels']
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix - SimpleConvNet Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def analyze_model_architecture(model):
    """Analyze and print model architecture details"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB):      {total_params * 4 / 1024 / 1024:.2f}")
    
    # Architecture details
    print(f"\nArchitecture Details:")
    print(f"Model Type:           SimpleConvNet")
    print(f"Input Size:           64x64x3")
    print(f"Feature Blocks:       4 CNN blocks")
    print(f"Final Features:       512")
    print(f"Classifier:           512 -> 256 -> 8")
    print(f"Number of Classes:    8")

def plot_performance_comparison(results, baseline_acc=83.84):
    """Plot performance comparison with baseline"""
    accuracy = results['accuracy']
    
    categories = ['Baseline\n(HybridFER-Max)', 'SimpleConvNet\n(This Model)']
    accuracies = [baseline_acc, accuracy]
    colors = ['skyblue', 'lightgreen' if accuracy > baseline_acc else 'orange']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7)
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement/degradation text
    improvement = accuracy - baseline_acc
    if improvement > 0:
        plt.text(0.5, max(accuracies) - 10, f'Improvement: +{improvement:.2f}%', 
                ha='center', fontsize=12, color='green', fontweight='bold')
    else:
        plt.text(0.5, max(accuracies) - 10, f'Change: {improvement:.2f}%', 
                ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison_working.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Performance comparison saved to performance_comparison_working.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate SimpleConvNet Model')
    parser.add_argument('--model_path', type=str, default='best_working_fer.pth',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--ferplus_csv', type=str, 
                       default='./FERPlus-master/fer2013new.csv',
                       help='Path to FERPlus CSV file')
    parser.add_argument('--fer2013_csv', type=str, 
                       default='./fer2013.csv',
                       help='Path to original FER2013 CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default='PrivateTest',
                       choices=['Training', 'PublicTest', 'PrivateTest'],
                       help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"❌ Model checkpoint not found: {args.model_path}")
        return
    
    if not os.path.exists(args.ferplus_csv):
        print(f"❌ FERPlus CSV not found: {args.ferplus_csv}")
        return
        
    if not os.path.exists(args.fer2013_csv):
        print(f"❌ FER2013 CSV not found: {args.fer2013_csv}")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_working_model(args.model_path, device)
    
    # Analyze architecture
    analyze_model_architecture(model)
    
    # Create dataset and dataloader
    test_transform = create_test_transforms()
    
    test_dataset = FERPlusDataset(
        csv_file=args.ferplus_csv,
        fer2013_csv_file=args.fer2013_csv,
        split=args.split,
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"\nDataset: {len(test_dataset)} samples in {args.split} split")
    
    # Emotion names
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, args.split)
    
    # Calculate detailed metrics
    metrics = calculate_detailed_metrics(results, emotion_names)
    
    # Create visualizations
    plot_confusion_matrix(results, emotion_names)
    plot_performance_comparison(results)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: SimpleConvNet (CNN Architecture)")
    print(f"Dataset Split: {args.split}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Compare with baseline
    baseline = 83.84
    improvement = results['accuracy'] - baseline
    if improvement > 0:
        print(f"🎉 IMPROVEMENT: +{improvement:.2f}% vs baseline!")
    elif improvement > -5:
        print(f"📊 Close to baseline: {improvement:.2f}% difference")
    else:
        print(f"📉 Below baseline: {improvement:.2f}% difference")
    
    # Practical assessment
    if results['accuracy'] > 75:
        print("✅ EXCELLENT: Model performs very well!")
    elif results['accuracy'] > 65:
        print("✅ GOOD: Model performs reasonably well")
    elif results['accuracy'] > 50:
        print("⚠️  FAIR: Model works but has room for improvement")
    else:
        print("❌ POOR: Model needs significant improvement")
    
    print("="*60)

if __name__ == "__main__":
    main() 