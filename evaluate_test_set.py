#!/usr/bin/env python3
"""
Test Set Evaluation Script for GReFEL
Evaluates both EMA model and best checkpoint on test set with detailed metrics.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

from models.grefel import GReFEL, EnhancedGReFELLoss
from datasets.fer_datasets import FERPlusDataset


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint with weights_only=False for compatibility with our training checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    # Create model with same architecture
    model = GReFEL(
        num_classes=8,
        feature_dim=args.feature_dim,
        num_anchors=args.num_anchors,
        drop_rate=args.drop_rate
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['best_acc']:.4f}")
    return model, args


def evaluate_model(model, test_loader, criterion, device='cuda', model_name="Model"):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    total_loss = 0
    total_samples = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f'Testing {model_name}'):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            logits = outputs['logits']
            
            # Compute loss
            is_soft_labels = targets.dim() > 1 and targets.size(1) > 1
            loss_dict = criterion(outputs, targets, is_soft_labels=is_soft_labels)
            loss = loss_dict['total_loss']
            
            # Get predictions
            preds = logits.argmax(dim=1)
            
            # Handle soft/hard labels
            if is_soft_labels:
                target_labels = targets.argmax(dim=1)
            else:
                target_labels = targets.long()
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_labels.cpu().numpy())
            all_logits.extend(F.softmax(logits, dim=1).cpu().numpy())
            
            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Per-class accuracy
            for pred, target in zip(preds.cpu().numpy(), target_labels.cpu().numpy()):
                class_total[target] += 1
                if pred == target:
                    class_correct[target] += 1
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)
    
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Per-class accuracies
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    class_accuracies = {}
    for i in range(8):
        if class_total[i] > 0:
            class_accuracies[emotion_names[i]] = class_correct[i] / class_total[i]
        else:
            class_accuracies[emotion_names[i]] = 0.0
    
    # Classification report
    class_report = classification_report(
        all_targets, all_preds, 
        target_names=emotion_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'class_accuracies': class_accuracies,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'targets': all_targets,
        'logits': all_logits
    }


def print_detailed_results(results, model_name):
    """Print detailed evaluation results."""
    print(f"\n{'='*60}")
    print(f"{model_name} - Test Set Results")
    print(f"{'='*60}")
    
    print(f"Overall Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Average Test Loss: {results['loss']:.4f}")
    
    print(f"\nPer-Class Accuracies:")
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    for emotion in emotion_names:
        acc = results['class_accuracies'][emotion]
        print(f"  {emotion:12}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nDetailed Classification Report:")
    print("-" * 60)
    report = results['classification_report']
    
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)
    
    for emotion in emotion_names:
        if emotion.lower() in report:
            metrics = report[emotion.lower()]
            print(f"{emotion:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1-score']:<10.4f} {int(metrics['support']):<8}")
    
    # Overall metrics
    print("-" * 60)
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    print(f"{'Macro Avg':<12} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} "
          f"{macro_avg['f1-score']:<10.4f} {'-':<8}")
    print(f"{'Weighted Avg':<12} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} "
          f"{weighted_avg['f1-score']:<10.4f} {'-':<8}")


def save_results(results_dict, output_dir):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary results
    summary = {}
    for model_name, results in results_dict.items():
        summary[model_name] = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'class_accuracies': {k: float(v) for k, v in results['class_accuracies'].items()},
            'macro_f1': float(results['classification_report']['macro avg']['f1-score']),
            'weighted_f1': float(results['classification_report']['weighted avg']['f1-score'])
        }
    
    # Save summary
    with open(os.path.join(output_dir, 'test_evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results for each model
    for model_name, results in results_dict.items():
        model_dir = os.path.join(output_dir, model_name.lower().replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        # Save classification report
        with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
            json.dump(results['classification_report'], f, indent=2)
        
        # Save confusion matrix
        np.save(os.path.join(model_dir, 'confusion_matrix.npy'), results['confusion_matrix'])
        
        # Save predictions and logits
        np.save(os.path.join(model_dir, 'predictions.npy'), results['predictions'])
        np.save(os.path.join(model_dir, 'targets.npy'), results['targets'])
        np.save(os.path.join(model_dir, 'logits.npy'), results['logits'])
    
    print(f"\nResults saved to: {output_dir}")


def compare_models(results_dict):
    """Compare results between different models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    models = list(results_dict.keys())
    if len(models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    print(f"{'Metric':<20} {models[0]:<15} {models[1]:<15} {'Difference':<15}")
    print("-" * 70)
    
    # Overall accuracy
    acc1 = results_dict[models[0]]['accuracy']
    acc2 = results_dict[models[1]]['accuracy']
    diff = acc2 - acc1
    print(f"{'Accuracy':<20} {acc1:<15.4f} {acc2:<15.4f} {diff:<+15.4f}")
    
    # Macro F1
    f1_1 = results_dict[models[0]]['classification_report']['macro avg']['f1-score']
    f1_2 = results_dict[models[1]]['classification_report']['macro avg']['f1-score']
    diff = f1_2 - f1_1
    print(f"{'Macro F1':<20} {f1_1:<15.4f} {f1_2:<15.4f} {diff:<+15.4f}")
    
    # Per-class comparison
    print(f"\nPer-Class Accuracy Differences ({models[1]} - {models[0]}):")
    print("-" * 50)
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    for emotion in emotion_names:
        acc1 = results_dict[models[0]]['class_accuracies'][emotion]
        acc2 = results_dict[models[1]]['class_accuracies'][emotion]
        diff = acc2 - acc1
        status = "✅" if diff > 0 else "❌" if diff < 0 else "➖"
        print(f"  {emotion:<12}: {diff:+.4f} {status}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GReFEL models on test set')
    parser.add_argument('--data_dir', type=str, default='./FERPlus', help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--use_soft_labels', action='store_true', default=True, help='Use soft labels for evaluation')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Create test dataset
    test_dataset = FERPlusDataset(
        root_dir=args.data_dir,
        split='test',  # Use test split
        use_soft_labels=args.use_soft_labels
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Create loss function (same as training)
    criterion = EnhancedGReFELLoss(
        num_classes=8,
        label_smoothing=0.0,  # No smoothing for evaluation
        geo_weight=0.01,
        reliability_weight=0.1
    )
    
    results_dict = {}
    
    # Evaluate best checkpoint model
    best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"\nLoading best checkpoint: {best_checkpoint_path}")
        model, model_args = load_model_from_checkpoint(best_checkpoint_path, args.device)
        results = evaluate_model(model, test_loader, criterion, args.device, "Best Checkpoint")
        results_dict["Best Checkpoint"] = results
        print_detailed_results(results, "Best Checkpoint")
    else:
        print(f"❌ Best checkpoint not found: {best_checkpoint_path}")
    
    # Evaluate EMA model
    ema_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_ema_model.pth')
    if os.path.exists(ema_checkpoint_path):
        print(f"\nLoading EMA checkpoint: {ema_checkpoint_path}")
        ema_model, ema_args = load_model_from_checkpoint(ema_checkpoint_path, args.device)
        ema_results = evaluate_model(ema_model, test_loader, criterion, args.device, "EMA Model")
        results_dict["EMA Model"] = ema_results
        print_detailed_results(ema_results, "EMA Model")
    else:
        print(f"❌ EMA checkpoint not found: {ema_checkpoint_path}")
    
    # Compare models if both exist
    if len(results_dict) == 2:
        compare_models(results_dict)
    
    # Save results
    if results_dict:
        save_results(results_dict, args.output_dir)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        for model_name, results in results_dict.items():
            print(f"{model_name}: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    else:
        print("❌ No valid checkpoints found for evaluation!")


if __name__ == "__main__":
    main() 