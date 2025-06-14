#!/usr/bin/env python3
"""
Compute class weights for balanced training
"""

import torch
import numpy as np
from collections import Counter
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from datasets.fer_datasets import get_loaders

def compute_class_weights(args):
    """Compute class weights from training data"""
    print("📊 Computing class weights from training data...")
    
    # Get data loaders
    train_loader, _ = get_loaders(args)
    
    # Collect all labels
    all_labels = []
    
    for _, labels in train_loader:
        if labels.dim() > 1 and labels.size(1) > 1:
            # Soft labels - use argmax
            label_indices = torch.argmax(labels, dim=1)
        else:
            # Hard labels
            label_indices = labels.long()
        
        all_labels.extend(label_indices.cpu().numpy())
    
    # Count class frequencies
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    print(f"Total samples: {total_samples}")
    print("Class distribution:")
    
    emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    
    for i in range(8):
        count = class_counts.get(i, 0)
        percentage = (count / total_samples) * 100
        print(f"  {emotion_names[i]}: {count} samples ({percentage:.1f}%)")
    
    # Compute inverse frequency weights
    class_weights = []
    for i in range(8):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total_samples / (8 * count)  # Inverse frequency
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Normalize weights so they sum to num_classes
    class_weights = class_weights / class_weights.mean()
    
    print("\nComputed class weights:")
    for i, weight in enumerate(class_weights):
        print(f"  {emotion_names[i]}: {weight:.3f}")
    
    return class_weights

def main():
    parser = argparse.ArgumentParser(description='Compute class weights')
    parser.add_argument('--data_dir', type=str, default='./FERPlus', help='Data directory')
    parser.add_argument('--dataset', type=str, default='ferplus', choices=['ferplus', 'rafdb', 'fer2013'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--use_soft_labels', action='store_true', default=True, help='Use soft labels')
    
    args = parser.parse_args()
    
    try:
        class_weights = compute_class_weights(args)
        
        # Save weights to file
        torch.save(class_weights, 'class_weights.pt')
        print(f"\n✅ Class weights saved to 'class_weights.pt'")
        
        # Print code to use in training
        print("\n📋 To use these weights in training, add this to your loss initialization:")
        print("```python")
        print("class_weights = torch.load('class_weights.pt')")
        print("criterion = EnhancedGReFELLoss(")
        print("    num_classes=8,")
        print("    label_smoothing=0.11,")
        print("    class_weights=class_weights")
        print(")")
        print("```")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error computing class weights: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 