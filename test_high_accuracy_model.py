#!/usr/bin/env python3
# Script to evaluate the high-accuracy emotion recognition model

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import cv2
from PIL import Image

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

# Import from our high accuracy training script
try:
    from high_accuracy_training import (
        EMOTIONS, load_data, AdvancedEmotionDataset, EnsembleModel
    )
    print("Successfully imported from high_accuracy_training")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure high_accuracy_training.py is in the current directory.")
    sys.exit(1)

def evaluate_model(
    model_path,
    test_dir="./extracted/emotion/test",
    output_dir="./results",
    batch_size=32,
    image_size=224,
    backbones=['efficientnet_b0', 'resnet18']
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {test_dir}...")
    test_paths, test_labels = load_data(test_dir)
    print(f"Loaded {len(test_paths)} test samples")
    
    # Create test dataset
    test_dataset = AdvancedEmotionDataset(
        test_paths, test_labels, mode='test', image_size=image_size
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    print(f"Creating model with backbones: {backbones}")
    model = EnsembleModel(num_classes=len(EMOTIONS), backbones=backbones)
    
    # Load model weights
    print(f"Loading model weights from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    all_preds = []
    all_labels = []
    all_confidences = []
    
    # Progress bar
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(inputs)
            
            # Get predictions and confidence scores
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Macro F1 Score: {macro_f1:.2f}%")
    print(f"Weighted F1 Score: {weighted_f1:.2f}%")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, 
                                 target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))],
                                 digits=4)
    print(report)
    
    # Per-class metrics
    print("\nDetailed Per-Class Metrics:")
    print("----------------------------")
    for class_idx in range(len(EMOTIONS)):
        class_mask = all_labels == class_idx
        if np.sum(class_mask) > 0:
            class_samples = np.sum(class_mask)
            class_acc = np.mean(all_preds[class_mask] == class_idx) * 100
            class_pred_mask = all_preds == class_idx
            class_precision = np.sum(all_labels[class_pred_mask] == class_idx) / max(np.sum(class_pred_mask), 1) * 100
            class_recall = np.sum(all_preds[class_mask] == class_idx) / np.sum(class_mask) * 100
            class_f1 = 2 * (class_precision * class_recall) / max((class_precision + class_recall), 1)
            class_confidence = np.mean(all_confidences[class_mask]) * 100
            
            print(f"{EMOTIONS[class_idx]}:")
            print(f"  Samples: {class_samples}")
            print(f"  Accuracy: {class_acc:.2f}%")
            print(f"  Precision: {class_precision:.2f}%")
            print(f"  Recall: {class_recall:.2f}%")
            print(f"  F1 Score: {class_f1:.2f}%")
            print(f"  Mean Confidence: {class_confidence:.2f}%")
    
    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(EMOTIONS))
    plt.xticks(tick_marks, [EMOTIONS[i] for i in range(len(EMOTIONS))], rotation=45)
    plt.yticks(tick_marks, [EMOTIONS[i] for i in range(len(EMOTIONS))])
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                   horizontalalignment="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'high_accuracy_confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'high_accuracy_confusion_matrix.png')}")
    
    # Plot prediction distribution vs actual distribution
    plt.figure(figsize=(12, 6))
    
    # Count actual and predicted instances for each class
    actual_counts = np.bincount(all_labels, minlength=len(EMOTIONS))
    pred_counts = np.bincount(all_preds, minlength=len(EMOTIONS))
    
    # Convert to percentages
    actual_pct = actual_counts / len(all_labels) * 100
    pred_pct = pred_counts / len(all_preds) * 100
    
    # Plot
    x = np.arange(len(EMOTIONS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, actual_pct, width, label='Actual Distribution')
    rects2 = ax.bar(x + width/2, pred_pct, width, label='Predicted Distribution')
    
    # Add labels and title
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Actual vs Predicted Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([EMOTIONS[i] for i in range(len(EMOTIONS))])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    print(f"Distribution comparison saved to {os.path.join(output_dir, 'distribution_comparison.png')}")
    
    # Return key metrics
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate high-accuracy emotion recognition model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--backbones', type=str, nargs='+', 
                      default=['efficientnet_b0', 'resnet18'],
                      help='Backbone architectures used in the model')
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        backbones=args.backbones
    )
    
    print("\nEvaluation completed!")
    print(f"Final metrics: Accuracy={metrics['accuracy']:.2f}%, F1={metrics['macro_f1']:.2f}%")

if __name__ == "__main__":
    main() 