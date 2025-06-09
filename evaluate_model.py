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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import importlib.util
import sys
from itertools import cycle
warnings.filterwarnings('ignore')

# Optional import for torchsummary
try:
    from torchsummary import summary
    HAS_TORCHSUMMARY = True
except ImportError:
    HAS_TORCHSUMMARY = False

# Dynamically import model classes from train.py
def import_model_from_train():
    """Dynamically import model classes from train.py"""
    spec = importlib.util.spec_from_file_location("train", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = train_module
    spec.loader.exec_module(train_module)
    return train_module


def show_model_architecture(model, input_size=(3, 112, 112)):
    """Display detailed model architecture"""
    print(f"\n{'='*80}")
    print(f"MODEL ARCHITECTURE DETAILS")
    print(f"{'='*80}")
    
    # Basic model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Name: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Model configuration
    if hasattr(model, 'num_classes'):
        print(f"Number of Classes: {model.num_classes}")
    if hasattr(model, 'embed_dim'):
        print(f"Embedding Dimension: {model.embed_dim}")
    if hasattr(model, 'depth'):
        print(f"Transformer Depth: {model.depth}")
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'num_patches'):
        print(f"Number of Patches: {model.patch_embed.num_patches}")
    
    print(f"\n{'='*80}")
    print(f"LAYER-BY-LAYER BREAKDOWN")
    print(f"{'='*80}")
    
    # Show major components
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:20} | {str(type(module).__name__):25} | Parameters: {params:,}")
        
        # Show sub-modules for transformer blocks
        if name == 'blocks' and hasattr(module, '__len__'):
            print(f"  └── {len(module)} Transformer Blocks:")
            for i, block in enumerate(module[:3]):  # Show first 3 blocks
                block_params = sum(p.numel() for p in block.parameters())
                print(f"      Block {i:2d} | Parameters: {block_params:,}")
            if len(module) > 3:
                print(f"      ... and {len(module)-3} more blocks")
    
    # Try to show model summary if torchsummary is available
    if HAS_TORCHSUMMARY:
        try:
            print(f"\n{'='*80}")
            print(f"DETAILED MODEL SUMMARY")
            print(f"{'='*80}")
            summary(model, input_size, device=next(model.parameters()).device.type)
        except Exception as e:
            print(f"\nNote: Could not display detailed summary: {e}")
    else:
        print(f"\n{'='*80}")
        print(f"DETAILED MODEL SUMMARY (UNAVAILABLE)")
        print(f"{'='*80}")
        print("Install torchsummary for detailed layer-by-layer summary:")
        print("pip install torchsummary")


class FERPlusDataset:
    """
    FERPlus Dataset class for loading and processing FER2013+ data
    """
    
    def __init__(self, csv_file, fer2013_csv_file, split='PrivateTest', 
                 transform=None, min_votes=2, quality_threshold=8):
        """
        Args:
            csv_file: Path to fer2013new.csv (FERPlus annotations)
            fer2013_csv_file: Path to original fer2013.csv (contains image pixels)
            split: 'Training', 'PublicTest', or 'PrivateTest'
            transform: Image transformations
            min_votes: Minimum votes required for a sample
            quality_threshold: Minimum quality score (max votes - unknown votes)
        """
        self.csv_file = csv_file
        self.fer2013_csv_file = fer2013_csv_file
        self.split = split
        self.transform = transform
        self.min_votes = min_votes
        self.quality_threshold = quality_threshold
        
        # FERPlus emotion mapping (8 classes)
        self.emotion_map = {
            0: 'neutral',
            1: 'happiness', 
            2: 'surprise',
            3: 'sadness',
            4: 'anger',
            5: 'disgust',
            6: 'fear',
            7: 'contempt'
        }
        
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        # Load and process data
        self.load_data()
        
    def load_data(self):
        """Load and process FERPlus dataset"""
        print(f"Loading FERPlus data for {self.split} split...")
        
        # Load FERPlus annotations
        ferplus_df = pd.read_csv(self.csv_file)
        
        # Load original FER2013 data (contains pixel data)
        fer2013_df = pd.read_csv(self.fer2013_csv_file)
        
        # Filter by split
        if self.split == 'Training':
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'Training']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'Training']
        elif self.split == 'PublicTest':
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'PublicTest']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'PublicTest']
        else:  # PrivateTest
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'PrivateTest']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'PrivateTest']
        
        # Reset indices for proper alignment
        ferplus_split = ferplus_split.reset_index(drop=True)
        fer2013_split = fer2013_split.reset_index(drop=True)
        
        # Process samples
        self.samples = []
        self.labels = []
        self.soft_labels = []
        
        # Ensure both dataframes have the same length
        min_length = min(len(ferplus_split), len(fer2013_split))
        
        for idx in range(min_length):
            try:
                ferplus_row = ferplus_split.iloc[idx]
                fer2013_row = fer2013_split.iloc[idx]
                
                # Get emotion vote counts
                emotion_votes = [ferplus_row[col] for col in self.emotion_columns]
                emotion_votes = [int(v) if pd.notna(v) else 0 for v in emotion_votes]
                
                # Quality filtering
                total_votes = sum(emotion_votes)
                unknown_votes = int(ferplus_row['unknown']) if pd.notna(ferplus_row['unknown']) else 0
                quality_score = total_votes - unknown_votes
                
                if total_votes < self.min_votes or quality_score < self.quality_threshold:
                    continue
                
                # Get pixel string and convert to image
                pixel_string = fer2013_row['pixels']
                
                if pd.notna(pixel_string):
                    pixels = np.array([int(pixel) for pixel in pixel_string.split()])
                    image = pixels.reshape(48, 48).astype(np.uint8)
                    
                    # Create hard label (majority vote)
                    hard_label = np.argmax(emotion_votes)
                    
                    # Create soft label (probability distribution)
                    soft_label = np.array(emotion_votes, dtype=np.float32)
                    if soft_label.sum() > 0:
                        soft_label = soft_label / soft_label.sum()
                    else:
                        # Fallback to uniform distribution
                        soft_label = np.ones(len(emotion_votes), dtype=np.float32) / len(emotion_votes)
                    
                    self.samples.append(image)
                    self.labels.append(hard_label)
                    self.soft_labels.append(soft_label)
                    
            except (IndexError, ValueError, KeyError) as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} {self.split} samples")
        
        # Print class distribution
        if len(self.labels) > 0:
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"Class distribution in {self.split}:")
            for emotion_idx, count in zip(unique, counts):
                if emotion_idx < len(self.emotion_columns):
                    emotion_name = self.emotion_columns[emotion_idx]
                    print(f"  {emotion_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get image and label
        image = self.samples[idx]
        hard_label = self.labels[idx]
        soft_label = self.soft_labels[idx]
        
        # Convert grayscale to RGB
        image = np.stack([image, image, image], axis=2)
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)


def create_test_transform():
    """Create test transform for evaluation"""
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, device, train_module):
    """Load the trained model from .pth file"""
    print(f"Loading model from {model_path}...")
    
    # Load the checkpoint to inspect its structure
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to extract model configuration from the checkpoint
    # This is a simplified approach - ideally, config should be saved with the model
    print("Analyzing model checkpoint...")
    
    # Get model class from train module
    HybridFERMax = getattr(train_module, 'HybridFERMax')
    
    # Create model with default architecture (should match training config)
    # In a production setting, model config should be saved with checkpoint
    model = HybridFERMax(
        img_size=112,
        patch_size=16,
        num_classes=8,
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(device)
    
    # Load the saved weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded successfully!")
    
    # Show detailed model architecture
    show_model_architecture(model)
    
    return model


def plot_multiclass_roc_curves(y_true, y_proba, emotion_names):
    """Plot ROC curves for each class in multiclass setting"""
    n_classes = len(emotion_names)
    
    # Convert labels to one-hot encoding
    y_true_onehot = np.eye(n_classes)[y_true]
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(12, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
    
    for i, color in zip(range(n_classes), colors):
        if np.sum(y_true_onehot[:, i]) > 0:  # Only plot if class exists in test set
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
            roc_auc[i] = roc_auc_score(y_true_onehot[:, i], y_proba[:, i])
            
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{emotion_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_proba.ravel())
    roc_auc["micro"] = roc_auc_score(y_true_onehot, y_proba, average="micro")
    
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
    
    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if i in fpr:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    roc_auc["macro"] = np.mean([roc_auc[i] for i in range(n_classes) if i in roc_auc])
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle=':', linewidth=4,
             label=f'Macro-avg (AUC = {roc_auc["macro"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves - One-vs-Rest')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('evaluation_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc


def plot_multiclass_precision_recall_curves(y_true, y_proba, emotion_names):
    """Plot Precision-Recall curves for each class"""
    n_classes = len(emotion_names)
    
    # Convert labels to one-hot encoding
    y_true_onehot = np.eye(n_classes)[y_true]
    
    # Compute precision-recall curve and average precision for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    plt.figure(figsize=(12, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
    
    for i, color in zip(range(n_classes), colors):
        if np.sum(y_true_onehot[:, i]) > 0:  # Only plot if class exists in test set
            precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_proba[:, i])
            average_precision[i] = average_precision_score(y_true_onehot[:, i], y_proba[:, i])
            
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{emotion_names[i]} (AP = {average_precision[i]:.3f})')
    
    # Compute micro-average precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_onehot.ravel(), y_proba.ravel())
    average_precision["micro"] = average_precision_score(y_true_onehot, y_proba, average="micro")
    
    plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-avg (AP = {average_precision["micro"]:.3f})')
    
    # Compute macro-average
    macro_ap = np.mean([average_precision[i] for i in range(n_classes) if i in average_precision])
    plt.axhline(y=macro_ap, color='navy', linestyle=':', linewidth=4,
                label=f'Macro-avg (AP = {macro_ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('evaluation_precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return average_precision


def compute_multiclass_metrics(y_true, y_pred, y_proba, emotion_names):
    """Compute comprehensive multiclass metrics"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MULTICLASS EVALUATION METRICS")
    print(f"{'='*80}")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Macro-averaged metrics
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nMACRO-AVERAGED METRICS:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    
    # Micro-averaged metrics
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print(f"\nMICRO-AVERAGED METRICS:")
    print(f"  Precision: {micro_precision:.4f}")
    print(f"  Recall:    {micro_recall:.4f}")
    print(f"  F1-Score:  {micro_f1:.4f}")
    
    # Weighted-averaged metrics
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\nWEIGHTED-AVERAGED METRICS:")
    print(f"  Precision: {weighted_precision:.4f}")
    print(f"  Recall:    {weighted_recall:.4f}")
    print(f"  F1-Score:  {weighted_f1:.4f}")
    
    # Multiclass AUC scores
    try:
        # Convert labels to one-hot for AUC calculation
        n_classes = len(emotion_names)
        y_true_onehot = np.eye(n_classes)[y_true]
        
        macro_auc = roc_auc_score(y_true_onehot, y_proba, average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true_onehot, y_proba, average='micro', multi_class='ovr')
        weighted_auc = roc_auc_score(y_true_onehot, y_proba, average='weighted', multi_class='ovr')
        
        print(f"\nMULTICLASS AUC SCORES:")
        print(f"  Macro AUC:    {macro_auc:.4f}")
        print(f"  Micro AUC:    {micro_auc:.4f}")
        print(f"  Weighted AUC: {weighted_auc:.4f}")
        
    except Exception as e:
        print(f"\nNote: Could not compute AUC scores: {e}")
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def plot_class_distribution_and_performance(y_true, y_pred, emotion_names):
    """Plot class distribution and per-class performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class distribution in test set
    unique, counts = np.unique(y_true, return_counts=True)
    ax1.bar(range(len(unique)), counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Emotion Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution in Test Set')
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels([emotion_names[i] for i in unique], rotation=45)
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    bars = ax2.bar(range(len(emotion_names)), class_accuracies, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Emotion Classes')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_xticks(range(len(emotion_names)))
    ax2.set_xticklabels(emotion_names, rotation=45)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Per-class F1-scores
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    bars = ax3.bar(range(len(emotion_names)), f1_scores, color='lightgreen', alpha=0.7)
    ax3.set_xlabel('Emotion Classes')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Per-Class F1-Score')
    ax3.set_xticks(range(len(emotion_names)))
    ax3.set_xticklabels(emotion_names, rotation=45)
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Prediction distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax4.bar(range(len(unique_pred)), counts_pred, color='gold', alpha=0.7)
    ax4.set_xlabel('Emotion Classes')
    ax4.set_ylabel('Number of Predictions')
    ax4.set_title('Prediction Distribution')
    ax4.set_xticks(range(len(unique_pred)))
    ax4.set_xticklabels([emotion_names[i] for i in unique_pred], rotation=45)
    
    plt.tight_layout()
    plt.savefig('evaluation_class_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, dataloader, device, emotion_names):
    """Evaluate model on test set with comprehensive multiclass metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, hard_labels, soft_labels in tqdm(dataloader, desc='Testing'):
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Compute comprehensive multiclass metrics
    metrics = compute_multiclass_metrics(all_labels, all_predictions, all_probabilities, emotion_names)
    
    # Detailed classification report
    print(f"\n{'='*80}")
    print(f"DETAILED CLASSIFICATION REPORT")
    print(f"{'='*80}")
    print(classification_report(all_labels, all_predictions, target_names=emotion_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title(f'Confusion Matrix - Test Accuracy: {metrics["accuracy"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot class distribution and performance analysis
    plot_class_distribution_and_performance(all_labels, all_predictions, emotion_names)
    
    # Plot multiclass ROC curves
    roc_auc_scores = plot_multiclass_roc_curves(all_labels, all_probabilities, emotion_names)
    
    # Plot multiclass Precision-Recall curves
    ap_scores = plot_multiclass_precision_recall_curves(all_labels, all_probabilities, emotion_names)
    
    # Per-class accuracy
    print(f"\n{'='*80}")
    print(f"PER-CLASS PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    f1_scores = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    for i, emotion in enumerate(emotion_names):
        support = np.sum(all_labels == i)
        print(f"{emotion:10} | Acc: {class_accuracies[i]:.4f} | "
              f"F1: {f1_scores[i]:.4f} | Support: {support:4d}")
    
    print(f"\n{'='*80}")
    print(f"FILES SAVED:")
    print(f"  • evaluation_confusion_matrix.png")
    print(f"  • evaluation_class_analysis.png") 
    print(f"  • evaluation_roc_curves.png")
    print(f"  • evaluation_precision_recall_curves.png")
    print(f"{'='*80}")
    
    return metrics, all_predictions, all_labels, all_probabilities


def main():
    parser = argparse.ArgumentParser(description='Evaluate HybridFER-Max model with comprehensive multiclass metrics')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the saved .pth model file')
    parser.add_argument('--ferplus_csv', type=str, default='./FERPlus-master/fer2013new.csv',
                       help='Path to FERPlus annotations CSV')
    parser.add_argument('--fer2013_csv', type=str, default='./fer2013.csv',
                       help='Path to original FER2013 CSV with pixel data')
    parser.add_argument('--split', type=str, default='PrivateTest', 
                       choices=['Training', 'PublicTest', 'PrivateTest'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--show_architecture_only', action='store_true',
                       help='Only show model architecture without evaluation')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
    if not os.path.exists('train.py'):
        print(f"Error: train.py not found! This file is needed to import model classes.")
        return
    
    # Dynamically import model classes from train.py
    print("Importing model classes from train.py...")
    train_module = import_model_from_train()
    
    # Load model
    model = load_model(args.model_path, device, train_module)
    
    if args.show_architecture_only:
        print("\n✅ Model architecture displayed successfully!")
        return
    
    # Check dataset files
    if not os.path.exists(args.ferplus_csv):
        print(f"Error: FERPlus CSV {args.ferplus_csv} not found!")
        return
        
    if not os.path.exists(args.fer2013_csv):
        print(f"Error: FER2013 CSV {args.fer2013_csv} not found!")
        return
    
    # Get dataset class from train module
    FERPlusDataset = getattr(train_module, 'FERPlusDataset')
    
    # Create test dataset
    test_transform = create_test_transform()
    test_dataset = FERPlusDataset(
        csv_file=args.ferplus_csv,
        fer2013_csv_file=args.fer2013_csv,
        split=args.split,
        transform=test_transform
    )
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Emotion names
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    # Evaluate model
    metrics, predictions, labels, probabilities = evaluate_model(
        model, test_loader, device, emotion_names
    )
    
    print(f"\n🎉 MULTICLASS EVALUATION COMPLETED!")
    print(f"📊 Comprehensive metrics and visualizations have been generated.")


if __name__ == "__main__":
    main()


# Usage examples:
# 
# 1. Comprehensive multiclass evaluation:
#    python evaluate_model.py --model_path best_hybridfer_max.pth
#
# 2. Show only model architecture:
#    python evaluate_model.py --model_path best_hybridfer_max.pth --show_architecture_only
#
# 3. Evaluate on different split:
#    python evaluate_model.py --model_path best_hybridfer_max.pth --split PublicTest
#
# 4. Custom batch size:
#    python evaluate_model.py --model_path best_hybridfer_max.pth --batch_size 32