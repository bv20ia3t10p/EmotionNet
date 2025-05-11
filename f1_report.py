#!/usr/bin/env python3
# F1 Score Report Generator for Emotion Recognition Model

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Fix import path issues
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

# Try to import the model
try:
    from balanced_training import EnhancedResEmoteNet, EmotionDataset, load_data, EMOTIONS
    print("Successfully imported classes from balanced_training")
except ImportError:
    print("Error importing from balanced_training, defining classes here")
    
    # Define EMOTIONS dictionary
    EMOTIONS = {
        0: 'angry',
        1: 'disgust',
        2: 'fear', 
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    # Import necessary modules
    import cv2
    from PIL import Image
    
    # Define EmotionDataset class
    class EmotionDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Read image
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    # Return a placeholder image
                    img = np.zeros((48, 48), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                # Return a placeholder image
                img = np.zeros((48, 48), dtype=np.uint8)
            
            # Apply standard transforms
            if self.transform:
                img = Image.fromarray(img)
                img = self.transform(img)
            
            return img, label
    
    # Define load_data function
    def load_data(data_dir):
        """
        Load images and labels from directory
        """
        image_paths = []
        labels = []
        
        # Scan through data directory
        for emotion_idx, emotion in EMOTIONS.items():
            emotion_dir = os.path.join(data_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} not found, skipping...")
                continue
            
            # Get all image files in this emotion directory
            files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Found {len(files)} images for {emotion}")
            
            # Add to dataset
            image_paths.extend(files)
            labels.extend([emotion_idx] * len(files))
        
        return image_paths, np.array(labels)
    
    # Import torchvision models for fallback model
    import torchvision.models as models
    
    # Define a simplified version of the model for compatibility
    class EnhancedResEmoteNet(nn.Module):
        def __init__(self, num_classes=7, dropout_rate=0.4, backbone='resnet18', 
                     use_fpn=True, use_landmarks=True, use_contrastive=True):
            super(EnhancedResEmoteNet, self).__init__()
            
            # Load pretrained backbone model
            if backbone == 'resnet18':
                base_model = models.resnet18(pretrained=True)
                feature_dim = 512
            elif backbone == 'efficientnet_b0':
                base_model = models.efficientnet_b0(pretrained=True)
                feature_dim = 1280
            elif backbone == 'mobilenet_v3_small':
                base_model = models.mobilenet_v3_small(pretrained=True)
                feature_dim = 576
            
            # Use most of the model except the final layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            
            # Simple classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )
            
            # Custom feature_extractor attribute for compatibility
            self.feature_extractor = self.backbone
            self.fpn = nn.Identity()
            
        def forward(self, x):
            # Feature extraction
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            
            # Classification
            logits = self.classifier(features)
            
            return logits

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data and generate detailed F1 score reports
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            
            # Handle different return types from the model
            outputs = model(inputs)
            
            # Extract logits based on model output type
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    # For models that return logits, valence, arousal
                    logits = outputs[0]
                else:
                    # For other tuple returns, assume first element is logits
                    logits = outputs[0]
            else:
                # For models that return only logits
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate overall metrics
    accuracy = np.mean(all_preds == all_labels) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Macro F1 Score: {macro_f1:.2f}%")
    print(f"Weighted F1 Score: {weighted_f1:.2f}%")
    
    # Generate and print classification report
    print("\nDetailed Classification Report:")
    target_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Detailed per-class metrics
    print("\nDetailed Per-Class Metrics:")
    print("----------------------------")
    for class_idx, class_name in EMOTIONS.items():
        # Count true instances of this class
        class_total = np.sum(all_labels == class_idx)
        if class_total == 0:
            print(f"{class_name}: No test samples")
            continue
        
        # Calculate metrics for this class
        class_mask = all_labels == class_idx
        class_correct = np.sum((all_preds == class_idx) & class_mask)
        class_acc = class_correct / class_total * 100
        
        # Binary classification metrics
        y_true_bin = all_labels == class_idx
        y_pred_bin = all_preds == class_idx
        
        # Calculate individual F1 score
        class_f1 = f1_score(y_true_bin, y_pred_bin) * 100
        
        # Calculate precision and recall 
        class_precision = report[class_name]['precision'] * 100
        class_recall = report[class_name]['recall'] * 100
        
        # Get predicted probability distribution for this class
        class_probs = all_probs[:, class_idx][class_mask]
        mean_confidence = np.mean(class_probs) * 100 if len(class_probs) > 0 else 0
        
        # Print metrics
        print(f"{class_name}:")
        print(f"  Samples: {class_total}")
        print(f"  Accuracy: {class_acc:.2f}%")
        print(f"  Precision: {class_precision:.2f}%")
        print(f"  Recall: {class_recall:.2f}%")
        print(f"  F1 Score: {class_f1:.2f}%")
        print(f"  Mean Confidence: {mean_confidence:.2f}%")
    
    # Create and save confusion matrix
    create_confusion_matrix(all_labels, all_preds, "detailed_confusion_matrix.png")
    
    # Create and save F1 score bar chart
    create_f1_bar_chart(report, "f1_scores.png")
    
    return report

def create_confusion_matrix(true_labels, pred_labels, save_path):
    """Create and save a detailed confusion matrix visualization"""
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot normalized confusion matrix
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
              xticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))],
              yticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    
    # Plot raw counts
    ax2 = plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))],
              yticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Raw Counts Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")

def create_f1_bar_chart(report, save_path):
    """Create and save a bar chart of F1 scores for each class"""
    # Extract F1 scores for each class
    classes = list(EMOTIONS.values())
    f1_scores = [report[class_name]['f1-score'] * 100 for class_name in classes]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, f1_scores, color='skyblue')
    
    # Add value labels above bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}%', ha='center', va='bottom')
    
    # Add average line
    plt.axhline(y=report['macro avg']['f1-score'] * 100, color='r', linestyle='-', 
               label=f"Macro Avg: {report['macro avg']['f1-score']*100:.1f}%")
    
    # Add labels and title
    plt.ylim(0, 105)  # Set y-axis limit to accommodate labels
    plt.xlabel('Emotion Class')
    plt.ylabel('F1 Score (%)')
    plt.title('F1 Scores by Emotion Class')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"F1 score chart saved to {save_path}")

def compare_augmentation_impact(model, test_loader, device, test_dir, transform):
    """
    Compare model performance on original vs augmented images
    to evaluate if face-preserving augmentations help or hurt
    """
    model.eval()
    
    # Import necessary modules if not already imported
    import albumentations as A
    import cv2
    import random
    from PIL import Image
    
    # Create face-preserving augmentation pipeline
    face_aug = A.Compose([
        # Non-geometric transformations that preserve facial structure
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.GaussNoise(var_limit=(5, 20), p=1.0),
        
        # Very gentle blur that won't remove important facial details
        A.GaussianBlur(blur_limit=1, p=0.5),
        
        # Very minimal geometric transformations to preserve facial alignment
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0,
                          border_mode=cv2.BORDER_CONSTANT, value=0),
    ])
    
    # Create more aggressive augmentation pipeline
    aggressive_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        A.GaussNoise(p=1.0),
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=35, p=1.0),
    ])
    
    # Reload test data
    test_paths, test_labels = load_data(test_dir)
    print(f"Comparing augmentation impact using {len(test_paths)} test samples")
    
    # Prepare test image batches with different augmentation strategies
    results = {}
    augmentation_types = {
        'original': None,  # No augmentation
        'face_preserving': face_aug,  # Our face-preserving augmentation
        'aggressive': aggressive_aug  # More aggressive augmentation
    }
    
    for aug_name, aug_pipeline in augmentation_types.items():
        print(f"\nEvaluating with {aug_name} augmentation...")
        
        all_preds = []
        all_labels = []
        
        # Process each test image
        for img_path, label in tqdm(zip(test_paths, test_labels), total=len(test_paths)):
            # Read and preprocess the image
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip failed images
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply augmentation if specified
                if aug_pipeline is not None:
                    img = aug_pipeline(image=img)["image"]
                
                # Convert to tensor
                img_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    
                    # Handle different return types
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    _, pred = torch.max(logits, 1)
                
                all_preds.append(pred.item())
                all_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Calculate metrics for this augmentation strategy
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        # Generate class-specific F1 scores
        class_f1_scores = {}
        for class_idx, class_name in EMOTIONS.items():
            # Binary classification metrics
            y_true_bin = np.array(all_labels) == class_idx
            y_pred_bin = np.array(all_preds) == class_idx
            
            if np.sum(y_true_bin) > 0:  # Only calculate if class exists in true labels
                class_f1 = f1_score(y_true_bin, y_pred_bin) * 100
                class_f1_scores[class_name] = class_f1
            else:
                class_f1_scores[class_name] = 0.0
        
        # Store results for this augmentation strategy
        results[aug_name] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'class_f1_scores': class_f1_scores
        }
        
        print(f"{aug_name} - Accuracy: {accuracy:.2f}%, Macro F1: {macro_f1:.2f}%")
        for class_name, f1 in class_f1_scores.items():
            print(f"  {class_name}: F1 = {f1:.2f}%")
    
    # Create comparative visualizations
    create_augmentation_comparison_chart(results, "augmentation_comparison.png")
    
    return results

def create_augmentation_comparison_chart(results, save_path):
    """Create and save a bar chart comparing F1 scores across augmentation strategies"""
    # Extract data for plotting
    aug_names = list(results.keys())
    class_names = list(EMOTIONS.values())
    
    # Prepare figure
    plt.figure(figsize=(15, 10))
    
    # Plot overall metrics
    plt.subplot(2, 1, 1)
    accuracy_values = [results[aug]['accuracy'] for aug in aug_names]
    f1_values = [results[aug]['macro_f1'] for aug in aug_names]
    
    x = np.arange(len(aug_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracy_values, width, label='Accuracy')
    plt.bar(x + width/2, f1_values, width, label='Macro F1')
    
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Score (%)')
    plt.title('Overall Performance by Augmentation Strategy')
    plt.xticks(x, aug_names)
    plt.legend()
    
    # Plot class-wise F1 scores
    plt.subplot(2, 1, 2)
    
    # Create grouped bar chart
    bar_width = 0.8 / len(aug_names)
    x = np.arange(len(class_names))
    
    for i, aug_name in enumerate(aug_names):
        class_f1_values = [results[aug_name]['class_f1_scores'][class_name] for class_name in class_names]
        offset = (i - len(aug_names)/2 + 0.5) * bar_width
        plt.bar(x + offset, class_f1_values, bar_width, label=aug_name)
    
    plt.xlabel('Emotion Class')
    plt.ylabel('F1 Score (%)')
    plt.title('Class F1 Scores by Augmentation Strategy')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Augmentation comparison chart saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate detailed F1 score reports for emotion recognition model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test data directory')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                      choices=['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                      help='Backbone architecture used in the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--compare_aug', action='store_true', help='Compare performance on augmented vs unaugmented images')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations for test data
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    test_paths, test_labels = load_data(args.test_dir)
    print(f"Loaded {len(test_paths)} test samples")
    
    # Create test dataset
    test_dataset = EmotionDataset(
        test_paths, test_labels, transform=transform
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    try:
        model = EnhancedResEmoteNet(
            num_classes=7,
            backbone=args.backbone
        )
    except Exception as e:
        print(f"Error creating model with parameters: {e}")
        print("Trying with minimal parameters...")
        model = EnhancedResEmoteNet(
            num_classes=7
        )
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model weights from {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using model with random initialization")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Evaluate model and generate reports
    evaluate_model(model, test_loader, device)
    
    # Optionally compare augmentation impact
    if args.compare_aug:
        compare_augmentation_impact(model, test_loader, device, args.test_dir, transform)

if __name__ == '__main__':
    main() 