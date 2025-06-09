import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import warnings
import importlib.util
import sys

warnings.filterwarnings('ignore')

# Import model from train.py
def import_model_from_train():
    """Dynamically import model classes from train.py"""
    spec = importlib.util.spec_from_file_location("train_module", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train_module"] = train_module
    spec.loader.exec_module(train_module)
    return train_module

# Use the proper FERPlus dataset that handles both CSV files
def create_dataset(split, transform):
    """Create dataset using the same approach as train.py"""
    train_module = import_model_from_train()
    return train_module.FERPlusDataset(
        csv_file='FERPlus-master/fer2013new.csv',
        fer2013_csv_file='fer2013.csv',  # Original CSV is in root directory
        split=split,
        transform=transform,
        min_votes=2
    )

def calculate_metrics(all_preds, all_labels, emotion_names):
    """Calculate comprehensive metrics including F1 scores"""
    # Convert to numpy arrays
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu().numpy()
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.cpu().numpy()
    
    # Ensure we have numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # F1 scores
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class F1 scores
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Precision and Recall
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Create per-class metrics dict
    per_class_metrics = {}
    for i, emotion in enumerate(emotion_names):
        mask = (all_labels == i)
        support = np.sum(mask)
            
        if support > 0:
            per_class_metrics[emotion] = {
                'f1': f1_per_class[i],
                'precision': precision_score(all_labels == i, all_preds == i, zero_division=0),
                'recall': recall_score(all_labels == i, all_preds == i, zero_division=0),
                'support': support
            }
        else:
            per_class_metrics[emotion] = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 0}
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'per_class': per_class_metrics
    }

def print_metrics(metrics, epoch, phase):
    """Print detailed metrics in a nice format"""
    print(f"\n{'='*60}")
    print(f"{phase.upper()} METRICS - EPOCH {epoch}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"F1 Macro:         {metrics['f1_macro']:.4f}")
    print(f"F1 Micro:         {metrics['f1_micro']:.4f}")
    print(f"F1 Weighted:      {metrics['f1_weighted']:.4f}")
    print(f"Precision Macro:  {metrics['precision_macro']:.4f}")
    print(f"Recall Macro:     {metrics['recall_macro']:.4f}")
    
    print(f"\nPer-Class Performance:")
    print("-" * 60)
    print(f"{'Class':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
    print("-" * 60)
    for emotion, scores in metrics['per_class'].items():
        print(f"{emotion:<12} {scores['f1']:<8.3f} {scores['precision']:<10.3f} "
              f"{scores['recall']:<8.3f} {scores['support']:<8}")
    print("=" * 60)

def conservative_finetune():
    """Conservative fine-tuning that preserves model performance"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Conservative Fine-tuning for HybridFER-Max')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (conservative)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Very low learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--min_improvement', type=float, default=0.001, help='Minimum improvement threshold')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import model architecture
    train_module = import_model_from_train()
    
    # Load pretrained model
    print(f"Loading pretrained model from {args.pretrained_path}...")
    model = train_module.HybridFERMax(
        img_size=112,
        patch_size=16,
        num_classes=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1
    )
    
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        original_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"✅ Loaded model with original validation accuracy: {original_val_acc:.4f}")
    else:
        model.load_state_dict(checkpoint)
        original_val_acc = 0.0
        print("✅ Pretrained model loaded successfully!")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # **CONSERVATIVE DATA SETUP**
    # Minimal augmentation to preserve original data distribution
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced augmentation
        transforms.RandomRotation(degrees=5),     # Minimal rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Light color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*60)
    print("CONSERVATIVE FINE-TUNING SETUP")
    print("="*60)
    
    # Load datasets - NO AUGMENTATION OF MINORITY CLASSES
    train_dataset = create_dataset('Training', train_transform)
    val_dataset = create_dataset('PublicTest', val_transform)
    
    # Create data loaders - STANDARD SAMPLING
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # **CONSERVATIVE OPTIMIZER SETUP**
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,  # Very low LR
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # Gentle learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate/10
    )
    
    # **STANDARD LOSS FUNCTION**
    criterion = nn.CrossEntropyLoss()
    
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("="*60)
    
    # Training tracking
    best_val_acc = original_val_acc
    best_f1_macro = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (images, hard_labels, soft_labels) in enumerate(train_pbar):
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, hard_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions for metrics
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(hard_labels.cpu().numpy())
            
            # Update progress bar
            current_acc = (np.array(train_preds) == np.array(train_labels)).mean()
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2%}'
            })
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(train_preds, train_labels, emotion_names)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, hard_labels, soft_labels in val_loader:
                images = images.to(device)
                hard_labels = hard_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, hard_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(hard_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(val_preds, val_labels, emotion_names)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']:.2%})")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']:.2%})")
        print(f"Train F1 Macro: {train_metrics['f1_macro']:.4f}")
        print(f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Print detailed metrics every 5 epochs or if new best
        if epoch % 5 == 0 or val_metrics['accuracy'] > best_val_acc:
            print_metrics(train_metrics, epoch, "TRAINING")
            print_metrics(val_metrics, epoch, "VALIDATION")
        
        # **CONSERVATIVE EARLY STOPPING**
        improvement = val_metrics['accuracy'] - best_val_acc
        f1_improvement = val_metrics['f1_macro'] - best_f1_macro
        
        if improvement > args.min_improvement:
            best_val_acc = val_metrics['accuracy']
            best_f1_macro = val_metrics['f1_macro']
            patience_counter = 0
            
            # Save best model
            save_path = 'checkpoints/conservative_finetuned_best.pth'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': best_val_acc,
                'val_f1': best_f1_macro,
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['f1_macro'],
                'original_val_acc': original_val_acc,
                'args': args
            }, save_path)
            
            print(f"✅ New best model saved! Val Acc: {best_val_acc:.4f}, Val F1: {best_f1_macro:.4f}")
            print(f"   Improvement over original: {best_val_acc - original_val_acc:+.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
            if improvement < 0:
                print(f"⚠️  Performance dropped by {-improvement:.4f}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"Best validation F1: {best_f1_macro:.4f}")
            break
        
        print("-" * 80)
    
    print(f"\n🎯 CONSERVATIVE FINE-TUNING COMPLETED!")
    print(f"Original validation accuracy: {original_val_acc:.4f}")
    print(f"Final validation accuracy: {best_val_acc:.4f}")
    print(f"Net improvement: {best_val_acc - original_val_acc:+.4f}")
    print(f"Best validation F1 macro: {best_f1_macro:.4f}")
    
    if best_val_acc > original_val_acc:
        print("✅ Fine-tuning was successful!")
    else:
        print("❌ Fine-tuning did not improve performance")

if __name__ == "__main__":
    conservative_finetune()