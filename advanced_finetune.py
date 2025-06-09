import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
import math
import random
from collections import defaultdict

warnings.filterwarnings('ignore')

# Import model from train.py
def import_model_from_train():
    """Dynamically import model classes from train.py"""
    spec = importlib.util.spec_from_file_location("train_module", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train_module"] = train_module
    spec.loader.exec_module(train_module)
    return train_module

# Advanced Focal Loss with Dynamic Gamma
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, epoch=0):
        # Dynamic gamma - increases focus on hard examples over time
        dynamic_gamma = self.gamma + (epoch * 0.1)
        dynamic_gamma = min(dynamic_gamma, 5.0)  # Cap at 5.0
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** dynamic_gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** dynamic_gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Smart Minority Class Augmentation
class SmartMinorityAugmentation:
    def __init__(self, target_classes=['disgust', 'contempt', 'fear'], augment_factor=3):
        self.target_classes = target_classes
        self.augment_factor = augment_factor
        self.emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                             'anger', 'disgust', 'fear', 'contempt']
        
    def create_synthetic_samples(self, dataset, class_idx, num_samples):
        """Create high-quality synthetic samples for minority class"""
        # Find all samples of this class
        class_indices = []
        for i in range(len(dataset)):
            _, hard_label, _ = dataset[i]
            if hard_label == class_idx:
                class_indices.append(i)
        
        if len(class_indices) < 2:
            return []
        
        synthetic_samples = []
        for _ in range(num_samples):
            # Pick two random samples from same class
            idx1, idx2 = random.sample(class_indices, 2)
            img1, _, soft1 = dataset[idx1]
            img2, _, soft2 = dataset[idx2]
            
            # Advanced mixup with careful blending
            lambda_val = np.random.beta(0.4, 0.4)  # More conservative mixing
            
            # Mix images in tensor space (after transform)
            mixed_img = lambda_val * img1 + (1 - lambda_val) * img2
            mixed_soft = lambda_val * soft1 + (1 - lambda_val) * soft2
            
            synthetic_samples.append((mixed_img, class_idx, mixed_soft))
        
        return synthetic_samples

# Progressive Unfreezing Strategy
class ProgressiveUnfreezing:
    def __init__(self, model, unfreeze_schedule=[5, 10, 15]):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.current_stage = 0
        
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False
            
        # Only unfreeze head initially
        for param in model.head_global.parameters():
            param.requires_grad = True
        for param in model.head_local.parameters():
            param.requires_grad = True
        for param in model.head_micro.parameters():
            param.requires_grad = True
            
    def step(self, epoch):
        if self.current_stage < len(self.unfreeze_schedule):
            if epoch >= self.unfreeze_schedule[self.current_stage]:
                self._unfreeze_next_layer()
                self.current_stage += 1
                
    def _unfreeze_next_layer(self):
        if self.current_stage == 0:
            # Unfreeze top transformer blocks
            for i in range(-3, 0):  # Last 3 blocks
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True
            print("🔓 Unfroze top 3 transformer blocks")
            
        elif self.current_stage == 1:
            # Unfreeze middle transformer blocks
            for i in range(-6, -3):  # Middle 3 blocks
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True
            print("🔓 Unfroze middle transformer blocks")
            
        elif self.current_stage == 2:
            # Unfreeze all remaining layers
            for param in self.model.parameters():
                param.requires_grad = True
            print("🔓 Unfroze all layers")

# Class-specific Learning Rate Optimizer
class ClassSpecificOptimizer:
    def __init__(self, model, base_lr=1e-5, minority_lr_boost=3.0):
        self.model = model
        self.base_lr = base_lr
        self.minority_lr_boost = minority_lr_boost
        
        # Create parameter groups with different learning rates
        self.param_groups = []
        
        # Higher LR for heads dealing with minority classes
        minority_head_params = list(model.head_global.parameters()) + \
                              list(model.head_local.parameters()) + \
                              list(model.head_micro.parameters())
        
        self.param_groups.append({
            'params': minority_head_params,
            'lr': base_lr * minority_lr_boost,
            'name': 'minority_heads'
        })
        
        # Normal LR for other parameters
        other_params = []
        head_param_ids = {id(p) for p in minority_head_params}
        for param in model.parameters():
            if id(param) not in head_param_ids:
                other_params.append(param)
                
        self.param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'backbone'
        })
        
    def create_optimizer(self):
        return optim.AdamW(self.param_groups, weight_decay=1e-4)# Create enhanced dataset
def create_enhanced_dataset(split, transform):
    """Create dataset with smart minority augmentation"""
    train_module = import_model_from_train()
    base_dataset = train_module.FERPlusDataset(
        csv_file='FERPlus-master/fer2013new.csv',
        fer2013_csv_file='fer2013.csv',
        split=split,
        transform=transform,
        min_votes=2
    )
    
    if split == 'Training':
        # Apply smart augmentation for minority classes
        augmenter = SmartMinorityAugmentation()
        
        # Count samples per class
        class_counts = defaultdict(int)
        for i in range(len(base_dataset)):
            _, hard_label, _ = base_dataset[i]
            class_counts[hard_label] += 1
            
        print(f"Original class distribution:")
        emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                        'anger', 'disgust', 'fear', 'contempt']
        for i, name in enumerate(emotion_names):
            print(f"  {name}: {class_counts[i]} samples")
        
        # Identify minority classes (< 500 samples)
        minority_threshold = 500
        synthetic_samples = []
        
        for class_idx, count in class_counts.items():
            if count < minority_threshold:
                target_count = min(minority_threshold, count * 3)  # 3x boost
                num_synthetic = target_count - count
                
                print(f"Augmenting {emotion_names[class_idx]}: {count} → {target_count} samples")
                synthetic = augmenter.create_synthetic_samples(base_dataset, class_idx, num_synthetic)
                synthetic_samples.extend(synthetic)
        
        # Create enhanced dataset
        class EnhancedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, synthetic_samples):
                self.base_dataset = base_dataset
                self.synthetic_samples = synthetic_samples
                
            def __len__(self):
                return len(self.base_dataset) + len(self.synthetic_samples)
                
            def __getitem__(self, idx):
                if idx < len(self.base_dataset):
                    return self.base_dataset[idx]
                else:
                    synthetic_idx = idx - len(self.base_dataset)
                    return self.synthetic_samples[synthetic_idx]
        
        return EnhancedDataset(base_dataset, synthetic_samples)
    
    return base_dataset

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
    print(f"\n{'='*70}")
    print(f"{phase.upper()} METRICS - EPOCH {epoch}")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"F1 Macro:         {metrics['f1_macro']:.4f}")
    print(f"F1 Micro:         {metrics['f1_micro']:.4f}")
    print(f"F1 Weighted:      {metrics['f1_weighted']:.4f}")
    print(f"Precision Macro:  {metrics['precision_macro']:.4f}")
    print(f"Recall Macro:     {metrics['recall_macro']:.4f}")
    
    print(f"\nPer-Class Performance:")
    print("-" * 70)
    print(f"{'Class':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
    print("-" * 70)
    for emotion, scores in metrics['per_class'].items():
        print(f"{emotion:<12} {scores['f1']:<8.3f} {scores['precision']:<10.3f} "
              f"{scores['recall']:<8.3f} {scores['support']:<8}")
    print("=" * 70)

def advanced_finetune():
    """Advanced fine-tuning with multiple sophisticated techniques"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Advanced Fine-tuning for HybridFER-Max')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=2e-5, help='Base learning rate')
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience')
    parser.add_argument('--min_improvement', type=float, default=0.005, help='Minimum improvement threshold')
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Focal loss gamma')
    
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
    
    checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        original_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"✅ Loaded model with original validation accuracy: {original_val_acc:.4f}")
    else:
        model.load_state_dict(checkpoint)
        original_val_acc = 0.8384  # Known good baseline
        print("✅ Pretrained model loaded successfully!")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Enhanced data transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*70)
    print("🚀 ADVANCED FINE-TUNING SETUP")
    print("="*70)
    
    # Load enhanced datasets
    train_dataset = create_enhanced_dataset('Training', train_transform)
    val_dataset = create_enhanced_dataset('PublicTest', val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Advanced loss function with class weights
    class_counts = torch.tensor([8967, 7366, 3312, 2989, 2142, 151, 575, 132], dtype=torch.float)
    class_weights = class_counts.max() / class_counts  # Inverse frequency
    class_weights = class_weights.to(device)
    
    criterion = AdaptiveFocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    
    # Progressive unfreezing
    progressive_unfreezer = ProgressiveUnfreezing(model, unfreeze_schedule=[5, 15, 25])
    
    # Class-specific optimizer
    optimizer_creator = ClassSpecificOptimizer(model, base_lr=args.base_lr, minority_lr_boost=2.0)
    optimizer = optimizer_creator.create_optimizer()
    
    # Learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.base_lr/10
    )
    
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    print(f"🎯 Advanced techniques enabled:")
    print(f"  ✅ Progressive unfreezing (stages: 5, 15, 25)")
    print(f"  ✅ Adaptive focal loss (gamma: {args.focal_gamma})")
    print(f"  ✅ Smart minority augmentation")
    print(f"  ✅ Class-specific learning rates")
    print(f"  ✅ Cosine annealing with warm restarts")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("="*70)
    
    # Training tracking
    best_val_acc = original_val_acc
    best_f1_macro = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Progressive unfreezing
        progressive_unfreezer.step(epoch)
        
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
            # Use adaptive focal loss with epoch information
            loss = criterion(outputs, hard_labels, epoch=epoch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                loss = criterion(outputs, hard_labels, epoch=epoch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(hard_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
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
            print_metrics(val_metrics, epoch, "VALIDATION")
        
        # Check for improvement
        improvement = val_metrics['accuracy'] - best_val_acc
        f1_improvement = val_metrics['f1_macro'] - best_f1_macro
        
        if improvement > args.min_improvement:
            best_val_acc = val_metrics['accuracy']
            best_f1_macro = val_metrics['f1_macro']
            patience_counter = 0
            
            # Save best model
            save_path = 'checkpoints/advanced_finetuned_best.pth'
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
            
            print(f"🚀 NEW BEST MODEL! Val Acc: {best_val_acc:.4f}, Val F1: {best_f1_macro:.4f}")
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
    
    print(f"\n🎯 ADVANCED FINE-TUNING COMPLETED!")
    print(f"Original validation accuracy: {original_val_acc:.4f}")
    print(f"Final validation accuracy: {best_val_acc:.4f}")
    print(f"Net improvement: {best_val_acc - original_val_acc:+.4f}")
    print(f"Best validation F1 macro: {best_f1_macro:.4f}")
    
    if best_val_acc > original_val_acc + 0.01:  # At least 1% improvement
        print("🚀 Advanced fine-tuning was successful!")
    elif best_val_acc > original_val_acc:
        print("✅ Modest improvement achieved")
    else:
        print("❌ No significant improvement")

if __name__ == "__main__":
    advanced_finetune()