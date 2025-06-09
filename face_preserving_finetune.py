import sys
import os
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def import_model_from_train():
    """Import the model and dataset classes from train.py"""
    spec = importlib.util.spec_from_file_location("train", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module

class FacePreservingFocalLoss(nn.Module):
    """Face-preserving focal loss with moderate parameters"""
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super(FacePreservingFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Gentle focal weight
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FacePreservingAugmentation:
    """Face-preserving augmentation that maintains facial expression integrity"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        
    def mixup_data(self, x1, y1, x2, y2):
        """Create gentle mixup of two samples"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam
    
    def create_synthetic_samples(self, dataset, target_class, num_samples):
        """Create synthetic samples using gentle mixup"""
        # Find all samples of target class
        target_indices = []
        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            if label == target_class:
                target_indices.append(i)
        
        if len(target_indices) < 2:
            return []
        
        synthetic_samples = []
        for _ in range(num_samples):
            # Pick two random samples from the same class
            idx1, idx2 = np.random.choice(target_indices, 2, replace=True)
            img1, label1, _ = dataset[idx1]
            img2, label2, _ = dataset[idx2]
            
            # Create gentle mixup
            mixed_img, _, _, _ = self.mixup_data(img1, label1, img2, label2)
            soft_label = torch.zeros(8)
            soft_label[target_class] = 1.0
            
            synthetic_samples.append((mixed_img, target_class, soft_label))
        
        return synthetic_samples

def create_face_preserving_dataset(split, transform):
    """Create dataset with face-preserving minority augmentation"""
    train_module = import_model_from_train()
    base_dataset = train_module.FERPlusDataset(
        csv_file='FERPlus-master/fer2013new.csv',
        fer2013_csv_file='fer2013.csv',
        split=split,
        transform=transform,
        min_votes=2
    )
    
    if split == 'Training':
        # Apply face-preserving augmentation for minority classes
        augmenter = FacePreservingAugmentation()
        
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
        
        # Gentle minority augmentation (1.5x boost)
        minority_threshold = 200
        synthetic_samples = []
        
        for class_idx, count in class_counts.items():
            if count < minority_threshold:
                target_count = min(minority_threshold, int(count * 1.5))  # 1.5x boost
                num_synthetic = target_count - count
                
                print(f"Gently augmenting {emotion_names[class_idx]}: {count} → {target_count} samples")
                synthetic = augmenter.create_synthetic_samples(base_dataset, class_idx, num_synthetic)
                synthetic_samples.extend(synthetic)
        
        # Create enhanced dataset
        class FacePreservingDataset(torch.utils.data.Dataset):
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
        
        return FacePreservingDataset(base_dataset, synthetic_samples)
    
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

def face_preserving_finetune():
    """Face-preserving fine-tuning without layer freezing"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Face-Preserving Fine-tuning for HybridFER-Max')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=5e-6, help='Base learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--min_improvement', type=float, default=0.002, help='Minimum improvement threshold')
    parser.add_argument('--focal_gamma', type=float, default=1.5, help='Focal loss gamma')
    parser.add_argument('--dropout', type=float, default=0.12, help='Dropout rate')
    
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
        dropout=args.dropout
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
    
    # NO LAYER FREEZING - All parameters trainable from start
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}/{total_params:,} (100%)")
    
    # Face-preserving data transforms
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.3),  # Gentle - faces are naturally asymmetric
        transforms.RandomRotation(degrees=5),    # Very gentle rotation to preserve expression
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),  # Gentle lighting
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),  # Minimal geometric
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*70)
    print("🎭 FACE-PRESERVING FINE-TUNING SETUP")
    print("="*70)
    
    # Load face-preserving datasets
    train_dataset = create_face_preserving_dataset('Training', train_transform)
    val_dataset = create_face_preserving_dataset('PublicTest', val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Gentle class weights
    class_counts = torch.tensor([8967, 7366, 3312, 2989, 2142, 151, 575, 132], dtype=torch.float)
    raw_weights = class_counts.max() / class_counts
    # Even more gentle smoothing
    class_weights = torch.pow(raw_weights, 0.3)  # Power of 0.3 for very gentle weighting
    class_weights = class_weights.to(device)
    
    criterion = FacePreservingFocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    
    # Gentle optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=1e-4,  # Moderate weight decay
        betas=(0.9, 0.999)
    )
    
    # Gentle learning rate schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=4, min_lr=1e-7
    )
    
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    print(f"🎭 Face-preserving techniques enabled:")
    print(f"  ✅ NO layer freezing - all parameters trainable")
    print(f"  ✅ Face-preserving augmentations (gentle transforms)")
    print(f"  ✅ Gentle focal loss (gamma: {args.focal_gamma})")
    print(f"  ✅ Gentle minority augmentation (1.5x boost)")
    print(f"  ✅ Gentle class weights (power 0.3)")
    print(f"  ✅ Conservative learning rate ({args.base_lr})")
    print(f"Gentle class weights: {class_weights.cpu().numpy()}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("="*70)    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)  # Very gentle clipping
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
                loss = criterion(outputs, hard_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(hard_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_metrics = calculate_metrics(val_preds, val_labels, emotion_names)
        
        # Update scheduler
        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']:.2%})")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']:.2%})")
        print(f"Train F1 Macro: {train_metrics['f1_macro']:.4f}")
        print(f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"Gap: {train_metrics['accuracy'] - val_metrics['accuracy']:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Print detailed metrics every 3 epochs or if new best
        if epoch % 3 == 0 or val_metrics['accuracy'] > best_val_acc:
            print_metrics(val_metrics, epoch, "VALIDATION")
        
        # Check for improvement
        improvement = val_metrics['accuracy'] - best_val_acc
        
        if improvement > args.min_improvement:
            best_val_acc = val_metrics['accuracy']
            best_f1_macro = val_metrics['f1_macro']
            patience_counter = 0
            
            # Save best model
            save_path = 'checkpoints/face_preserving_best.pth'
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
            
            print(f"🎭 NEW BEST MODEL! Val Acc: {best_val_acc:.4f}, Val F1: {best_f1_macro:.4f}")
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
    
    print(f"\n🎭 FACE-PRESERVING FINE-TUNING COMPLETED!")
    print(f"Original validation accuracy: {original_val_acc:.4f}")
    print(f"Final validation accuracy: {best_val_acc:.4f}")
    print(f"Net improvement: {best_val_acc - original_val_acc:+.4f}")
    print(f"Best validation F1 macro: {best_f1_macro:.4f}")
    
    if best_val_acc > original_val_acc + 0.01:  # At least 1% improvement
        print("🚀 Face-preserving fine-tuning was successful!")
    elif best_val_acc > original_val_acc:
        print("✅ Modest improvement achieved")
    else:
        print("❌ No significant improvement")

if __name__ == "__main__":
    face_preserving_finetune()