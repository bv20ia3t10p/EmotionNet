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

class BalancedFocalLoss(nn.Module):
    """Balanced focal loss with moderate gamma for gentle focus on hard examples"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Moderate focal weight - not too aggressive
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ModerateMinorityAugmentation:
    """Moderate augmentation for minority classes using mixup"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def mixup_data(self, x1, y1, x2, y2):
        """Create mixup of two samples"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam
    
    def create_synthetic_samples(self, dataset, target_class, num_samples):
        """Create synthetic samples using moderate mixup"""
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
            
            # Create mixup
            mixed_img, _, _, _ = self.mixup_data(img1, label1, img2, label2)
            soft_label = torch.zeros(8)
            soft_label[target_class] = 1.0
            
            synthetic_samples.append((mixed_img, target_class, soft_label))
        
        return synthetic_samples

class ConservativeProgressiveUnfreezing:
    """More conservative progressive unfreezing"""
    def __init__(self, model, unfreeze_schedule=[10, 20, 35]):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.unfrozen_stages = 0
        
        # Initially freeze all transformer blocks
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Always keep prediction heads unfrozen
        for param in self.model.head_global.parameters():
            param.requires_grad = True
        for param in self.model.head_local.parameters():
            param.requires_grad = True
        for param in self.model.head_micro.parameters():
            param.requires_grad = True
            
        # Keep feature fusion unfrozen
        if hasattr(self.model, 'feature_fusion'):
            for param in self.model.feature_fusion.parameters():
                param.requires_grad = True
    
    def step(self, epoch):
        """Unfreeze layers based on schedule"""
        if self.unfrozen_stages < len(self.unfreeze_schedule):
            if epoch >= self.unfreeze_schedule[self.unfrozen_stages]:
                self._unfreeze_next_stage()
                self.unfrozen_stages += 1
    
    def _unfreeze_next_stage(self):
        """Unfreeze next stage of transformer blocks"""
        if hasattr(self.model, 'blocks'):
            if self.unfrozen_stages == 0:
                # Unfreeze last 2 transformer blocks
                for param in self.model.blocks[-2:].parameters():
                    param.requires_grad = True
                print("🔓 Unfroze last 2 transformer blocks")
            elif self.unfrozen_stages == 1:
                # Unfreeze 4 more blocks (total 6)
                for param in self.model.blocks[-6:].parameters():
                    param.requires_grad = True
                print("🔓 Unfroze 6 transformer blocks total")
            elif self.unfrozen_stages == 2:
                # Unfreeze all transformer blocks
                for param in self.model.blocks.parameters():
                    param.requires_grad = True
                # Also unfreeze patch embedding and other components
                for param in self.model.patch_embed.parameters():
                    param.requires_grad = True
                print("🔓 Unfroze all transformer blocks and patch embedding")

def create_balanced_dataset(split, transform):
    """Create dataset with moderate minority augmentation"""
    train_module = import_model_from_train()
    base_dataset = train_module.FERPlusDataset(
        csv_file='FERPlus-master/fer2013new.csv',
        fer2013_csv_file='fer2013.csv',
        split=split,
        transform=transform,
        min_votes=2
    )
    
    if split == 'Training':
        # Apply moderate augmentation for minority classes
        augmenter = ModerateMinorityAugmentation()
        
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
        
        # Conservative minority augmentation (2x instead of 3x)
        minority_threshold = 300  # Lower threshold
        synthetic_samples = []
        
        for class_idx, count in class_counts.items():
            if count < minority_threshold:
                target_count = min(minority_threshold, count * 2)  # 2x boost (was 3x)
                num_synthetic = target_count - count
                
                print(f"Moderately augmenting {emotion_names[class_idx]}: {count} → {target_count} samples")
                synthetic = augmenter.create_synthetic_samples(base_dataset, class_idx, num_synthetic)
                synthetic_samples.extend(synthetic)
        
        # Create enhanced dataset
        class BalancedDataset(torch.utils.data.Dataset):
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
        
        return BalancedDataset(base_dataset, synthetic_samples)
    
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
def balanced_finetune():
    """Balanced fine-tuning with moderate techniques to prevent overfitting"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Balanced Fine-tuning for HybridFER-Max')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=1e-5, help='Base learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_improvement', type=float, default=0.003, help='Minimum improvement threshold')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    
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
        dropout=args.dropout  # Increased dropout for regularization
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
    
    # Moderate data transforms - less aggressive than before
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.4),  # Reduced from 0.5
        transforms.RandomRotation(degrees=10),   # Reduced from 15
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),  # Reduced
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),  # Less aggressive
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*70)
    print("🎯 BALANCED ADVANCED FINE-TUNING SETUP")
    print("="*70)
    
    # Load balanced datasets
    train_dataset = create_balanced_dataset('Training', train_transform)
    val_dataset = create_balanced_dataset('PublicTest', val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Moderate class weights - less extreme than before
    class_counts = torch.tensor([8967, 7366, 3312, 2989, 2142, 151, 575, 132], dtype=torch.float)
    raw_weights = class_counts.max() / class_counts
    # Smooth the weights to be less extreme
    class_weights = torch.sqrt(raw_weights)  # Square root to moderate the weights
    class_weights = class_weights.to(device)
    
    criterion = BalancedFocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    
    # Conservative progressive unfreezing
    progressive_unfreezer = ConservativeProgressiveUnfreezing(model, unfreeze_schedule=[10, 20, 35])
    
    # Conservative optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr,
        weight_decay=2e-4,  # Increased weight decay for regularization
        betas=(0.9, 0.999)
    )
    
    # Gentle learning rate schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-7
    )
    
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    print(f"🎯 Balanced techniques enabled:")
    print(f"  ✅ Conservative progressive unfreezing (10, 20, 35)")
    print(f"  ✅ Balanced focal loss (gamma: {args.focal_gamma})")
    print(f"  ✅ Moderate minority augmentation (2x boost)")
    print(f"  ✅ Increased dropout ({args.dropout})")
    print(f"  ✅ Higher weight decay (2e-4)")
    print(f"  ✅ Reduce LR on plateau")
    print(f"Smoothed class weights: {class_weights.cpu().numpy()}")
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
            loss = criterion(outputs, hard_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gentler clipping
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
        
        # Print detailed metrics every 5 epochs or if new best
        if epoch % 5 == 0 or val_metrics['accuracy'] > best_val_acc:
            print_metrics(val_metrics, epoch, "VALIDATION")
        
        # Check for improvement
        improvement = val_metrics['accuracy'] - best_val_acc
        
        if improvement > args.min_improvement:
            best_val_acc = val_metrics['accuracy']
            best_f1_macro = val_metrics['f1_macro']
            patience_counter = 0
            
            # Save best model
            save_path = 'checkpoints/balanced_finetuned_best.pth'
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
    
    print(f"\n🎯 BALANCED FINE-TUNING COMPLETED!")
    print(f"Original validation accuracy: {original_val_acc:.4f}")
    print(f"Final validation accuracy: {best_val_acc:.4f}")
    print(f"Net improvement: {best_val_acc - original_val_acc:+.4f}")
    print(f"Best validation F1 macro: {best_f1_macro:.4f}")
    
    if best_val_acc > original_val_acc + 0.01:  # At least 1% improvement
        print("🚀 Balanced fine-tuning was successful!")
    elif best_val_acc > original_val_acc:
        print("✅ Modest improvement achieved")
    else:
        print("❌ No significant improvement")

if __name__ == "__main__":
    balanced_finetune()