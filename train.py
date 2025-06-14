import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import copy
import argparse

from models.grefel import GReFEL, EnhancedGReFELLoss
from datasets.fer_datasets import FERPlusDataset, RAFDBDataset, FER2013Dataset, get_loaders, CutMixAugmentation
from torch.utils.data import DataLoader

def update_ema(ema_model, model, decay):
    """Update exponential moving average of model parameters."""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GReFEL on emotion recognition datasets')
    # Dataset
    parser.add_argument('--dataset', type=str, default='ferplus', choices=['ferplus', 'rafdb', 'fer2013'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./FERPlus', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (optimized for GReFEL)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--use_soft_labels', action='store_true', default=True, help='Use probability distribution from FERPlus votes (default)')
    parser.add_argument('--use_hard_labels', dest='use_soft_labels', action='store_false', help='Use majority vote from FERPlus')
    
    # Model (optimized for GReFEL paper specifications)
    parser.add_argument('--feature_dim', type=int, default=768, help='Feature dimension (ViT-Base)')
    parser.add_argument('--num_anchors', type=int, default=10, help='Number of learnable anchors')
    parser.add_argument('--drop_rate', type=float, default=0.15, help='Dropout rate')
    
    # Training (optimized hyperparameters)
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Peak learning rate (more conservative for stability)')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay (further reduced for stability)')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='Warmup epochs (longer for stability)')
    parser.add_argument('--label_smoothing', type=float, default=0.11, help='Label smoothing factor')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    
    # Advanced training options
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay for model averaging')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    
    return parser.parse_args()

def get_loaders(args):
    """Create train and validation data loaders based on dataset and arguments."""
    if args.dataset.lower() == 'ferplus':
        train_dataset = FERPlusDataset(
            root_dir=args.data_dir,
            split='train',
            use_soft_labels=args.use_soft_labels
        )
        val_dataset = FERPlusDataset(
            root_dir=args.data_dir,
            split='val',
            use_soft_labels=args.use_soft_labels
        )
    elif args.dataset.lower() == 'rafdb':
        train_dataset = RAFDBDataset(
            root_dir=args.data_dir,
            split='train'
        )
        val_dataset = RAFDBDataset(
            root_dir=args.data_dir,
            split='val'
        )
    elif args.dataset.lower() == 'fer2013':
        train_dataset = FER2013Dataset(
            root_dir=args.data_dir,
            split='train'
        )
        val_dataset = FER2013Dataset(
            root_dir=args.data_dir,
            split='val'
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, scaler, ema_model, args):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        # Apply mixup augmentation
        if args.mixup_alpha > 0 and np.random.rand() < 0.5:
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(args.device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = targets, targets[index]
            
            optimizer.zero_grad()
            
            outputs = model(mixed_images)
            is_soft_labels = targets.dim() > 1 and targets.size(1) > 1
            
            # Compute losses for mixup
            loss_dict_a = criterion(outputs, targets_a, is_soft_labels=is_soft_labels)
            loss_dict_b = criterion(outputs, targets_b, is_soft_labels=is_soft_labels)
            
            # Combine losses with mixup weights
            loss = lam * loss_dict_a['total_loss'] + (1 - lam) * loss_dict_b['total_loss']
        else:
            optimizer.zero_grad()
            
            outputs = model(images)
            is_soft_labels = targets.dim() > 1 and targets.size(1) > 1
            loss_dict = criterion(outputs, targets, is_soft_labels=is_soft_labels)
            loss = loss_dict['total_loss']
        
        # Check for NaN loss with detailed debugging
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at batch {batch_idx}")
            print(f"  Logits range: [{outputs['logits'].min().item():.4f}, {outputs['logits'].max().item():.4f}]")
            print(f"  Geo loss: {outputs['geo_loss'].item():.6f}")
            print(f"  Reliability range: [{outputs['reliability'].min().item():.4f}, {outputs['reliability'].max().item():.4f}]")
            print("  Skipping batch...")
            continue
        
        # Simplified gradient handling without scaler complications
        try:
            # Standard backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_inf = False
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    if torch.isnan(param_norm) or torch.isinf(param_norm):
                        has_nan_inf = True
                        break
                    total_norm += param_norm.item() ** 2
            
            if has_nan_inf:
                print(f"Warning: NaN/Inf gradients detected at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                continue
            
            total_norm = total_norm ** (1. / 2)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step
            optimizer.step()
            
        except RuntimeError as e:
            print(f"Warning: Runtime error at batch {batch_idx}: {e}, skipping...")
            optimizer.zero_grad()
            continue
        
        # Update EMA model
        if ema_model is not None:
            update_ema(ema_model, model, args.ema_decay)
        
        # Compute accuracy
        preds = outputs['logits'].argmax(dim=1)
        if targets.dim() > 1 and targets.size(1) > 1:
            # Soft labels - compare with argmax
            target_labels = targets.argmax(dim=1)
        else:
            # Hard labels
            target_labels = targets.long()
        acc = (preds == target_labels).float().mean()
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples,
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item() if 'grad_norm' in locals() else 0.0
        })
    
    return total_loss / total_samples, total_acc / total_samples

def validate(model, loader, criterion, args):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
            images = images.to(args.device)
            targets = targets.to(args.device)
            
            outputs = model(images)
            is_soft_labels = targets.dim() > 1 and targets.size(1) > 1
            loss_dict = criterion(outputs, targets, is_soft_labels=is_soft_labels)
            loss = loss_dict['total_loss']
            
            preds = outputs['logits'].argmax(dim=1)
            if targets.dim() > 1 and targets.size(1) > 1:
                # Soft labels - compare with argmax
                target_labels = targets.argmax(dim=1)
            else:
                # Hard labels
                target_labels = targets.long()
            acc = (preds == target_labels).float().mean()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_labels.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return (
        total_loss / total_samples,
        total_acc / total_samples,
        class_acc
    )

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = GReFEL(
        num_classes=8,
        feature_dim=args.feature_dim,
        num_anchors=args.num_anchors,
        drop_rate=args.drop_rate
    ).to(args.device)
    
    # Get data loaders
    train_loader, val_loader = get_loaders(args)
    
    # Enhanced loss function with adaptive weighting
    criterion = EnhancedGReFELLoss(
        num_classes=8,
        label_smoothing=args.label_smoothing,
        geo_weight=0.01,
        reliability_weight=0.1
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with cosine annealing (no warmup)
    def lr_lambda(epoch):
        return 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # EMA model for better generalization
    ema_model = copy.deepcopy(model) if args.ema_decay > 0 else None
    
    # Training loop
    best_acc = 0
    best_ema_acc = 0
    
    print(f"Training with optimized hyperparameters:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Label smoothing: {args.label_smoothing}")
    print(f"  - Mixup alpha: {args.mixup_alpha}")
    print(f"  - Soft labels: {args.use_soft_labels}")
    print(f"  - Feature dim: {args.feature_dim}")
    print(f"  - No warmup (removed for stability)")
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, None, ema_model, args
        )
        
        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, args
        )
        
        # Validate EMA model if available
        ema_val_acc = 0
        if ema_model is not None:
            ema_val_loss, ema_val_acc, ema_class_acc = validate(
                ema_model, val_loader, criterion, args
            )
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': args,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save best EMA model
        if ema_model is not None and ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema_model.state_dict(),
                'best_acc': best_ema_acc,
                'args': args,
            }, os.path.join(args.output_dir, 'best_ema_model.pth'))
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict() if ema_model else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'ema_val_acc': ema_val_acc,
                'args': args,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})')
        if ema_model is not None:
            print(f'EMA Val Acc: {ema_val_acc:.4f} (Best: {best_ema_acc:.4f})')
        print('Class Accuracies:')
        for i, acc in enumerate(class_acc):
            emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
            print(f'{emotion_names[i]}: {acc:.4f}')

if __name__ == '__main__':
    main() 