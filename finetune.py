#!/usr/bin/env python3
"""
Fine-tuning script for GReFEL
Loads from best checkpoint and saves fine-tuned model with distinct name
"""

import os
import torch
import argparse
import copy
import numpy as np
from train import train_epoch, validate, get_loaders, update_ema
from models.grefel import GReFEL, EnhancedGReFELLoss

def parse_finetune_args():
    parser = argparse.ArgumentParser(description='Fine-tune GReFEL from best checkpoint')
    
    # Base training args (inherited from successful run)
    parser.add_argument('--dataset', type=str, default='ferplus', choices=['ferplus', 'rafdb', 'fer2013'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./FERPlus', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (optimized for GReFEL)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--use_soft_labels', action='store_true', default=True, help='Use probability distribution from FERPlus votes (default)')
    parser.add_argument('--use_hard_labels', dest='use_soft_labels', action='store_false', help='Use majority vote from FERPlus')
    
    # Fine-tuning specific parameters (optimized based on successful training)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint to fine-tune from')
    parser.add_argument('--finetune_name', type=str, default='finetuned', help='Name suffix for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=50, help='Fine-tuning epochs')
    
    # Fine-tuning hyperparameters - based on successful original training but slightly adjusted
    parser.add_argument('--lr', type=float, default=2e-5, help='Fine-tuning learning rate (lower than original 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.007, help='Weight decay (slightly higher for fine-tuning)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs (shorter for fine-tuning)')
    
    # Regularization (stronger for fine-tuning to prevent overfitting)
    parser.add_argument('--drop_rate', type=float, default=0.18, help='Dropout rate (slightly higher)')
    parser.add_argument('--label_smoothing', type=float, default=0.13, help='Label smoothing (slightly higher)')
    parser.add_argument('--mixup_alpha', type=float, default=0.25, help='Mixup alpha (slightly higher)')
    
    # EMA and other settings
    parser.add_argument('--ema_decay', type=float, default=0.9998, help='EMA decay (slightly faster updates)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency')
    
    # Technical settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--num_anchors', type=int, default=10)
    
    return parser.parse_args()

def main():
    args = parse_finetune_args()
    
    # Create output directory
    output_dir = f"checkpoints/finetune_{args.finetune_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔥 Fine-tuning GReFEL from checkpoint: {args.checkpoint_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load checkpoint
    print(f"📋 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    
    # Initialize model and load weights
    model = GReFEL(
        feature_dim=args.feature_dim,
        num_anchors=args.num_anchors,
        drop_rate=args.drop_rate  # Use new dropout rate
    ).to(args.device)
    
    # Load state dict from the checkpoint's model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📋 Loaded model from epoch {checkpoint['epoch']} with best accuracy: {checkpoint['best_acc']:.2f}%")
    
    # Create EMA model copy
    ema_model = copy.deepcopy(model)
    
    # Get data loaders
    train_loader, val_loader = get_loaders(args)
    
    # Create criterion and optimizer
    criterion = EnhancedGReFELLoss(
        num_classes=8,  # FERPlus has 8 emotion classes
        label_smoothing=args.label_smoothing,
        geo_weight=0.01,
        reliability_weight=0.1
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup from 20% to 100% of base lr
            return 0.2 + 0.8 * (epoch / args.warmup_epochs)
        # Cosine decay from 100% to 20% of base lr
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.2 + 0.8 * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize best accuracies
    best_acc = 0
    best_ema_acc = 0
    
    print(f"\n🚀 Starting fine-tuning with parameters:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr} (original was 5e-5)")
    print(f"  - Weight decay: {args.weight_decay} (original was 0.005)")
    print(f"  - Label smoothing: {args.label_smoothing} (original was 0.11)")
    print(f"  - Mixup alpha: {args.mixup_alpha} (original was 0.2)")
    print(f"  - Dropout rate: {args.drop_rate} (original was 0.15)")
    print(f"  - EMA decay: {args.ema_decay} (original was 0.9999)")
    print(f"  - Warmup epochs: {args.warmup_epochs}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            None, ema_model, args
        )
        
        # Validate with both models
        val_loss, val_acc, val_class_acc = validate(model, val_loader, criterion, args)
        ema_val_loss, ema_val_acc, ema_val_class_acc = validate(ema_model, val_loader, criterion, args)
        
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
                'original_checkpoint': args.checkpoint_path
            }, os.path.join(output_dir, f'best_model_{args.finetune_name}.pth'))
        
        # Save best EMA model
        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema_model.state_dict(),
                'best_acc': best_ema_acc,
                'args': args,
                'original_checkpoint': args.checkpoint_path
            }, os.path.join(output_dir, f'best_ema_model_{args.finetune_name}.pth'))
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'ema_val_acc': ema_val_acc,
                'args': args,
                'original_checkpoint': args.checkpoint_path
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}_{args.finetune_name}.pth'))
        
        # Print metrics in same format as train.py
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})')
        print(f'EMA Val Acc: {ema_val_acc:.4f} (Best: {best_ema_acc:.4f})')
        print('Class Accuracies:')
        emotion_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
        for i, acc in enumerate(val_class_acc):
            print(f'{emotion_names[i]}: {acc:.4f}')
    
    print("\n✅ Fine-tuning completed!")
    print("📊 Final Results:")
    print(f"   - Best model accuracy: {best_acc:.4f}")
    print(f"   - Best EMA accuracy: {best_ema_acc:.4f}")
    print(f"📁 Models saved to: {output_dir}")

if __name__ == '__main__':
    main() 