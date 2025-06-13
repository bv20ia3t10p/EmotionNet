import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from models.grefel import GReFEL, SimplifiedGReFELLoss
from datasets.fer_datasets import FERPlusDataset
from utils.metrics import accuracy_score, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--num_anchors', type=int, default=10)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    return parser.parse_args()

def get_loaders(args):
    train_dataset = FERPlusDataset(args.data_dir, split='train')
    val_dataset = FERPlusDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, scaler, args):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, targets in pbar:
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute accuracy
        preds = outputs['logits'].argmax(dim=1)
        acc = (preds == targets).float().mean()
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples
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
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            preds = outputs['logits'].argmax(dim=1)
            acc = (preds == targets).float().mean()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
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
    
    # Loss and optimizer
    criterion = SimplifiedGReFELLoss(num_classes=8)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, args
        )
        
        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, args
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
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
        print('Class Accuracies:')
        for i, acc in enumerate(class_acc):
            print(f'Class {i}: {acc:.4f}')

if __name__ == '__main__':
    main() 