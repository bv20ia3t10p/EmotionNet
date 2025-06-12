import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import GReFEL
from dataset import FERPlusDataset

class Logger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics file
        self.metrics_file = os.path.join(output_dir, 'metrics.csv')
        self.metrics = []
        
        # Create log file
        self.log_file = os.path.join(output_dir, 'training.log')
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f'[{timestamp}] {message}\n')
        print(f'[{timestamp}] {message}')
        
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        self.metrics.append(metrics)
        
        # Save metrics to CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_file, index=False)
        
        # Plot metrics
        self.plot_metrics()
        
    def plot_metrics(self):
        df = pd.DataFrame(self.metrics)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot loss
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(df['epoch'], df['train_acc'], label='Train Acc')
        ax2.plot(df['epoch'], df['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics.png'))
        plt.close()

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            logits, loss = model(images, labels)
            
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_loss = total_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    return total_loss / len(val_loader), 100. * correct / total

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up logger
    logger = Logger(args.output_dir)
    logger.log(f'Using device: {device}')
    
    # Create datasets
    train_dataset = FERPlusDataset(args.train_dir, train=True)
    val_dataset = FERPlusDataset(args.val_dir, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    logger.log(f'Train dataset size: {len(train_dataset)}')
    logger.log(f'Val dataset size: {len(val_dataset)}')
    
    # Create model
    model = GReFEL(num_classes=args.num_classes, num_anchors=args.num_anchors).to(device)
    
    # Set up loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # Set up learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Set up mixed precision training with new API
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop with early stopping
    best_val_acc = 0
    patience = 500
    patience_counter = 0
    
    # Main training loop with progress bar
    pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.log(f'Epoch {epoch+1}/{args.epochs}:')
        logger.log(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.log(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.log_metrics(epoch+1, train_loss, train_acc, val_loss, val_acc)
        
        # Update main progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%'
        })
        
        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.log(f'Saved new best model with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
        }, os.path.join(args.output_dir, 'latest_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--num_anchors', type=int, default=10, help='Number of anchors')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    args.train_dir = os.path.abspath(args.train_dir)
    args.val_dir = os.path.abspath(args.val_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 