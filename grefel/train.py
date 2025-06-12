import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from model import EnhancedGReFEL
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

def train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        try:
            # Forward pass
            logits, loss = model(images, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                continue
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm for monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Skip step if gradients are NaN
            if torch.isnan(grad_norm):
                print(f"NaN gradients detected at batch {batch_idx}. Skipping update.")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%',
                'grad_norm': f'{grad_norm:.2f}'
            })
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Clear GPU cache periodically
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
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
            
            # Forward pass
            logits, _ = model(images)  # Unpack tuple, ignore loss
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
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Set up logger
    logger = Logger(args.output_dir)
    logger.log(f'Using device: {device}')
    
    # Create datasets with enhanced augmentations
    train_dataset = FERPlusDataset(
        args.train_dir, 
        train=True
    )
    val_dataset = FERPlusDataset(args.val_dir, train=False)
    
    # Create data loaders with pin memory and persistent workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    logger.log(f'Train dataset size: {len(train_dataset)}')
    logger.log(f'Val dataset size: {len(val_dataset)}')
    
    # Create enhanced model
    model = EnhancedGReFEL(
        num_classes=args.num_classes,
        num_anchors=args.num_anchors
    ).to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Use Kaiming initialization for linear layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Use Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Apply weight initialization to non-pretrained parts
    model.multiscale.apply(init_weights)
    model.attention.apply(init_weights)
    model.reliability_module.apply(init_weights)
    model.classifier.apply(init_weights)
    
    # Set up loss function with class weights to handle imbalance
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Separate parameter groups for better optimization
    no_decay = ['bias', 'LayerNorm.weight', 'norm', 'layer_norm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.backbone.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'lr': args.lr * 0.01,  # Much lower LR for pretrained backbone
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.backbone.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'lr': args.lr * 0.01,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if not n.startswith('backbone') 
                      and not any(nd in n for nd in no_decay)],
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if not n.startswith('backbone') 
                      and any(nd in n for nd in no_decay)],
            'lr': args.lr,
            'weight_decay': 0.0
        }
    ]
    
    # Initialize optimizer with gradient clipping
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Enhanced learning rate scheduler with longer warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Minimum LR ratio increased
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop with improved early stopping
    best_val_acc = 0
    best_epoch = 0
    patience = args.patience
    patience_counter = 0
    
    # Main training loop with progress bar
    pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, max_grad_norm=args.max_grad_norm
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics and current learning rate
        current_lr = scheduler.get_last_lr()[0]
        logger.log(f'Epoch {epoch+1}/{args.epochs}:')
        logger.log(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.log(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.log(f'Learning Rate: {current_lr:.2e}')
        logger.log_metrics(epoch+1, train_loss, train_acc, val_loss, val_acc)
        
        # Update main progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',
            'lr': f"{current_lr:.2e}"
        })
        
        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.log(f'Saved new best model with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log(f'Early stopping triggered after {epoch+1} epochs. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}')
                break
        
        # Save latest model
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../FERPlus/data/FER2013Train', help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='../FERPlus/data/FER2013Valid', help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/enhanced_grefel_run2', help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--num_anchors', type=int, default=16, help='Number of anchors')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    args.train_dir = os.path.abspath(args.train_dir)
    args.val_dir = os.path.abspath(args.val_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 