import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import math

from models.grefel_plus import GReFELPlusPlus, GReFELPlusPlusLoss
from datasets.fer_datasets import FERPlusDataset, RAFDBDataset, FER2013Dataset
from utils.metrics import MetricsTracker, BatchMetricsAggregator

# Reuse emotion classes from train.py
from train import EMOTION_CLASSES

def get_dataset(name, data_dir, split):
    if name == 'ferplus':
        return FERPlusDataset(data_dir, split)
    elif name == 'rafdb':
        return RAFDBDataset(data_dir, split)
    else:
        return FER2013Dataset(data_dir, split)

def train_epoch(model, train_loader, criterion, optimizer, device, metrics_aggregator):
    model.train()
    metrics_aggregator.reset()
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup augmentation
        mixed_images, labels_a, labels_b, lam = train_loader.dataset.mixup_data(images, labels)
        mixed_images = mixed_images.to(device)
        
        optimizer.zero_grad()
        probabilities, _, geometry_loss, confidence = model(mixed_images)
        
        # Calculate loss with mixup
        loss_a = criterion(probabilities, labels_a, geometry_loss, confidence)
        loss_b = criterion(probabilities, labels_b, geometry_loss, confidence)
        loss = lam * loss_a + (1 - lam) * loss_b
        
        loss.backward()
        optimizer.step()
        
        # For metrics, use original labels
        _, predicted = torch.max(probabilities.data, 1)
        metrics_aggregator.update(loss.item(), predicted, labels)
        
        current_loss = metrics_aggregator.total_loss / metrics_aggregator.num_batches
        pbar.set_postfix({'loss': current_loss})
    
    return metrics_aggregator.get_aggregated_metrics()

def validate(model, val_loader, criterion, device, metrics_aggregator):
    model.eval()
    metrics_aggregator.reset()
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            probabilities, _, geometry_loss, confidence = model(images)
            loss = criterion(probabilities, labels, geometry_loss, confidence)
            
            _, predicted = torch.max(probabilities.data, 1)
            metrics_aggregator.update(loss.item(), predicted, labels)
    
    return metrics_aggregator.get_aggregated_metrics()

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    def lr_lambda(current_step):
        current_epoch = current_step / steps_per_epoch
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, warmup_epochs))
        # Cosine decay
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ferplus', 'rafdb', 'fer2013'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--num_anchors', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--smoothing', type=float, default=0.11)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='checkpoints_plus')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics tracker and aggregators
    metrics_tracker = MetricsTracker(
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        class_names=EMOTION_CLASSES[args.dataset]
    )
    train_aggregator = BatchMetricsAggregator(
        args.num_classes,
        class_names=EMOTION_CLASSES[args.dataset]
    )
    val_aggregator = BatchMetricsAggregator(
        args.num_classes,
        class_names=EMOTION_CLASSES[args.dataset]
    )
    
    # Create datasets and dataloaders
    train_dataset = get_dataset(args.dataset, args.data_dir, 'train')
    val_dataset = get_dataset(args.dataset, args.data_dir, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Create model and move to device
    model = GReFELPlusPlus(
        num_classes=args.num_classes,
        pretrained=True,
        feature_dim=1024,  # For ViT-Large
        num_anchors=args.num_anchors,
        num_heads=args.num_heads,
        smoothing=args.smoothing,
        temperature=args.temperature,
        backbone='vit_large_patch16_224'
    ).to(device)
    
    # Create criterion, optimizer and scheduler
    criterion = GReFELPlusPlusLoss(
        num_classes=args.num_classes,
        alpha=0.4,
        beta=0.2,
        gamma=2.0
    ).to(device)
    
    # Use AdamW with weight decay
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + cosine scheduler
    steps_per_epoch = len(train_loader)
    scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        args.warmup_epochs,
        args.epochs,
        steps_per_epoch
    )
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, train_aggregator
        )
        metrics_tracker.update_metrics('train', train_metrics, epoch)
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, val_aggregator
        )
        metrics_tracker.update_metrics('val', val_metrics, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Print and save metrics
        metrics_tracker.print_epoch_metrics(epoch)
        metrics_tracker.save_metrics()
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics_tracker.history,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'Saved new best model with validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main() 