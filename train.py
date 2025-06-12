import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.grefel import GReFEL, GReFELLoss
from datasets.fer_datasets import FERPlusDataset, RAFDBDataset, FER2013Dataset
from utils.metrics import MetricsTracker, BatchMetricsAggregator

EMOTION_CLASSES = {
    'ferplus': ['neutral', 'happiness', 'surprise', 'sadness', 
                'anger', 'disgust', 'fear', 'contempt'],
    'rafdb': ['surprise', 'fear', 'disgust', 'happiness',
              'sadness', 'anger', 'neutral'],
    'fer2013': ['angry', 'disgust', 'fear', 'happy',
                'sad', 'surprise', 'neutral']
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['ferplus', 'rafdb', 'fer2013'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--num_anchors', type=int, default=10)
    parser.add_argument('--smoothing', type=float, default=0.11)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    return parser.parse_args()

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
        
        optimizer.zero_grad()
        probabilities, _, center_loss = model(images)
        
        loss = criterion(probabilities, labels, center_loss)
        loss.backward()
        optimizer.step()
        
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
            
            probabilities, _, center_loss = model(images)
            loss = criterion(probabilities, labels, center_loss)
            
            _, predicted = torch.max(probabilities.data, 1)
            metrics_aggregator.update(loss.item(), predicted, labels)
    
    return metrics_aggregator.get_aggregated_metrics()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        class_names=EMOTION_CLASSES[args.dataset]
    )
    
    # Create metrics aggregators
    train_aggregator = BatchMetricsAggregator()
    val_aggregator = BatchMetricsAggregator()
    
    # Create datasets and dataloaders
    train_dataset = get_dataset(args.dataset, args.data_dir, 'train')
    val_dataset = get_dataset(args.dataset, args.data_dir, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model and move to device
    model = GReFEL(
        num_classes=args.num_classes,
        num_anchors=args.num_anchors,
        smoothing=args.smoothing
    ).to(args.device)
    
    # Create criterion, optimizer and scheduler
    criterion = GReFELLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, 
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}')
        
        # Train
        train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, args.device, train_aggregator
        )
        train_metrics = metrics_tracker.update_metrics(
            'train', train_loss, train_preds, train_targets, epoch
        )
        
        # Validate
        val_loss, val_preds, val_targets = validate(
            model, val_loader, criterion, args.device, val_aggregator
        )
        val_metrics = metrics_tracker.update_metrics(
            'val', val_loss, val_preds, val_targets, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print and save metrics
        metrics_tracker.print_epoch_metrics(epoch, train_metrics, val_metrics)
        metrics_tracker.save_metrics()
        
        # Save best model
        if metrics_tracker.should_save_model(val_metrics['accuracy']):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics_tracker.history,
            }, os.path.join(args.output_dir, 'best_model.pth'))

if __name__ == '__main__':
    main() 