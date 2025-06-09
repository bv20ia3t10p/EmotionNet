#!/usr/bin/env python3
"""
Fine-tuning script for SimpleConvNet model
Improves upon the base model from train_working_fix.py
"""

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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the SimpleConvNet model
try:
    from train_working_fix import SimpleConvNet, FERPlusDataset, FocalLoss
    print("✅ Successfully imported SimpleConvNet model")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    exit(1)

def create_enhanced_transforms():
    """Enhanced transforms for fine-tuning"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=8, fill=128),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_aggressive_sampler(dataset):
    """More aggressive weighted sampler"""
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    hard_labels = []
    for i in range(len(dataset)):
        _, hard_label, _ = dataset[i]
        hard_labels.append(hard_label)
    
    hard_labels = np.array(hard_labels)
    class_counts = np.bincount(hard_labels, minlength=8)
    
    total_samples = len(hard_labels)
    class_weights = total_samples / (8 * class_counts + 1e-6)
    
    # More aggressive weighting
    class_weights[5] *= 5  # disgust
    class_weights[7] *= 5  # contempt  
    class_weights[6] *= 3  # fear
    class_weights[3] *= 1.5  # sadness
    
    print("Fine-tuning class weights:")
    for i, (emotion, weight) in enumerate(zip(emotion_columns, class_weights)):
        print(f"  {emotion}: {weight:.3f} (count: {class_counts[i]})")
    
    sample_weights = class_weights[hard_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    return sampler

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))

def load_pretrained_model(checkpoint_path, device):
    """Load pre-trained model"""
    print(f"Loading pre-trained model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleConvNet(num_classes=8).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Pre-training val_acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model state dict")
    
    return model

def train_epoch_finetune(model, dataloader, criterion, optimizer, device, epoch):
    """Fine-tuning training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Finetune Epoch {epoch}')
    for batch_idx, (images, hard_labels, soft_labels) in enumerate(pbar):
        images = images.to(device)
        hard_labels = hard_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, hard_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch_finetune(model, dataloader, criterion, device):
    """Validation epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for images, hard_labels, soft_labels in pbar:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, hard_labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained model
    pretrained_path = 'best_working_fer.pth'
    if not os.path.exists(pretrained_path):
        print(f"❌ Pre-trained model not found: {pretrained_path}")
        print("Please run train_working_fix.py first.")
        return
    
    model = load_pretrained_model(pretrained_path, device)
    
    # Enhanced transforms
    train_transform, val_transform = create_enhanced_transforms()
    
    # Datasets
    print("Creating datasets for fine-tuning...")
    train_dataset = FERPlusDataset(
        csv_file='./FERPlus-master/fer2013new.csv',
        fer2013_csv_file='./fer2013.csv',
        split='Training',
        transform=train_transform
    )
    
    val_dataset = FERPlusDataset(
        csv_file='./FERPlus-master/fer2013new.csv',
        fer2013_csv_file='./fer2013.csv',
        split='PublicTest',
        transform=val_transform
    )
    
    # Aggressive sampling
    train_sampler = create_aggressive_sampler(train_dataset)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Fine-tuning setup
    print("Setting up fine-tuning...")
    
    # Loss functions
    label_smooth = LabelSmoothingLoss(num_classes=8, smoothing=0.1)
    class_weights = torch.tensor([1.0, 1.0, 1.8, 2.0, 1.8, 8.0, 4.0, 8.0]).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.5)
    
    # Optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
    
    # Training
    print("Starting fine-tuning...")
    best_val_acc = 0.0
    patience = 25
    patience_counter = 0
    
    # Get starting accuracy
    checkpoint = torch.load(pretrained_path, map_location=device)
    if 'val_acc' in checkpoint:
        best_val_acc = checkpoint['val_acc']
        print(f"Starting from: {best_val_acc:.2f}%")
    
    for epoch in range(1, 51):
        # Alternate loss functions
        criterion = focal_loss if epoch % 3 == 0 else label_smooth
        
        train_loss, train_acc = train_epoch_finetune(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_epoch_finetune(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Finetune Epoch {epoch}/50")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            print(f"🎯 New best: {val_acc:.2f}% (+{improvement:.2f}%)")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_working_finetuned.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        if val_acc > 80.0:
            print(f"Reached excellent accuracy: {val_acc:.2f}%!")
            break
    
    print(f"Fine-tuning completed! Best: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main() 