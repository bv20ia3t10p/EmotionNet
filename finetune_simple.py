#!/usr/bin/env python3
"""
Simple Fine-tuning for SimpleConvNet
Improves the 67.64% baseline with targeted adjustments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# Import from your working model
from train_working_fix import SimpleConvNet, FERPlusDataset, FocalLoss

def create_finetune_transforms():
    """More aggressive but still face-preserving augmentations"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=10, fill=128),  # More rotation
        transforms.RandomHorizontalFlip(p=0.5),  # Higher flip chance
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_pretrained_model(checkpoint_path, device):
    """Load your trained model for fine-tuning"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleConvNet(num_classes=8).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Starting val_acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    return model, checkpoint.get('val_acc', 0.0)

def create_aggressive_sampler(dataset):
    """More aggressive sampling for minority classes"""
    hard_labels = []
    for i in range(len(dataset)):
        _, hard_label, _ = dataset[i]
        hard_labels.append(hard_label)
    
    hard_labels = np.array(hard_labels)
    class_counts = np.bincount(hard_labels, minlength=8)
    
    # Calculate weights - boost minority classes more
    total_samples = len(hard_labels)
    class_weights = total_samples / (8 * class_counts + 1e-6)
    
    # Extra aggressive boost for rare classes
    class_weights[5] *= 10  # disgust
    class_weights[7] *= 10  # contempt
    class_weights[6] *= 5   # fear
    class_weights[3] *= 2   # sadness (was underperforming)
    
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    print("Fine-tuning class weights:")
    for i, (emotion, weight) in enumerate(zip(emotion_names, class_weights)):
        print(f"  {emotion}: {weight:.2f} (count: {class_counts[i]})")
    
    sample_weights = class_weights[hard_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your trained model
    model_path = 'best_working_fer.pth'
    model, starting_acc = load_pretrained_model(model_path, device)
    
    print(f"\n🎯 FINE-TUNING SETUP")
    print(f"Starting accuracy: {starting_acc:.2f}%")
    print(f"Target: 75%+ (improvement of {75-starting_acc:.2f}%+)")
    
    # Enhanced transforms
    train_transform, val_transform = create_finetune_transforms()
    
    # Load datasets
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
    
    # Smaller batch size for fine-tuning
    train_loader = DataLoader(train_dataset, batch_size=24, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Fine-tuning setup
    print("\n🔧 FINE-TUNING CONFIGURATION")
    
    # Lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    
    # Enhanced class weights
    class_weights = torch.tensor([1.0, 1.0, 2.0, 2.5, 2.0, 15.0, 8.0, 15.0]).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=3.0)  # Higher gamma
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
    
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Focal Loss gamma: 3.0")
    print(f"Max epochs: 50")
    
    # Training
    best_val_acc = starting_acc
    patience = 20
    patience_counter = 0
    
    print(f"\n🚀 STARTING FINE-TUNING...")
    
    for epoch in range(1, 51):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        # Print results
        print(f"\nEpoch {epoch}/50:")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        print(f"  Best:  {best_val_acc:.2f}%")
        print(f"  LR:    {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss/len(val_loader)
            }, 'best_working_finetuned.pth')
            
            print(f"  🎯 NEW BEST: {val_acc:.2f}% (+{improvement:.2f}%)")
            
            # Check milestones
            if val_acc > 75:
                print(f"  🏆 EXCELLENT: Surpassed 75% target!")
            elif val_acc > starting_acc + 5:
                print(f"  ✅ GOOD: +{val_acc-starting_acc:.1f}% improvement!")
                
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        print("-" * 50)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
        
        # Success stopping
        if val_acc > 80:
            print(f"🎉 SUCCESS: Reached {val_acc:.2f}%!")
            break
    
    print(f"\n🏁 FINE-TUNING COMPLETE!")
    print(f"Starting accuracy: {starting_acc:.2f}%")
    print(f"Final accuracy:    {best_val_acc:.2f}%")
    print(f"Improvement:       +{best_val_acc-starting_acc:.2f}%")
    
    if best_val_acc > starting_acc + 3:
        print("✅ Fine-tuning was successful!")
    else:
        print("⚠️  Limited improvement - may need different approach")

if __name__ == "__main__":
    main() 