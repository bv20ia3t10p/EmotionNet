import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from collections import Counter
import pandas as pd
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class FERPlusDataset(torch.utils.data.Dataset):
    """FERPlus dataset loader that combines FERPlus labels with FER2013 images"""
    
    def __init__(self, csv_file, fer2013_csv_file, split='Training', transform=None):
        # Load FERPlus annotations
        self.ferplus_df = pd.read_csv(csv_file)
        
        # Load FER2013 data
        self.fer2013_df = pd.read_csv(fer2013_csv_file)
        
        # Filter by split
        self.ferplus_df = self.ferplus_df[self.ferplus_df['Usage'] == split].reset_index(drop=True)
        self.fer2013_df = self.fer2013_df[self.fer2013_df['Usage'] == split].reset_index(drop=True)
        
        self.transform = transform
        
        # FERPlus emotion columns (8 emotions)
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        
        print(f"Loaded {len(self.ferplus_df)} samples for {split}")
        
        # Print class distribution
        hard_labels = []
        for idx in range(len(self.ferplus_df)):
            emotion_scores = self.ferplus_df.iloc[idx][self.emotion_columns].values.astype(float)
            hard_label = np.argmax(emotion_scores)
            hard_labels.append(hard_label)
        
        unique, counts = np.unique(hard_labels, return_counts=True)
        for emotion_idx, count in zip(unique, counts):
            print(f"  {self.emotion_columns[emotion_idx]}: {count}")
    
    def __len__(self):
        return len(self.ferplus_df)
    
    def __getitem__(self, idx):
        # Get image from FER2013
        pixels = self.fer2013_df.iloc[idx]['pixels']
        
        # Convert pixel string to image
        pixel_array = np.array([int(pixel) for pixel in pixels.split()], dtype=np.uint8)
        image = pixel_array.reshape(48, 48)
        
        # Convert to RGB PIL Image
        image = Image.fromarray(image).convert('RGB')
        
        # Get emotion labels from FERPlus
        emotion_scores = self.ferplus_df.iloc[idx][self.emotion_columns].values.astype(float)
        
        # Hard label (most voted emotion)
        hard_label = np.argmax(emotion_scores)
        
        # Soft labels (normalized vote distribution)
        soft_labels = emotion_scores / (emotion_scores.sum() + 1e-8)
        
        if self.transform:
            image = self.transform(image)
        
        return image, hard_label, torch.FloatTensor(soft_labels)

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleConvNet, self).__init__()
        
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        
        # Fourth conv block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

# Conservative face-preserving augmentations
def get_conservative_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced from 0.5
        transforms.RandomRotation(degrees=5),    # Reduced from 10
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Reduced
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_class_weights(dataset):
    # Get hard labels from FERPlusDataset
    targets = []
    for i in range(len(dataset)):
        _, hard_label, _ = dataset[i]
        targets.append(hard_label)
    
    counter = Counter(targets)
    total = len(targets)
    
    # Calculate inverse frequency weights with smoothing
    weights = torch.zeros(8)  # 8 classes for FERPlus
    for class_idx, count in counter.items():
        # Add smoothing factor to prevent extreme weights
        weights[class_idx] = total / (count + 10)
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Cap maximum weight to prevent extreme imbalance
    weights = torch.clamp(weights, min=0.5, max=3.0)
    
    return weights

def create_weighted_sampler(dataset, class_weights):
    targets = [dataset.samples[i][1] for i in range(len(dataset))]
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, hard_labels, soft_labels in val_loader:
            inputs, hard_labels = inputs.to(device), hard_labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, hard_labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += hard_labels.size(0)
            correct += (predicted == hard_labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_conservative_transforms()
    
    # Load datasets using the same approach as working script
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
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Calculate class weights and create weighted sampler
    class_weights = get_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")
    
    train_sampler = create_weighted_sampler(train_dataset, class_weights)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Reduced batch size for more stable gradients
        sampler=train_sampler,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    # Load pre-trained model
    print("🔄 Loading pre-trained model...")
    model = SimpleConvNet(num_classes=8)
    checkpoint_path = 'best_working_fer.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
        print(f"Previous best accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    model = model.to(device)
    
    # Conservative loss function and optimizer
    criterion = FocalLoss(alpha=1, gamma=1.5, reduction='mean')  # Reduced gamma
    
    # Much more conservative optimizer settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-6,        # Very low learning rate
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # More conservative scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.8,      # Less aggressive reduction
        patience=5,      # More patience
        verbose=True,
        min_lr=1e-7
    )
    
    # Training parameters
    num_epochs = 500      # Fewer epochs
    best_val_acc = 67.64 # Previous best
    patience = 50        # More patience
    patience_counter = 0
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints/conservative_finetune'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("🚀 STARTING CONSERVATIVE FINE-TUNING...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (inputs, hard_labels, soft_labels) in enumerate(progress_bar):
            inputs, hard_labels = inputs.to(device), hard_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, hard_labels)
            
            loss.backward()
            
            # Conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += hard_labels.size(0)
            correct += (predicted == hard_labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.1f}%'
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        print("Validating...", end=" ")
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print("✅")
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            
            print(f"\n🎉 NEW BEST MODEL! Validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Best:  {best_val_acc:.2f}%")
        print(f"  LR:    {current_lr:.2e}")
        print(f"  ⏳ Patience: {patience_counter}/{patience}")
        print("-" * 50)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered! No improvement for {patience} epochs.")
            break
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"\n🏁 Training completed!")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved to: {checkpoint_dir}/best_model.pth")

if __name__ == "__main__":
    main() 