import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings('ignore')

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ===== SIMPLIFIED EFFECTIVE ARCHITECTURE =====

class EfficientPatchEmbedding(nn.Module):
    """Simplified but effective patch embedding for 64x64 images"""
    def __init__(self, img_size=64, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # For 64x64 images with 16x16 patches = 4x4 = 16 patches
        self.grid_size = img_size // patch_size  # 4
        self.num_patches = self.grid_size * self.grid_size  # 16
        
        # Single-scale patch embedding (simpler is better)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Extract patches
        x = self.patch_embed(x)  # [B, embed_dim, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 16, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        return self.dropout(self.norm(x))

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance"""
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Class-specific alpha
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EffectiveFER(nn.Module):
    """Simplified but effective FER model for 64x64 images"""
    
    def __init__(self, img_size=64, patch_size=16, num_classes=8, embed_dim=512, 
                 depth=8, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Efficient patch embedding
        self.patch_embed = EfficientPatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Standard transformer blocks (simpler is better)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,  # Smaller MLP
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Simple classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 16, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 17, embed_dim]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Class token
        
        return self.head(cls_output)# ===== DATA LOADING (Same as before) =====
class FERPlusDataset(Dataset):
    """FERPlus Dataset class"""
    def __init__(self, csv_file, fer2013_csv_file, image_dir=None, split='Training', 
                 transform=None, min_votes=2, quality_threshold=8):
        self.csv_file = csv_file
        self.fer2013_csv_file = fer2013_csv_file
        self.image_dir = image_dir
        self.split = split
        self.transform = transform
        self.min_votes = min_votes
        self.quality_threshold = quality_threshold
        
        self.emotion_map = {
            0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
            4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'
        }
        
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        self.load_data()
        
    def load_data(self):
        """Load and process FERPlus dataset"""
        print(f"Loading FERPlus data for {self.split} split...")
        
        ferplus_df = pd.read_csv(self.csv_file)
        fer2013_df = pd.read_csv(self.fer2013_csv_file)
        
        if self.split == 'Training':
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'Training']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'Training']
        elif self.split == 'PublicTest':
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'PublicTest']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'PublicTest']
        else:
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'PrivateTest']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'PrivateTest']
        
        ferplus_split = ferplus_split.reset_index(drop=True)
        fer2013_split = fer2013_split.reset_index(drop=True)
        
        self.samples = []
        self.labels = []
        self.soft_labels = []
        
        min_length = min(len(ferplus_split), len(fer2013_split))
        
        for idx in range(min_length):
            try:
                ferplus_row = ferplus_split.iloc[idx]
                fer2013_row = fer2013_split.iloc[idx]
                
                emotion_votes = [ferplus_row[col] for col in self.emotion_columns]
                emotion_votes = [int(v) if pd.notna(v) else 0 for v in emotion_votes]
                
                total_votes = sum(emotion_votes)
                unknown_votes = int(ferplus_row['unknown']) if pd.notna(ferplus_row['unknown']) else 0
                quality_score = total_votes - unknown_votes
                
                if total_votes < self.min_votes or quality_score < self.quality_threshold:
                    continue
                
                pixel_string = fer2013_row['pixels']
                
                if pd.notna(pixel_string):
                    pixels = np.array([int(pixel) for pixel in pixel_string.split()])
                    image = pixels.reshape(48, 48).astype(np.uint8)
                    
                    hard_label = np.argmax(emotion_votes)
                    
                    soft_label = np.array(emotion_votes, dtype=np.float32)
                    if soft_label.sum() > 0:
                        soft_label = soft_label / soft_label.sum()
                    else:
                        soft_label = np.ones(len(emotion_votes), dtype=np.float32) / len(emotion_votes)
                    
                    self.samples.append(image)
                    self.labels.append(hard_label)
                    self.soft_labels.append(soft_label)
                    
            except (IndexError, ValueError, KeyError) as e:
                continue
        
        print(f"Loaded {len(self.samples)} {self.split} samples")
        
        if len(self.labels) > 0:
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"Class distribution in {self.split}:")
            for emotion_idx, count in zip(unique, counts):
                if emotion_idx < len(self.emotion_columns):
                    emotion_name = self.emotion_columns[emotion_idx]
                    print(f"  {emotion_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        hard_label = self.labels[idx]
        soft_label = self.soft_labels[idx]
        
        image = np.stack([image, image, image], axis=2)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)# ===== TRAINING UTILITIES =====
def create_effective_transforms():
    """Face-preserving transforms optimized for 64x64"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # Very gentle face-preserving augmentations
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=3),  # Minimal rotation
        transforms.ColorJitter(brightness=0.05, contrast=0.05),  # Very gentle
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_balanced_sampler(dataset):
    """Create weighted sampler to handle class imbalance"""
    labels = dataset.labels
    class_counts = np.bincount(labels)
    
    # Calculate weights (inverse frequency)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    
    # Boost minority classes even more
    minority_boost = {5: 10.0, 6: 5.0, 7: 10.0}  # disgust, fear, contempt
    for i, label in enumerate(labels):
        if label in minority_boost:
            sample_weights[i] *= minority_boost[label]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_epoch_effective(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Effective training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
    
    for batch_idx, (images, hard_labels, soft_labels) in enumerate(pbar):
        images = images.to(device)
        hard_labels = hard_labels.to(device)
        soft_labels = soft_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Focal loss for class imbalance + soft targets
        focal_loss = criterion(outputs, hard_labels)
        soft_loss = F.kl_div(F.log_softmax(outputs, dim=1), soft_labels, reduction='batchmean')
        
        # Combined loss
        loss = 0.8 * focal_loss + 0.2 * soft_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gentle clipping
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch_effective(model, dataloader, criterion, device):
    """Effective validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, hard_labels, soft_labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, hard_labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * correct / total, all_predictions, all_labels

def train_effective_model(ferplus_csv_path, fer2013_csv_path, 
                         num_epochs=50, batch_size=64, learning_rate=1e-3):
    """Main training function for effective model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform = create_effective_transforms()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FERPlusDataset(
        csv_file=ferplus_csv_path,
        fer2013_csv_file=fer2013_csv_path,
        split='Training',
        transform=train_transform
    )
    
    val_dataset = FERPlusDataset(
        csv_file=ferplus_csv_path,
        fer2013_csv_file=fer2013_csv_path,
        split='PublicTest',
        transform=val_transform
    )
    
    test_dataset = FERPlusDataset(
        csv_file=ferplus_csv_path,
        fer2013_csv_file=fer2013_csv_path,
        split='PrivateTest',
        transform=val_transform
    )    
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty.")
        return None, 0
    
    # Create balanced sampler for training
    balanced_sampler = create_balanced_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=balanced_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create effective model (much smaller and better)
    print("Creating Effective FER model...")
    model = EffectiveFER(
        img_size=64,
        patch_size=16,
        num_classes=8,
        embed_dim=512,  # Smaller than 768
        depth=8,        # Fewer layers than 12
        num_heads=8,    # Proper number of heads
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate class weights for focal loss
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=alpha, gamma=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print("Starting Effective training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch_effective(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch_effective(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_effective_fer.pth')
            print(f"🎯 New best validation accuracy: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 50)    
    # Test on best model
    print("Testing on best model...")
    checkpoint = torch.load('best_effective_fer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_preds, test_labels = validate_epoch_effective(
        model, test_loader, criterion, device
    )
    
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=emotion_names))
    
    return model, test_acc

if __name__ == "__main__":
    FERPLUS_CSV = "./FERPlus-master/fer2013new.csv"
    FER2013_CSV = "./fer2013.csv"
    
    if not os.path.exists(FERPLUS_CSV) or not os.path.exists(FER2013_CSV):
        print("Error: Required data files not found!")
        exit(1)
    
    print("=" * 60)
    print("EFFECTIVE FER TRAINING - SIMPLIFIED ARCHITECTURE")
    print("=" * 60)
    
    model, final_accuracy = train_effective_model(
        ferplus_csv_path=FERPLUS_CSV,
        fer2013_csv_path=FER2013_CSV,
        num_epochs=500,
        batch_size=64,
        learning_rate=1e-3
    )
    
    if model is not None:
        print(f"\nEffective training completed!")
        print(f"Final test accuracy: {final_accuracy:.2f}%")
        print("🎯 Target: Beat 83.84% baseline!")
        
        if final_accuracy > 83.84:
            print("🎉 SUCCESS! Beat the baseline!")
        else:
            print("💪 Keep improving to beat baseline!")
    else:
        print("Training failed.")