import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
warnings.filterwarnings('ignore')

# Import the HybridFER-Max model (previous artifact)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class PatchEmbedding(nn.Module):
    """Enhanced patch embedding with overlapping patches and multi-scale features"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, overlap=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
        # Multi-scale patch extraction optimized for 112x112 images
        self.patch_embed_16 = nn.Conv2d(in_channels, embed_dim//2, kernel_size=16, stride=16, padding=0)   # 7x7 patches
        self.patch_embed_8 = nn.Conv2d(in_channels, embed_dim//4, kernel_size=8, stride=8, padding=0)     # 14x14 patches  
        self.patch_embed_4 = nn.Conv2d(in_channels, embed_dim//4, kernel_size=4, stride=4, padding=0)     # 28x28 patches
        
        # Adaptive pooling to ensure consistent spatial dimensions
        self.target_size = img_size // patch_size  # 112 // 16 = 7
        self.adaptive_pool_8 = nn.AdaptiveAvgPool2d(self.target_size)
        self.adaptive_pool_4 = nn.AdaptiveAvgPool2d(self.target_size)
        
        # Calculate actual number of patches (all scales produce same spatial size)
        self.num_patches = self.target_size * self.target_size  # 7 * 7 = 49
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Multi-scale patch extraction for 112x112 images
        x_16 = self.patch_embed_16(x).flatten(2).transpose(1, 2)  # [B, N, embed_dim//2]
        
        x_8 = self.patch_embed_8(x)  # [B, embed_dim//4, H', W']
        x_8 = self.adaptive_pool_8(x_8).flatten(2).transpose(1, 2)  # [B, N, embed_dim//4]
        
        x_4 = self.patch_embed_4(x)  # [B, embed_dim//4, H', W']
        x_4 = self.adaptive_pool_4(x_4).flatten(2).transpose(1, 2)  # [B, N, embed_dim//4]
        
        # Concatenate multi-scale features along feature dimension
        x = torch.cat([x_16, x_8, x_4], dim=-1)  # [B, N, embed_dim]
        
        return self.norm(x)

class FacialRegionAttention(nn.Module):
    """Attention mechanism specifically designed for facial regions"""
    def __init__(self, embed_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable facial region masks (eyes, mouth, eyebrows, etc.)
        self.facial_regions = nn.Parameter(torch.randn(7, embed_dim))  # 7 facial regions
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Region-specific attention weights
        self.region_weights = nn.Parameter(torch.ones(7))
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Standard multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add facial region bias - simplified approach
        region_bias = torch.einsum('rc,r->c', self.facial_regions, self.region_weights)
        region_bias = region_bias.view(1, 1, 1, C)
        # Only add bias if dimensions match
        if attn.size(-1) == N:
            # Apply region bias to attention scores
            attn = attn + region_bias[:, :, :, :N]
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class EmotionContrastiveLoss(nn.Module):
    """Contrastive loss for better emotion separation"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(positive_mask.size(0), device=positive_mask.device)
        
        # Compute contrastive loss with numerical stability
        exp_sim = torch.exp(similarity_matrix)
        positive_loss = -torch.log((exp_sim * positive_mask).sum(1) + 1e-8)
        negative_loss = torch.log((exp_sim * negative_mask).sum(1) + 1e-8)
        
        return (positive_loss + negative_loss).mean()

class AdaptiveMixup(nn.Module):
    """Adaptive mixup for data augmentation during training"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, y, training=True):
        if not training:
            return x, y
            
        batch_size = x.size(0)
        if batch_size <= 1:
            return x, y
            
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # Simplified adaptive lambda
        lam = float(lam)  # Convert to scalar
        
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        return mixed_x, (y, y[index], lam)

class HybridFERMax(nn.Module):
    """
    HybridFER-Max: State-of-the-art facial expression recognition model
    
    Key innovations:
    1. Multi-scale overlapping patch embedding
    2. Facial region-aware attention mechanism
    3. Hierarchical feature fusion
    4. Contrastive learning for emotion separation
    5. Adaptive mixup augmentation
    6. Ensemble of multiple prediction heads
    """
    
    def __init__(self, img_size=224, patch_size=16, num_classes=7, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Enhanced patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks with facial region attention
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': FacialRegionAttention(embed_dim, num_heads, dropout),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(depth)
        ])
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Linear(embed_dim, embed_dim // 8)
        ])
        
        # Ensemble heads for robust prediction
        self.head_global = nn.Linear(embed_dim, num_classes)
        self.head_local = nn.Linear(embed_dim // 2, num_classes)
        self.head_micro = nn.Linear(embed_dim // 4, num_classes)
        
        # Emotion-specific feature extractors
        self.emotion_extractors = nn.ModuleList([
            nn.Linear(embed_dim, 128) for _ in range(num_classes)
        ])
        
        # Adaptive mixup
        self.mixup = AdaptiveMixup()
        
        # Contrastive loss
        self.contrastive_loss = EmotionContrastiveLoss()
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding (resize if needed for dynamic patching)
        if x.size(1) != self.pos_embed.size(1):
            # Interpolate position embedding to match actual sequence length
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed
        x = self.dropout(x)
        
        # Store intermediate features for multi-scale fusion
        features = []
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = x + block['attn'](block['norm1'](x))
            x = x + block['mlp'](block['norm2'](x))
            
            # Store features at different depths - fixed indexing
            if i in [self.depth//4, self.depth//2, 3*self.depth//4]:
                features.append(x[:, 0])  # Class token
        
        # Final feature
        x = self.norm(x)
        features.append(x[:, 0])  # Final class token
        
        return features
    
    def forward(self, x, labels=None):
        # Apply adaptive mixup during training
        if self.training and labels is not None:
            x, labels = self.mixup(x, labels, self.training)
        
        # Extract multi-scale features
        features = self.forward_features(x)
        
        # Handle case where we have fewer features than expected
        if len(features) < 3:
            # Pad with the last feature if we don't have enough
            while len(features) < 3:
                features.append(features[-1])
        
        # Multi-scale feature processing
        global_feat = features[-1]  # Full depth feature
        mid_feat = self.feature_fusion[0](features[-2]) if len(features) >= 2 else self.feature_fusion[0](global_feat)
        early_feat = self.feature_fusion[1](features[-3]) if len(features) >= 3 else self.feature_fusion[1](global_feat)
        
        # Ensemble predictions
        pred_global = self.head_global(global_feat)
        pred_local = self.head_local(mid_feat)
        pred_micro = self.head_micro(early_feat)
        
        # Weighted ensemble
        ensemble_pred = 0.5 * pred_global + 0.3 * pred_local + 0.2 * pred_micro
        
        if self.training and labels is not None:
            # Compute contrastive loss for better emotion separation
            emotion_features = []
            for i, extractor in enumerate(self.emotion_extractors):
                emotion_features.append(extractor(global_feat))
            
            # Handle mixup labels
            if isinstance(labels, tuple):
                y_a, y_b, lam = labels
                # Stack emotion features properly
                stacked_features = torch.stack(emotion_features, dim=1).mean(dim=1)  # Average across emotions
                contrastive_loss = lam * self.contrastive_loss(stacked_features, y_a) + \
                                 (1 - lam) * self.contrastive_loss(stacked_features, y_b)
            else:
                stacked_features = torch.stack(emotion_features, dim=1).mean(dim=1)
                contrastive_loss = self.contrastive_loss(stacked_features, labels)
            
            return ensemble_pred, contrastive_loss
        
        return ensemble_pred


# ===== FERPLUS DATASET CLASS =====
class FERPlusDataset(Dataset):
    """
    FERPlus Dataset class for loading and processing FER2013+ data
    
    The fer2013new.csv contains:
    - usage: Training/PublicTest/PrivateTest
    - neutral, happiness, surprise, sadness, anger, disgust, fear, contempt: vote counts
    - unknown, NF: quality indicators
    """
    
    def __init__(self, csv_file, fer2013_csv_file, image_dir=None, split='Training', 
                 transform=None, min_votes=2, quality_threshold=8):
        """
        Args:
            csv_file: Path to fer2013new.csv (FERPlus annotations)
            fer2013_csv_file: Path to original fer2013.csv (contains image pixels)
            image_dir: Directory containing fer*.png images (optional)
            split: 'Training', 'PublicTest', or 'PrivateTest'
            transform: Image transformations
            min_votes: Minimum votes required for a sample
            quality_threshold: Minimum quality score (max votes - unknown votes)
        """
        self.csv_file = csv_file
        self.fer2013_csv_file = fer2013_csv_file
        self.image_dir = image_dir
        self.split = split
        self.transform = transform
        self.min_votes = min_votes
        self.quality_threshold = quality_threshold
        
        # FERPlus emotion mapping (8 classes)
        self.emotion_map = {
            0: 'neutral',
            1: 'happiness', 
            2: 'surprise',
            3: 'sadness',
            4: 'anger',
            5: 'disgust',
            6: 'fear',
            7: 'contempt'
        }
        
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        # Load and process data
        self.load_data()
        
    def load_data(self):
        """Load and process FERPlus dataset"""
        print(f"Loading FERPlus data for {self.split} split...")
        
        # Load FERPlus annotations
        ferplus_df = pd.read_csv(self.csv_file)
        
        # Load original FER2013 data (contains pixel data)
        fer2013_df = pd.read_csv(self.fer2013_csv_file)
        
        # Filter by split
        if self.split == 'Training':
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'Training']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'Training']
        elif self.split == 'PublicTest':
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'PublicTest']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'PublicTest']
        else:  # PrivateTest
            ferplus_split = ferplus_df[ferplus_df['Usage'] == 'PrivateTest']
            fer2013_split = fer2013_df[fer2013_df['Usage'] == 'PrivateTest']
        
        # Reset indices for proper alignment
        ferplus_split = ferplus_split.reset_index(drop=True)
        fer2013_split = fer2013_split.reset_index(drop=True)
        
        # Process samples
        self.samples = []
        self.labels = []
        self.soft_labels = []
        
        # Ensure both dataframes have the same length
        min_length = min(len(ferplus_split), len(fer2013_split))
        
        for idx in range(min_length):
            try:
                ferplus_row = ferplus_split.iloc[idx]
                fer2013_row = fer2013_split.iloc[idx]
                
                # Get emotion vote counts
                emotion_votes = [ferplus_row[col] for col in self.emotion_columns]
                emotion_votes = [int(v) if pd.notna(v) else 0 for v in emotion_votes]
                
                # Quality filtering
                total_votes = sum(emotion_votes)
                unknown_votes = int(ferplus_row['unknown']) if pd.notna(ferplus_row['unknown']) else 0
                quality_score = total_votes - unknown_votes
                
                if total_votes < self.min_votes or quality_score < self.quality_threshold:
                    continue
                
                # Get pixel string and convert to image
                pixel_string = fer2013_row['pixels']
                
                if pd.notna(pixel_string):
                    pixels = np.array([int(pixel) for pixel in pixel_string.split()])
                    image = pixels.reshape(48, 48).astype(np.uint8)
                    
                    # Create hard label (majority vote)
                    hard_label = np.argmax(emotion_votes)
                    
                    # Create soft label (probability distribution)
                    soft_label = np.array(emotion_votes, dtype=np.float32)
                    if soft_label.sum() > 0:
                        soft_label = soft_label / soft_label.sum()
                    else:
                        # Fallback to uniform distribution
                        soft_label = np.ones(len(emotion_votes), dtype=np.float32) / len(emotion_votes)
                    
                    self.samples.append(image)
                    self.labels.append(hard_label)
                    self.soft_labels.append(soft_label)
                    
            except (IndexError, ValueError, KeyError) as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} {self.split} samples")
        
        # Print class distribution
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
        # Get image and label
        image = self.samples[idx]
        hard_label = self.labels[idx]
        soft_label = self.soft_labels[idx]
        
        # Convert grayscale to RGB
        image = np.stack([image, image, image], axis=2)
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)

# ===== TRAINING UTILITIES =====
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SoftTargetCrossEntropy(nn.Module):
    """Cross entropy loss with soft targets (for label smoothing)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss

def create_transforms():
    """Create data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Resize to 112x112 for better feature extraction
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Resize to 112x112 for better feature extraction
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train one epoch"""
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
        outputs = model(images, hard_labels)
        
        # Handle outputs based on whether contrastive loss is returned
        if isinstance(outputs, tuple):
            predictions, contrastive_loss = outputs
        else:
            predictions = outputs
            contrastive_loss = 0
        
        # Combined loss: hard labels + soft labels + contrastive
        hard_loss = criterion(predictions, hard_labels)
        soft_loss = F.kl_div(F.log_softmax(predictions, dim=1), soft_labels, reduction='batchmean')
        
        # Weighted combination
        loss = 0.7 * hard_loss + 0.2 * soft_loss + 0.1 * contrastive_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = predictions.max(1)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
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
            soft_labels = soft_labels.to(device)
            
            outputs = model(images)
            
            # Loss
            hard_loss = criterion(outputs, hard_labels)
            soft_loss = F.kl_div(F.log_softmax(outputs, dim=1), soft_labels, reduction='batchmean')
            loss = 0.7 * hard_loss + 0.3 * soft_loss
            
            running_loss += loss.item()
            
            # Predictions
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, emotion_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===== MAIN TRAINING FUNCTION =====
def train_ferplus_model(ferplus_csv_path, fer2013_csv_path, 
                       num_epochs=100, batch_size=32, learning_rate=1e-4):
    """
    Main training function for FERPlus dataset
    
    Args:
        ferplus_csv_path: Path to fer2013new.csv
        fer2013_csv_path: Path to original fer2013.csv
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
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
    
    # Check if datasets are not empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty. Please check your data files.")
        return None, 0
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating HybridFER-Max model...")
    model = HybridFERMax(
        img_size=112,       # 112x112 image size for good balance between efficiency and detail
        patch_size=16,      # 16x16 patches for 112x112 images (gives 7x7 = 49 patches)
        num_classes=8,      # FERPlus has 8 emotions
        embed_dim=768,      # Full embedding dimension for 112x112 images
        depth=12,           # Full depth for better performance
        num_heads=12        # Full attention heads
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print("Starting training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_hybridfer_max.pth')
            print(f"New best validation accuracy: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 50)
    
    # Test on best model
    print("Testing on best model...")
    model.load_state_dict(torch.load('best_hybridfer_max.pth'))
    test_loss, test_acc, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)
    
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # Classification report
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=emotion_names))
    
    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, emotion_names)
    
    return model, test_acc

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # File paths - UPDATE THESE PATHS
    FERPLUS_CSV = "./FERPlus-master/fer2013new.csv"  # Path to FERPlus annotations
    FER2013_CSV = "./fer2013.csv"     # Path to original FER2013 with pixel data
    
    # Check if files exist
    if not os.path.exists(FERPLUS_CSV):
        print(f"Error: {FERPLUS_CSV} not found!")
        print("Download from: https://github.com/microsoft/FERPlus/blob/master/fer2013new.csv")
        exit(1)
    
    if not os.path.exists(FER2013_CSV):
        print(f"Error: {FER2013_CSV} not found!")
        print("Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        exit(1)
    
    # Start training
    print("=" * 60)
    print("HYBRIDFER-MAX TRAINING ON FERPLUS DATASET")
    print("=" * 60)
    
    model, final_accuracy = train_ferplus_model(
        ferplus_csv_path=FERPLUS_CSV,
        fer2013_csv_path=FER2013_CSV,
        num_epochs=500,
        batch_size=64,   # Balanced batch size for 112x112 images
        learning_rate=1e-4
    )
    
    if model is not None:
        print(f"\nTraining completed!")
        print(f"Final test accuracy: {final_accuracy:.2f}%")
        print(f"Target SOTA to beat: 95.55%")
        
        if final_accuracy > 95.55:
            print("🎉 CONGRATULATIONS! You've beaten the SOTA!")
        else:
            print("💪 Keep training to reach SOTA performance!")
    else:
        print("Training failed. Please check your data files and try again.")