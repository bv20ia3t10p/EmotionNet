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
import math
warnings.filterwarnings('ignore')

# Import the data loading from train.py (keeping it identical)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ===== UPGRADED ARCHITECTURE COMPONENTS =====

class AdvancedPatchEmbedding(nn.Module):
    """Advanced multi-scale patch embedding with learnable aggregation for 64x64 images"""
    def __init__(self, img_size=64, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # For 64x64 images, calculate correct patch dimensions
        self.grid_size = img_size // patch_size  # 64//16 = 4
        self.num_patches = self.grid_size * self.grid_size  # 4*4 = 16 patches
        
        # Multi-scale patch extraction optimized for 64x64 images
        self.patch_16 = nn.Conv2d(in_channels, embed_dim//3, kernel_size=16, stride=16)  # 4x4 patches
        self.patch_8 = nn.Conv2d(in_channels, embed_dim//3, kernel_size=8, stride=8)    # 8x8 patches  
        self.patch_4 = nn.Conv2d(in_channels, embed_dim//3, kernel_size=4, stride=4)    # 16x16 patches
        
        # Adaptive pooling to ensure consistent spatial dimensions (4x4)
        self.spatial_pool_8 = nn.AdaptiveAvgPool2d(self.grid_size)  # Pool to 4x4
        self.spatial_pool_4 = nn.AdaptiveAvgPool2d(self.grid_size)  # Pool to 4x4
        
        # Cross-scale feature fusion with attention  
        self.scale_attention = nn.MultiheadAttention(embed_dim//3, num_heads=4, batch_first=True)  # Fixed: embed_dim//3 must be divisible by num_heads
        
        # Learnable positional encoding for 64x64 images
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Multi-scale feature extraction for 64x64 images
        x_16 = self.patch_16(x)  # [B, embed_dim//3, 4, 4]
        x_8 = self.spatial_pool_8(self.patch_8(x))  # [B, embed_dim//3, 4, 4]
        x_4 = self.spatial_pool_4(self.patch_4(x))  # [B, embed_dim//3, 4, 4]
        
        # Flatten to sequence format
        x_16_flat = x_16.flatten(2).transpose(1, 2)  # [B, 16, embed_dim//3]
        x_8_flat = x_8.flatten(2).transpose(1, 2)    # [B, 16, embed_dim//3]
        x_4_flat = x_4.flatten(2).transpose(1, 2)    # [B, 16, embed_dim//3]
        
        # Apply cross-scale attention
        x_8_attn, _ = self.scale_attention(x_8_flat, x_16_flat, x_16_flat)
        x_4_attn, _ = self.scale_attention(x_4_flat, x_16_flat, x_16_flat)
        
        # Combine multi-scale features
        combined = torch.cat([x_16_flat, x_8_attn, x_4_attn], dim=-1)  # [B, 16, embed_dim]
        
        # Add positional encoding
        combined = combined + self.pos_encoding
        
        return self.dropout(self.norm(combined))
        
class SqueezeExciteAttention(nn.Module):
    """Squeeze-and-Excitation enhanced attention mechanism"""
    def __init__(self, embed_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard attention components
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation for channel attention
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1 = nn.Linear(embed_dim, embed_dim // 4)
        self.se_fc2 = nn.Linear(embed_dim // 4, embed_dim)
        
        # Learnable emotion-specific attention patterns
        self.emotion_queries = nn.Parameter(torch.randn(8, embed_dim))  # 8 emotions
        self.emotion_attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Standard multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        
        # Squeeze-and-Excitation channel attention
        x_pooled = self.se_pool(x_attn.transpose(1, 2)).squeeze(-1)  # [B, C]
        se_weights = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(x_pooled))))  # [B, C]
        x_se = x_attn * se_weights.unsqueeze(1)  # [B, N, C]
        
        # Emotion-specific attention
        emotion_queries = self.emotion_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 8, C]
        emotion_attn, _ = self.emotion_attention(emotion_queries, x_se, x_se)  # [B, 8, C]
        
        # Combine with residual connection
        x_final = x + x_se
        
        return x_final, emotion_attn

class PyramidFeatureFusion(nn.Module):
    """Pyramid feature fusion for multi-level representations"""
    def __init__(self, embed_dim, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        
        # Pyramid projections
        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // (2**i)) for i in range(num_levels)
        ])
        
        # Cross-level attention
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim // (2**i), num_heads=max(1, 12 // (2**i)), batch_first=True)
            for i in range(num_levels)
        ])
        
        # Feature refinement
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim // (2**i), embed_dim // (2**i)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // (2**i), embed_dim // (2**i))
            ) for i in range(num_levels)
        ])
        
    def forward(self, features_list):
        """
        features_list: List of features from different depths
        """
        if len(features_list) < self.num_levels:
            # Pad with last feature if needed
            while len(features_list) < self.num_levels:
                features_list.append(features_list[-1])
        
        pyramid_features = []
        
        for i in range(self.num_levels):
            # Project to pyramid level
            feat = self.projections[i](features_list[i])
            
            # Cross-level attention if not first level
            if i > 0 and len(pyramid_features) > 0:
                # Use previous level as query, current as key/value
                prev_feat = pyramid_features[-1]
                # Interpolate dimensions if needed
                if prev_feat.size(-1) != feat.size(-1):
                    prev_feat_proj = nn.Linear(prev_feat.size(-1), feat.size(-1)).to(feat.device)
                    prev_feat = prev_feat_proj(prev_feat)
                
                feat_attn, _ = self.cross_attention[i](feat, prev_feat, prev_feat)
                feat = feat + feat_attn
            
            # Refine features
            feat = feat + self.refinement[i](feat)
            pyramid_features.append(feat)
        
        return pyramid_features
class AdaptiveEnsembleHead(nn.Module):
    """Adaptive ensemble prediction head with uncertainty estimation"""
    def __init__(self, feature_dims, num_classes=8, ensemble_size=5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_classes = num_classes
        
        # Multiple prediction heads for different feature levels
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_classes) for dim in feature_dims
        ])
        
        # Uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(dim, 1) for dim in feature_dims
        ])
        
        # Adaptive weight predictor
        total_dim = sum(feature_dims)
        self.weight_predictor = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim // 2, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, pyramid_features):
        predictions = []
        uncertainties = []
        
        # Get predictions and uncertainties from each head
        for i, (head, unc_head, feat) in enumerate(zip(self.heads, self.uncertainty_heads, pyramid_features)):
            pred = head(feat)
            unc = torch.sigmoid(unc_head(feat))  # Uncertainty in [0, 1]
            predictions.append(pred)
            uncertainties.append(unc)
        
        # Compute adaptive weights
        combined_features = torch.cat(pyramid_features, dim=-1)
        adaptive_weights = self.weight_predictor(combined_features)  # [B, num_heads]
        
        # Weighted ensemble with uncertainty
        ensemble_pred = 0
        total_weight = 0
        
        for i, (pred, unc, weight) in enumerate(zip(predictions, uncertainties, adaptive_weights.unbind(-1))):
            # Use uncertainty to modulate weights (lower uncertainty = higher weight)
            confidence = 1 - unc.mean(dim=-1, keepdim=True)  # [B, 1]
            final_weight = weight.unsqueeze(-1) * confidence  # [B, 1]
            
            ensemble_pred += final_weight * pred
            total_weight += final_weight
        
        # Normalize
        ensemble_pred = ensemble_pred / (total_weight + 1e-8)
        
        # Apply temperature scaling
        ensemble_pred = ensemble_pred / self.temperature
        
        return ensemble_pred, predictions, uncertainties

class AdvancedContrastiveLoss(nn.Module):
    """Advanced contrastive loss with hard negative mining"""
    def __init__(self, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal
        diag_mask = torch.eye(features.size(0), device=features.device)
        positive_mask = positive_mask - diag_mask
        
        # Hard negative mining
        hard_negatives = sim_matrix * negative_mask
        hard_neg_values, _ = hard_negatives.topk(k=min(10, hard_negatives.size(1)), dim=1)
        
        # Positive pairs
        positive_pairs = sim_matrix * positive_mask
        pos_sum = positive_pairs.sum(dim=1)
        pos_count = positive_mask.sum(dim=1)
        pos_mean = pos_sum / (pos_count + 1e-8)
        
        # Contrastive loss with hard negatives
        neg_exp_sum = torch.exp(hard_neg_values).sum(dim=1)
        pos_exp = torch.exp(pos_mean)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
        
        return loss.mean()# ===== UPGRADED MAIN MODEL =====
        
class UltraAdvancedFER(nn.Module):
    """Ultra-Advanced Facial Expression Recognition Model"""
    
    def __init__(self, img_size=64, patch_size=16, num_classes=8, embed_dim=768, 
                 depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Advanced patch embedding
        self.patch_embed = AdvancedPatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Advanced transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': SqueezeExciteAttention(embed_dim, num_heads, dropout),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout)
                ),
                'drop_path': nn.Dropout(dropout * (i / depth))  # Stochastic depth
            })
            self.blocks.append(block)
        
        # Pyramid feature fusion
        feature_dims = [embed_dim, embed_dim//2, embed_dim//4, embed_dim//8]
        self.pyramid_fusion = PyramidFeatureFusion(embed_dim, num_levels=4)
        
        # Adaptive ensemble head
        self.ensemble_head = AdaptiveEnsembleHead(feature_dims, num_classes)
        
        # Advanced losses
        self.contrastive_loss = AdvancedContrastiveLoss()
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
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
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Store features at different depths
        features = []
        emotion_attentions = []
        
        # Transformer blocks with advanced attention
        for i, block in enumerate(self.blocks):
            # Pre-norm and attention
            x_norm = block['norm1'](x)
            x_attn, emotion_attn = block['attn'](x_norm)
            x = x + block['drop_path'](x_attn)
            
            # MLP block
            x = x + block['drop_path'](block['mlp'](block['norm2'](x)))
            
            # Store features at key depths
            if i in [self.depth//4, self.depth//2, 3*self.depth//4, self.depth-1]:
                features.append(x[:, 0])  # Class token
                emotion_attentions.append(emotion_attn)
        
        # Final normalization
        features[-1] = self.norm(features[-1])
        
        return features, emotion_attentions
    
    def forward(self, x, labels=None):
        # Extract multi-level features
        features, emotion_attentions = self.forward_features(x)
        
        # Pyramid feature fusion
        pyramid_features = self.pyramid_fusion(features)
        
        # Adaptive ensemble prediction
        predictions, individual_preds, uncertainties = self.ensemble_head(pyramid_features)
        
        if self.training and labels is not None:
            # Compute advanced contrastive loss
            contrastive_loss = self.contrastive_loss(features[-1], labels)
            
            return predictions, {
                'contrastive_loss': contrastive_loss,
                'individual_predictions': individual_preds,
                'uncertainties': uncertainties,
                'emotion_attentions': emotion_attentions
            }
        
        return predictions# ===== KEEP SAME DATA LOADING FROM TRAIN.PY =====
class FERPlusDataset(Dataset):
    """Same data loading as train.py - keeping identical"""
    def __init__(self, csv_file, fer2013_csv_file, image_dir=None, split='Training', 
                 transform=None, min_votes=2, quality_threshold=8):
        self.csv_file = csv_file
        self.fer2013_csv_file = fer2013_csv_file
        self.image_dir = image_dir
        self.split = split
        self.transform = transform
        self.min_votes = min_votes
        self.quality_threshold = quality_threshold
        
        # FERPlus emotion mapping (8 classes)
        self.emotion_map = {
            0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
            4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'
        }
        
        self.emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
        
        # Load and process data
        self.load_data()
        
    def load_data(self):
        """Same data loading logic as train.py"""
        print(f"Loading FERPlus data for {self.split} split...")
        
        # Load FERPlus annotations
        ferplus_df = pd.read_csv(self.csv_file)
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
        
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)# ===== ADVANCED TRAINING UTILITIES =====
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def create_advanced_transforms():
    """Face-preserving advanced data transforms for 64x64 images"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # Face-preserving augmentations - very gentle
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability for faces
        transforms.RandomRotation(degrees=5),    # Minimal rotation to preserve facial structure
        # Gentle lighting changes that preserve facial features
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        # Minimal geometric transforms to preserve face proportions
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Very minimal random erasing to avoid destroying facial features
        transforms.RandomErasing(p=0.05, scale=(0.01, 0.03), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch_advanced(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Advanced training epoch with multiple losses"""
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
        
        if isinstance(outputs, tuple):
            predictions, aux_outputs = outputs
            contrastive_loss = aux_outputs['contrastive_loss']
        else:
            predictions = outputs
            contrastive_loss = 0
        
        # Multiple loss components
        hard_loss = criterion(predictions, hard_labels)
        soft_loss = F.kl_div(F.log_softmax(predictions, dim=1), soft_labels, reduction='batchmean')
        
        # Combined loss
        loss = 0.6 * hard_loss + 0.3 * soft_loss + 0.1 * contrastive_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = predictions.max(1)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total
def validate_epoch_advanced(model, dataloader, criterion, device):
    """Advanced validation with uncertainty estimation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, hard_labels, soft_labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                predictions, aux_outputs = outputs
                uncertainties = aux_outputs.get('uncertainties', [])
            else:
                predictions = outputs
                uncertainties = []
            
            loss = criterion(predictions, hard_labels)
            running_loss += loss.item()
            
            _, predicted = predictions.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
            if uncertainties:
                avg_uncertainty = torch.stack(uncertainties).mean(dim=0).cpu().numpy()
                all_uncertainties.extend(avg_uncertainty)
    
    return running_loss / len(dataloader), 100. * correct / total, all_predictions, all_labels, all_uncertainties

# ===== MAIN TRAINING FUNCTION =====
def train_ultraadvanced_model(ferplus_csv_path, fer2013_csv_path, 
                             num_epochs=100, batch_size=32, learning_rate=1e-4):
    """Main training function for Ultra-Advanced FER model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create advanced transforms
    train_transform, val_transform = create_advanced_transforms()
    
    # Create datasets (same as train.py)
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create ultra-advanced model
    print("Creating Ultra-Advanced FER model...")
    model = UltraAdvancedFER(
        img_size=64,
        patch_size=16,
        num_classes=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.15  # More reasonable dropout for 64x64 images
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Advanced loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Advanced scheduler with warmup
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=num_epochs,
        steps_per_epoch=len(train_loader), pct_start=warmup_epochs/num_epochs
    )
    
    # Training loop
    print("Starting Ultra-Advanced training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_uncertainties = validate_epoch_advanced(
            model, val_loader, criterion, device
        )        
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_ultraadvanced_fer.pth')
            print(f"New best validation accuracy: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        if val_uncertainties:
            print(f"Avg Uncertainty: {np.mean(val_uncertainties):.4f}")
        print("-" * 50)
    
    # Test on best model
    print("Testing on best model...")
    checkpoint = torch.load('best_ultraadvanced_fer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_preds, test_labels, test_uncertainties = validate_epoch_advanced(
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
    print("ULTRA-ADVANCED FER TRAINING")
    print("=" * 60)
    
    model, final_accuracy = train_ultraadvanced_model(
        ferplus_csv_path=FERPLUS_CSV,
        fer2013_csv_path=FER2013_CSV,
        num_epochs=500,
        batch_size=64,
        learning_rate=1e-4
    )
    
    if model is not None:
        print(f"\nUltra-Advanced training completed!")
        print(f"Final test accuracy: {final_accuracy:.2f}%")
        print(f"Target to beat: 95.55%")
        
        if final_accuracy > 95.55:
            print("🚀 SOTA ACHIEVED!")
        else:
            print("💪 Close to SOTA - continue training!")
    else:
        print("Training failed.")