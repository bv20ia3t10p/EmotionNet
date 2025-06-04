import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class WindowCrossAttention(nn.Module):
    """Window-based Cross-Attention mechanism for landmark-image fusion"""
    
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from landmark, Key and Value from image
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, landmark_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        B, N, C = image_features.shape
        
        # Generate Q, K, V
        q = self.q_proj(landmark_features).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(image_features).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(image_features).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        if attn.shape[-1] == relative_position_bias.shape[-1]:
            attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer block with window-based cross-attention"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, window_size: int = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = WindowCrossAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, landmark_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        image_features = image_features + self.cross_attn(
            self.norm1(landmark_features), self.norm1(image_features)
        )
        
        # MLP
        image_features = image_features + self.mlp(self.norm2(image_features))
        
        return image_features

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction with Vision Transformer"""
    
    def __init__(self, img_size: int = 224, patch_sizes: List[int] = [28, 14, 7], 
                 embed_dim: int = 512, num_heads: int = 8, depth: int = 6):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        
        # Patch embedding layers for different scales
        self.patch_embeds = nn.ModuleList()
        for patch_size in patch_sizes:
            num_patches = (img_size // patch_size) ** 2
            self.patch_embeds.append(nn.Linear(patch_size * patch_size * 3, embed_dim))
        
        # Position embeddings
        self.pos_embeds = nn.ParameterList()
        for patch_size in patch_sizes:
            num_patches = (img_size // patch_size) ** 2
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, num_patches, embed_dim)))
        
        # Transformer blocks for each scale
        self.transformer_blocks = nn.ModuleList()
        for _ in patch_sizes:
            blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads) for _ in range(depth)
            ])
            self.transformer_blocks.append(blocks)
        
        # Final fusion transformer
        self.fusion_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
            for _ in range(2)
        ])
        
        # Landmark feature extractor (simplified)
        self.landmark_extractor = nn.Sequential(
            nn.Linear(68 * 2, 256),  # 68 facial landmarks with x,y coordinates
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    
    def patchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Convert image to patches"""
        B, C, H, W = x.shape
        assert H == W and H % patch_size == 0
        
        num_patches_per_side = H // patch_size
        patches = x.view(B, C, num_patches_per_side, patch_size, num_patches_per_side, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, num_patches_per_side * num_patches_per_side, C * patch_size * patch_size)
        
        return patches
    
    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224)
            landmarks: (B, 68, 2) - facial landmarks
        """
        B = images.shape[0]
        
        # Extract landmark features
        landmark_features = self.landmark_extractor(landmarks.view(B, -1))
        landmark_features = landmark_features.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Multi-scale feature extraction
        scale_features = []
        
        for i, patch_size in enumerate(self.patch_sizes):
            # Patchify image
            patches = self.patchify(images, patch_size)  # (B, num_patches, patch_dim)
            
            # Patch embedding
            patch_features = self.patch_embeds[i](patches)  # (B, num_patches, embed_dim)
            
            # Add position embedding
            patch_features = patch_features + self.pos_embeds[i]
            
            # Apply transformer blocks with cross-attention
            for block in self.transformer_blocks[i]:
                patch_features = block(landmark_features, patch_features)
            
            # Global average pooling
            scale_feature = patch_features.mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
            scale_features.append(scale_feature)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_features, dim=1)  # (B, num_scales, embed_dim)
        
        # Final fusion with transformer
        for layer in self.fusion_transformer:
            multi_scale_features = layer(multi_scale_features)
        
        # Final feature embedding
        final_embedding = multi_scale_features.mean(dim=1)  # (B, embed_dim)
        
        return final_embedding

class GeometryAwareAnchors(nn.Module):
    """Geometry-aware anchor points for reliability balancing"""
    
    def __init__(self, num_classes: int, num_anchors_per_class: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_class = num_anchors_per_class
        self.embed_dim = embed_dim
        
        # Trainable anchor points
        self.anchors = nn.Parameter(
            torch.randn(num_classes, num_anchors_per_class, embed_dim) * 0.02
        )
        
        # Anchor labels (one-hot)
        anchor_labels = torch.zeros(num_classes, num_anchors_per_class, num_classes)
        for i in range(num_classes):
            anchor_labels[i, :, i] = 1.0
        self.register_buffer('anchor_labels', anchor_labels)
        
    def forward(self, embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute similarity-based label correction
        
        Args:
            embeddings: (B, embed_dim)
            temperature: Temperature for softmax
        
        Returns:
            label_correction: (B, num_classes)
        """
        B = embeddings.shape[0]
        
        # Compute distances to all anchors
        embeddings_expanded = embeddings.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, embed_dim)
        anchors_expanded = self.anchors.unsqueeze(0)  # (1, num_classes, num_anchors_per_class, embed_dim)
        
        # Euclidean distance
        distances = torch.norm(
            embeddings_expanded - anchors_expanded, dim=-1
        )  # (B, num_classes, num_anchors_per_class)
        
        # Convert to similarities
        similarities = torch.exp(-distances / temperature)  # (B, num_classes, num_anchors_per_class)
        
        # Normalize similarities
        similarities_flat = similarities.view(B, -1)  # (B, num_classes * num_anchors_per_class)
        similarities_normalized = F.softmax(similarities_flat, dim=-1)
        similarities_normalized = similarities_normalized.view(B, self.num_classes, self.num_anchors_per_class)
        
        # Compute weighted label correction
        anchor_labels_expanded = self.anchor_labels.unsqueeze(0)  # (1, num_classes, num_anchors_per_class, num_classes)
        
        label_correction = torch.sum(
            similarities_normalized.unsqueeze(-1) * anchor_labels_expanded,
            dim=(1, 2)
        )  # (B, num_classes)
        
        return label_correction

class ReliabilityBalancing(nn.Module):
    """Reliability balancing module with anchor and attention correction"""
    
    def __init__(self, num_classes: int, embed_dim: int, num_anchors_per_class: int = 10, num_heads: int = 8):
        super().__init__()
        self.num_classes = num_classes
        
        # Geometry-aware anchors
        self.anchors = GeometryAwareAnchors(num_classes, num_anchors_per_class, embed_dim)
        
        # Multi-head self-attention for attentive correction
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_projection = nn.Linear(embed_dim, num_classes)
        
    def confidence_function(self, label_dist: torch.Tensor) -> torch.Tensor:
        """Compute confidence using normalized entropy"""
        # Normalized entropy
        entropy = -torch.sum(label_dist * torch.log(label_dist + 1e-8), dim=-1)
        normalized_entropy = entropy / math.log(self.num_classes)
        confidence = 1 - normalized_entropy
        return confidence
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, embed_dim)
            
        Returns:
            geometric_correction: (B, num_classes)
            attentive_correction: (B, num_classes)
        """
        B = embeddings.shape[0]
        
        # Geometric correction using anchors
        geometric_correction = self.anchors(embeddings)
        
        # Attentive correction using self-attention
        embeddings_seq = embeddings.unsqueeze(1)  # (B, 1, embed_dim)
        attn_output, _ = self.self_attention(embeddings_seq, embeddings_seq, embeddings_seq)
        attentive_correction = self.attention_projection(attn_output.squeeze(1))  # (B, num_classes)
        attentive_correction = F.softmax(attentive_correction, dim=-1)
        
        return geometric_correction, attentive_correction

class GReFELModel(nn.Module):
    """Complete GReFEL model for facial expression recognition"""
    
    def __init__(self, num_classes: int = 8, img_size: int = 224, embed_dim: int = 512,
                 num_heads: int = 8, depth: int = 6, num_anchors_per_class: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(
            img_size=img_size,
            patch_sizes=[28, 14, 7],
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth
        )
        
        # Primary classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )
        
        # Reliability balancing
        self.reliability_balancing = ReliabilityBalancing(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_anchors_per_class=num_anchors_per_class,
            num_heads=num_heads
        )
        
    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, 224, 224)
            landmarks: (B, 68, 2)
            
        Returns:
            Dictionary containing:
                - embeddings: Feature embeddings
                - primary_logits: Primary classifier logits
                - primary_probs: Primary classifier probabilities
                - geometric_correction: Geometric label correction
                - attentive_correction: Attentive label correction
                - final_probs: Final corrected probabilities
        """
        # Extract features
        embeddings = self.feature_extractor(images, landmarks)
        
        # Primary classification
        primary_logits = self.classifier(embeddings)
        primary_probs = F.softmax(primary_logits, dim=-1)
        
        # Reliability balancing
        geometric_correction, attentive_correction = self.reliability_balancing(embeddings)
        
        # Compute confidences
        primary_confidence = self.reliability_balancing.confidence_function(primary_probs)
        geometric_confidence = self.reliability_balancing.confidence_function(geometric_correction)
        attentive_confidence = self.reliability_balancing.confidence_function(attentive_correction)
        
        # Combine corrections with confidence weighting
        total_geometric_confidence = geometric_confidence + attentive_confidence + 1e-8
        geometric_weight = geometric_confidence / total_geometric_confidence
        attentive_weight = attentive_confidence / total_geometric_confidence
        
        combined_correction = (
            geometric_weight.unsqueeze(-1) * geometric_correction +
            attentive_weight.unsqueeze(-1) * attentive_correction
        )
        
        # Final label distribution
        correction_confidence = self.reliability_balancing.confidence_function(combined_correction)
        total_confidence = primary_confidence + correction_confidence + 1e-8
        
        primary_weight = primary_confidence / total_confidence
        correction_weight = correction_confidence / total_confidence
        
        final_probs = (
            primary_weight.unsqueeze(-1) * primary_probs +
            correction_weight.unsqueeze(-1) * combined_correction
        )
        
        return {
            'embeddings': embeddings,
            'primary_logits': primary_logits,
            'primary_probs': primary_probs,
            'geometric_correction': geometric_correction,
            'attentive_correction': attentive_correction,
            'final_probs': final_probs
        }

class GReFELLoss(nn.Module):
    """Combined loss function for GReFEL training"""
    
    def __init__(self, lambda_cls: float = 1.0, lambda_anchor: float = 1.0, lambda_center: float = 1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_anchor = lambda_anchor
        self.lambda_center = lambda_center
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, model: GReFELModel) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels (B,)
            model: GReFEL model for accessing anchors
            
        Returns:
            Dictionary of losses
        """
        B = targets.shape[0]
        num_classes = outputs['final_probs'].shape[-1]
        
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets, num_classes).float()
        
        # Classification loss (negative log-likelihood)
        cls_loss = F.cross_entropy(outputs['primary_logits'], targets)
        
        # Anchor loss - maximize distance between different anchors
        anchors = model.reliability_balancing.anchors.anchors  # (num_classes, num_anchors_per_class, embed_dim)
        anchor_distances = []
        
        for i in range(anchors.shape[0]):
            for j in range(i + 1, anchors.shape[0]):
                for ki in range(anchors.shape[1]):
                    for kj in range(anchors.shape[1]):
                        dist = torch.norm(anchors[i, ki] - anchors[j, kj], p=2) ** 2
                        anchor_distances.append(dist)
        
        if anchor_distances:
            anchor_loss = -torch.mean(torch.stack(anchor_distances))
        else:
            anchor_loss = torch.tensor(0.0, device=anchors.device)
        
        # Center loss - minimize distance between embeddings and corresponding class anchors
        embeddings = outputs['embeddings']  # (B, embed_dim)
        center_losses = []
        
        for i in range(B):
            target_class = targets[i].item()
            class_anchors = anchors[target_class]  # (num_anchors_per_class, embed_dim)
            
            # Find closest anchor for this embedding
            distances = torch.norm(embeddings[i:i+1] - class_anchors, dim=-1)
            min_distance = torch.min(distances)
            center_losses.append(min_distance ** 2)
        
        center_loss = torch.mean(torch.stack(center_losses))
        
        # Total loss
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_anchor * anchor_loss +
            self.lambda_center * center_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'anchor_loss': anchor_loss,
            'center_loss': center_loss
        }

# Example usage and training setup
def create_dummy_data(batch_size: int = 8, num_classes: int = 8):
    """Create dummy data for testing"""
    images = torch.randn(batch_size, 3, 224, 224)
    landmarks = torch.randn(batch_size, 68, 2)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, landmarks, labels

def main():
    """Example training setup"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    model = GReFELModel(num_classes=8, embed_dim=512).to(device)
    criterion = GReFELLoss(lambda_cls=1.0, lambda_anchor=1.0, lambda_center=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Create dummy data
    images, landmarks, labels = create_dummy_data()
    images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
    
    # Forward pass
    model.train()
    outputs = model(images, landmarks)
    
    # Compute losses
    losses = criterion(outputs, labels, model)
    
    # Backward pass
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()
    
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Classification Loss: {losses['cls_loss'].item():.4f}")
    print(f"Anchor Loss: {losses['anchor_loss'].item():.4f}")
    print(f"Center Loss: {losses['center_loss'].item():.4f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(images, landmarks)
        predictions = torch.argmax(outputs['final_probs'], dim=-1)
        accuracy = (predictions == labels).float().mean()
        print(f"Accuracy: {accuracy.item():.4f}")

if __name__ == "__main__":
    main() 