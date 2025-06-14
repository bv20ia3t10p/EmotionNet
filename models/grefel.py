import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import timm
import math

class EnhancedGeometryAwareModule(nn.Module):
    """Enhanced Geometry-aware module with multi-scale attention and adaptive anchors"""
    def __init__(self, feature_dim: int, num_anchors: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_anchors = num_anchors
        
        # Multi-scale learnable anchors
        self.anchors = nn.Parameter(torch.randn(num_anchors, feature_dim))
        nn.init.xavier_uniform_(self.anchors, gain=0.1)
        
        # Adaptive anchor refinement
        self.anchor_refiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Tanh()  # Bounded refinement
        )
        
        # Multi-head attention for anchor-feature interaction
        self.num_heads = 8
        self.head_dim = feature_dim // self.num_heads
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Geometry loss components with much smaller initial weights
        self.diversity_weight = nn.Parameter(torch.tensor(0.001))
        self.center_weight = nn.Parameter(torch.tensor(0.001))
        
        # Initialize projections
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.constant_(module.bias, 0)
            
        # Initialize anchor refiner
        for module in self.anchor_refiner.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        
        # Normalize inputs for stability
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # Refine anchors adaptively
        anchor_refinements = self.anchor_refiner(self.anchors)
        refined_anchors = self.anchors + 0.1 * anchor_refinements  # Small refinement
        anchors_norm = F.normalize(refined_anchors, p=2, dim=1, eps=1e-8)
        
        # Multi-head attention between features and anchors
        queries = self.query_proj(features_norm).view(batch_size, self.num_heads, self.head_dim)
        keys = self.key_proj(anchors_norm).view(self.num_anchors, self.num_heads, self.head_dim)
        values = self.value_proj(anchors_norm).view(self.num_anchors, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.einsum('bhd,khd->bhk', queries, keys) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, H, K]
        
        # Apply attention to values
        attended_values = torch.einsum('bhk,khd->bhd', attention_weights, values)
        attended_values = attended_values.contiguous().view(batch_size, -1)
        
        # Project back to feature dimension
        attended_features = self.out_proj(attended_values)
        
        # Enhance features with residual connection
        enhanced_features = features_norm + 0.1 * attended_features
        enhanced_features = self.feature_enhancer(enhanced_features)
        
        # Add residual connection
        final_features = features_norm + 0.2 * enhanced_features
        
        # Compute geometry losses with learnable weights
        # 1. Diversity loss: encourage anchors to be different
        anchor_distances = torch.pdist(anchors_norm, p=2)
        if anchor_distances.numel() > 0:
            diversity_loss = torch.exp(-anchor_distances.mean().clamp(min=1e-6))
        else:
            diversity_loss = torch.tensor(0.0, device=features.device)
        
        # 2. Center loss: encourage features to be close to assigned anchors
        # Use average attention weights across heads
        avg_attention = attention_weights.mean(dim=1)  # [B, K]
        feature_anchor_distances = torch.cdist(features_norm.unsqueeze(1), anchors_norm.unsqueeze(0))
        feature_anchor_distances = feature_anchor_distances.squeeze(1)  # [B, K]
        
        assigned_distances = (avg_attention * feature_anchor_distances).sum(dim=1)  # [B]
        center_loss = assigned_distances.mean()
        
        # Combined geometry loss with learnable weights and better scaling
        geo_loss = (torch.abs(self.diversity_weight) * diversity_loss + 
                   torch.abs(self.center_weight) * center_loss)
        geo_loss = torch.clamp(geo_loss, min=0, max=0.01)  # Much smaller clamp
        
        return final_features, geo_loss

class EnhancedReliabilityBalancingModule(nn.Module):
    """Enhanced reliability balancing with uncertainty estimation and adaptive weighting"""
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Multi-layer reliability estimation network
        self.reliability_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(feature_dim // 4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output reliability score [0, 1]
        )
        
        # Uncertainty estimation through Monte Carlo Dropout
        self.uncertainty_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(0.2)  # Higher dropout for uncertainty estimation
            ) for _ in range(3)
        ])
        
        # Adaptive weighting based on prediction confidence
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement for reliable samples
        self.feature_refiner = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize all networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def estimate_uncertainty(self, features: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """Estimate uncertainty using Monte Carlo Dropout"""
        uncertainties = []
        
        for _ in range(num_samples):
            # Apply uncertainty layers with dropout
            uncertain_features = features
            for layer in self.uncertainty_layers:
                uncertain_features = layer(uncertain_features)
            
            # Compute prediction variance as uncertainty measure
            feature_variance = torch.var(uncertain_features, dim=1, keepdim=True)
            uncertainties.append(feature_variance)
        
        # Average uncertainty across samples
        uncertainty = torch.stack(uncertainties).mean(dim=0)
        return uncertainty
    
    def forward(self, features: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        
        # Normalize features for stability
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # Estimate reliability score
        reliability_scores = self.reliability_net(features_norm)  # [B, 1]
        
        # Estimate uncertainty
        uncertainty_scores = self.estimate_uncertainty(features_norm, num_samples=3)  # [B, 1]
        
        # Estimate confidence from predictions
        prediction_probs = F.softmax(predictions, dim=1)
        max_probs = torch.max(prediction_probs, dim=1, keepdim=True)[0]  # [B, 1]
        entropy = -torch.sum(prediction_probs * torch.log(prediction_probs + 1e-8), dim=1, keepdim=True)  # [B, 1]
        
        # Normalize entropy to [0, 1] range
        max_entropy = math.log(self.num_classes)
        normalized_entropy = entropy / max_entropy
        confidence_from_entropy = 1.0 - normalized_entropy
        
        # Combine confidence measures
        confidence_scores = self.confidence_estimator(prediction_probs)  # [B, 1]
        
        # Final reliability combines multiple factors
        # Higher reliability = high confidence + low uncertainty + high max prob
        combined_reliability = (
            0.4 * reliability_scores +
            0.3 * confidence_scores +
            0.2 * max_probs +
            0.1 * confidence_from_entropy
        )
        
        # Inverse relationship with uncertainty
        uncertainty_factor = torch.sigmoid(-5.0 * (uncertainty_scores - 0.1))  # Sigmoid to [0,1]
        final_reliability = combined_reliability * uncertainty_factor
        
        # Clamp reliability to reasonable range
        final_reliability = torch.clamp(final_reliability, min=0.1, max=0.95)
        
        # Refine features based on reliability
        # High reliability samples get enhanced features
        reliability_weights = final_reliability.expand(-1, self.feature_dim)
        refined_features = self.feature_refiner(features_norm)
        
        # Weighted combination: more reliable samples use refined features more
        enhanced_features = (
            reliability_weights * refined_features +
            (1.0 - reliability_weights) * features_norm
        )
        
        # Add residual connection
        final_features = features_norm + 0.1 * enhanced_features
        
        return final_features, final_reliability.squeeze(1)  # [B], [B]

class GReFEL(nn.Module):
    """Full GReFEL architecture with multi-scale features and geometry-aware learning"""
    def __init__(
        self,
        num_classes: int = 8,
        feature_dim: int = 768,
        num_anchors: int = 10,
        drop_rate: float = 0.15
    ):
        super().__init__()
        
        # Use standard ViT-Base without features_only to avoid spatial dimensions
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            num_classes=0  # Remove classification head
        )
        
        self.backbone_dim = 768
        self.feature_dim = feature_dim
        
        # Since we can't easily get multi-scale from standard ViT, 
        # we'll create multiple projections from the final features
        self.multi_scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.backbone_dim),
                nn.Linear(self.backbone_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(drop_rate * 0.5),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(drop_rate * 0.5)
            ) for _ in range(3)
        ])
        
        # Initialize projections
        for proj in self.multi_scale_projections:
            for module in proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # Slightly larger gain
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        # Geometry-aware modules for each "scale"
        self.geometry_modules = nn.ModuleList([
            EnhancedGeometryAwareModule(feature_dim, num_anchors) for _ in range(3)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        # Initialize fusion
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Slightly larger gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Reliability balancing module
        self.reliability_module = EnhancedReliabilityBalancingModule(feature_dim, num_classes)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Initialize classifier with proper scaling
        for i, module in enumerate(self.classifier.modules()):
            if isinstance(module, nn.Linear):
                if i == len(list(self.classifier.modules())) - 1:  # Last layer
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from ViT backbone
        backbone_features = self.backbone(x)  # [B, 768]
        
        # Create "multi-scale" features by applying different projections
        processed_features = []
        total_geo_loss = 0
        
        for i, projection in enumerate(self.multi_scale_projections):
            # Apply different projections to simulate multi-scale
            projected = projection(backbone_features)
            
            # Apply geometry-aware module
            geo_features, geo_loss = self.geometry_modules[i](projected)
            
            processed_features.append(geo_features)
            total_geo_loss += geo_loss
        
        # Fuse multi-scale features
        fused_features = torch.cat(processed_features, dim=1)  # [B, feature_dim * 3]
        fused_features = self.fusion(fused_features)  # [B, feature_dim]
        
        # Reliability estimation
        reliability, reliability_score = self.reliability_module(fused_features, self.classifier(fused_features))
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'reliability': reliability,
            'reliability_score': reliability_score,
            'geo_loss': total_geo_loss / 3,  # Average geometry loss
            'features': fused_features
        }

class EnhancedGReFELLoss(nn.Module):
    """Enhanced GReFEL loss with adaptive weighting and improved reliability handling"""
    def __init__(self, num_classes: int, label_smoothing: float = 0.11, 
                 geo_weight: float = 0.01, reliability_weight: float = 0.1,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.geo_weight = geo_weight
        self.reliability_weight = reliability_weight
        
        # Class weights for handling imbalanced datasets
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            # Default balanced weights
            self.register_buffer('class_weights', torch.ones(num_classes))
        
        # Adaptive loss weights (learnable) with smaller initial values
        self.adaptive_geo_weight = nn.Parameter(torch.tensor(geo_weight * 0.1))  # Much smaller
        self.adaptive_reliability_weight = nn.Parameter(torch.tensor(reliability_weight * 0.1))
        
        # Temperature for reliability-based weighting
        self.temperature = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, 
                is_soft_labels: bool = False) -> Dict[str, torch.Tensor]:
        logits = outputs['logits']
        reliability = outputs['reliability']
        reliability_score = outputs['reliability_score']
        geo_loss = outputs['geo_loss']
        
        batch_size = logits.size(0)
        
        # Classification loss with reliability weighting
        if is_soft_labels:
            # For soft labels, use KL divergence
            log_probs = F.log_softmax(logits, dim=1)
            
            # Apply label smoothing to soft targets
            if self.label_smoothing > 0:
                # Smooth the soft labels
                smoothed_targets = targets * (1 - self.label_smoothing) + \
                                 self.label_smoothing / self.num_classes
            else:
                smoothed_targets = targets
            
            # KL divergence loss
            kl_loss = F.kl_div(log_probs, smoothed_targets, reduction='none').sum(dim=1)
            
            # Weight by reliability scores
            reliability_weights = torch.sigmoid(reliability_score * self.temperature)
            weighted_kl_loss = kl_loss * reliability_weights
            classification_loss = weighted_kl_loss.mean()
            
        else:
            # For hard labels, use cross-entropy with label smoothing
            if self.label_smoothing > 0:
                # Create one-hot encoding
                one_hot = torch.zeros_like(logits)
                targets_long = targets.long()
                # Ensure targets are 1D for scatter operation
                if targets_long.dim() > 1:
                    targets_long = targets_long.squeeze()
                one_hot.scatter_(1, targets_long.unsqueeze(1), 1)
                
                # Apply label smoothing
                smoothed_targets = one_hot * (1 - self.label_smoothing) + \
                                 self.label_smoothing / self.num_classes
                
                # Cross-entropy with smoothed labels
                log_probs = F.log_softmax(logits, dim=1)
                ce_loss = -(smoothed_targets * log_probs).sum(dim=1)
            else:
                targets_long = targets.long()
                if targets_long.dim() > 1:
                    targets_long = targets_long.squeeze()
                ce_loss = F.cross_entropy(logits, targets_long, weight=self.class_weights, reduction='none')
            
            # Weight by reliability scores
            reliability_weights = torch.sigmoid(reliability_score * self.temperature)
            weighted_ce_loss = ce_loss * reliability_weights
            classification_loss = weighted_ce_loss.mean()
        
        # Reliability loss: encourage high reliability for confident predictions
        # and low reliability for uncertain predictions
        pred_probs = F.softmax(logits, dim=1)
        max_probs = torch.max(pred_probs, dim=1)[0]
        
        # Target reliability should be high when prediction is confident
        target_reliability = torch.sigmoid(5.0 * (max_probs - 0.5))  # Sigmoid mapping
        
        # MSE loss between predicted and target reliability
        reliability_loss = F.mse_loss(reliability_score, target_reliability)
        
        # Adaptive geometry loss weighting
        adaptive_geo_loss = torch.abs(self.adaptive_geo_weight) * geo_loss
        adaptive_reliability_loss = torch.abs(self.adaptive_reliability_weight) * reliability_loss
        
        # Total loss with adaptive weighting and better numerical stability
        total_loss = (classification_loss + 
                     adaptive_geo_loss + 
                     adaptive_reliability_loss)
        
        # Check for NaN/Inf before clamping
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Return a safe fallback loss
            total_loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
        else:
            # Clamp total loss to prevent explosion
            total_loss = torch.clamp(total_loss, min=0, max=5.0)  # Reduced max clamp
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'geometry_loss': adaptive_geo_loss,
            'reliability_loss': adaptive_reliability_loss,
            'avg_reliability': reliability_score.mean(),
            'avg_confidence': max_probs.mean()
        } 