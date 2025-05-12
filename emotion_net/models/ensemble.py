"""Ensemble model for emotion recognition."""

import torch
import torch.nn as nn
import timm
from .model_parts import SimpleChannelAttention

class EnsembleModel(nn.Module):
    """Ensemble of multiple backbone models with attention fusion."""
    
    def __init__(self, num_classes=7, backbones=None, drop_path_rate=0.1, pretrained=True):
        super(EnsembleModel, self).__init__()
        if backbones is None:
            backbones = ['efficientnet_b0', 'resnet18']
        
        self.backbone_names = backbones
        self.backbones = nn.ModuleList()
        self.neck = nn.ModuleList()
        backbone_out_dims = []
        
        # Create backbones
        for backbone_name in backbones:
            if 'efficientnet' in backbone_name:
                model = timm.create_model(
                    backbone_name, 
                    pretrained=pretrained, 
                    drop_path_rate=drop_path_rate
                )
                out_dim = model.classifier.in_features
                model.classifier = nn.Identity()
                
            elif 'resnet' in backbone_name:
                model = timm.create_model(
                    backbone_name, 
                    pretrained=pretrained, 
                    drop_path_rate=drop_path_rate
                )
                out_dim = model.fc.in_features
                model.fc = nn.Identity()
                
            elif 'vit' in backbone_name:
                model = timm.create_model(
                    backbone_name, 
                    pretrained=pretrained, 
                    drop_path_rate=drop_path_rate
                )
                out_dim = model.head.in_features
                model.head = nn.Identity()
                
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            self.backbones.append(model)
            backbone_out_dims.append(out_dim)
            
            # Create neck for each backbone (reduce dimensions and add regularization)
            # Use a simpler neck architecture with less dropout for faster convergence
            neck = nn.Sequential(
                nn.Linear(out_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),  # Change to ReLU for faster convergence
                nn.Dropout(0.3)  # Reduced dropout
            )
            self.neck.append(neck)
        
        # Use a fixed averaging approach initially instead of learned attention
        # This avoids the need to learn attention weights at the beginning
        self.use_fixed_averaging = True  # This will be used in forward
        
        # Attention mechanism to weight models (will be used after some convergence)
        self.attention = nn.Sequential(
            nn.Linear(512 * len(backbones), len(backbones)),
            nn.Softmax(dim=1)
        )
        
        # Final classifier - simpler architecture for faster initial convergence
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)  # Direct projection to classes
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from each backbone
        features = []
        for i, backbone in enumerate(self.backbones):
            feat = backbone(x)
            feat = self.neck[i](feat)
            features.append(feat)
        
        # Use simple averaging for initial training epochs
        if self.use_fixed_averaging:
            # Simple average of features
            weighted_features = torch.stack(features).mean(dim=0)
        else:
            # Later can switch to learned attention mechanism
            # Concatenate features for attention
            concat_features = torch.cat(features, dim=1)
            
            # Get attention weights
            weights = self.attention(concat_features)
            
            # Apply attention weights
            weighted_features = torch.zeros_like(features[0])
            for i, feat in enumerate(features):
                weighted_features += feat * weights[:, i].unsqueeze(1)
        
        # Final classification
        logits = self.classifier(weighted_features)
        
        return logits, weighted_features 