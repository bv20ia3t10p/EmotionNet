"""Ensemble model for emotion recognition."""

import torch
import torch.nn as nn
import timm

class EnsembleModel(nn.Module):
    """Ensemble of multiple backbone models with attention fusion."""
    
    def __init__(self, num_classes=7, backbones=None):
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
                model = timm.create_model(backbone_name, pretrained=True)
                out_dim = model.classifier.in_features
                model.classifier = nn.Identity()
                
            elif 'resnet' in backbone_name:
                model = timm.create_model(backbone_name, pretrained=True)
                out_dim = model.fc.in_features
                model.fc = nn.Identity()
                
            elif 'vit' in backbone_name:
                model = timm.create_model(backbone_name, pretrained=True)
                out_dim = model.head.in_features
                model.head = nn.Identity()
                
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            self.backbones.append(model)
            backbone_out_dims.append(out_dim)
            
            # Create neck for each backbone (reduce dimensions and add regularization)
            neck = nn.Sequential(
                nn.Linear(out_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.neck.append(neck)
        
        # Attention mechanism to weight models
        self.attention = nn.Sequential(
            nn.Linear(512 * len(backbones), len(backbones)),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Get features from each backbone
        features = []
        for i, backbone in enumerate(self.backbones):
            feat = backbone(x)
            feat = self.neck[i](feat)
            features.append(feat)
        
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