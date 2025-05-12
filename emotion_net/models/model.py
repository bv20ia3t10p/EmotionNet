"""Emotion recognition model with configurable backbone."""

import torch
import torch.nn as nn
import torchvision.models as models

class EmotionNet(nn.Module):
    """Emotion recognition model with configurable backbone."""
    
    def __init__(self, num_classes=7, backbone='resnet50', pretrained=True):
        """Initialize model with specified backbone.
        
        Args:
            num_classes (int): Number of emotion classes
            backbone (str): Backbone architecture ('resnet50', 'efficientnet_b0', etc.)
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final FC layer
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            in_features = self.backbone.classifier[3].in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Classification
        logits = self.classifier(features)
        
        return logits 