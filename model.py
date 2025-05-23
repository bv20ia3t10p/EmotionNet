import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.nn import TransformerEncoderLayer


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Add normalization for better stability
        self.query_norm = nn.BatchNorm2d(in_channels // 8)
        self.key_norm = nn.BatchNorm2d(in_channels // 8)
        self.value_norm = nn.BatchNorm2d(in_channels)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Generate query, key, and value with normalization
        proj_query = self.query_norm(self.query(x)).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_norm(self.key(x)).view(batch_size, -1, height * width)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_norm(self.value(x)).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # Apply residual connection with learned gamma
        out = self.gamma * out + x
        return out


class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        
        # Load pre-trained MobileNetV3 instead of EfficientNet for better efficiency
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Get the number of features from the last layer
        backbone_output_features = self.backbone.classifier[0].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Identity()
        
        # Enhanced channel attention with residual connection
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(backbone_output_features, backbone_output_features // 8, kernel_size=1),
            nn.BatchNorm2d(backbone_output_features // 8),
            nn.ReLU(),
            nn.Dropout2d(0.4),  # Increased dropout
            nn.Conv2d(backbone_output_features // 8, backbone_output_features, kernel_size=1),
            nn.BatchNorm2d(backbone_output_features),
            nn.Sigmoid()
        )
        
        # Improved self-attention layer with more features
        self.self_attention = SelfAttention(backbone_output_features)
        
        # Transformer encoder with stronger regularization
        encoder_layer = TransformerEncoderLayer(
            d_model=backbone_output_features,
            nhead=8,
            dim_feedforward=backbone_output_features*4,
            dropout=0.4,  # Increased dropout
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(backbone_output_features, backbone_output_features),
            nn.LayerNorm(backbone_output_features),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Final classifier with stronger regularization and residual connections
        self.fc1 = nn.Linear(backbone_output_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Extract features using backbone
        x = self.backbone.features(x)
        
        # Apply channel attention with residual connection
        identity = x
        channel_weights = self.channel_attention(x)
        x = x * channel_weights + identity  # Residual connection
        
        # Apply self-attention
        x = self.self_attention(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Apply transformer encoder for global context
        x_trans = x.unsqueeze(1)  # Add sequence dimension
        x_trans = self.transformer_encoder(x_trans)
        x_trans = x_trans.squeeze(1)  # Remove sequence dimension
        
        # Feature fusion with residual connection
        x = self.fusion(x_trans) + x  # Residual connection
        
        # Classification with residual connections
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x


def get_model(num_classes=7):
    """
    Returns the EmotionNet model with the specified number of classes.
    
    Args:
        num_classes (int): Number of emotion classes to predict
    
    Returns:
        EmotionNet: The constructed model
    """
    return EmotionNet(num_classes=num_classes) 