import torch
import torch.nn as nn

class SimpleChannelAttention(nn.Module):
    def __init__(self, num_features, reduction_ratio=16):
        super(SimpleChannelAttention, self).__init__()
        self.num_features = num_features
        reduced_features = max(1, num_features // reduction_ratio) # Ensure reduced_features is at least 1
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(num_features, reduced_features),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_features, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x has shape (batch_size, num_features)
        # Global average pooling is not explicitly needed here as we operate on already pooled features.
        # The MLP will learn to derive channel importance from the feature vector itself.
        attention_scores = self.attention_mlp(x)
        return x * attention_scores 