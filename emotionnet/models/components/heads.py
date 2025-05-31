"""
Classification heads and pooling modules for EmotionNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module for multi-scale context"""
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPooling, self).__init__()
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        for size in sizes[1:]:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = []
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        return torch.cat(out, dim=1)


class EmotionSpecificHead(nn.Module):
    """Emotion-specific classification head with multi-task learning"""
    def __init__(self, in_features, num_classes=7):
        super(EmotionSpecificHead, self).__init__()
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Emotion-specific branches
        self.emotion_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ) for _ in range(num_classes)
        ])
        
        # Global classifier
        self.global_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        shared = self.shared_fc(x)
        
        # Get emotion-specific predictions
        emotion_scores = []
        for branch in self.emotion_branches:
            emotion_scores.append(branch(shared))
        emotion_scores = torch.cat(emotion_scores, dim=1)
        
        # Get global predictions
        global_scores = self.global_classifier(shared)
        
        # Combine predictions
        combined_scores = 0.7 * global_scores + 0.3 * emotion_scores
        
        return combined_scores 