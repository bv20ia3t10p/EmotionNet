"""
Loss Functions for EmotionNet Training
Contains various loss functions for handling class imbalance and improving training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance by focusing on hard examples
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothing(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(pred, dim=-1), true_dist, reduction='batchmean')


class AdvancedCrossEntropyLoss(nn.Module):
    """Advanced Cross Entropy with label smoothing and temperature scaling for SOTA"""
    def __init__(self, num_classes=7, label_smoothing=0.1, temperature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
    def forward(self, inputs, targets):
        # Apply temperature scaling
        inputs = inputs / self.temperature
        
        # Label smoothing
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                        self.label_smoothing / self.num_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss 