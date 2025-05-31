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
    
    Enhanced version with:
    - Class weighting to handle class imbalance
    - Automatic weight calculation based on inverse frequency
    - Temperature scaling option for better gradients
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', weight=None, temperature=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight
        self.temperature = temperature
        
    def forward(self, inputs, targets):
        # Apply temperature scaling if needed
        if self.temperature != 1.0:
            inputs = inputs / self.temperature
        
        if self.weight is not None:
            # Use provided class weights
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        else:
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


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss for better generalization
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        
        # Convert targets to one-hot and apply smoothing
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Calculate loss
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss that uses effective number of samples to reweight loss
    Based on "Class-Balanced Loss Based on Effective Number of Samples"
    
    Automatically calculates the weights based on sample frequency
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, loss_type='focal'):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # Calculate effective number of samples
        effective_num = 1.0 - beta ** samples_per_class
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights
        weights = weights / weights.sum() * len(samples_per_class)
        self.weights = weights
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, weight=weights)
        elif loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, inputs, targets):
        return self.criterion(inputs, targets) 