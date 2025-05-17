"""
Focal Loss implementation for emotion recognition.
Implementation adapted from https://github.com/clcarwin/focal_loss_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableFocalLoss(nn.Module):
    """
    Stable version of Focal Loss that handles corner cases and NaN values.
    
    Focal Loss is described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0, eps=1e-8):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
                                      Can be a 1D tensor of size C (number of classes).
            gamma (float, optional): A constant, as described in the paper.
                                     Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                                       Defaults to 'mean'.
            label_smoothing (float, optional): Label smoothing factor.
                                              Defaults to 0.0.
            eps (float, optional): Small constant for numerical stability.
                                  Defaults to 1e-8.
        """
        super(StableFocalLoss, self).__init__()
        if gamma < 0:
            raise ValueError("Gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): The model's unnormalized logits, tensor of shape (N, C).
            targets (Tensor): The ground truth targets, tensor of shape (N).
                              Each value must be in [0, C-1].
        Returns:
            Tensor: The focal loss.
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            # Create one-hot encoding of targets
            one_hot = F.one_hot(targets, num_classes).float()
            # Apply label smoothing
            smooth_targets = (1 - self.label_smoothing) * one_hot + self.label_smoothing / num_classes
            
            # Use log_softmax for numerical stability
            log_probs = F.log_softmax(inputs, dim=1)
            # Compute smoothed cross entropy loss
            loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            # Standard cross entropy without label smoothing
            loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities using softmax, but handle overflow/underflow
        # First, make inputs more numerically stable by subtracting max
        inputs_stable = inputs - inputs.max(dim=1, keepdim=True)[0].detach()
        exp_inputs = torch.exp(inputs_stable)
        probs = exp_inputs / (exp_inputs.sum(dim=1, keepdim=True) + self.eps)
        
        # Get probability for the correct class
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weight - add eps to avoid NaN when target_probs is 0
        focal_weight = torch.pow(1 - target_probs + self.eps, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * loss

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Get alpha for each sample
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Handle NaN values
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

# Keep the original FocalLoss for backward compatibility
class FocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
                                      Can be a 1D tensor of size C (number of classes).
            gamma (float, optional): A constant, as described in the paper.
                                     Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                                       Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        if gamma < 0:
            raise ValueError("Gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): The model's unnormalized logits, tensor of shape (N, C).
            targets (Tensor): The ground truth targets, tensor of shape (N).
                              Each value must be in [0, C-1].
        Returns:
            Tensor: The focal loss.
        """
        # Calculate Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Get alpha for each sample
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}") 