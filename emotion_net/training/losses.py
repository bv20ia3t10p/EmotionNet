import torch
import torch.nn as nn
import torch.nn.functional as F

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