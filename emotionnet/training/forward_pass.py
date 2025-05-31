"""
Forward pass handling for EmotionNet training.
"""

import torch


class ForwardPassHandler:
    """Handles forward pass logic with auxiliary outputs and mixed labels."""
    
    def __init__(self, criterion):
        self.criterion = criterion
    
    def compute_loss(self, model, inputs, targets_a, targets_b, lam, is_mixed, accumulation_steps):
        """Compute loss for forward pass with optional auxiliary outputs."""
        model_output = model(inputs)
        
        if isinstance(model_output, tuple):
            outputs, aux_outputs = model_output
            main_loss = self._compute_mixed_loss(outputs, targets_a, targets_b, lam, is_mixed)
            aux_loss = self._compute_mixed_loss(aux_outputs, targets_a, targets_b, lam, is_mixed)
            loss = main_loss + 0.1 * aux_loss  # Reduced auxiliary loss weight
        else:
            outputs = model_output
            loss = self._compute_mixed_loss(outputs, targets_a, targets_b, lam, is_mixed)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        return loss, outputs
    
    def _compute_mixed_loss(self, outputs, targets_a, targets_b, lam, is_mixed):
        """Compute loss for mixed or regular targets."""
        if is_mixed:
            return lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
        else:
            return self.criterion(outputs, targets_a)
    
    def create_sam_closure(self, model, inputs, targets_a, targets_b, lam, is_mixed, accumulation_steps):
        """Create closure function for SAM optimizer."""
        def closure():
            # Recompute forward pass and loss for SAM
            model_output_sam = model(inputs)
            
            if isinstance(model_output_sam, tuple):
                outputs_sam, aux_outputs_sam = model_output_sam
                main_loss_sam = self._compute_mixed_loss(outputs_sam, targets_a, targets_b, lam, is_mixed)
                aux_loss_sam = self._compute_mixed_loss(aux_outputs_sam, targets_a, targets_b, lam, is_mixed)
                loss_sam = main_loss_sam + 0.1 * aux_loss_sam
            else:
                outputs_sam = model_output_sam
                loss_sam = self._compute_mixed_loss(outputs_sam, targets_a, targets_b, lam, is_mixed)
            
            loss_sam = loss_sam / accumulation_steps
            loss_sam.backward()
            return loss_sam
        
        return closure 