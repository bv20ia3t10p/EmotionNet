"""
Training-time augmentation handler for EmotionNet.
"""

import torch
from ..data.augmentation import MixUp, CutMix, RandAugment


class TrainingAugmentationHandler:
    """Handles training-time augmentation strategies."""
    
    def __init__(self, config):
        self.config = config
        
        # Check if augmentation config exists
        has_aug_config = hasattr(config, 'augmentation')
        
        # Initialize augmentation methods with safe defaults if config attributes don't exist
        mixup_alpha = getattr(config.augmentation, 'mixup_alpha', 0.2) if has_aug_config else 0.2
        self.mixup = MixUp(alpha=mixup_alpha)
        
        cutmix_alpha = getattr(config.augmentation, 'cutmix_alpha', 1.0) if has_aug_config else 1.0
        self.cutmix = CutMix(alpha=cutmix_alpha)
        
        randaug_n = getattr(config.augmentation, 'randaugment_n', 2) if has_aug_config else 2
        randaug_m = getattr(config.augmentation, 'randaugment_m', 9) if has_aug_config else 9
        self.randaugment = RandAugment(n=randaug_n, m=randaug_m)
        
        # Class-specific augmentation probabilities
        self.class_aug_probs = {
            0: 0.5,   # Angry
            1: 0.3,   # Disgust
            2: 0.6,   # Fear
            3: 0.3,   # Happy
            4: 0.5,   # Sad
            5: 0.3,   # Surprise
            6: 0.5    # Neutral
        }
        
        # For FERPlus dataset (8 classes)
        if hasattr(config.model, 'num_classes') and config.model.num_classes == 8:
            self.class_aug_probs[7] = 0.4  # Contempt
    
    def apply_class_specific_augmentation(self, inputs, targets):
        """Apply class-specific RandAugment based on emotion type."""
        for i in range(inputs.size(0)):
            label = targets[i].item()
            aug_prob = self.class_aug_probs.get(label, 0.5)
            if torch.rand(1) < aug_prob:
                inputs[i:i+1] = self.randaugment(inputs[i:i+1])
        return inputs
    
    def apply_mixup_cutmix(self, inputs, targets):
        """Apply MixUp or CutMix augmentation."""
        # Check if augmentations are enabled in configuration safely
        has_aug_config = hasattr(self.config, 'augmentation')
        
        mixup_enabled = (
            getattr(self.config.augmentation, 'use_mixup', False) 
            if has_aug_config else False
        )
        
        cutmix_enabled = (
            getattr(self.config.augmentation, 'use_cutmix', False)
            if has_aug_config else False
        )
        
        # Get alpha values safely 
        mixup_alpha = (
            getattr(self.config.augmentation, 'mixup_alpha', 0.2)
            if has_aug_config else 0.2
        )
        
        cutmix_alpha = (
            getattr(self.config.augmentation, 'cutmix_alpha', 1.0)
            if has_aug_config else 1.0
        )
        
        use_mixup = mixup_enabled and mixup_alpha > 0 and torch.rand(1) < 0.5
        use_cutmix = cutmix_enabled and cutmix_alpha > 0 and torch.rand(1) < 0.5
        
        lam = 1.0
        targets_a = targets
        targets_b = targets
        
        if use_mixup and not use_cutmix:
            inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
        elif use_cutmix and not use_mixup:
            inputs, targets_a, targets_b, lam = self.cutmix(inputs, targets)
        
        return inputs, targets_a, targets_b, lam, (use_mixup or use_cutmix) 