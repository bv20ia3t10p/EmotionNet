import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random

class DynamicAugmentation:
    """
    Dynamically adjusts augmentation strength based on class performance.
    Underperforming classes receive stronger augmentation.
    """
    def __init__(self, num_classes=7, base_strength=0.5):
        self.num_classes = num_classes
        self.base_strength = base_strength
        self.class_strengths = [base_strength] * num_classes
        self.min_strength = 0.3
        self.max_strength = 1.0
        
        # Initialize augmentation pools
        self.weak_augs = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        
        self.medium_augs = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            A.RandomShadow(p=0.2),
            A.CoarseDropout(max_holes=4, max_height=8, max_width=8, p=0.2),
        ])
        
        self.strong_augs = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2),
                A.GridDistortion(distort_limit=0.2),
                A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10),
            ], p=0.3),
            A.RandomShadow(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.4),
            A.RandomGamma(p=0.3),
        ])
        
        # Mixup/CutMix probabilities
        self.mixup_prob = 0.3
        self.cutmix_prob = 0.3
        self.alpha = 0.4
    
    def update_class_strengths(self, class_f1_scores):
        """
        Update augmentation strength based on F1 scores.
        Lower F1 scores get stronger augmentation.
        """
        for i, f1 in enumerate(class_f1_scores):
            # Scale strength inversely proportional to F1
            inverse_strength = 1.0 - f1  # When F1 is low, inverse_strength is high
            # Smooth changes with moving average
            self.class_strengths[i] = 0.7 * self.class_strengths[i] + 0.3 * inverse_strength
            # Clip to valid range
            self.class_strengths[i] = max(self.min_strength, min(self.max_strength, self.class_strengths[i]))
        
        # Log the updated strengths
        print("Updated augmentation strengths by class:")
        for i, strength in enumerate(self.class_strengths):
            print(f"Class {i}: {strength:.3f}")
    
    def __call__(self, img, label=None):
        """
        Apply dynamic augmentation based on class label.
        If label is None, applies medium augmentation.
        """
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
        
        # Choose augmentation strength based on class
        if label is not None:
            strength = self.class_strengths[label]
        else:
            strength = self.base_strength
        
        # Apply augmentation based on strength
        if strength < 0.4:
            augmented = self.weak_augs(image=img_np)['image']
        elif strength < 0.7:
            augmented = self.medium_augs(image=img_np)['image']
        else:
            augmented = self.strong_augs(image=img_np)['image']
        
        # Convert back to PIL
        if isinstance(img, Image.Image):
            return Image.fromarray(augmented)
        return augmented

class MixAugmentation:
    """
    Implements Mixup and CutMix augmentations with dynamic probabilities.
    """
    def __init__(self, alpha=0.4, cutmix_alpha=1.0):
        self.alpha = alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = 0.5
        self.cutmix_prob = 0.3
    
    def update_probabilities(self, epoch, total_epochs):
        """
        Update mixup/cutmix probabilities based on training progress.
        """
        progress = epoch / total_epochs
        # Increase mixup probability as training progresses
        self.mixup_prob = min(0.8, 0.3 + progress * 0.5)
        # Decrease cutmix probability as training progresses
        self.cutmix_prob = max(0.1, 0.5 - progress * 0.4)
    
    def mixup(self, inputs, targets):
        """
        Apply mixup augmentation to a batch.
        """
        batch_size = inputs.size(0)
        
        # Generate mixup coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1-lam)  # Ensure lam >= 0.5 for better stability
        
        # Create shuffled indices
        index = torch.randperm(batch_size).to(inputs.device)
        
        # Mix the inputs
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        
        # Return mixed inputs, pairs of targets, and lambda
        return mixed_inputs, targets, targets[index], lam
    
    def cutmix(self, inputs, targets):
        """
        Apply cutmix augmentation to a batch.
        """
        batch_size = inputs.size(0)
        
        # Generate random box parameters
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Create shuffled indices
        index = torch.randperm(batch_size).to(inputs.device)
        
        # Get dimensions
        _, _, h, w = inputs.shape
        
        # Create random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Get random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Calculate box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        inputs_copy = inputs.clone()
        inputs_copy[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return inputs_copy, targets, targets[index], lam
    
    def __call__(self, inputs, targets, epoch=0, total_epochs=30):
        """
        Apply either mixup, cutmix, or no augmentation based on probabilities.
        """
        self.update_probabilities(epoch, total_epochs)
        
        r = np.random.rand()
        if r < self.mixup_prob:
            return self.mixup(inputs, targets)
        elif r < self.mixup_prob + self.cutmix_prob:
            return self.cutmix(inputs, targets)
        else:
            # No mix augmentation, return original with dummy values
            return inputs, targets, targets, 1.0

def get_transform(phase, img_size=224, dynamic_aug=None):
    """
    Get transformation pipeline with optional dynamic augmentation.
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 