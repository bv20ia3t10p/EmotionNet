"""
Data augmentation utilities for EmotionNet.

Contains advanced augmentation techniques for better model generalization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MixUp:
    """
    MixUp augmentation for better generalization.
    
    Mixes pairs of examples and their labels linearly.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Mixup interpolation strength parameter
        """
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch of images and labels.
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda)
        """
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels, labels, 1.0
        
        # Generate mix ratio
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam


class CutMix:
    """
    CutMix augmentation for better localization.
    
    Cuts and pastes patches between training images.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: CutMix interpolation strength parameter
        """
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch of images and labels.
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda)
        """
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels, labels, 1.0
        
        # Generate mix ratio
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Get image dimensions
        _, _, H, W = images.size()
        
        # Generate random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling of center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        images_clone = images.clone()
        images_clone[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images_clone, labels, labels[index], lam


class RandAugment:
    """
    RandAugment implementation for emotion recognition.
    
    Applies N random augmentations with magnitude M.
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        """
        Args:
            n: Number of augmentations to apply
            m: Magnitude of augmentations (0-10 scale)
        """
        self.n = n
        self.m = m
        
    def __call__(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply RandAugment to input tensor.
        
        Args:
            x: Input tensor
            labels: Optional labels (unused in this implementation)
            
        Returns:
            Augmented tensor
        """
        # Apply n random augmentations with magnitude m
        device = x.device
        
        # Define augmentation operations
        ops = [
            self.rotate, self.translate, self.brightness, 
            self.contrast, self.sharpness, self.gaussian_noise
        ]
        
        for _ in range(self.n):
            op = np.random.choice(ops)
            x = op(x)
            
        return torch.clamp(x, 0, 1)
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        angle = (self.m / 10) * 30  # Max 30 degrees
        angle = np.random.uniform(-angle, angle)
        k = int(angle // 90)
        if k != 0:
            return torch.rot90(x, k=k, dims=[2, 3])
        return x
    
    def translate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random translation."""
        shift = int((self.m / 10) * 4)  # Max 4 pixels
        shift_x = np.random.randint(-shift, shift + 1)
        shift_y = np.random.randint(-shift, shift + 1)
        return torch.roll(x, shifts=(shift_x, shift_y), dims=(2, 3))
    
    def brightness(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random brightness adjustment."""
        factor = 1 + (self.m / 10) * 0.3 * np.random.uniform(-1, 1)
        return x * factor
    
    def contrast(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random contrast adjustment."""
        factor = 1 + (self.m / 10) * 0.3 * np.random.uniform(-1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - mean) * factor + mean
    
    def sharpness(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random sharpness adjustment."""
        if np.random.random() < 0.5:
            kernel = torch.tensor([[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]).float().to(x.device)
            return F.conv2d(x, kernel, padding=1)
        return x
    
    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise_level = (self.m / 10) * 0.1
        noise = torch.randn_like(x) * noise_level
        return x + noise


class TestTimeAugmentation:
    """
    Test Time Augmentation for improved inference.
    
    Averages predictions over multiple augmented versions of the input.
    """
    
    def __init__(self, n_transforms: int = 5):
        """
        Args:
            n_transforms: Number of augmented versions to generate
        """
        self.n_transforms = n_transforms
        
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA and return averaged predictions.
        
        Args:
            model: PyTorch model
            x: Input tensor
            
        Returns:
            Averaged predictions
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]  # Use main output for multi-output models
            predictions.append(torch.softmax(pred, dim=1))
            
            # Augmented predictions
            for _ in range(self.n_transforms - 1):
                # Apply simple augmentations
                aug_x = x.clone()
                
                # Random horizontal flip
                if torch.rand(1) < 0.5:
                    aug_x = torch.flip(aug_x, dims=[3])
                
                # Random rotation
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    aug_x = torch.rot90(aug_x, k=k, dims=[2, 3])
                
                # Small random translation
                shift_x = torch.randint(-2, 3, (1,)).item()
                shift_y = torch.randint(-2, 3, (1,)).item()
                aug_x = torch.roll(aug_x, shifts=(shift_x, shift_y), dims=(2, 3))
                
                pred = model(aug_x)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Use main output for multi-output models
                predictions.append(torch.softmax(pred, dim=1))
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)


class ClassSpecificAugmentation:
    """
    Apply different augmentation probabilities based on class labels.
    
    Useful for handling class imbalance by augmenting minority classes more heavily.
    """
    
    def __init__(self, class_aug_probs: dict, augmenter: RandAugment):
        """
        Args:
            class_aug_probs: Dictionary mapping class indices to augmentation probabilities
            augmenter: Augmentation function to apply
        """
        self.class_aug_probs = class_aug_probs
        self.augmenter = augmenter
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Apply class-specific augmentation.
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            Augmented images
        """
        augmented_images = images.clone()
        
        for i in range(images.size(0)):
            label = labels[i].item()
            aug_prob = self.class_aug_probs.get(label, 0.5)
            
            if torch.rand(1) < aug_prob:
                augmented_images[i:i+1] = self.augmenter(images[i:i+1])
        
        return augmented_images 