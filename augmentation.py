"""
Data Augmentation Modules for EmotionNet
Contains various augmentation strategies for preventing overfitting and handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as F_t
import numpy as np


class GridMask(nn.Module):
    """Enhanced GridMask augmentation for stronger regularization"""
    def __init__(self, d_range=(24, 72), r=0.6, p=0.4):
        super().__init__()
        self.d_range = d_range
        self.r = r
        self.p = p
        
    def forward(self, x):
        if torch.rand(1) > self.p:
            return x
            
        b, c, h, w = x.shape
        d = torch.randint(self.d_range[0], self.d_range[1], (1,)).item()
        l = int(d * self.r)
        
        # Create grid mask with random rotation
        mask = torch.ones_like(x)
        angle = torch.randint(-30, 30, (1,)).item()  # Random rotation angle
        
        for i in range(0, h, d):
            for j in range(0, w, d):
                if torch.rand(1) < 0.5:
                    # Apply mask with random offset
                    offset_i = torch.randint(-2, 3, (1,)).item()
                    offset_j = torch.randint(-2, 3, (1,)).item()
                    i_pos = min(max(i + offset_i, 0), h - l)
                    j_pos = min(max(j + offset_j, 0), w - l)
                    mask[:, :, i_pos:i_pos+l, j_pos:j_pos+l] = 0
        
        # Apply random rotation to mask
        if angle != 0:
            mask = F_t.rotate(mask, angle)
        
        return x * mask


class ReducedAugmentation48(nn.Module):
    """Reduced augmentation for 48x48 images with minimal transformations"""
    def __init__(self, img_size=48, p=0.4):
        super().__init__()
        self.img_size = img_size
        self.p = p
        
        # Minimal geometric augmentations
        self.minimal_geometric = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
            transforms.RandomRotation(10),            # Reduced rotation
            transforms.RandomAffine(
                degrees=5,                            # Minimal rotation
                translate=(0.05, 0.05),              # Minimal translation
                scale=(0.95, 1.05),                  # Minimal scaling
                shear=5                              # Minimal shear
            ),
        ])
        
        # Minimal color augmentations
        self.minimal_color = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1,                      # Reduced brightness
                contrast=0.1,                       # Reduced contrast
                saturation=0.05,                    # Minimal saturation
                hue=0.02                            # Minimal hue
            ),
        ])
        
        # Very light noise
        self.light_noise = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))  # Reduced blur
            ], p=0.1),  # Very low probability
        ])
        
        # Minimal erasing
        self.minimal_erasing = transforms.RandomErasing(
            p=0.1,                                   # Very low probability
            scale=(0.01, 0.05),                     # Smaller scale
            ratio=(0.5, 2.0), 
            value='random'
        )
        
        # Reduced class-specific augmentation probabilities (UPDATED for Angry/Neutral fix)
        self.class_aug_probs = {
            0: 0.8,  # Angry - DRAMATICALLY INCREASED (was 0.5) - complete failure needs aggressive aug
            1: 0.1,  # Disgust - REDUCED (was 0.2) - performing too well, confusing with angry
            2: 0.5,  # Fear - REDUCED (was 0.6) - moderate performance
            3: 0.3,  # Happy - REDUCED (was 0.4) - good performance
            4: 0.2,  # Sad - DRAMATICALLY REDUCED (was 0.6) - too dominant, confusing other classes
            5: 0.3,  # Surprise - REDUCED (was 0.4) - good performance
            6: 0.9   # Neutral - DRAMATICALLY INCREASED (was 0.3) - complete failure needs aggressive aug
        }
        
    def apply_geometric_aug(self, x):
        """Apply minimal geometric augmentations"""
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        augmented = []
        for i in range(batch_size):
            img = x[i].cpu()
            img_pil = transforms.ToPILImage()(img)
            img_pil = self.minimal_geometric(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)
            augmented.append(img_tensor.to(device, dtype=dtype))
        
        return torch.stack(augmented)
    
    def apply_color_aug(self, x):
        """Apply minimal color augmentations"""
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        augmented = []
        for i in range(batch_size):
            img = x[i].cpu()
            img_pil = transforms.ToPILImage()(img)
            img_pil = self.minimal_color(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)
            augmented.append(img_tensor.to(device, dtype=dtype))
        
        return torch.stack(augmented)
    
    def forward(self, x, labels=None):
        if labels is None:
            # Apply minimal augmentation if no labels provided
            if torch.rand(1) < self.p:
                # Apply geometric augmentations (30% chance)
                if torch.rand(1) < 0.3:
                    x = self.apply_geometric_aug(x)
                
                # Apply color augmentations (20% chance)
                if torch.rand(1) < 0.2:
                    x = self.apply_color_aug(x)
                
                # Apply light noise (10% chance)
                if torch.rand(1) < 0.1:
                    x = self.light_noise(x)
                
                # Apply minimal erasing (10% chance)
                if torch.rand(1) < 0.1:
                    x = self.minimal_erasing(x)
            
            return torch.clamp(x, 0, 1)
        
        # Class-specific minimal augmentation
        batch_size = x.size(0)
        augmented_batch = []
        
        for i in range(batch_size):
            img = x[i:i+1]  # Keep batch dimension
            label = labels[i].item()
            aug_prob = self.class_aug_probs.get(label, 0.3)
            
            if torch.rand(1) < aug_prob:
                # Apply geometric augmentations (reduced probability)
                if torch.rand(1) < 0.4:
                    img = self.apply_geometric_aug(img)
                
                # Apply color augmentations (reduced probability)
                if torch.rand(1) < 0.3:
                    img = self.apply_color_aug(img)
                
                # Apply light noise (very low probability)
                if torch.rand(1) < 0.1:
                    img = self.light_noise(img)
                
                # Apply minimal erasing (very low probability)
                if torch.rand(1) < 0.1:
                    img = self.minimal_erasing(img)
            
            augmented_batch.append(img.squeeze(0))
        
        result = torch.stack(augmented_batch)
        return torch.clamp(result, 0, 1)


class EnhancedAntiOverfittingAugmentation(nn.Module):
    """Enhanced augmentation specifically designed to prevent overfitting and handle class imbalance"""
    def __init__(self, img_size=64, p=0.8):
        super().__init__()
        self.img_size = img_size
        self.p = p
        
        # Strong geometric augmentations to prevent overfitting
        self.strong_geometric = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomRotation(25),
            transforms.RandomAffine(
                degrees=20,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=15
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        ])
        
        # Enhanced color augmentations for generalization
        self.enhanced_color = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAutocontrast(p=0.4),
            transforms.RandomEqualize(p=0.4),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])
        
        # Noise and blur for robustness
        self.noise_blur = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.3),
        ])
        
        # Multiple erasing strategies
        self.erasing_strategies = [
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.0), value='random'),
            transforms.RandomErasing(p=0.2, scale=(0.05, 0.15), ratio=(0.5, 2.0), value=0),
            transforms.RandomErasing(p=0.2, scale=(0.03, 0.08), ratio=(0.8, 1.2), value=1),
        ]
        
        # Grid mask for structured occlusion
        self.grid_mask = GridMask(d_range=(8, 32), r=0.6, p=0.3)
        
        # Class-specific augmentation probabilities for class imbalance (UPDATED for Angry/Neutral fix)
        self.class_aug_probs = {
            0: 0.95, # Angry - DRAMATICALLY INCREASED (was 0.9) - complete failure
            1: 0.2,  # Disgust - DRAMATICALLY REDUCED (was 0.4) - performing too well, confusing with angry
            2: 0.8,  # Fear - REDUCED (was 0.95) - moderate performance
            3: 0.7,  # Happy - REDUCED (was 0.85) - good performance
            4: 0.3,  # Sad - DRAMATICALLY REDUCED (was 0.95) - too dominant, confusing other classes
            5: 0.7,  # Surprise - REDUCED (was 0.85) - good performance
            6: 0.95  # Neutral - DRAMATICALLY INCREASED (was 0.8) - complete failure
        }
        
    def apply_geometric_aug(self, x):
        """Apply geometric augmentations with proper device handling"""
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        augmented = []
        for i in range(batch_size):
            img = x[i].cpu()
            img_pil = transforms.ToPILImage()(img)
            img_pil = self.strong_geometric(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)
            augmented.append(img_tensor.to(device, dtype=dtype))
        
        return torch.stack(augmented)
    
    def apply_color_aug(self, x):
        """Apply color augmentations with proper device handling"""
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        augmented = []
        for i in range(batch_size):
            img = x[i].cpu()
            img_pil = transforms.ToPILImage()(img)
            img_pil = self.enhanced_color(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)
            augmented.append(img_tensor.to(device, dtype=dtype))
        
        return torch.stack(augmented)
    
    def apply_erasing_aug(self, x):
        """Apply random erasing augmentation"""
        erasing_transform = np.random.choice(self.erasing_strategies)
        return erasing_transform(x)
    
    def forward(self, x, labels=None):
        if labels is None:
            # Apply standard augmentation if no labels provided
            if torch.rand(1) < self.p:
                # Apply geometric augmentations (70% chance)
                if torch.rand(1) < 0.7:
                    x = self.apply_geometric_aug(x)
                
                # Apply color augmentations (60% chance)
                if torch.rand(1) < 0.6:
                    x = self.apply_color_aug(x)
                
                # Apply noise and blur (40% chance)
                if torch.rand(1) < 0.4:
                    x = self.noise_blur(x)
                
                # Apply erasing (50% chance)
                if torch.rand(1) < 0.5:
                    x = self.apply_erasing_aug(x)
                
                # Apply grid mask (30% chance)
                if torch.rand(1) < 0.3:
                    x = self.grid_mask(x)
            
            return torch.clamp(x, 0, 1)
        
        # Class-specific augmentation for handling class imbalance
        batch_size = x.size(0)
        augmented_batch = []
        
        for i in range(batch_size):
            img = x[i:i+1]  # Keep batch dimension
            label = labels[i].item()
            aug_prob = self.class_aug_probs.get(label, 0.7)
            
            if torch.rand(1) < aug_prob:
                # Apply geometric augmentations (reduced probability)
                if torch.rand(1) < 0.4:
                    img = self.apply_geometric_aug(img)
                
                # Apply color augmentations (reduced probability)
                if torch.rand(1) < 0.3:
                    img = self.apply_color_aug(img)
                
                # Apply noise and blur
                if torch.rand(1) < 0.5:
                    img = self.noise_blur(img)
                
                # Apply erasing (higher probability for underrepresented classes)
                erasing_prob = 0.7 if label in [2, 4] else 0.4  # Fear and Sad get more erasing
                if torch.rand(1) < erasing_prob:
                    img = self.apply_erasing_aug(img)
                
                # Apply grid mask
                if torch.rand(1) < 0.4:
                    img = self.grid_mask(img)
            
            augmented_batch.append(img.squeeze(0))
        
        result = torch.stack(augmented_batch)
        return torch.clamp(result, 0, 1)


class ConservativeClassSpecificAugmentation(nn.Module):
    """Conservative class-specific augmentation optimized for custom ResEmoteNet with 64x64 images"""
    def __init__(self, img_size=64):
        super().__init__()
        self.img_size = img_size
        
        # Define class-specific augmentation probabilities (UPDATED for Angry/Neutral fix)
        self.class_aug_probs = {
            'fear': 0.5,      # REDUCED (was 0.6) - moderate performance
            'sad': 0.2,       # DRAMATICALLY REDUCED (was 0.6) - too dominant, confusing other classes
            'angry': 0.8,     # DRAMATICALLY INCREASED (was 0.5) - complete failure needs aggressive aug
            'neutral': 0.9,   # DRAMATICALLY INCREASED (was 0.5) - complete failure needs aggressive aug
            'disgust': 0.2,   # REDUCED (was 0.4) - performing too well, confusing with angry
            'happy': 0.3,     # REDUCED (was 0.4) - good performance
            'surprise': 0.3   # REDUCED (was 0.4) - good performance
        }
        
        # Emotion label mapping
        self.emotion_map = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        
        # Conservative geometric augmentations
        self.conservative_geometric = transforms.Compose([
            transforms.RandomRotation(15, fill=0),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                fill=0
            )
        ])
        
        # Conservative color augmentations
        self.conservative_color = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            )
        ])
        
        # Mild noise and blur
        self.mild_effects = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])
    
    def apply_augmentation(self, img, aug_type):
        """Apply specific augmentation type with proper device handling"""
        device = img.device
        dtype = img.dtype
        
        if aug_type == 'minimal':
            img_cpu = img.cpu()
            img_pil = transforms.ToPILImage()(img_cpu)
            img_pil = transforms.RandomHorizontalFlip(p=0.3)(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)
            return img_tensor.to(device, dtype=dtype)
            
        elif aug_type == 'geometric':
            img_cpu = img.cpu()
            img_pil = transforms.ToPILImage()(img_cpu)
            img_aug = self.conservative_geometric(img_pil)
            img_tensor = transforms.ToTensor()(img_aug)
            return img_tensor.to(device, dtype=dtype)
            
        elif aug_type == 'color':
            img_cpu = img.cpu()
            img_pil = transforms.ToPILImage()(img_cpu)
            img_aug = self.conservative_color(img_pil)
            img_tensor = transforms.ToTensor()(img_aug)
            return img_tensor.to(device, dtype=dtype)
            
        elif aug_type == 'effects':
            img_cpu = img.cpu()
            img_pil = transforms.ToPILImage()(img_cpu)
            img_aug = self.mild_effects(img_pil)
            img_tensor = transforms.ToTensor()(img_aug)
            return img_tensor.to(device, dtype=dtype)
            
        elif aug_type == 'combined':
            img_cpu = img.cpu()
            img_pil = transforms.ToPILImage()(img_cpu)
            # Apply multiple augmentations
            img_aug = self.conservative_geometric(img_pil)
            img_aug = self.conservative_color(img_aug)
            img_aug = self.mild_effects(img_aug)
            img_tensor = transforms.ToTensor()(img_aug)
            return img_tensor.to(device, dtype=dtype)
        else:
            return img
    
    def forward(self, x, labels=None):
        if labels is None or not self.training:
            return x
        
        batch_size = x.size(0)
        augmented_batch = []
        
        for i in range(batch_size):
            img = x[i]
            label = labels[i].item()
            emotion = self.emotion_map.get(label, 'neutral')
            aug_prob = self.class_aug_probs.get(emotion, 0.3)
            
            if torch.rand(1).item() < aug_prob:
                # Choose augmentation type based on emotion
                if emotion in ['fear', 'sad']:  # Enhanced classes
                    aug_types = ['combined', 'geometric', 'color']
                    aug_type = torch.randint(0, len(aug_types), (1,)).item()
                    aug_type = aug_types[aug_type]
                elif emotion in ['angry', 'neutral']:  # Moderate classes
                    aug_types = ['geometric', 'color', 'effects']
                    aug_type = torch.randint(0, len(aug_types), (1,)).item()
                    aug_type = aug_types[aug_type]
                else:  # Standard classes
                    aug_types = ['minimal', 'geometric', 'color']
                    aug_type = torch.randint(0, len(aug_types), (1,)).item()
                    aug_type = aug_types[aug_type]
                
                img = self.apply_augmentation(img, aug_type)
            
            augmented_batch.append(img)
        
        return torch.stack(augmented_batch)


class ExtremeAugmentation64(nn.Module):
    """Extreme augmentation for 64x64 images with less severe transforms"""
    def __init__(self, config=None):
        super(ExtremeAugmentation64, self).__init__()
        
        # Use the updated less severe configuration
        from constants import EXTREME_AUGMENTATION_CONFIG
        self.config = config or EXTREME_AUGMENTATION_CONFIG
        
        # Less severe geometric transforms
        self.geometric_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=self.config['geometric_transforms']['rotation_range']),  # ±10°
            transforms.RandomAffine(
                degrees=0,
                translate=(self.config['geometric_transforms']['translation_range'], 
                          self.config['geometric_transforms']['translation_range']),  # ±8%
                scale=self.config['geometric_transforms']['scale_range'],  # (0.92, 1.08)
                shear=self.config['geometric_transforms']['shear_range']   # ±5°
            ),
        ])
        
        # Less severe color transforms
        self.color_transforms = transforms.Compose([
            transforms.ColorJitter(
                brightness=self.config['color_transforms']['brightness_range'],  # ±15%
                contrast=self.config['color_transforms']['contrast_range'],      # ±15%
                saturation=self.config['color_transforms']['saturation_range'], # ±10%
                hue=self.config['color_transforms']['hue_range']                 # ±3%
            ),
        ])
        
        # Reduced noise and effects
        self.noise_std = self.config['noise_and_effects']['gaussian_noise_std']  # 0.02
        self.salt_pepper_prob = self.config['noise_and_effects']['salt_pepper_prob']  # 0.005
        self.motion_blur_prob = self.config['noise_and_effects']['motion_blur_prob']  # 0.08
        self.gaussian_blur_prob = self.config['noise_and_effects']['gaussian_blur_prob']  # 0.08
        
        # Reduced erasing and masking
        self.random_erasing = transforms.RandomErasing(
            p=self.config['erasing_and_masking']['random_erasing_prob'],  # 0.10
            scale=(0.02, 0.15),  # Smaller erasing areas
            ratio=(0.3, 3.3),
            value=0
        )
        
        # Class-specific probabilities (UPDATED for Angry/Neutral fix)
        self.class_probs = {
            'angry': 0.9,    # DRAMATICALLY INCREASED - complete failure needs aggressive aug
            'disgust': 0.2,  # DRAMATICALLY REDUCED - performing too well, confusing with angry
            'fear': 0.6,     # REDUCED - moderate performance
            'happy': 0.4,    # REDUCED - good performance
            'neutral': 0.95, # DRAMATICALLY INCREASED - complete failure needs aggressive aug
            'sad': 0.25,     # DRAMATICALLY REDUCED - too dominant, confusing other classes
            'surprise': 0.4  # REDUCED - good performance
        }
        
    def forward(self, x, emotion_label=None):
        """Apply less severe augmentation based on emotion class"""
        if emotion_label is None:
            # Apply augmentation to entire batch with default probability
            if torch.rand(1).item() < self.config['overall_probability']:
                return self._apply_augmentation_to_batch(x)
            return x
        
        # Handle batch processing
        batch_size = x.size(0)
        augmented_batch = []
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        for i in range(batch_size):
            img = x[i:i+1]  # Keep batch dimension
            label = emotion_label[i].item() if emotion_label[i].dim() == 0 else emotion_label[i].item()
            
            # Determine augmentation probability based on emotion
            if label < len(emotion_names):
                emotion_name = emotion_names[label]
                aug_prob = self.class_probs.get(emotion_name, self.config['overall_probability'])
            else:
                aug_prob = self.config['overall_probability']
            
            # Apply augmentation with emotion-specific probability
            if torch.rand(1).item() < aug_prob:
                img = self._apply_augmentation_to_single(img)
            
            augmented_batch.append(img.squeeze(0))  # Remove batch dimension
        
        return torch.stack(augmented_batch)
    
    def _apply_augmentation_to_batch(self, x):
        """Apply augmentation to entire batch"""
        # Apply geometric transforms with reduced probability
        if torch.rand(1).item() < self.config['geometric_transforms']['probability']:  # 0.40
            x = self.geometric_transforms(x)
        
        # Apply color transforms with reduced probability
        if torch.rand(1).item() < self.config['color_transforms']['probability']:  # 0.35
            x = self.color_transforms(x)
        
        # Apply noise and effects with reduced probability
        if torch.rand(1).item() < self.config['noise_and_effects']['probability']:  # 0.25
            # Reduced gaussian noise
            if torch.rand(1).item() < 0.3:  # Reduced from 0.5
                noise = torch.randn_like(x) * self.noise_std
                x = torch.clamp(x + noise, 0, 1)
            
            # Reduced motion blur
            if torch.rand(1).item() < self.motion_blur_prob:
                x = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))(x)
            
            # Reduced gaussian blur
            if torch.rand(1).item() < self.gaussian_blur_prob:
                x = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))(x)
        
        # Apply erasing with reduced probability
        if torch.rand(1).item() < self.config['erasing_and_masking']['probability']:  # 0.20
            x = self.random_erasing(x)
        
        return x
    
    def _apply_augmentation_to_single(self, x):
        """Apply augmentation to a single image (with batch dimension)"""
        # Apply geometric transforms with reduced probability
        if torch.rand(1).item() < self.config['geometric_transforms']['probability']:  # 0.40
            x = self.geometric_transforms(x)
        
        # Apply color transforms with reduced probability
        if torch.rand(1).item() < self.config['color_transforms']['probability']:  # 0.35
            x = self.color_transforms(x)
        
        # Apply noise and effects with reduced probability
        if torch.rand(1).item() < self.config['noise_and_effects']['probability']:  # 0.25
            # Reduced gaussian noise
            if torch.rand(1).item() < 0.3:  # Reduced from 0.5
                noise = torch.randn_like(x) * self.noise_std
                x = torch.clamp(x + noise, 0, 1)
            
            # Reduced motion blur
            if torch.rand(1).item() < self.motion_blur_prob:
                x = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))(x)
            
            # Reduced gaussian blur
            if torch.rand(1).item() < self.gaussian_blur_prob:
                x = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))(x)
        
        # Apply erasing with reduced probability
        if torch.rand(1).item() < self.config['erasing_and_masking']['probability']:  # 0.20
            x = self.random_erasing(x)
        
        return x 