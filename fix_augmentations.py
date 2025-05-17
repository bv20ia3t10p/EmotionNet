#!/usr/bin/env python3
"""Fix the augmentations.py file with proper encoding"""

import os

def fix_augmentations_file():
    # Path to the file
    filepath = "emotion_net/data/augmentations.py"
    
    print(f"Fixing {filepath}...")
    
    # Create a clean version of the file
    clean_content = '''"""Custom augmentations for facial emotion recognition."""

import numpy as np
import torch
import math
import random
import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip, ShiftScaleRotate,
    Normalize, RandomGamma, GaussianBlur, ElasticTransform, GaussNoise,
    GridDistortion, RandomResizedCrop, CoarseDropout, RGBShift, CLAHE
)
from albumentations.pytorch import ToTensorV2

class FacialPartErasing:
    """
    Randomly erase facial parts like eyes, mouth, or nose.
    This helps the model focus on other regions and improves robustness.
    """
    def __init__(self, p=0.5, area_ratio=(0.02, 0.2), eye_chance=0.5, mouth_chance=0.5, nose_chance=0.3):
        self.p = p
        self.min_area_ratio, self.max_area_ratio = area_ratio
        self.eye_chance = eye_chance
        self.mouth_chance = mouth_chance
        self.nose_chance = nose_chance
        
    def __call__(self, image):
        if random.random() > self.p:
            return image
            
        h, w = image.shape[:2]
        
        # Approximate facial landmark positions based on typical face proportions
        # These are rough estimations - in production, a landmark detector would be better
        
        # Define eye regions (left and right)
        if random.random() < self.eye_chance:
            eye_height = int(h * 0.12)
            eye_width = int(w * 0.2)
            eye_top = int(h * 0.25)  # Eyes are typically around 25% from top
            
            # Randomly erase left or right or both eyes
            if random.random() < 0.6:  # Both eyes
                left_eye_left = int(w * 0.25) - eye_width // 2
                right_eye_left = int(w * 0.75) - eye_width // 2
                
                # Fill with random noise or gray
                if random.random() < 0.5:
                    noise = np.random.randint(0, 256, (eye_height, eye_width, 3), dtype=np.uint8)
                    image[eye_top:eye_top+eye_height, left_eye_left:left_eye_left+eye_width] = noise
                    image[eye_top:eye_top+eye_height, right_eye_left:right_eye_left+eye_width] = noise
                else:
                    # Use mean color of the image
                    mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
                    image[eye_top:eye_top+eye_height, left_eye_left:left_eye_left+eye_width] = mean_color
                    image[eye_top:eye_top+eye_height, right_eye_left:right_eye_left+eye_width] = mean_color
            else:  # Single eye
                eye_left = int(w * (0.25 if random.random() < 0.5 else 0.75)) - eye_width // 2
                
                if random.random() < 0.5:
                    noise = np.random.randint(0, 256, (eye_height, eye_width, 3), dtype=np.uint8)
                    image[eye_top:eye_top+eye_height, eye_left:eye_left+eye_width] = noise
                else:
                    mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
                    image[eye_top:eye_top+eye_height, eye_left:eye_left+eye_width] = mean_color
        
        # Define mouth region
        if random.random() < self.mouth_chance:
            mouth_height = int(h * 0.15)
            mouth_width = int(w * 0.4)
            mouth_top = int(h * 0.7)  # Mouth is typically around 70% from top
            mouth_left = int(w * 0.5) - mouth_width // 2
            
            if random.random() < 0.5:
                noise = np.random.randint(0, 256, (mouth_height, mouth_width, 3), dtype=np.uint8)
                image[mouth_top:mouth_top+mouth_height, mouth_left:mouth_left+mouth_width] = noise
            else:
                mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
                image[mouth_top:mouth_top+mouth_height, mouth_left:mouth_left+mouth_width] = mean_color
                
        # Define nose region
        if random.random() < self.nose_chance:
            nose_height = int(h * 0.12)
            nose_width = int(w * 0.2)
            nose_top = int(h * 0.45)  # Nose is typically around 45% from top
            nose_left = int(w * 0.5) - nose_width // 2
            
            if random.random() < 0.5:
                noise = np.random.randint(0, 256, (nose_height, nose_width, 3), dtype=np.uint8)
                image[nose_top:nose_top+nose_height, nose_left:nose_left+nose_width] = noise
            else:
                mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
                image[nose_top:nose_top+nose_height, nose_left:nose_left+nose_width] = mean_color
                
        return image

class NoiseInjection:
    """
    Adds various types of noise to improve model robustness.
    """
    def __init__(self, p=0.5, noise_types=['gaussian', 'speckle', 'salt_pepper']):
        self.p = p
        self.noise_types = noise_types
        
    def __call__(self, image):
        if random.random() > self.p:
            return image
            
        # Select a noise type randomly
        noise_type = random.choice(self.noise_types)
        
        # Convert image to float for operations
        img_float = image.astype(np.float32) / 255.0
        
        if noise_type == 'gaussian':
            # Add Gaussian noise
            mean = 0
            var = 0.01 * random.random()
            sigma = var ** 0.5
            noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
            noisy_img = img_float + noise
            
        elif noise_type == 'speckle':
            # Add speckle noise
            noise = np.random.randn(*image.shape).astype(np.float32) * 0.15
            noisy_img = img_float + img_float * noise
            
        elif noise_type == 'salt_pepper':
            # Add salt and pepper noise
            s_vs_p = 0.5
            amount = 0.05 * random.random()
            noisy_img = img_float.copy()
            
            # Salt
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
            noisy_img[coords[0], coords[1], :] = 1
            
            # Pepper
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
            noisy_img[coords[0], coords[1], :] = 0
        
        # Clip and convert back to uint8
        noisy_img = np.clip(noisy_img, 0, 1) * 255
        return noisy_img.astype(np.uint8)

def get_training_transforms(image_size=224, augmentation_strength="standard"):
    """
    Returns training transforms with different augmentation strengths.
    
    Args:
        image_size: Target image size
        augmentation_strength: "light", "standard", or "strong"
    """
    common_transforms = [
        HorizontalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if augmentation_strength == "light":
        return Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
            ShiftScaleRotate(p=0.3, rotate_limit=10, scale_limit=0.1, border_mode=cv2.BORDER_CONSTANT),
            GaussianBlur(p=0.2, blur_limit=(3, 5)),
            GaussNoise(p=0.2, var_limit=(10.0, 40.0)),
            *common_transforms
        ])
    elif augmentation_strength == "strong":
        # Build a pipeline of custom and albumentations transforms
        custom_transforms = [
            # First apply custom transforms that work on numpy arrays
            lambda x: FacialPartErasing(p=0.3)(x),
            lambda x: NoiseInjection(p=0.4)(x)
        ]
        
        album_transforms = Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
            ShiftScaleRotate(p=0.6, rotate_limit=20, scale_limit=0.2, border_mode=cv2.BORDER_CONSTANT),
            RandomBrightnessContrast(p=0.6, brightness_limit=0.3, contrast_limit=0.3),
            RGBShift(p=0.3, r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            GaussianBlur(p=0.3, blur_limit=(3, 7)),
            GaussNoise(p=0.4, var_limit=(10.0, 50.0)),
            CoarseDropout(p=0.3, max_holes=8, max_height=8, max_width=8),
            ElasticTransform(p=0.2, alpha=50, sigma=5, alpha_affine=5),
            GridDistortion(p=0.2, distort_limit=0.1),
            CLAHE(p=0.3, clip_limit=2.0),
            *common_transforms
        ])
        
        # Create a function to apply custom transforms followed by albumentations
        def apply_transforms(image):
            for transform in custom_transforms:
                image = transform(image)
            return album_transforms(image=image)["image"]
        
        return apply_transforms
    else:  # "standard" - default
        return Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, border_mode=cv2.BORDER_CONSTANT),
            RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            GaussianBlur(p=0.3, blur_limit=(3, 5)),
            GaussNoise(p=0.3, var_limit=(10.0, 50.0)),
            CoarseDropout(p=0.2, max_holes=8, max_height=8, max_width=8),
            *common_transforms
        ])

def get_validation_transforms(image_size=224):
    """Returns validation/test transforms (normalization + resize only)."""
    return Compose([
        RandomResizedCrop(height=image_size, width=image_size, scale=(0.9, 1.0)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_test_time_augmentation(image_size=224, num_augments=5):
    """
    Returns a function to perform test-time augmentation.
    This applies several moderate augmentations to the image,
    runs prediction on each, and averages the results.
    """
    # Define TTA transforms
    transforms = [
        # 1. Original image
        Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.9, 1.0)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # 2. Horizontal flip
        Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.9, 1.0)),
            HorizontalFlip(p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # 3. Slight brightness/contrast change
        Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.9, 1.0)),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # 4. Small rotation
        Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.9, 1.0)),
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=5, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # 5. Small zoom
        Compose([
            RandomResizedCrop(height=image_size, width=image_size, scale=(0.85, 0.95)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]
    
    def tta_func(model, image):
        # Apply transforms and collect predictions
        preds = []
        for i in range(min(num_augments, len(transforms))):
            # Apply transform
            augmented = transforms[i](image=image)["image"]
            # Add batch dimension and move to device
            augmented = augmented.unsqueeze(0).to(next(model.parameters()).device)
            # Get prediction
            with torch.no_grad():
                output = model(augmented)
                # If model returns a list, get the first element (main logits)
                if isinstance(output, list):
                    output = output[0]
                # Apply softmax to get probabilities
                probs = F.softmax(output, dim=1)
                preds.append(probs)
        
        # Average predictions
        avg_preds = torch.mean(torch.stack(preds), dim=0)
        return avg_preds
    
    return tta_func

def get_transforms(image_size=224, augmentation_strength="standard", is_training=True):
    """
    Main function to get transforms based on training/validation mode and strength.
    """
    if is_training:
        return get_training_transforms(image_size, augmentation_strength)
    else:
        return get_validation_transforms(image_size)'''
    
    # Write the clean content to the file
    try:
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(clean_content)
        print(f"✅ Successfully fixed {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

if __name__ == "__main__":
    fix_augmentations_file() 