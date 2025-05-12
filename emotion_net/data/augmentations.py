"""Augmentation definitions for emotion datasets."""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from emotion_net.config.constants import IMAGE_MEAN, IMAGE_STD

def get_transforms(mode='train', image_size=224, dataset_name=None):
    """Get augmentations for different modes.
    
    Args:
        mode: 'train', 'val', or 'test'
        image_size: target image size
        dataset_name: optional dataset name for dataset-specific augmentations
    """
    if mode == 'train':
        # Enhanced augmentation pipeline for RAF-DB and general training
        aug_pipeline = [
            A.Resize(height=image_size, width=image_size),  # Use Resize instead of LongestMaxSize+PadIfNeeded
            A.HorizontalFlip(p=0.5),
            # Improved geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=1.0),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-20, 20),
                    shear=(-10, 10),
                    p=1.0
                ),
            ], p=0.7),
            # Enhanced color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.7),
            # Noise and blur for robustness
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                # Fix parameters for GaussNoise
                A.GaussNoise(p=1.0),
            ], p=0.5),
            # Using CoarseDropout instead of Cutout
            A.CoarseDropout(
                max_holes=8, 
                max_height=int(image_size * 0.1), 
                max_width=int(image_size * 0.1),
                min_holes=1,
                min_height=int(image_size * 0.05),
                min_width=int(image_size * 0.05),
                fill_value=0, 
                p=0.5
            ),
            # Grid distortion and elastic transform occasionally
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=25, p=1.0),  # Fixed alpha_affine parameter
            ], p=0.3),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]
        
        # Apply dataset-specific augmentations
        if dataset_name == 'rafdb':
            # For RAF-DB, we'll use slightly less aggressive augmentations
            # since the model is struggling to learn the basic patterns
            return A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
                ToTensorV2(),
            ])
        else:
            # For other datasets (FER2013, etc.)
            return A.Compose(aug_pipeline)
            
    else: # 'val' or 'test'
        # Keep transforms simple for validation/testing
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]) 