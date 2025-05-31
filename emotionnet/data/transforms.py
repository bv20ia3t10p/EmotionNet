"""
Data transforms for FER2013 dataset.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms


def get_transforms(img_size=224, is_train=False):
    """Get data transforms for training or validation."""
    
    if is_train:
        # Enhanced training transforms with albumentations
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.Equalize(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, 
                        scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(p=0.3),
            ], p=0.4),
            A.OneOf([
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.02, 0.1), hole_width_range=(0.02, 0.1), p=0.3),
                A.GridDropout(ratio=0.1, holes_number_xy=(3, 3), p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485], std=[0.229]),  # Single channel normalization for grayscale
            ToTensorV2(),
        ])
    else:
        # Validation/test transforms - minimal processing
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485], std=[0.229]),  # Single channel normalization for grayscale
            ToTensorV2(),
        ])
    
    return transform


def get_basic_transforms(img_size=224):
    """Get basic PyTorch transforms as fallback."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Single channel normalization
    ]) 