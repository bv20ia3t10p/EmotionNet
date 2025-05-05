import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from config import *

# More advanced train transforms with richer augmentation strategies
def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.7),
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 25.0))
        ], p=0.4),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=2, min_height=8, min_width=8, p=0.3),
        A.GridDistortion(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# More robust validation transforms including TTA-friendly normalization
def get_val_transform():
    return A.Compose([
        A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Test-time augmentation transforms for more robust inference
def get_tta_transforms():
    return [
        # Original center crop
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Slightly brighter
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Slightly darker
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Small rotation
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Rotate(limit=5, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ] 