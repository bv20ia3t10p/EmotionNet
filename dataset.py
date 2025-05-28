import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from typing import Optional, Tuple, List
from torchvision.transforms import functional as F
import random

# Emotion labels mapping for FER2013
EMOTION_MAP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad", 
    5: "Surprise",
    6: "Neutral"
}

# Advanced augmentation imports
try:
    from torchvision.transforms import RandAugment, AugMix
    HAS_ADVANCED_AUGMENT = True
except ImportError:
    HAS_ADVANCED_AUGMENT = False

class MixUp:
    """MixUp augmentation for better generalization"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels
        
        # Generate mix ratio
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return mixed images and both sets of labels with lambda
        return mixed_images, labels, labels[index], lam

class CutMix:
    """CutMix augmentation for better localization"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels
        
        # Generate mix ratio
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Get image dimensions
        _, _, H, W = images.size()
        
        # Generate random box
        cut_rat = np.sqrt(1. - lam)
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

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', use_extra_augmentation=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val' or 'test'
            use_extra_augmentation (bool): Whether to use extra augmentation
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.use_extra_augmentation = use_extra_augmentation
        
        # Split the data for training, validation and testing
        if mode == 'train':
            self.data = self.data[self.data[' Usage'] == 'Training']
        elif mode == 'val':
            self.data = self.data[self.data[' Usage'] == 'PublicTest']
        elif mode == 'test':
            self.data = self.data[self.data[' Usage'] == 'PrivateTest']
            
        self.pixels = self.data[' pixels'].tolist()
        self.emotions = self.data['emotion'].tolist()
        
        # Print class distribution
        self.print_class_distribution()
        
    def __len__(self):
        return len(self.emotions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Convert pixels string to image
        img_array = np.array([int(p) for p in self.pixels[idx].split()]).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        emotion = self.emotions[idx]
        
        # Convert PIL Image to numpy array for albumentations
        img_array = np.array(image)
        
        # Keep as single channel (grayscale)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
            
        # Apply transforms using albumentations
        if self.transform:
            augmented = self.transform(image=img_array)
            image = augmented['image']
        else:
            image = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            
        return image, emotion
    
    def print_class_distribution(self):
        """Print the distribution of classes in the dataset"""
        class_counts = pd.Series(self.emotions).value_counts().sort_index()
        print(f"\nClass distribution for {self.mode} set:")
        for emotion, count in class_counts.items():
            print(f"{EMOTION_MAP[emotion]}: {count}")

def get_data_loaders(train_csv, val_csv, test_csv, img_dir, batch_size=32, img_size=224, 
                     num_workers=4, use_weighted_sampler=True):
    """
    Create data loaders with optional weighted sampling for handling class imbalance
    """
    # Create datasets
    train_dataset = FER2013Dataset(
        csv_file=train_csv,
        root_dir=img_dir,
        transform=get_transforms(img_size, is_train=True),
        mode='train',
        use_extra_augmentation=True
    )
    val_dataset = FER2013Dataset(
        csv_file=val_csv,
        root_dir=img_dir,
        transform=get_transforms(img_size, is_train=False),
        mode='val'
    )
    test_dataset = FER2013Dataset(
        csv_file=test_csv,
        root_dir=img_dir,
        transform=get_transforms(img_size, is_train=False),
        mode='test'
    )
    # Calculate class weights for weighted sampling using only the training split
    class_counts = pd.Series(train_dataset.emotions).value_counts()
    total_samples = len(train_dataset)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in class_counts.items()}
    # Create weighted sampler if requested
    if use_weighted_sampler:
        weights = [class_weights[label] for label in train_dataset.emotions]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

def get_transforms(img_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 