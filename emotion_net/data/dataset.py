"""Dataset and data loading utilities for emotion recognition."""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from emotion_net.config.constants import IMAGE_MEAN, IMAGE_STD, EMOTIONS

def load_data(data_dir, emotions):
    """Load image paths and labels from directory structure."""
    paths = []
    labels = []
    for emotion_idx, emotion_name in emotions.items():
        emotion_dir = os.path.join(data_dir, emotion_name)
        if not os.path.exists(emotion_dir):
            print(f"Warning: {emotion_dir} does not exist.")
            continue
        
        for img_file in os.listdir(emotion_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(emotion_dir, img_file)
                paths.append(img_path)
                labels.append(emotion_idx)
    
    return paths, labels

class AdvancedEmotionDataset(Dataset):
    """Advanced dataset class with strong augmentations for emotion recognition."""
    
    def __init__(self, image_paths, labels, mode='train', image_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        self.image_size = image_size
        self.classes = list(EMOTIONS.values())  # Add classes attribute
        
        # Define strong augmentations for training
        if mode == 'train':
            self.transform = A.Compose([
                # Resize with padding to maintain aspect ratio
                A.LongestMaxSize(max_size=image_size),
                A.Resize(height=image_size, width=image_size),  # Simple resize instead of PadIfNeeded
                
                # Very mild spatial transformations that preserve facial features
                A.HorizontalFlip(p=0.3),  # Reduced probability
                A.Affine(
                    scale=(0.98, 1.02),  # Reduced scale range
                    translate_percent=(-0.02, 0.02),  # Reduced translation
                    rotate=(-5, 5),  # Reduced rotation
                    p=0.3  # Reduced probability
                ),
                
                # Color and lighting augmentations (more important for facial recognition)
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,  # Reduced brightness variation
                        contrast_limit=0.1,    # Reduced contrast variation
                        p=1.0
                    ),
                    A.CLAHE(
                        clip_limit=1.5,  # Reduced clip limit
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                    A.GaussianBlur(
                        blur_limit=3,
                        p=1.0
                    ),
                ], p=0.5),
                
                # Normalize and convert to tensor
                A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
                ToTensorV2(),
            ])
        else:
            # Minimal transforms for validation and test
            self.transform = A.Compose([
                # Simple resize for validation/test
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        try:
            # Try to use OpenCV for faster loading
            img = cv2.imread(img_path)
            if img is None:
                raise Exception("Image loading failed")
                
            # Check if the image is grayscale (1 channel) and convert to RGB if needed
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback to PIL
            try:
                pil_img = Image.open(img_path)
                # Ensure RGB mode (convert from grayscale if needed)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img = np.array(pil_img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image and the label if loading fails
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        
        return img, label 