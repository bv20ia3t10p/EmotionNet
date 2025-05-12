"""Dataset and data loading utilities for emotion recognition."""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

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
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                
                # Very mild spatial transformations that preserve facial features
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    p=0.5
                ),
                
                # Color and lighting augmentations (more important for facial recognition)
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, 7),
                        p=1.0
                    ),
                ], p=0.7),
                
                # Add Cutout/RandomErasing for regularization
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

def split_data(paths, labels, val_ratio=0.1, seed=42):
    """Split data into train and validation sets."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get indices for splitting
    indices = np.arange(len(paths))
    np.random.shuffle(indices)
    
    # Calculate split point
    split = int(len(indices) * (1 - val_ratio))
    
    # Split indices
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Split data
    # Use pandas DataFrames for easier handling later
    train_df = pd.DataFrame({'path': [paths[i] for i in train_indices], 'emotion': [labels[i] for i in train_indices]})
    val_df = pd.DataFrame({'path': [paths[i] for i in val_indices], 'emotion': [labels[i] for i in val_indices]})
    
    return train_df, val_df # Return DataFrames

def load_and_split_data(data_dir, emotions, test_dir=None, image_size=224):
    """Load image paths and labels from directory structure and split into training and validation sets."""
    paths, labels = load_data(data_dir, emotions)
    
    # Split the data into training and validation sets
    train_df, val_df = split_data(paths, labels)
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Extract paths and labels from DataFrames
    train_paths = train_df['path'].tolist()
    train_labels_list = train_df['emotion'].tolist()
    val_paths = val_df['path'].tolist()
    val_labels_list = val_df['emotion'].tolist()

    # Instantiate datasets correctly
    train_dataset = AdvancedEmotionDataset(train_paths, train_labels_list, mode='train', image_size=image_size)
    val_dataset = AdvancedEmotionDataset(val_paths, val_labels_list, mode='val', image_size=image_size)
    
    # Extract train labels for sampler (already done from df)
    # train_labels = train_df['emotion'].tolist() # Keep this variable name for consistency below

    # Load test data if test_dir is provided
    if test_dir:
        # Use the existing load_data function for test data
        test_paths, test_labels_list = load_data(test_dir, EMOTIONS) 
        # Instantiate test dataset correctly
        test_dataset = AdvancedEmotionDataset(test_paths, test_labels_list, mode='test', image_size=image_size)
        print(f"Test samples: {len(test_dataset)}")
        # Return train_labels_list used for sampler (originally named train_labels)
        return train_dataset, val_dataset, test_dataset, train_labels_list 
    else:
        # Return train_labels_list used for sampler (originally named train_labels)
        return train_dataset, val_dataset, None, train_labels_list 