"""
FER2013 Dataset class for EmotionNet.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from ..utils.constants import EMOTION_MAP


class FER2013Dataset(Dataset):
    """FER2013 dataset for emotion recognition."""
    
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val' or 'test'
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Filter data based on mode
        self._filter_data_by_mode()
        
        # Extract pixels and emotions
        self.pixels = self.data['pixels'].tolist()
        self.emotions = self.data['emotion'].tolist()
        
        # Print class distribution
        self._print_class_distribution()
    
    def _filter_data_by_mode(self):
        """Filter data based on train/val/test mode."""
        mode_mapping = {
            'train': 'Training',
            'val': 'PublicTest', 
            'test': 'PrivateTest'
        }
        
        if self.mode in mode_mapping:
            self.data = self.data[self.data['Usage'] == mode_mapping[self.mode]]
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset."""
        class_counts = pd.Series(self.emotions).value_counts().sort_index()
        print(f"\nClass distribution for {self.mode} set:")
        for emotion, count in class_counts.items():
            emotion_name = EMOTION_MAP.get(emotion, f"Unknown({emotion})")
            print(f"{emotion_name}: {count}")
    
    def __len__(self):
        return len(self.emotions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Convert pixels string to image
        img_array = np.array([int(p) for p in self.pixels[idx].split()]).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(img_array)
        emotion = self.emotions[idx]
        
        # Convert PIL Image to numpy array for transforms
        img_array = np.array(image)
        
        # Ensure single channel format
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
            
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img_array)
            image = augmented['image']
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            
        return image, emotion 