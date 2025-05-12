"""Base Dataset class."""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Import from refactored modules
from .augmentations import get_transforms

# --- Base Dataset ---

class BaseEmotionDataset(Dataset):
    """Base class for emotion datasets, handles image loading and transforms."""
    def __init__(self, image_paths, labels, classes, mode='train', image_size=224, dataset_name=None):
        if len(image_paths) != len(labels):
             raise ValueError(f"Mismatch between number of image paths ({len(image_paths)}) and labels ({len(labels)}).")
        self.image_paths = image_paths
        self.labels = labels
        self.classes = classes # List of class names
        self.mode = mode
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.transform = get_transforms(mode, image_size, dataset_name)
        print(f"  BaseEmotionDataset created for mode '{mode}' with {len(self)} samples.")
        
        # Print class distribution for better debugging
        if len(self) > 0:
            self._print_class_distribution()

    def __len__(self):
        return len(self.image_paths)
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset for debugging."""
        if not self.labels or not self.classes:
            return
            
        # Count instances of each class
        class_counts = {}
        for label in self.labels:
            if label >= 0 and label < len(self.classes):
                class_name = self.classes[label]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
        
        # Print the distribution
        print(f"  Class distribution in {self.mode} dataset:")
        total = len(self.labels)
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"    {class_name}: {count} samples ({percentage:.1f}%)")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Check if path exists
        if not os.path.exists(img_path):
            print(f"Error: Image path does not exist: {img_path}")
            # Create a placeholder tensor
            img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return img, torch.tensor(label, dtype=torch.long)

        try:
            # Try PIL first - most robust and reliable
            pil_img = Image.open(img_path).convert('RGB')
            img = np.array(pil_img)
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]

            return img, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder tensor
            img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return img, torch.tensor(label, dtype=torch.long)