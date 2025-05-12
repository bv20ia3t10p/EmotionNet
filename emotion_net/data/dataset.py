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

        try:
            # Use OpenCV first
            img = cv2.imread(img_path)
            if img is None: raise cv2.error("cv2.imread failed") # More specific error
            # Ensure 3 channels (convert grayscale to RGB)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4: # Handle RGBA
                 img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else: # Assume BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as e_cv2:
            # Fallback to PIL if OpenCV fails
            try:
                pil_img = Image.open(img_path).convert('RGB')
                img = np.array(pil_img)
            except Exception as e_pil:
                print(f"Error loading image {img_path} with both OpenCV and PIL: {e_cv2}, {e_pil}")
                # Return a placeholder tensor
                img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                # Ensure label is returned even if image loading fails
                return img, torch.tensor(label, dtype=torch.long)

        # Apply transformations
        if self.transform:
            try:
                transformed = self.transform(image=img)
                img = transformed["image"]
            except Exception as e_transform:
                 print(f"Error applying transform to image {img_path}: {e_transform}")
                 # Return a placeholder tensor
                 img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                 return img, torch.tensor(label, dtype=torch.long)

        return img, torch.tensor(label, dtype=torch.long) # Ensure label is tensor