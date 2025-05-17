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
        self.transforms = get_transforms(mode, image_size, dataset_name)
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
        """Get item at index idx.
        
        Args:
            idx: Index of item to get
            
        Returns:
            tuple: (image, label) where label is index of the label class
        """
        try:
            label = self.labels[idx]
            
            # Get image path
            if hasattr(self, 'image_paths'):
                img_path = self.image_paths[idx]
                
                # Verify path exists
                if not os.path.exists(img_path):
                    print(f"Error: Image path does not exist: {img_path}")
                    # Use fallback image or raise error
                    if hasattr(self, 'fallback_image'):
                        return self.fallback_image, label
                    else:
                        # Try using a different image from the same class if available
                        same_class_indices = [i for i, l in enumerate(self.labels) if l == label and i != idx]
                        if same_class_indices:
                            fallback_idx = same_class_indices[0]
                            print(f"Using fallback image from same class at index {fallback_idx}")
                            return self.__getitem__(fallback_idx)
                        else:
                            # Create a dummy grayscale image
                            print(f"Creating dummy {self.image_size}x{self.image_size} image for class {label}")
                            dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                            if self.transforms:
                                dummy_img = self.transforms(Image.fromarray(dummy_img))
                            return dummy_img, label
                
                # Try loading with OpenCV first (faster)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"cv2.imread returned None for {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                except Exception as cv_error:
                    # Fallback to PIL
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = np.array(img)
                    except Exception as pil_error:
                        error_msg = f"Error loading image {img_path} with both OpenCV and PIL: cv2.imread failed, {pil_error}"
                        print(error_msg)
                        # Use fallback strategy
                        same_class_indices = [i for i, l in enumerate(self.labels) if l == label and i != idx]
                        if same_class_indices:
                            fallback_idx = same_class_indices[0]
                            return self.__getitem__(fallback_idx)
                        else:
                            # Create a dummy grayscale image
                            dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                            if self.transforms:
                                dummy_img = self.transforms(Image.fromarray(dummy_img))
                            return dummy_img, label
                        
            # If no image_paths, use images from attribute
            elif hasattr(self, 'images'):
                img = self.images[idx]
                
            # Should never happen if class is correctly implemented
            else:
                raise ValueError("Dataset has neither image_paths nor images attribute")
            
            # Apply transformations
            if self.transforms:
                # Convert to PIL Image if numpy array
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                
                # Apply transforms
                img = self.transforms(img)
                
            return img, label
            
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {str(e)}")
            # Create a placeholder image and return it with the label
            dummy_img = torch.zeros(3, self.image_size, self.image_size)
            return dummy_img, label