import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import albumentations as A

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

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', use_extra_augmentation=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val' or 'test'
            use_extra_augmentation (bool): Whether to use extra augmentation for under-represented classes
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
        
        # Track original indices to retrieve the correct pixel data
        self.indices = list(range(len(self.emotions)))
        
        # Create extra augmentations for the Disgust class which is severely underrepresented
        if mode == 'train' and use_extra_augmentation:
            self._oversample_minority_classes()
            
        # Print class distribution
        self.print_class_distribution()
        
        # Setup extra augmentations for rare classes
        self.extra_augment = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2),
                A.GridDistortion(distort_limit=0.1),
                A.ElasticTransform(alpha=0.3),
            ], p=0.2),
        ])
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the original index (important for oversampled data)
        original_idx = self.indices[idx]
        
        # Convert pixels string to image
        img_array = np.array([int(p) for p in self.pixels[original_idx].split()]).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        emotion = self.emotions[original_idx]
        
        # Apply extra augmentation for underrepresented classes in training mode
        if self.mode == 'train' and self.use_extra_augmentation:
            # Apply more augmentation for Disgust class (emotion == 1)
            if emotion == 1:  # Disgust
                # Convert PIL Image to numpy array for albumentations
                img_array = np.array(image)
                # Apply additional augmentations
                augmented = self.extra_augment(image=img_array)
                # Convert back to PIL Image
                image = Image.fromarray(augmented['image'])
        
        # Convert grayscale to RGB (3 channels) before applying transforms
        if image.mode == 'L':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, emotion
    
    def _oversample_minority_classes(self):
        """
        Oversample minority classes to balance the dataset
        """
        # Count instances of each class
        class_counts = pd.Series(self.emotions).value_counts().sort_index()
        max_count = class_counts.max()
        
        # Store the original length before oversampling
        original_length = len(self.emotions)
        
        # Store additional samples here
        additional_indices = []
        additional_emotions = []
        additional_pixels = []
        
        # For each class, especially focusing on Disgust
        for emotion_class, count in class_counts.items():
            # Don't oversample if it's already the majority class
            if count == max_count:
                continue
                
            # Calculate how many samples to add
            if emotion_class == 1:  # Disgust class gets special treatment
                # More moderate oversampling for Disgust class
                target = min(int(max_count * 0.5), count * 4)  # At most 4x original count or 50% of max
                samples_to_add = target - count
            else:
                # For other minority classes, add a more moderate number of examples
                samples_to_add = int((max_count - count) * 0.3)  # Add 30% of the difference
                
            # Get indices of this class
            class_indices = [i for i, e in enumerate(self.emotions) if e == emotion_class]
            
            # Randomly sample from these indices with replacement
            if samples_to_add > 0:
                sampled_indices = np.random.choice(class_indices, size=samples_to_add, replace=True)
                
                for idx in sampled_indices:
                    additional_indices.append(idx)
                    additional_emotions.append(self.emotions[idx])
                    additional_pixels.append(self.pixels[idx])
        
        # Extend the original data with the oversampled data
        self.indices.extend(additional_indices)
        self.emotions.extend(additional_emotions)
        self.pixels.extend(additional_pixels)
        
        print(f"\nClass balancing: Added {len(additional_indices)} additional samples")
        print(f"Original dataset size: {original_length}, New size: {len(self.emotions)}")
    
    def print_class_distribution(self):
        """
        Print the class distribution for this dataset
        """
        if hasattr(self, 'distribution_printed'):
            return
            
        emotion_counts = pd.Series(self.emotions).value_counts().sort_index()
        total = len(self.emotions)
        
        print(f"\nClass distribution for {self.mode} set:")
        print("-" * 50)
        print(f"{'Emotion':<10} {'Count':<8} {'Percentage':<12} {'Label'}")
        print("-" * 50)
        
        for emotion_idx, count in emotion_counts.items():
            percentage = count / total * 100
            emotion_name = EMOTION_MAP[emotion_idx]
            print(f"{emotion_idx:<10} {count:<8} {percentage:>6.2f}%       {emotion_name}")
            
        print("-" * 50)
        print(f"Total: {total} images\n")
        
        # Mark as printed to avoid duplicate output
        self.distribution_printed = True


def get_dataloaders(csv_file, root_dir, batch_size=8, num_workers=4, balance_classes=False):
    """
    Returns dataloaders for train, validation and test sets.
    
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with the dataset.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for the dataloaders.
        balance_classes (bool): Whether to use class balancing techniques.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with oversampling for minority classes in training set
    train_dataset = FER2013Dataset(
        csv_file=csv_file, 
        root_dir=root_dir, 
        transform=train_transform, 
        mode='train',
        use_extra_augmentation=True  # Enable extra augmentation for rare classes
    )
    
    val_dataset = FER2013Dataset(
        csv_file=csv_file, 
        root_dir=root_dir, 
        transform=val_transform, 
        mode='val',
        use_extra_augmentation=False
    )
    
    test_dataset = FER2013Dataset(
        csv_file=csv_file, 
        root_dir=root_dir, 
        transform=val_transform, 
        mode='test',
        use_extra_augmentation=False
    )
    
    # Create sampler for handling class imbalance if requested
    if balance_classes:
        print("Using weighted sampling to handle class imbalance")
        train_sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True  # Drop the last incomplete batch to avoid BatchNorm issues
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True  # Drop the last incomplete batch to avoid BatchNorm issues
        )
    
    # Validation and test loaders (no balancing needed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def get_balanced_sampler(dataset):
    """
    Create a weighted sampler to handle class imbalance
    """
    # Get all labels from the dataset
    labels = dataset.emotions
    
    # Count instances per class
    class_counts = np.bincount(labels)
    
    # Calculate weights inversely proportional to class frequencies
    weights = 1.0 / class_counts
    
    # Give moderate weight to the Disgust class (index 1)
    disgust_multiplier = 3.0  # Reduced from 10x to 3x
    weights[1] *= disgust_multiplier
    
    # Assign weights to samples
    sample_weights = np.array([weights[label] for label in labels])
    
    # Create and return the sampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


# Utility function to convert grayscale to RGB (not needed anymore since we convert in the dataset)
def convert_grayscale_to_rgb(tensor):
    """
    Convert a grayscale tensor to RGB by repeating the channel.
    
    Args:
        tensor (torch.Tensor): Grayscale tensor with shape [batch_size, 1, height, width]
        
    Returns:
        torch.Tensor: RGB tensor with shape [batch_size, 3, height, width]
    """
    return tensor.repeat(1, 3, 1, 1) 