"""
FERPlus Dataset Loader

The FERPlus dataset is an extension of FER2013 with 8 emotion classes:
neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
"""

import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FERPlusDataset(Dataset):
    """
    Dataset for FERPlus with different label processing modes:
    - majority: use the emotion with the majority vote
    - probability: convert votes to probability distribution 
    - multi_target: treat emotions with >30% votes as targets
    
    Enhanced with:
    - Class distribution tracking
    - Support for oversampling and class balancing
    - Minority class-specific augmentations
    """
    def __init__(self, data_dir, partition, mode='majority', transform=None):
        """
        Args:
            data_dir: FERPlus data directory
            partition: 'Training', 'PublicTest' or 'PrivateTest'
            mode: 'majority', 'probability', or 'multi_target'
            transform: image transformations
        """
        self.data_dir = data_dir
        self.partition = partition
        self.mode = mode
        self.transform = transform
        self.emotion_map = {
            'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
            'anger': 4, 'disgust': 5, 'fear': 6, 'contempt': 7
        }
        
        # Map partition names to FERPlus folder names
        partition_map = {
            'Training': 'FER2013Train',
            'PublicTest': 'FER2013Valid',
            'PrivateTest': 'FER2013Test'
        }
        
        # Path to the data folder containing images
        self.folder_path = os.path.join(os.path.join(data_dir, 'data'), partition_map[partition])
        
        # Path to the labels CSV file (fer2013new.csv in the root directory)
        self.labels_path = os.path.join(data_dir, 'fer2013new.csv')
        
        print(f"Loading FERPlus {partition} labels from: {self.labels_path}")
        
        # Mapping from emotion index to emotion name for logging
        self.idx_to_emotion = {v: k for k, v in self.emotion_map.items()}
        
        # Load labels from fer2013new.csv
        self.data = []
        self.class_counts = [0] * 8  # Count of each emotion class
        
        with open(self.labels_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            
            for row in reader:
                usage = row[0]
                
                # Skip entries not in the current partition
                if usage != self.partition:
                    continue
                    
                image_file = row[1]
                
                # Skip rows with empty image file (some entries are invalid)
                if not image_file:
                    continue
                
                # FERPlus label format: 
                # Usage, Image name, neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
                try:
                    emotion_votes = [int(v) for v in row[2:10]]  # 8 emotions (columns 2-9)
                except ValueError:
                    print(f"Error parsing line: {row}")
                    continue
                
                # Process the labels according to the mode
                if self.mode == 'majority':
                    # Find the emotion with the majority votes
                    max_votes = max(emotion_votes)
                    if max_votes > 0:
                        label = emotion_votes.index(max_votes)
                        # Track class distribution
                        self.class_counts[label] += 1
                    else:
                        # Skip images with no clear emotion
                        continue
                    
                    processed_label = label  # Single integer label for majority mode
                
                elif self.mode == 'probability':
                    # Convert votes to probability distribution
                    total_votes = sum(emotion_votes)
                    if total_votes > 0:
                        processed_label = [v / total_votes for v in emotion_votes]
                        # Track most common class for distribution statistics
                        max_idx = processed_label.index(max(processed_label))
                        self.class_counts[max_idx] += 1
                    else:
                        # Skip images with no votes
                        continue
                
                elif self.mode == 'multi_target':
                    # Treat emotions with >30% votes as targets
                    total_votes = sum(emotion_votes)
                    if total_votes > 0:
                        # Calculate percentage of votes for each emotion
                        vote_percents = [v / total_votes for v in emotion_votes]
                        # Emotions with >30% votes are targets
                        processed_label = [1 if p > 0.3 else 0 for p in vote_percents]
                        # If no emotion has >30% votes, use the max
                        if sum(processed_label) == 0:
                            max_idx = vote_percents.index(max(vote_percents))
                            processed_label[max_idx] = 1
                            self.class_counts[max_idx] += 1
                        else:
                            # For multi-target, increment all present classes
                            for i, present in enumerate(processed_label):
                                if present:
                                    self.class_counts[i] += 1
                    else:
                        # Skip images with no votes
                        continue
                
                # Store the image path and processed label
                self.data.append((os.path.join(self.folder_path, image_file), processed_label))
        
        # Define minority classes (those with fewer samples)
        self.minority_classes = self._get_minority_classes()
        
        print(f"Loaded {len(self.data)} valid samples for {partition}")
        print(f"Class distribution:")
        for i in range(8):
            emotion_name = self.idx_to_emotion[i]
            print(f"   - {emotion_name}: {self.class_counts[i]} samples")
    
    def _get_minority_classes(self):
        """
        Identify minority classes for special augmentation handling
        """
        # Classes with less than 10% of average count are considered minority
        avg_count = sum(self.class_counts) / len(self.class_counts)
        threshold = avg_count * 0.2
        return [i for i, count in enumerate(self.class_counts) if count < threshold]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = np.array(image)
        
        # Apply more aggressive augmentation for minority classes
        if self.transform and self.mode == 'majority':
            if isinstance(label, int) and label in self.minority_classes:
                # Use stronger augmentation for minority classes
                augmented = self._apply_minority_augmentation(image)
            else:
                augmented = self.transform(image=image)
            image = augmented['image']
        elif self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.mode == 'majority':
            # Return integer label for majority mode
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Return multi-class or probability labels
            return image, torch.tensor(label, dtype=torch.float32)
    
    def _apply_minority_augmentation(self, image):
        """Apply stronger augmentation for minority classes"""
        minority_transform = A.Compose([
            A.Resize(48, 48),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.RandomGamma(gamma_limit=(80, 120), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.GaussNoise(p=0.6),
                A.GaussianBlur(blur_limit=3, p=0.6),
                A.MotionBlur(blur_limit=5, p=0.6),
            ], p=0.8),
            A.OneOf([
                A.RandomRotate90(p=0.6),
                A.Rotate(limit=40, p=0.6),
            ], p=0.7),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.6),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.6),
                A.OpticalDistortion(distort_limit=0.2, p=0.6),
            ], p=0.7),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        return minority_transform(image=image)


def get_transforms(img_size=48, is_training=True):
    """
    Get data transformations for FERPlus dataset
    """
    if is_training:
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(),
                # Remove HueSaturationValue as it's not applicable for grayscale images
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.GaussNoise(),  # Fixed: removed invalid var_limit parameter
            ], p=0.5),
            A.OneOf([
                A.RandomRotate90(),
                A.Rotate(limit=30),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=1),
            ], p=0.3),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        return train_transform
    else:
        val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        return val_transform


def get_loss_fn(mode):
    """
    Get appropriate loss function based on the training mode
    """
    import torch.nn as nn
    from ..training.losses import FocalLoss
    
    if mode == 'majority':
        # Use Focal Loss with higher gamma for minority classes
        return FocalLoss(gamma=2.0)
    elif mode == 'probability':
        # KL divergence for probability distributions
        return nn.KLDivLoss(reduction='batchmean')
    elif mode == 'multi_target':
        # Binary cross entropy for multi-label classification
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_ferplus_data_loaders(data_dir, batch_size=32, img_size=48, num_workers=4, mode='majority'):
    """
    Create data loaders for FERPlus dataset
    
    Args:
        data_dir: FERPlus data directory
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
        mode: Label processing mode ('majority', 'probability', or 'multi_target')
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Creating FERPlus data loaders with mode: {mode}")
    train_transform = get_transforms(img_size=img_size, is_training=True)
    val_transform = get_transforms(img_size=img_size, is_training=False)
    
    train_dataset = FERPlusDataset(
        data_dir=data_dir,
        partition='Training',
        mode=mode,
        transform=train_transform
    )
    
    val_dataset = FERPlusDataset(
        data_dir=data_dir,
        partition='PublicTest',
        mode=mode,
        transform=val_transform
    )
    
    test_dataset = FERPlusDataset(
        data_dir=data_dir,
        partition='PrivateTest',
        mode=mode,
        transform=val_transform
    )
    
    # Create oversampling weights for training data to handle class imbalance
    if mode == 'majority':
        class_counts = train_dataset.class_counts
        # Calculate class weights inversely proportional to class frequency
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        # Create sample weights (weight for each sample in dataset)
        sample_weights = [0] * len(train_dataset)
        for idx in range(len(train_dataset)):
            _, label = train_dataset.data[idx]
            sample_weights[idx] = class_weights[label].item()
        
        # Create a weighted sampler that oversamples minority classes
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        print(f"Using oversampling with weighted sampler for balanced training")
        print(f"Class weights: {[f'{w:.2f}' for w in class_weights.numpy()]}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use weighted sampler instead of shuffle
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
    
    # Store class distribution in config format for loss functions
    class_distribution = {'class_counts': train_dataset.class_counts}
    
    return train_loader, val_loader, test_loader, class_distribution 