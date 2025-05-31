"""
Data loader utilities for EmotionNet.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def create_weighted_sampler(dataset):
    """Create weighted sampler for handling class imbalance."""
    class_counts = pd.Series(dataset.emotions).value_counts()
    total_samples = len(dataset)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in class_counts.items()}
    
    weights = [class_weights[label] for label in dataset.emotions]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def get_data_loaders(train_csv, val_csv, test_csv, img_dir, batch_size=32, 
                     img_size=224, num_workers=4, use_weighted_sampler=True):
    """
    Create data loaders with optional weighted sampling for handling class imbalance.
    """
    from .dataset import FER2013Dataset
    from .transforms import get_transforms
    
    # Create datasets
    train_dataset = FER2013Dataset(
        csv_file=train_csv,
        root_dir=img_dir,
        transform=get_transforms(img_size, is_train=True),
        mode='train'
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
    
    # Create data loaders
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
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