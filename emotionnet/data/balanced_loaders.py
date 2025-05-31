"""
Balanced data loader utilities for EmotionNet.
Uses balanced training set but ORIGINAL validation and test sets for proper evaluation.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader


def get_balanced_data_loaders(balanced_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Create data loaders with balanced TRAINING set but original VAL/TEST sets.
    This ensures proper evaluation on the natural distribution.
    """
    from .balanced_dataset import BalancedFER2013Dataset
    from .dataset import FER2013Dataset
    from .transforms import get_transforms
    
    print("   🎯 Using balanced TRAINING set + original VAL/TEST sets")
    
    # Balanced training dataset
    train_dataset = BalancedFER2013Dataset(
        csv_file=f'{balanced_dir}/train.csv',
        root_dir=balanced_dir,
        transform=get_transforms(img_size, is_train=True),
        mode='train'
    )
    
    # ORIGINAL validation and test datasets (not balanced)
    val_dataset = FER2013Dataset(
        csv_file='fer2013/fer2013.csv',
        root_dir='fer2013',
        transform=get_transforms(img_size, is_train=False),
        mode='val'
    )
    test_dataset = FER2013Dataset(
        csv_file='fer2013/fer2013.csv',
        root_dir='fer2013',
        transform=get_transforms(img_size, is_train=False),
        mode='test'
    )
    
    print("   📊 Dataset composition:")
    print(f"      - Training: Balanced ({len(train_dataset)} samples)")
    print(f"      - Validation: Original distribution ({len(val_dataset)} samples)")
    print(f"      - Test: Original distribution ({len(test_dataset)} samples)")
    
    # Create data loaders (no weighted sampling - training is balanced)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Regular shuffle instead of weighted sampling
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