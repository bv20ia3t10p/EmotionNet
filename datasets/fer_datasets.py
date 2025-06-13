import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple, List

class BaseDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._get_default_transforms()
        
    def _get_default_transforms(self) -> transforms.Compose:
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),  # Resize larger for random crop
                transforms.RandomCrop(224),     # Random crop to final size
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),  # Increased rotation range
                transforms.ColorJitter(
                    brightness=0.4,  # Increased color enhancement
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),  # Increased translation
                    scale=(0.8, 1.2),      # Increased scaling
                    shear=15               # Added shear
                ),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Added perspective
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Performs mixup on the input and target."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # This should be implemented by child classes
        raise NotImplementedError

class FERPlusDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._get_default_transforms()
        
        # Load annotations
        self.samples = self._load_annotations()
        
    def _get_default_transforms(self) -> transforms.Compose:
        if self.split == 'train':
            return transforms.Compose([
                # Resize with slight variation to preserve face proportions
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224, padding=8, padding_mode='reflect'),
                
                # Gentle horizontal flip (faces are roughly symmetric)
                transforms.RandomHorizontalFlip(p=0.5),
                
                # Very mild rotation to preserve facial structure
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=(-5, 5), fill=0)
                ], p=0.3),
                
                # Subtle lighting and contrast changes (preserve facial features)
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.2,  # Reduced from 0.4
                        contrast=0.2,    # Reduced from 0.4
                        saturation=0.1,  # Reduced from 0.4
                        hue=0.02         # Reduced from 0.1
                    )
                ], p=0.6),
                
                # Very gentle affine transforms (preserve face geometry)
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=0,           # No rotation (handled separately)
                        translate=(0.05, 0.05),  # Reduced from (0.1, 0.1)
                        scale=(0.95, 1.05),      # Reduced from (0.8, 1.2)
                        shear=0              # No shear to preserve face shape
                    )
                ], p=0.3),
                
                # Mild perspective changes (very conservative)
                transforms.RandomApply([
                    transforms.RandomPerspective(distortion_scale=0.1, p=1.0)  # Reduced from 0.3
                ], p=0.2),
                
                # Gaussian blur occasionally (simulates focus variations)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
                ], p=0.1),
                
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Random erasing with face-aware parameters
                transforms.RandomErasing(
                    p=0.1,           # Low probability
                    scale=(0.02, 0.1),  # Small patches only
                    ratio=(0.3, 3.3),
                    value='random'
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def _load_annotations(self) -> list:
        """Load FERPlus annotations from processed directory."""
        annotations = []
        split_file = os.path.join(self.root_dir, 'processed', f'{self.split}_ferplus.csv')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f'Cannot find {split_file}. Please ensure the data is preprocessed.')
        
        df = pd.read_csv(split_file)
        for _, row in df.iterrows():
            img_path = os.path.join(self.root_dir, 'processed', 'images', row['image'])
            label = row['label']  # Changed from 'emotion' to 'label'
            annotations.append((img_path, label))
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class RAFDBDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(root_dir, split, transform)
        
        # Load annotations
        list_path = os.path.join(root_dir, f'{split}_list.txt')
        with open(list_path, 'r') as f:
            self.samples = [line.strip().split(' ') for line in f]
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(int(label), dtype=torch.long)
        return image, label

class FER2013Dataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(root_dir, split, transform)
        
        # Load annotations
        csv_path = os.path.join(root_dir, 'fer2013.csv')
        df = pd.read_csv(csv_path)
        
        # Split data
        if split == 'train':
            self.data = df[df['Usage'] == 'Training']
        elif split == 'val':
            self.data = df[df['Usage'] == 'PublicTest']
        else:
            self.data = df[df['Usage'] == 'PrivateTest']
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pixels = self.data.iloc[idx]['pixels'].split()
        image = np.array(pixels, dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.data.iloc[idx]['emotion'], dtype=torch.long)
        return image, label 