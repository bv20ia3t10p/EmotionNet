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
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])

    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
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

class FERPlusDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(root_dir, split, transform)
        
        # Load annotations from processed directory
        processed_dir = os.path.join(root_dir, 'processed')
        csv_path = os.path.join(processed_dir, f'{split}_ferplus.csv')
        self.data = pd.read_csv(csv_path)
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.root_dir, 'processed', 'images', self.data.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
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