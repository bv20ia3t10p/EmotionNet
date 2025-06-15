import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple, List
import random
import cv2

class BaseDataset(Dataset):
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
        
    def _get_default_transforms(self) -> transforms.Compose:
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),  # Direct resize to target size
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
                transforms.Resize((224, 224)),  # Direct resize to target size
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
        transform: Optional[transforms.Compose] = None,
        use_soft_labels: bool = True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._get_default_transforms()
        self.use_soft_labels = use_soft_labels
        
        # Load annotations
        self.samples = self._load_annotations()
        
    def _get_default_transforms(self) -> transforms.Compose:
        if self.split == 'train':
            return transforms.Compose([
                # Direct resize to target size
                transforms.Resize((224, 224)),
                
                # Face-preserving augmentations
                FacePreservingAugmentation(p=0.6),  # 60% chance of advanced augmentation
                
                # Gentle geometric transforms
                transforms.RandomHorizontalFlip(p=0.5),
                
                # Very mild rotation to preserve facial structure
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=(-8, 8), fill=0)  # Slightly increased range
                ], p=0.4),
                
                # Enhanced color augmentation for emotion recognition
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.25,  # Slightly increased for emotion emphasis
                        contrast=0.25,    # Slightly increased for emotion emphasis
                        saturation=0.15,  # Moderate saturation changes
                        hue=0.03         # Very subtle hue changes
                    )
                ], p=0.7),  # Higher probability for color changes
                
                # Gentle affine transforms (preserve face geometry)
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=0,           # No rotation (handled separately)
                        translate=(0.08, 0.08),  # Slightly increased translation
                        scale=(0.92, 1.08),      # Slightly increased scale variation
                        shear=0              # No shear to preserve face shape
                    )
                ], p=0.4),
                
                # Enhanced perspective changes
                transforms.RandomApply([
                    transforms.RandomPerspective(distortion_scale=0.15, p=1.0)  # Slightly increased
                ], p=0.3),
                
                # Gaussian blur for focus variations
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))  # Slightly increased sigma
                ], p=0.15),
                
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Enhanced random erasing with face-aware parameters
                transforms.RandomErasing(
                    p=0.15,          # Slightly increased probability
                    scale=(0.02, 0.12),  # Slightly larger patches
                    ratio=(0.3, 3.3),
                    value='random'
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),  # Direct resize to target size
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
        emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        prob_columns = [f'{col}_prob' for col in emotion_columns]
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.root_dir, 'processed', 'images', row['image'])
            hard_label = row['label']
            
            # Check if soft labels are available and requested
            if self.use_soft_labels and all(col in row for col in prob_columns):
                soft_labels = torch.tensor([row[col] for col in prob_columns], dtype=torch.float32)
                annotations.append((img_path, hard_label, soft_labels))
            else:
                annotations.append((img_path, hard_label, None))
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.samples[idx]) == 3:
            img_path, hard_label, soft_labels = self.samples[idx]
        else:
            img_path, hard_label = self.samples[idx]
            soft_labels = None
        
        # Load and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return appropriate label format
        if soft_labels is not None and self.use_soft_labels:
            return image, soft_labels
        else:
            return image, torch.tensor(hard_label, dtype=torch.long)

class RAFDBDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        val_split: float = 0.1  # 10% for validation
    ):
        super().__init__(root_dir, split, transform)
        
        # Load annotations from CSV
        if split == 'test':
            csv_path = os.path.join(root_dir, 'test_labels.csv')
            self.data = pd.read_csv(csv_path)
        else:
            # For train/val, load training data and split
            csv_path = os.path.join(root_dir, 'train_labels.csv')
            full_data = pd.read_csv(csv_path)
            
            # Ensure deterministic split
            np.random.seed(42)
            indices = np.random.permutation(len(full_data))
            val_size = int(len(full_data) * val_split)
            
            if split == 'train':
                # Use remaining 90% for training
                train_indices = indices[val_size:]
                self.data = full_data.iloc[train_indices].reset_index(drop=True)
            else:  # val
                # Use 10% for validation
                val_indices = indices[:val_size]
                self.data = full_data.iloc[val_indices].reset_index(drop=True)
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        # Get label first (1-based in RAF-DB)
        label = int(row['label'])
        
        # Images are organized in emotion label subdirectories
        img_path = os.path.join(self.root_dir, 'DATASET', 
                               'train' if self.split != 'test' else 'test',
                               str(label),  # Use emotion label as subdirectory
                               row['image'])
        
        # Load and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to 0-based for training
        label = torch.tensor(label - 1, dtype=torch.long)
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

class FacePreservingAugmentation:
    """Advanced augmentation that preserves facial features while enhancing emotion recognition"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # Convert to numpy for advanced processing
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
            
        # Apply one of several face-preserving augmentations
        aug_type = random.choice(['facial_emphasis', 'emotion_enhancement', 'lighting_variation'])
        
        if aug_type == 'facial_emphasis':
            # Enhance facial regions while preserving structure
            img_np = self._enhance_facial_regions(img_np)
        elif aug_type == 'emotion_enhancement':
            # Subtle contrast/brightness changes that emphasize emotions
            img_np = self._enhance_emotion_features(img_np)
        elif aug_type == 'lighting_variation':
            # Simulate different lighting conditions
            img_np = self._vary_lighting(img_np)
            
        return Image.fromarray(img_np.astype(np.uint8))
    
    def _enhance_facial_regions(self, img):
        """Enhance key facial regions (eyes, mouth) for emotion recognition"""
        # Apply subtle sharpening to enhance facial features
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        
        if len(img.shape) == 3:
            enhanced = np.zeros_like(img)
            for c in range(3):
                enhanced[:,:,c] = cv2.filter2D(img[:,:,c], -1, kernel)
        else:
            enhanced = cv2.filter2D(img, -1, kernel)
            
        # Blend with original to maintain naturalness
        return np.clip(0.7 * img + 0.3 * enhanced, 0, 255)
    
    def _enhance_emotion_features(self, img):
        """Enhance features that are important for emotion recognition"""
        # Subtle contrast enhancement
        contrast_factor = random.uniform(1.05, 1.15)
        enhanced = np.clip((img - 128) * contrast_factor + 128, 0, 255)
        
        # Add slight brightness variation
        brightness_factor = random.uniform(-5, 5)
        enhanced = np.clip(enhanced + brightness_factor, 0, 255)
        
        return enhanced
    
    def _vary_lighting(self, img):
        """Simulate different lighting conditions while preserving facial structure"""
        # Create subtle lighting gradient
        h, w = img.shape[:2]
        
        # Random lighting direction
        direction = random.choice(['top', 'bottom', 'left', 'right', 'center'])
        
        if direction == 'top':
            gradient = np.linspace(1.1, 0.9, h).reshape(-1, 1)
        elif direction == 'bottom':
            gradient = np.linspace(0.9, 1.1, h).reshape(-1, 1)
        elif direction == 'left':
            gradient = np.linspace(1.1, 0.9, w).reshape(1, -1)
        elif direction == 'right':
            gradient = np.linspace(0.9, 1.1, w).reshape(1, -1)
        else:  # center
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h//2, w//2
            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            gradient = 1.0 + 0.1 * (1 - distance / np.max(distance))
        
        if len(img.shape) == 3:
            gradient = np.expand_dims(gradient, axis=2)
            
        return np.clip(img * gradient, 0, 255)

class CutMixAugmentation:
    """CutMix augmentation adapted for facial emotion recognition"""
    
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, img1, img2, label1, label2):
        if random.random() > self.p:
            return img1, label1, label2, 1.0
            
        # Generate mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        H, W = img1.shape[-2:]
        
        # Generate random bounding box that avoids central facial features
        # Focus on peripheral areas to preserve main facial expressions
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Avoid center region (where main facial features are)
        center_margin = 0.3  # Avoid central 30% of image
        margin_w = int(W * center_margin / 2)
        margin_h = int(H * center_margin / 2)
        
        # Choose cut region from peripheral areas
        if random.random() < 0.5:
            # Top or bottom regions
            if random.random() < 0.5:
                # Top region
                cx = random.randint(cut_w // 2, W - cut_w // 2)
                cy = random.randint(cut_h // 2, margin_h)
            else:
                # Bottom region
                cx = random.randint(cut_w // 2, W - cut_w // 2)
                cy = random.randint(H - margin_h, H - cut_h // 2)
        else:
            # Left or right regions
            if random.random() < 0.5:
                # Left region
                cx = random.randint(cut_w // 2, margin_w)
                cy = random.randint(cut_h // 2, H - cut_h // 2)
            else:
                # Right region
                cx = random.randint(W - margin_w, W - cut_w // 2)
                cy = random.randint(cut_h // 2, H - cut_h // 2)
        
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)
        
        # Apply cutmix
        img1[..., bby1:bby2, bbx1:bbx2] = img2[..., bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return img1, label1, label2, lam

def get_loaders(args):
    """Create train and validation data loaders based on dataset and arguments."""
    from torch.utils.data import DataLoader
    
    if args.dataset.lower() == 'ferplus':
        train_dataset = FERPlusDataset(
            root_dir=args.data_dir,
            split='train',
            use_soft_labels=args.use_soft_labels
        )
        val_dataset = FERPlusDataset(
            root_dir=args.data_dir,
            split='val',
            use_soft_labels=args.use_soft_labels
        )
    elif args.dataset.lower() == 'rafdb':
        train_dataset = RAFDBDataset(
            root_dir=args.data_dir,
            split='train'
        )
        val_dataset = RAFDBDataset(
            root_dir=args.data_dir,
            split='val'
        )
    elif args.dataset.lower() == 'fer2013':
        train_dataset = FER2013Dataset(
            root_dir=args.data_dir,
            split='train'
        )
        val_dataset = FER2013Dataset(
            root_dir=args.data_dir,
            split='val'
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader 