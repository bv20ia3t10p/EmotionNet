"""
Advanced augmentation techniques for facial expression recognition
"""
import torch
import torch.nn as nn
import numpy as np
import random
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import os

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*3-channel images.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*ShiftScaleRotate is a special case of Affine.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*A new version of Albumentations is available.*")

# Disable Albumentations version check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

class AlbumentationsTransform:
    """Adapter to use Albumentations with PyTorch datasets"""
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, img):
        # Convert PIL Image to numpy array
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Apply transformations
        transformed = self.transform(image=img)
        return transformed["image"]

def get_advanced_transforms(mode='train', img_size=48):
    """Get advanced augmentation pipeline using Albumentations"""
    if mode == 'train':
        transform = A.Compose([
            # Pixel-level transforms
            A.Equalize(p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # Spatial transforms - use Affine instead of ShiftScaleRotate
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-20, 20), p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2),
            
            # Occlusion transforms - Fixed parameters for CoarseDropout
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
            A.CoarseDropout(p=0.2),
            
            # Noise transforms - Fixed parameters for GaussNoise
            A.GaussNoise(p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            
            # Gaussian blur and sharpening
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            ], p=0.3),
            
            # Normalization and conversion to tensor
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:  # val or test
        transform = A.Compose([
            A.Equalize(p=1.0),  # Always apply histogram equalization for test/val
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    return AlbumentationsTransform(transform)

def get_advanced_augmentations_torchvision(mode='train'):
    """Get advanced augmentation pipeline using torchvision transforms"""
    if mode == 'train':
        # PIL image transforms (applied before conversion to tensor)
        pil_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3)
        ]
        
        # Tensor transforms (applied after conversion to tensor)
        tensor_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ]
        
        transform = transforms.Compose(pil_transforms + tensor_transforms)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    return transform

class AdvancedAugmentationDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies advanced augmentations"""
    def __init__(self, dataset, transform=None, mixup_alpha=0.2, 
                 cutmix_alpha=1.0, use_mixup=True, use_cutmix=True,
                 face_alignment=False):
        self.dataset = dataset
        self.transform = transform
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.face_alignment = face_alignment
        self.face_cascade = None
        
        # Load face detector if face alignment is enabled
        if self.face_alignment:
            try:
                # Use OpenCV's Haar cascade for face detection
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                # Silently fail and disable face alignment
                self.face_alignment = False
                
    def __len__(self):
        return len(self.dataset)
    
    def align_face(self, img_array):
        """Apply face alignment to center the face"""
        if self.face_cascade is None:
            return img_array
            
        # Ensure image is a valid numpy array with proper type
        if not isinstance(img_array, np.ndarray):
            # If img_array is not a numpy array, try to convert it
            try:
                img_array = np.array(img_array)
            except:
                return img_array
        
        # Check if image is empty or invalid
        if img_array.size == 0 or img_array.ndim < 2:
            return img_array
                
        # Ensure image is grayscale for face detection
        if len(img_array.shape) == 3 and img_array.shape[2] > 1:
            try:
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            except Exception:
                # Return original image if conversion fails
                return img_array
        else:
            gray_img = img_array
            
        # Ensure image is uint8 for face detection
        if gray_img.dtype != np.uint8:
            try:
                if gray_img.max() <= 1.0:
                    gray_img = (gray_img * 255).astype(np.uint8)
                else:
                    gray_img = gray_img.astype(np.uint8)
            except:
                return img_array
            
        # Detect faces
        try:
            faces = self.face_cascade.detectMultiScale(
                gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        except Exception:
            return img_array
            
        # If a face is detected, crop to it
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Use the first detected face
            
            # Add some margin (20%)
            margin = int(0.2 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_array.shape[1] - x, w + 2*margin)
            h = min(img_array.shape[0] - y, h + 2*margin)
            
            # Crop and resize back to 48x48
            face_img = img_array[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            return face_img
        
        return img_array
    
    def preprocess(self, img):
        """Apply preprocessing before augmentation"""
        # Convert to numpy array if PIL Image
        try:
            if isinstance(img, Image.Image):
                # Convert to grayscale first if it's a PIL image
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img)
            elif isinstance(img, np.ndarray):
                # If it's already a numpy array
                img_array = img
            else:
                # For other types, try direct conversion to numpy
                img_array = np.array(img)
                
            # Ensure we have a proper 2D grayscale image
            if img_array.ndim > 2:
                # If shape is (H,W,C) with C > 1, convert to grayscale
                if img_array.shape[2] == 3:
                    # For RGB images
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                elif img_array.shape[2] == 4:
                    # For RGBA images
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
                elif img_array.shape[2] == 1:
                    # For single channel images in 3D format
                    img_array = img_array[:,:,0]
                else:
                    # For unknown format, just take first channel - but don't warn for common sizes
                    if img_array.shape != (1, 48, 48):
                        pass  # No warning to reduce console noise
                    img_array = img_array[:,:,0]
            
            # Ensure the array is 2D
            if img_array.ndim != 2:
                if img_array.ndim == 3 and img_array.shape[0] == 1:
                    # If it's a tensor-like format with shape (1, H, W)
                    img_array = img_array[0]
                else:
                    # Only warn for uncommon sizes to reduce noise
                    if img_array.shape != (1, 48, 48):
                        pass  # No warning to reduce console noise
                    # Try to reshape to 2D
                    try:
                        img_array = img_array.reshape(48, 48)
                    except:
                        # If reshape fails, create a blank image
                        img_array = np.zeros((48, 48), dtype=np.uint8)
            
            # Ensure correct data type for OpenCV operations (8-bit unsigned int)
            if img_array.dtype != np.uint8:
                # If float between 0-1, convert to 0-255 range
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    if np.max(img_array) <= 1.0 and np.max(img_array) > 0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                
            # Apply histogram equalization with error handling
            try:
                # Final check to ensure image is proper format for equalizeHist
                if img_array.ndim == 2 and img_array.dtype == np.uint8:
                    # Check for division by zero (all zeros or all the same value)
                    if np.min(img_array) != np.max(img_array):
                        img_array = cv2.equalizeHist(img_array)
                        
                        # Apply CLAHE
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img_array = clahe.apply(img_array)
            except Exception:
                pass  # Silently continue with original image
                
            # Convert back to PIL Image (grayscale mode 'L')
            img = Image.fromarray(img_array, mode='L')
                
        except Exception:
            # If everything fails, return a blank grayscale image
            img = Image.new('L', (48, 48), 0)
            
        return img
        
    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
            
            # Apply preprocessing
            img = self.preprocess(img)
            
            # Apply transforms if provided
            if self.transform:
                try:
                    img = self.transform(img)
                except Exception:
                    # Fall back to basic transformation
                    try:
                        if isinstance(img, Image.Image):
                            img = transforms.ToTensor()(img)
                            img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
                    except Exception:
                        # Last resort: return empty tensor
                        img = torch.zeros((1, 48, 48))
            
            # Ensure tensor has consistent dimensions - should be (C, H, W) = (1, 48, 48) for grayscale
            if isinstance(img, torch.Tensor):
                if img.dim() == 2:
                    # If we have a 2D tensor, add channel dimension
                    img = img.unsqueeze(0)
                
                if img.shape != (1, 48, 48):
                    if img.shape[0] == 3 and img.shape[1] == 48 and img.shape[2] == 48:
                        # Convert RGB to grayscale by taking the mean
                        img = img.mean(dim=0, keepdim=True)
                    elif img.shape[0] == 1 and img.shape[1] == 1 and img.shape[2] == 48:
                        # Reshape [1, 1, 48] to [1, 48, 48]
                        img = img.expand(1, 48, 48)
                    elif img.shape[0] == 1:
                        # Try to reshape to [1, 48, 48]
                        try:
                            img = img.reshape(1, 48, 48)
                        except:
                            img = torch.zeros((1, 48, 48))
                    else:
                        # If we have a tensor with completely wrong dimensions, replace it
                        img = torch.zeros((1, 48, 48))
            
            # Final check for NaN or Infinity values
            if isinstance(img, torch.Tensor) and (torch.isnan(img).any() or torch.isinf(img).any()):
                img = torch.zeros((1, 48, 48))
            
            return img, label
            
        except Exception:
            # Return empty image and original label as fallback
            return torch.zeros((1, 48, 48)), label
        
class TestTimeAugmentation:
    """Test-time augmentation for inference"""
    def __init__(self, model, num_augmentations=3, device='cuda'):
        self.model = model
        self.num_augmentations = num_augmentations
        self.device = device
        
        # Create a set of test-time augmentations - simplified for speed
        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        
    def predict(self, img):
        """Apply TTA and average predictions - optimized for speed"""
        self.model.eval()
        
        # Get base prediction first (without augmentation)
        with torch.no_grad():
            base_pred = self.model(img)
            
            # For speed, if num_augmentations is 1 or less, just return base prediction
            if self.num_augmentations <= 1:
                return base_pred
            
            # Apply a maximum of 3 augmentations regardless of requested amount (for speed)
            actual_augs = min(self.num_augmentations - 1, 2)
            
            # Prepare to collect all predictions
            all_preds = [base_pred]
            batch_size = img.size(0)
            
            # Process in batches for efficiency
            for _ in range(actual_augs):
                # Create augmented versions directly on GPU
                augmented_batch = torch.zeros_like(img)
                
                # Basic augmentations that can be done directly with tensors
                # Horizontal flip (faster than using albumentations)
                if random.random() > 0.5:
                    augmented_batch = img.flip(3)  # Horizontal flip
                else:
                    augmented_batch = img.clone()
                
                # Make prediction on augmented batch
                aug_pred = self.model(augmented_batch)
                all_preds.append(aug_pred)
            
            # Average all predictions
            avg_pred = torch.stack(all_preds).mean(dim=0)
            return avg_pred

# Define a safe custom collate function to handle tensors of different shapes
def safe_collate(batch):
    """Handle batches with inconsistent tensor shapes"""
    images = []
    labels = []
    
    for item in batch:
        try:
            img, label = item
            
            # Process tensor to ensure consistency
            if isinstance(img, torch.Tensor):
                # Check for NaN/Infinity
                if torch.isnan(img).any() or torch.isinf(img).any():
                    img = torch.zeros((1, 48, 48))
                    
                # Ensure proper shape
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                elif img.dim() == 3:
                    if img.shape[0] == 3:  # RGB
                        img = img.mean(dim=0, keepdim=True)
                    
                    # Reshape if necessary
                    if img.shape != (1, 48, 48):
                        try:
                            img = img.reshape(1, 48, 48)
                        except:
                            img = torch.zeros((1, 48, 48))
            else:
                # If not a tensor, create a zero tensor
                img = torch.zeros((1, 48, 48))
                
            images.append(img)
            labels.append(label)
        except Exception:
            # Skip problematic batch items
            continue
    
    # Handle empty batch
    if not images:
        return torch.zeros((0, 1, 48, 48)), torch.zeros(0)
        
    # Stack tensors
    try:
        images = torch.stack(images)
        labels = torch.tensor(labels)
    except Exception:
        # Fallback if stacking fails
        images = torch.zeros((len(images), 1, 48, 48))
        labels = torch.zeros(len(labels))
        
    return images, labels

# Advanced MixUp implementation
def advanced_mixup(x, y, alpha=0.2, device='cuda'):
    """Enhanced MixUp augmentation with custom distributions"""
    batch_size = x.size()[0]
    
    # Use a custom Beta distribution instead of simple Beta
    if random.random() < 0.8:  # 80% of the time use standard Beta
        lam = np.random.beta(alpha, alpha)
    else:  # 20% of the time use extreme values for higher diversity
        lam = np.random.choice([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    # Generate random index permutation
    index = torch.randperm(batch_size).to(device)
    
    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Get mixed labels
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# Advanced CutMix implementation
def advanced_cutmix(x, y, alpha=1.0, device='cuda'):
    """Enhanced CutMix augmentation with custom region selection"""
    batch_size = x.size()[0]
    
    # Use a custom Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Generate random index permutation
    index = torch.randperm(batch_size).to(device)
    
    # Get image dimensions
    _, _, h, w = x.shape
    
    # Random choice of cutout strategy
    strategy = random.choice(['center', 'random', 'quarter'])
    
    if strategy == 'center':
        # Center region cut
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Center coordinates
        cx = w // 2
        cy = h // 2
        
        # Boundary
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
    elif strategy == 'quarter':
        # Cut one quarter of the image
        quadrant = random.randint(0, 3)
        
        if quadrant == 0:  # Top-left
            bbx1, bby1, bbx2, bby2 = 0, 0, w//2, h//2
        elif quadrant == 1:  # Top-right
            bbx1, bby1, bbx2, bby2 = w//2, 0, w, h//2
        elif quadrant == 2:  # Bottom-left
            bbx1, bby1, bbx2, bby2 = 0, h//2, w//2, h
        else:  # Bottom-right
            bbx1, bby1, bbx2, bby2 = w//2, h//2, w, h
            
        # Adjust lambda based on the actual cut area
        lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
    else:  # Random strategy
        # Random rectangular region
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Randomly select cutout center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Boundary
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Create mixed images
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to reflect actual area ratio
    lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    # Get mixed labels
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam 