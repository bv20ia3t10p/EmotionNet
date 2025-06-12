import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FERPlusDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        """
        Args:
            data_dir (str): Path to the data directory containing images
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): Whether this is training set or not
        """
        self.data_dir = data_dir
        self.train = train
        
        # Read FER+ labels
        ferplus_path = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'fer2013new.csv')
        self.labels_df = pd.read_csv(ferplus_path)
        
        # Filter rows based on the directory (Training, PublicTest, or PrivateTest)
        dir_name = os.path.basename(data_dir)
        if 'Train' in dir_name:
            usage = 'Training'
        elif 'Valid' in dir_name or 'PublicTest' in dir_name:
            usage = 'PublicTest'
        else:
            usage = 'PrivateTest'
        self.labels_df = self.labels_df[self.labels_df['Usage'] == usage]
        
        # Filter out rows with missing image names
        self.labels_df = self.labels_df[self.labels_df['Image name'].notna()]
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
        # Define transforms
        if transform is None:
            if train:
                self.transform = A.Compose([
                    A.Resize(224, 224),  # Required size for ViT
                    A.HorizontalFlip(p=0.5),  # Mirror faces - safe augmentation
                    # Gentle brightness/contrast adjustments
                    A.OneOf([
                        A.RandomBrightnessContrast(
                            brightness_limit=0.1,  # Reduced from 0.2
                            contrast_limit=0.1,    # Reduced from 0.2
                            p=1.0
                        ),
                        A.RandomGamma(
                            gamma_limit=(90, 110),  # More conservative gamma range
                            p=1.0
                        ),
                    ], p=0.3),  # Reduced probability
                    # Subtle noise/blur - preserves features
                    A.OneOf([
                        A.GaussNoise(
                            p=0.25
                        ),
                        A.GaussianBlur(
                            blur_limit=(3, 3),  # Fixed small kernel
                            p=1.0
                        ),
                    ], p=0.2),
                    # Very mild geometric transforms
                    A.OneOf([
                        A.Affine(
                            translate_percent=(-0.05, 0.05),  # Small shifts only
                            scale=(-0.05, 0.05),             # Minimal scaling
                            rotate=(-10, 10),                # Limited rotation
                            shear=(-5, 5),                   # Mild shear
                            interpolation=cv2.INTER_LINEAR,  # Better interpolation
                            p=1.0
                        ),
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.1,     # Reduced distortion
                            border_mode=cv2.BORDER_CONSTANT,
                            p=1.0
                        ),
                    ], p=0.2),  # Reduced probability
                    # ImageNet normalization
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # If image reading fails, create a blank image
            img = np.zeros((224, 224), dtype=np.uint8)
        
        # Convert to 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        # Get label from FER+ CSV
        # The columns are: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
        img_idx = int(img_name.split('.')[0][3:])  # Extract index from filename (e.g., 'fer0000001.png' -> 1)
        # Find the corresponding row in labels_df using the Image name column
        label_row = self.labels_df[self.labels_df['Image name'] == img_name].iloc[0]
        emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        votes = [int(v) for v in label_row[emotion_columns]]
        
        # Get the emotion with maximum votes
        label = np.argmax(votes)
        
        return img, label 