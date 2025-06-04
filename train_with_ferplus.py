#!/usr/bin/env python3
"""
Train GReFEL with FERPlus dataset

This script is specifically designed to work with the FERPlus dataset structure:
- FERPlus-master/fer2013new.csv (main label file)
- FERPlus-master/data/Training/*.png (training images)
- FERPlus-master/data/PublicTest/*.png (validation images)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import time
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from grefel_implementation import GReFELModel, GReFELLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FERPlusDataset(Dataset):
    """FERPlus dataset loader for facial expression recognition"""
    
    def __init__(self, csv_file, base_dir, split='Training', transform=None):
        """
        Args:
            csv_file (str): Path to fer2013new.csv
            base_dir (str): Base directory containing data folders
            split (str): 'Training', 'PublicTest', or 'PrivateTest'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        
        # Filter by split and remove rows with missing image names
        split_data = self.df[self.df['Usage'] == split].copy()
        
        # Remove rows with missing or invalid image names
        split_data = split_data.dropna(subset=['Image name'])
        split_data = split_data[split_data['Image name'].astype(str).str.strip() != '']
        
        self.data = split_data.reset_index(drop=True)
        
        # FERPlus emotion mapping (based on CSV columns)
        # neutral,happiness,surprise,sadness,anger,disgust,fear,contempt
        self.emotion_map = {
            0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
            4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'
        }
        
        logger.info(f"Loaded {len(self.data)} samples from {split} split")
        
    def __len__(self):
        return len(self.data)
    
    def extract_landmarks(self, image):
        """Generate dummy landmarks for the image"""
        try:
            # Get image dimensions
            if isinstance(image, Image.Image):
                width, height = image.size
            else:
                height, width = image.shape[:2]
            
            # Generate realistic dummy landmarks for a centered face
            center_x, center_y = width // 2, height // 2
            face_width, face_height = width * 0.6, height * 0.8
            
            landmarks = []
            
            # Generate 68 facial landmarks in standard order
            # Face contour (17 points)
            for i in range(17):
                x = center_x + (i - 8) * face_width / 16
                y = center_y + abs(i - 8) * face_height / 32
                landmarks.append([x, y])
            
            # Eyebrows (10 points)
            for i in range(10):
                x = center_x - face_width/3 + i * face_width/15
                y = center_y - face_height/4
                landmarks.append([x, y])
            
            # Nose (9 points)
            for i in range(9):
                x = center_x + (i - 4) * face_width/20
                y = center_y - face_height/8 + i * face_height/16
                landmarks.append([x, y])
            
            # Eyes (12 points)
            for i in range(12):
                x = center_x - face_width/3 + i * face_width/6
                y = center_y - face_height/8
                landmarks.append([x, y])
            
            # Lips (20 points)
            for i in range(20):
                angle = i * 2 * np.pi / 20
                x = center_x + np.cos(angle) * face_width/6
                y = center_y + face_height/4 + np.sin(angle) * face_height/12
                landmarks.append([x, y])
            
            landmark_array = np.array(landmarks[:68])  # Ensure exactly 68 points
            
            # Normalize to [0, 1] range
            landmark_array[:, 0] /= width
            landmark_array[:, 1] /= height
            
        except Exception as e:
            logger.warning(f"Error generating landmarks: {e}")
            # Return center-based dummy landmarks
            landmark_array = np.random.rand(68, 2) * 0.1 + 0.45
            
        return landmark_array.astype(np.float32)
    
    def get_emotion_label(self, row):
        """Extract the dominant emotion from FERPlus annotations"""
        # Get emotion vote counts
        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 
                   'anger', 'disgust', 'fear', 'contempt']
        
        vote_counts = [row[emotion] for emotion in emotions]
        
        # Return the emotion with the highest vote count
        dominant_emotion = np.argmax(vote_counts)
        return dominant_emotion
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get row data
        row = self.data.iloc[idx]
        
        # Validate and load image
        img_name = row['Image name']
        if pd.isna(img_name) or img_name == '' or not isinstance(img_name, str):
            logger.error(f"Invalid image name at index {idx}: {img_name}")
            # Create a dummy image if name is invalid
            image = Image.new('RGB', (48, 48), color='gray')
        else:
            # Load image
            img_path = os.path.join(self.base_dir, 'data', self.split, str(img_name).strip())
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                # Create a dummy image if loading fails
                image = Image.new('RGB', (48, 48), color='gray')
        
        # Get emotion label
        emotion_label = self.get_emotion_label(row)
        
        # Extract facial landmarks
        landmarks = self.extract_landmarks(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'landmarks': torch.FloatTensor(landmarks),
            'label': torch.LongTensor([emotion_label])[0]
        }

class DataAugmentation:
    """Data augmentation for facial expression recognition"""
    
    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_grefel_with_ferplus():
    """Main training function for FERPlus dataset"""
    
    print("🎭 Training GReFEL with FERPlus Dataset")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_classes': 8,
        'batch_size': 32,
        'num_epochs': 100,
        'embed_dim': 512,
        'num_heads': 8,
        'depth': 6,
        'num_anchors_per_class': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 3e-4,
        'weight_decay': 0.01
    }
    
    print(f"Configuration: {config}")
    print(f"Using device: {config['device']}")
    
    # Dataset paths
    csv_file = 'FERPlus-master/fer2013new.csv'
    base_dir = 'FERPlus-master'
    
    # Check if dataset exists
    if not os.path.exists(csv_file):
        print(f"❌ FERPlus dataset not found at {csv_file}")
        print("Please ensure the FERPlus dataset is properly downloaded and extracted.")
        return
    
    # Load and check dataset
    df = pd.read_csv(csv_file)
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Training samples: {len(df[df['Usage'] == 'Training'])}")
    print(f"  PublicTest samples: {len(df[df['Usage'] == 'PublicTest'])}")
    print(f"  PrivateTest samples: {len(df[df['Usage'] == 'PrivateTest'])}")
    
    # Create datasets
    print("\n🏗️ Creating datasets...")
    train_dataset = FERPlusDataset(
        csv_file=csv_file,
        base_dir=base_dir,
        split='Training',
        transform=DataAugmentation.get_train_transforms()
    )
    
    val_dataset = FERPlusDataset(
        csv_file=csv_file,
        base_dir=base_dir,
        split='PublicTest',
        transform=DataAugmentation.get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    print("\n🧠 Creating GReFEL model...")
    model = GReFELModel(
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        depth=config['depth'],
        num_anchors_per_class=config['num_anchors_per_class']
    ).to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Loss and optimizer
    criterion = GReFELLoss(lambda_cls=1.0, lambda_anchor=1.0, lambda_center=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch in pbar:
            images = batch['image'].to(config['device'])
            landmarks = batch['landmarks'].to(config['device'])
            labels = batch['label'].to(config['device'])
            
            # Forward pass
            outputs = model(images, landmarks)
            losses = criterion(outputs, labels, model)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            total_train_loss += losses['total_loss'].item()
            train_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Cls': f"{losses['cls_loss'].item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for batch in pbar:
                images = batch['image'].to(config['device'])
                landmarks = batch['landmarks'].to(config['device'])
                labels = batch['label'].to(config['device'])
                
                # Forward pass
                outputs = model(images, landmarks)
                losses = criterion(outputs, labels, model)
                
                total_val_loss += losses['total_loss'].item()
                val_batches += 1
                
                # Get predictions
                predictions = torch.argmax(outputs['final_probs'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'Val Loss': f"{losses['total_loss'].item():.4f}"})
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        # Timing
        epoch_time = time.time() - start_time
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, 'grefel_ferplus_best.pth')
            print(f"  🌟 New best accuracy: {best_accuracy:.4f} - Model saved!")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, f'grefel_ferplus_epoch_{epoch+1}.pth')
            print(f"  💾 Checkpoint saved: epoch_{epoch+1}.pth")
        
        print("-" * 60)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([optimizer.param_groups[0]['lr'] * (0.99 ** i) for i in range(len(train_losses))], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ferplus_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, best_accuracy

def quick_test():
    """Quick test to verify dataset loading"""
    print("🔍 Quick test of FERPlus dataset loading...")
    
    csv_file = 'FERPlus-master/fer2013new.csv'
    base_dir = 'FERPlus-master'
    
    if not os.path.exists(csv_file):
        print(f"❌ FERPlus dataset not found at {csv_file}")
        return
    
    # Test dataset loading
    try:
        train_dataset = FERPlusDataset(
            csv_file=csv_file,
            base_dir=base_dir,
            split='Training',
            transform=DataAugmentation.get_val_transforms()
        )
        
        print(f"✅ Training dataset loaded: {len(train_dataset)} samples")
        
        # Test loading one sample
        sample = train_dataset[0]
        print(f"✅ Sample loaded successfully:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Landmarks shape: {sample['landmarks'].shape}")
        print(f"   Label: {sample['label'].item()}")
        
        val_dataset = FERPlusDataset(
            csv_file=csv_file,
            base_dir=base_dir,
            split='PublicTest',
            transform=DataAugmentation.get_val_transforms()
        )
        
        print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎭 GReFEL FERPlus Training")
    print("=" * 60)
    
    try:
        # Quick test first
        quick_test()
        
        # Ask user if they want to proceed with full training
        print("\n" + "=" * 60)
        response = input("Proceed with full training? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            train_grefel_with_ferplus()
        else:
            print("Training cancelled. Run again with 'y' to start training.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 