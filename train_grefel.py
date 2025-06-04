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
# import dlib
# import face_recognition
# Note: Using dummy landmarks for demo purposes

from grefel_implementation import GReFELModel, GReFELLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialExpressionDataset(Dataset):
    """Custom dataset for facial expression recognition with landmark extraction"""
    
    def __init__(self, csv_file, img_dir, transform=None, landmark_detector=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations
            img_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            landmark_detector: Face landmark detector (dlib predictor)
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.landmark_detector = landmark_detector
        
        # Emotion mapping for FER2013/FERPlus
        self.emotion_map = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral', 7: 'contempt'
        }
        
    def __len__(self):
        return len(self.annotations)
    
    def extract_landmarks(self, image):
        """Extract 68 facial landmarks using dummy generation (for demo purposes)"""
        try:
            # Generate realistic dummy landmarks for a centered face
            # In a real implementation, this would use face_recognition or dlib
            
            # Get image dimensions
            if isinstance(image, Image.Image):
                width, height = image.size
            else:
                height, width = image.shape[:2]
            
            # Generate normalized landmarks for a typical face layout
            # These are approximate positions for the 68 facial landmarks
            center_x, center_y = width // 2, height // 2
            face_width, face_height = width * 0.6, height * 0.8
            
            landmarks = []
            
            # Face contour (17 points)
            for i in range(17):
                x = center_x + (i - 8) * face_width / 16
                y = center_y + abs(i - 8) * face_height / 32
                landmarks.append([x, y])
            
            # Right eyebrow (5 points)
            for i in range(5):
                x = center_x - face_width/4 + i * face_width/8
                y = center_y - face_height/4
                landmarks.append([x, y])
            
            # Left eyebrow (5 points)
            for i in range(5):
                x = center_x + face_width/8 + i * face_width/8
                y = center_y - face_height/4
                landmarks.append([x, y])
            
            # Nose bridge and tip (9 points)
            for i in range(9):
                x = center_x + (i - 4) * face_width/20
                y = center_y - face_height/8 + i * face_height/16
                landmarks.append([x, y])
            
            # Right eye (6 points)
            for i in range(6):
                x = center_x - face_width/4 + i * face_width/12
                y = center_y - face_height/8
                landmarks.append([x, y])
            
            # Left eye (6 points)
            for i in range(6):
                x = center_x + face_width/6 + i * face_width/12
                y = center_y - face_height/8
                landmarks.append([x, y])
            
            # Outer lip (12 points)
            for i in range(12):
                angle = i * 2 * np.pi / 12
                x = center_x + np.cos(angle) * face_width/6
                y = center_y + face_height/4 + np.sin(angle) * face_height/12
                landmarks.append([x, y])
            
            # Inner lip (8 points)
            for i in range(8):
                angle = i * 2 * np.pi / 8
                x = center_x + np.cos(angle) * face_width/8
                y = center_y + face_height/4 + np.sin(angle) * face_height/16
                landmarks.append([x, y])
            
            landmark_array = np.array(landmarks[:68])  # Ensure exactly 68 points
            
            # Normalize to [0, 1] range
            landmark_array[:, 0] /= width
            landmark_array[:, 1] /= height
            
        except Exception as e:
            logger.warning(f"Error generating dummy landmarks: {e}")
            # Return center-based dummy landmarks if generation fails
            landmark_array = np.random.rand(68, 2) * 0.1 + 0.45  # Small variation around center
            
        return landmark_array.astype(np.float32)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        row = self.annotations.iloc[idx]
        
        # Handle different CSV formats
        if 'pixels' in row:
            # FER2013 format - pixels in CSV
            pixels = row['pixels'].split(' ')
            image = np.array(pixels, dtype=np.uint8).reshape(48, 48)
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
            image = Image.fromarray(image)
        else:
            # Image file format
            img_name = os.path.join(self.img_dir, str(row['image']))
            image = Image.open(img_name).convert('RGB')
        
        # Get emotion label
        if 'emotion' in row:
            label = int(row['emotion'])
        else:
            label = int(row.iloc[0])  # Assume first column is label
        
        # Extract facial landmarks
        landmarks = self.extract_landmarks(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'landmarks': torch.FloatTensor(landmarks),
            'label': torch.LongTensor([label])[0]
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

class EarlyStopping:
    """Early stopping to avoid overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class GReFELTrainer:
    """Complete training pipeline for GReFEL"""
    
    def __init__(self, model, train_loader, val_loader, num_classes=8, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        
        # Loss and optimizer
        self.criterion = GReFELLoss(lambda_cls=1.0, lambda_anchor=1.0, lambda_center=1.0)
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, landmarks)
            losses = self.criterion(outputs, labels, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Cls': f"{losses['cls_loss'].item():.4f}",
                'Anchor': f"{losses['anchor_loss'].item():.4f}",
                'Center': f"{losses['center_loss'].item():.4f}"
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, landmarks)
                losses = self.criterion(outputs, labels, self.model)
                
                total_loss += losses['total_loss'].item()
                
                # Get predictions
                predictions = torch.argmax(outputs['final_probs'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, num_epochs=100):
        """Complete training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_accuracy, val_predictions, val_labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(f'grefel_best_model.pth')
                logger.info(f"New best accuracy: {best_accuracy:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}")
        return best_accuracy
    
    def save_model(self, filepath):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, filepath)
    
    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model...")
        
        val_loss, val_accuracy, predictions, labels = self.validate()
        
        # Classification report
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        report = classification_report(labels, predictions, target_names=emotion_names[:self.num_classes])
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_names[:self.num_classes],
                   yticklabels=emotion_names[:self.num_classes])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return val_accuracy

def prepare_fer_dataset(csv_path, img_dir=None, test_size=0.2, random_state=42):
    """Prepare FER dataset for training"""
    from sklearn.model_selection import train_test_split
    
    # Read the dataset
    if csv_path.endswith('fer2013new.csv'):
        # FERPlus format
        df = pd.read_csv(csv_path)
        
        # Filter training and validation data
        train_df = df[df['Usage'] == 'Training'].copy()
        val_df = df[df['Usage'] == 'PublicTest'].copy()
    else:
        # Custom format - split manually
        df = pd.read_csv(csv_path)
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['emotion'])
    
    # Save splits
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    
    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    return train_df, val_df

def main():
    """Main training function"""
    # Configuration
    config = {
        'num_classes': 8,
        'batch_size': 32,
        'num_epochs': 100,
        'embed_dim': 512,
        'num_heads': 8,
        'depth': 6,
        'num_anchors_per_class': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {config}")
    logger.info(f"Using device: {config['device']}")
    
    # Prepare dataset
    csv_path = 'FERPlus-master/fer2013new.csv'  # Update path as needed
    img_dir = None  # For FER2013, images are in CSV
    
    if os.path.exists(csv_path):
        train_df, val_df = prepare_fer_dataset(csv_path, img_dir)
    else:
        logger.error(f"Dataset not found at {csv_path}")
        return
    
    # Create datasets
    train_dataset = FacialExpressionDataset(
        csv_file='train_split.csv',
        img_dir=img_dir,
        transform=DataAugmentation.get_train_transforms()
    )
    
    val_dataset = FacialExpressionDataset(
        csv_file='val_split.csv',
        img_dir=img_dir,
        transform=DataAugmentation.get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = GReFELModel(
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        depth=config['depth'],
        num_anchors_per_class=config['num_anchors_per_class']
    ).to(config['device'])
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = GReFELTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['num_classes'],
        device=config['device']
    )
    
    # Train model
    best_accuracy = trainer.train(num_epochs=config['num_epochs'])
    
    # Plot training history
    trainer.plot_training_history()
    
    # Final evaluation
    final_accuracy = trainer.evaluate_model()
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logger.info(f"Final validation accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    main() 