import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random
from sklearn.metrics import confusion_matrix, classification_report
import math

# Import models from separate files
from resemotenet import ResEmoteNet
from enhanced_resemotenet import EnhancedResEmoteNet

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Dataset class for FER2013
class FERDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, emotion_specific_transforms=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.emotion_specific_transforms = emotion_specific_transforms or {}
        self.samples = []
        
        # Load images and labels from directory
        for emotion_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_name)
                        self.samples.append((img_path, emotion_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Use emotion-specific transform if available
        emotion_name = EMOTIONS[label]
        if emotion_name in self.emotion_specific_transforms:
            image = self.emotion_specific_transforms[emotion_name](image)
        elif self.transform:
            image = self.transform(image)
        
        # For simplicity, we generate synthetic valence/arousal values based on emotion categories
        # In a real implementation, these would come from labeled data
        # Valence: -1 (negative) to 1 (positive)
        # Arousal: -1 (calm) to 1 (excited)
        valence_values = [-0.7, -0.6, -0.5, 0.8, -0.6, 0.4, 0.0]  # Approximate for 7 emotions
        arousal_values = [0.6, 0.3, 0.7, 0.5, -0.3, 0.8, -0.2]    # Approximate for 7 emotions
        
        valence = torch.tensor([valence_values[label]], dtype=torch.float32)
        arousal = torch.tensor([arousal_values[label]], dtype=torch.float32)
        
        return image, label, valence, arousal

# Advanced data augmentation
class StrongAugment:
    def __call__(self, img):
        img = np.array(img)
        
        # Apply a random combination of augmentations
        # 1. Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        
        # 2. Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-10, 10)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 3. Random Gaussian noise
        if random.random() > 0.7:
            row, col = img.shape
            mean = 0
            sigma = random.randint(1, 5)
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
            img = img + gauss
            img = np.clip(img, 0, 255)
        
        # 4. Occasional horizontal flip for data augmentation
        if random.random() > 0.5:
            img = cv2.flip(img, 1)  # 1 means horizontal flip
        
        return Image.fromarray(img.astype('uint8'))

# Mixup data augmentation
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs and targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function for original ResEmoteNet
def train_original(model, train_loader, optimizer, criterion_cls, criterion_reg, device, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for i, (images, labels, valence, arousal) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        valence = valence.to(device)
        arousal = arousal.to(device)
        
        # Apply mixup if enabled
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            
            # Forward pass
            logits, val_pred, ar_pred = model(images)
            
            # Calculate loss with mixup for classification
            cls_loss = mixup_criterion(criterion_cls, logits, labels_a, labels_b, lam)
            
            # Regression losses (simplified - not using mixup for regression targets)
            val_loss = criterion_reg(val_pred, valence)
            ar_loss = criterion_reg(ar_pred, arousal)
            
        else:
            # Forward pass
            logits, val_pred, ar_pred = model(images)
            
            # Calculate losses
            cls_loss = criterion_cls(logits, labels)
            val_loss = criterion_reg(val_pred, valence)
            ar_loss = criterion_reg(ar_pred, arousal)
        
        # Combined loss (weighted sum)
        loss = cls_loss + 0.2 * val_loss + 0.2 * ar_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        # For accuracy calculation, we use the emotion classification output
        _, predicted = logits.max(1)
        total += labels.size(0)
        
        if use_mixup:
            # Approximate accuracy calculation with mixup
            correct += (lam * predicted.eq(labels_a).sum().float() 
                      + (1 - lam) * predicted.eq(labels_b).sum().float())
        else:
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (i + 1), 
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total

# Training function for enhanced ResEmoteNet with deep supervision
def train_enhanced(model, train_loader, optimizer, criterion_cls, criterion_reg, device, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for i, (images, labels, valence, arousal) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        valence = valence.to(device)
        arousal = arousal.to(device)
        
        # Apply mixup if enabled
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            
            # Forward pass with deep supervision outputs
            logits, val_pred, ar_pred, aux_logits, aux_val, aux_ar = model(images)
            
            # Calculate loss with mixup for classification (main outputs)
            cls_loss = mixup_criterion(criterion_cls, logits, labels_a, labels_b, lam)
            
            # Auxiliary classification loss with mixup
            aux_cls_loss = mixup_criterion(criterion_cls, aux_logits, labels_a, labels_b, lam)
            
            # Regression losses (simplified - not using mixup for regression targets)
            val_loss = criterion_reg(val_pred, valence)
            ar_loss = criterion_reg(ar_pred, arousal)
            aux_val_loss = criterion_reg(aux_val, valence)
            aux_ar_loss = criterion_reg(aux_ar, arousal)
            
        else:
            # Forward pass with deep supervision outputs
            logits, val_pred, ar_pred, aux_logits, aux_val, aux_ar = model(images)
            
            # Calculate losses for main outputs
            cls_loss = criterion_cls(logits, labels)
            val_loss = criterion_reg(val_pred, valence)
            ar_loss = criterion_reg(ar_pred, arousal)
            
            # Calculate losses for auxiliary outputs
            aux_cls_loss = criterion_cls(aux_logits, labels)
            aux_val_loss = criterion_reg(aux_val, valence)
            aux_ar_loss = criterion_reg(aux_ar, arousal)
        
        # Combined loss with deep supervision (main + auxiliary outputs)
        loss = (cls_loss + 0.2 * val_loss + 0.2 * ar_loss) + 0.4 * (aux_cls_loss + 0.2 * aux_val_loss + 0.2 * aux_ar_loss)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        # For accuracy calculation, we use the emotion classification output
        _, predicted = logits.max(1)
        total += labels.size(0)
        
        if use_mixup:
            # Approximate accuracy calculation with mixup
            correct += (lam * predicted.eq(labels_a).sum().float() 
                      + (1 - lam) * predicted.eq(labels_b).sum().float())
        else:
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (i + 1), 
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total

# Validation function - supports both model variants
def validate(model, val_loader, criterion_cls, criterion_reg, device, model_type='original'):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, valence, arousal in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            valence = valence.to(device)
            arousal = arousal.to(device)
            
            # Forward pass - handle both model types
            if model_type == 'original':
                logits, val_pred, ar_pred = model(images)
                
                # Losses
                cls_loss = criterion_cls(logits, labels)
                val_loss_val = criterion_reg(val_pred, valence)
                ar_loss = criterion_reg(ar_pred, arousal)
                
                # Combined loss
                loss = cls_loss + 0.2 * val_loss_val + 0.2 * ar_loss
                
            else:  # enhanced model
                logits, val_pred, ar_pred = model(images)  # In eval mode, only returns main outputs
                
                # Losses for main outputs
                cls_loss = criterion_cls(logits, labels)
                val_loss_val = criterion_reg(val_pred, valence)
                ar_loss = criterion_reg(ar_pred, arousal)
                
                # Combined loss (no auxiliary outputs during evaluation)
                loss = cls_loss + 0.2 * val_loss_val + 0.2 * ar_loss
            
            val_loss += loss.item()
            
            # Accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Save predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=EMOTIONS)
    
    return val_loss / len(val_loader), 100. * correct / total, cm, report

def main():
    parser = argparse.ArgumentParser(description='Train Facial Emotion Recognition Models')
    parser.add_argument('--train_dir', type=str, default='./extracted/emotion/train', help='Path to training data')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--model', type=str, choices=['original', 'enhanced'], default='enhanced', 
                        help='Model to train: original ResEmoteNet or enhanced version')
    parser.add_argument('--focal_loss', action='store_true', help='Use focal loss for class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Data augmentation and normalization - enhanced to match pretrained model expectations
    # More balanced augmentation - not as aggressive
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Reduced rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Less aggressive scaling
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Less aggressive jitter 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # ImageNet stats for grayscale
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08))  # Reduced erasing
    ])
    
    # Add stronger augmentation for challenging emotions
    strong_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),  # More rotation for difficult classes
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # More aggressive scaling
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),  # More aggressive jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # More aggressive erasing
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet stats for grayscale
    ])
    
    # Create emotion-specific transforms for challenging emotions
    emotion_specific_transforms = {}
    for emotion in ['fear', 'disgust', 'sad']:
        emotion_specific_transforms[emotion] = strong_transform
    
    # Load datasets
    train_dataset = FERDataset(
        root_dir=args.train_dir, 
        mode='train', 
        transform=train_transform,
        emotion_specific_transforms=emotion_specific_transforms
    )
    test_dataset = FERDataset(root_dir=args.test_dir, mode='test', transform=test_transform)
    
    # Print dataset distribution for debugging
    class_counts = [0] * len(EMOTIONS)
    for _, label, _, _ in train_dataset:
        class_counts[label] += 1
    
    print(f'Class distribution: {class_counts}')
    print(f'Total training samples: {len(train_dataset)}')
    
    # Calculate sample weights for balanced sampling - highlight challenging classes
    sample_weights = [0] * len(train_dataset)
    
    # Calculate inverse frequency weights
    class_weights_list = [1.0/max(count, 1) for count in class_counts]
    
    # Give extra weight to challenging emotions (fear and sad based on logs)
    emotion_multipliers = {
        'fear': 2.0,  # Boost fear class which had low recall
        'sad': 1.8,   # Boost sad class which also performed poorly
        'disgust': 2.2 # Very few samples, needs strong boosting
    }
    
    for idx, emotion in enumerate(EMOTIONS):
        if emotion in emotion_multipliers:
            class_weights_list[idx] *= emotion_multipliers[emotion]
    
    # Normalize weights
    sum_weights = sum(class_weights_list)
    class_weights_list = [w/sum_weights * len(class_weights_list) for w in class_weights_list]
    
    print("Class weights for sampling:")
    for i, emotion in enumerate(EMOTIONS):
        print(f"  {emotion}: {class_weights_list[i]:.4f}")
    
    # Create class weights tensor for loss function
    if args.focal_loss:
        # For focal loss, we don't need class weights in the loss
        class_loss_weights = None
    else:
        # Use sqrt of class weights to avoid too aggressive weighting
        class_loss_weights = torch.tensor([np.sqrt(w) for w in class_weights_list], dtype=torch.float32, device=device)
    
    # Assign weights to samples
    for idx, (img_path, label) in enumerate(train_dataset.samples):
        sample_weights[idx] = class_weights_list[label]
    
    # Create WeightedRandomSampler to balance classes with focus on challenging ones
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders with weighted sampler for training
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize the selected model
    print(f'Initializing {args.model} model...')
    if args.model == 'original':
        model = ResEmoteNet(num_classes=len(EMOTIONS)).to(device)
    else:  # enhanced
        model = EnhancedResEmoteNet(num_classes=len(EMOTIONS), dropout_rate=0.65).to(device)  # Increased dropout rate
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Define loss functions
    # Use label smoothing for classification to prevent overconfidence
    if args.focal_loss:
        # Implement focal loss for better handling of class imbalance
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, label_smoothing=0.2):  # Increased label smoothing
                super(FocalLoss, self).__init__()
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                
            def forward(self, logits, targets):
                # Apply label smoothing
                num_classes = logits.size(1)
                smooth_targets = torch.zeros_like(logits).scatter_(
                    1, targets.unsqueeze(1), 1.0
                )
                if self.label_smoothing > 0:
                    smooth_targets = (1 - self.label_smoothing) * smooth_targets + \
                                    self.label_smoothing / num_classes
                
                # Calculate focal loss
                probs = F.softmax(logits, dim=1)
                probs_t = torch.gather(probs, 1, targets.unsqueeze(1))
                log_probs = F.log_softmax(logits, dim=1)
                
                # Focal term
                focal_term = (1 - probs_t) ** self.gamma
                
                # Loss calculation
                loss = -torch.mean(focal_term * torch.sum(smooth_targets * log_probs, dim=1))
                return loss
                
        criterion_cls = FocalLoss(gamma=args.focal_gamma, label_smoothing=0.2).to(device)  # Increased label smoothing
    else:
        # Standard cross entropy with class weights
        criterion_cls = nn.CrossEntropyLoss(weight=class_loss_weights, label_smoothing=0.2)  # Increased label smoothing
    
    criterion_reg = nn.SmoothL1Loss()  # Huber loss is more robust for regression
    
    # Define optimizer with different learning rates for different parts of the model
    if args.model == 'original':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)  # Increased weight decay
    else:
        # Define parameter groups with more granular control
        pretrained_params = []
        attention_params = []
        emotion_specific_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'base_model' in name:
                pretrained_params.append(param)
            elif 'attention' in name:
                attention_params.append(param)
            elif 'emotion_specific_paths' in name or 'fusion' in name:
                emotion_specific_params.append(param)
            else:
                classifier_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': 0.00001, 'weight_decay': 2e-5},  # Reduced LR, increased weight decay
            {'params': attention_params, 'lr': 0.0001, 'weight_decay': 2e-4},    # Reduced LR, increased weight decay 
            {'params': emotion_specific_params, 'lr': 0.0003, 'weight_decay': 3e-4},  # Reduced LR, increased weight decay
            {'params': classifier_params, 'lr': 0.0005, 'weight_decay': 3e-4}     # Reduced LR, increased weight decay
        ])
    
    # Better learning rate scheduler - using cosine annealing with restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Mixup function for data augmentation
    def mixup_data_advanced(x, y, alpha=0.2):
        '''Returns mixed inputs and targets with dynamic alpha'''
        batch_size = x.size()[0]
        
        # Use a slightly stronger alpha value to prevent overfitting
        alpha_value = min(0.4, alpha * (1 + epoch/10))  # Gradually increase alpha up to 0.4
        
        # Dynamically adjust alpha based on epoch
        lam = np.random.beta(alpha_value, alpha_value)
        
        # Generate permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix data
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    # Training function for the transfer learning model
    def train_transfer_model(model, train_loader, optimizer, criterion_cls, criterion_reg, device, epoch, use_mixup=True):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Track per-class performance
        class_correct = torch.zeros(len(EMOTIONS), device=device)
        class_total = torch.zeros(len(EMOTIONS), device=device)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for i, (images, labels, valence, arousal) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            valence = valence.to(device)
            arousal = arousal.to(device)
            
            # Apply mixup if enabled (but only after a few epochs of regular training)
            if use_mixup and epoch > 3:
                images, labels_a, labels_b, lam = mixup_data_advanced(images, labels, alpha=0.2)
                
                # Forward pass
                outputs = model(images)
                logits, val_pred, ar_pred = outputs[0], outputs[1], outputs[2]
                
                if len(outputs) > 3:  # If we have auxiliary outputs
                    aux_logits, aux_val, aux_ar = outputs[3], outputs[4], outputs[5]
                    
                    # Calculate losses with mixup
                    cls_loss = lam * criterion_cls(logits, labels_a) + (1 - lam) * criterion_cls(logits, labels_b)
                    aux_cls_loss = lam * criterion_cls(aux_logits, labels_a) + (1 - lam) * criterion_cls(aux_logits, labels_b)
                    
                    # Regression losses (simplified - not using mixup for regression targets)
                    val_loss = criterion_reg(val_pred, valence)
                    ar_loss = criterion_reg(ar_pred, arousal)
                    aux_val_loss = criterion_reg(aux_val, valence)
                    aux_ar_loss = criterion_reg(aux_ar, arousal)
                    
                    # Combined loss (emphasis on classification)
                    loss = cls_loss + 0.1 * val_loss + 0.1 * ar_loss + 0.4 * aux_cls_loss + 0.05 * aux_val_loss + 0.05 * aux_ar_loss
                else:
                    # Main loss with mixup
                    cls_loss = lam * criterion_cls(logits, labels_a) + (1 - lam) * criterion_cls(logits, labels_b)
                    val_loss = criterion_reg(val_pred, valence)
                    ar_loss = criterion_reg(ar_pred, arousal)
                    
                    # Combined loss
                    loss = cls_loss + 0.1 * val_loss + 0.1 * ar_loss
            else:
                # Standard forward pass without mixup
                outputs = model(images)
                logits, val_pred, ar_pred = outputs[0], outputs[1], outputs[2]
                
                if len(outputs) > 3:  # If we have auxiliary outputs
                    aux_logits, aux_val, aux_ar = outputs[3], outputs[4], outputs[5]
                    
                    # Calculate losses
                    cls_loss = criterion_cls(logits, labels)
                    val_loss = criterion_reg(val_pred, valence)
                    ar_loss = criterion_reg(ar_pred, arousal)
                    
                    aux_cls_loss = criterion_cls(aux_logits, labels)
                    aux_val_loss = criterion_reg(aux_val, valence)
                    aux_ar_loss = criterion_reg(aux_ar, arousal)
                    
                    # Combined loss (emphasis on classification)
                    loss = cls_loss + 0.1 * val_loss + 0.1 * ar_loss + 0.4 * aux_cls_loss + 0.05 * aux_val_loss + 0.05 * aux_ar_loss
                else:
                    # Calculate losses without auxiliary outputs
                    cls_loss = criterion_cls(logits, labels)
                    val_loss = criterion_reg(val_pred, valence)
                    ar_loss = criterion_reg(ar_pred, arousal)
                    
                    # Combined loss
                    loss = cls_loss + 0.1 * val_loss + 0.1 * ar_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            # For accuracy calculation (no mixup for accuracy calculation)
            _, predicted = logits.max(1)
            total += labels.size(0)
            
            if use_mixup and epoch > 3:
                # Approximate accuracy calculation with mixup
                correct += (lam * predicted.eq(labels_a).sum().float() + (1 - lam) * predicted.eq(labels_b).sum().float())
                
                # Per-class tracking is less accurate with mixup, but still informative
                for label_idx in range(len(EMOTIONS)):
                    mask_a = (labels_a == label_idx)
                    mask_b = (labels_b == label_idx)
                    class_total[label_idx] += lam * mask_a.sum().float() + (1 - lam) * mask_b.sum().float()
                    class_correct[label_idx] += lam * predicted[mask_a].eq(label_idx).sum().float() + \
                                               (1 - lam) * predicted[mask_b].eq(label_idx).sum().float()
            else:
                correct += predicted.eq(labels).sum().item()
                
                # Track per-class performance
                for label_idx in range(len(EMOTIONS)):
                    mask = (labels == label_idx)
                    class_total[label_idx] += mask.sum().item()
                    class_correct[label_idx] += predicted[mask].eq(label_idx).sum().item()
            
            # Update progress bar
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            progress_bar.set_postfix({
                'loss': running_loss / (i + 1), 
                'acc': 100. * correct / total,
                'lr': f"{current_lrs[0]:.6f}/{current_lrs[-1]:.6f}"
            })
        
        # Calculate per-class accuracy
        class_accuracy = 100. * class_correct / class_total.clamp(min=1)
        print("\nPer-class training accuracy:")
        for i, emotion in enumerate(EMOTIONS):
            print(f"  {emotion}: {class_accuracy[i].item():.2f}%")
        
        return running_loss / len(train_loader), 100. * correct / total
    
    # Update validate function to return per-class metrics
    def validate(model, val_loader, criterion_cls, criterion_reg, device, model_type='original'):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        # Track per-class performance
        class_correct = torch.zeros(len(EMOTIONS), device=device)
        class_total = torch.zeros(len(EMOTIONS), device=device)
        
        with torch.no_grad():
            for images, labels, valence, arousal in tqdm(val_loader, desc='Validating'):
                images = images.to(device)
                labels = labels.to(device)
                valence = valence.to(device)
                arousal = arousal.to(device)
                
                # Forward pass - handle both model types
                if model_type == 'original':
                    logits, val_pred, ar_pred = model(images)
                    
                    # Losses
                    cls_loss = criterion_cls(logits, labels)
                    val_loss_val = criterion_reg(val_pred, valence)
                    ar_loss = criterion_reg(ar_pred, arousal)
                    
                    # Combined loss
                    loss = cls_loss + 0.2 * val_loss_val + 0.2 * ar_loss
                    
                else:  # enhanced model
                    logits, val_pred, ar_pred = model(images)  # In eval mode, only returns main outputs
                    
                    # Losses for main outputs
                    cls_loss = criterion_cls(logits, labels)
                    val_loss_val = criterion_reg(val_pred, valence)
                    ar_loss = criterion_reg(ar_pred, arousal)
                    
                    # Combined loss (no auxiliary outputs during evaluation)
                    loss = cls_loss + 0.2 * val_loss_val + 0.2 * ar_loss
                
                val_loss += loss.item()
                
                # Accuracy
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Track per-class performance
                for label_idx in range(len(EMOTIONS)):
                    mask = (labels == label_idx)
                    class_total[label_idx] += mask.sum().item()
                    class_correct[label_idx] += predicted[mask].eq(label_idx).sum().item()
                
                # Save predictions and labels for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=EMOTIONS)
        
        # Calculate per-class accuracy
        class_accuracy = 100. * class_correct / class_total.clamp(min=1)
        print("\nPer-class validation accuracy:")
        for i, emotion in enumerate(EMOTIONS):
            print(f"  {emotion}: {class_accuracy[i].item():.2f}%")
        
        return val_loss / len(val_loader), 100. * correct / total, cm, report, class_accuracy
    
    # Training loop
    best_acc = 0.0
    patience = 7  # Reduced patience for earlier stopping
    patience_counter = 0
    best_epoch = 0
    
    # Use a more aggressive early stopping based on train-val gap
    max_train_val_gap = 7.0  # Maximum acceptable train-val accuracy gap
    
    # Initialize lists to track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    emotion_accs = {emotion: [] for emotion in EMOTIONS}  # Track per-emotion accuracy
    
    for epoch in range(args.epochs):
        # Train
        print(f'Training epoch {epoch+1}/{args.epochs}...')
        if args.model == 'original':
            train_loss, train_acc = train_original(model, train_loader, optimizer, criterion_cls, criterion_reg, 
                                                  device, epoch, use_mixup=args.use_mixup)
        else:  # enhanced
            train_loss, train_acc = train_transfer_model(model, train_loader, optimizer, criterion_cls, criterion_reg, 
                                                  device, epoch, use_mixup=args.use_mixup)
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Validate
        print('Validating...')
        val_loss, val_acc, confusion_mat, class_report, class_accuracy = validate(model, test_loader, criterion_cls, 
                                                                criterion_reg, device, model_type=args.model)
        
        # Keep track of metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Track per-emotion accuracy
        for i, emotion in enumerate(EMOTIONS):
            emotion_accs[emotion].append(class_accuracy[i].item())
        
        # Track class predictions to catch class-switching issue
        val_preds = []
        val_targets = []
        model.eval()
        with torch.no_grad():
            for images, labels, _, _ in test_loader:
                images = images.to(device)
                outputs = model(images)[0]  # First output is logits
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.numpy())
        
        # Count predictions per class
        pred_counts = [0] * len(EMOTIONS)
        for pred in val_preds:
            pred_counts[pred] += 1
        
        print("Predictions per class:")
        for i, emotion in enumerate(EMOTIONS):
            print(f"  {emotion}: {pred_counts[i]} ({pred_counts[i]/len(val_preds)*100:.1f}%)")
        
        # Get current learning rates
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        print(f'Current learning rates: {", ".join([f"{lr:.7f}" for lr in current_lrs])}')
        
        print(f'Epoch: {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('\nClassification Report:')
        print(class_report)
        
        # Check for overfitting
        train_val_gap = train_acc - val_acc
        print(f'Train-Val Accuracy Gap: {train_val_gap:.2f}%')
        
        # Early stopping based on train-val gap
        if train_val_gap > max_train_val_gap and epoch > 15:
            print(f"Early stopping triggered due to large train-validation gap: {train_val_gap:.2f}%")
            print(f"This suggests overfitting. Best validation metric: {best_acc:.2f}% at epoch {best_epoch+1}")
            break
        
        # Calculate per-class F1 scores
        classification_dict = classification_report(val_targets, val_preds, 
                                                  target_names=EMOTIONS, 
                                                  output_dict=True)
        
        # Save best model based on validation accuracy or balanced metrics
        # Weight challenging emotions more heavily in the decision
        weighted_acc = val_acc * 0.5  # Base validation accuracy weight
        
        # Add weighted contributions from challenging emotions
        for emotion in ['fear', 'disgust', 'sad']:
            emotion_idx = EMOTIONS.index(emotion)
            # No need to check class_total - class_accuracy already handles empty classes
            # Simply add the per-class accuracy, which will be 0 if no samples exist
            weighted_acc += class_accuracy[emotion_idx].item() * 0.5 / 3  # Split remaining 50% equally
        
        print(f'Weighted validation metric: {weighted_acc:.2f}%')
        
        if weighted_acc > best_acc:
            best_acc = weighted_acc
            best_epoch = epoch
            patience_counter = 0  # Reset counter
            
            # Save full model state
            torch.save({
                'epoch': epoch,
                'model_type': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'class_report': classification_dict,
                'per_class_accuracy': {EMOTIONS[i]: class_accuracy[i].item() for i in range(len(EMOTIONS))}
            }, os.path.join(args.save_dir, f'best_{args.model}_model.pth'))
            print(f'Best model saved with weighted metric: {weighted_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{patience} (best: {best_acc:.2f}% at epoch {best_epoch+1})')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_type': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'per_class_accuracy': {EMOTIONS[i]: class_accuracy[i].item() for i in range(len(EMOTIONS))}
            }, os.path.join(args.save_dir, f'{args.model}_checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping based on patience
        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            print(f'Best validation metric: {best_acc:.2f}% at epoch {best_epoch+1}')
            break
        
        # Apply L2 regularization adjustment based on training-validation gap
        if train_val_gap > max_train_val_gap and epoch > 10:  # If gap is too large, increase regularization
            print("Increasing L2 regularization to combat overfitting")
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = param_group['weight_decay'] * 1.2  # Increase weight decay by 20%
    
    # Final results
    print(f'Training completed! Best validation metric: {best_acc:.2f}% at epoch {best_epoch+1}')
    
    # Plot training curves
    try:
        epochs_range = range(1, len(train_losses) + 1)
        
        # Plot overall training metrics
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # Plot difficult emotion accuracies
        plt.subplot(2, 2, 3)
        for emotion in ['fear', 'disgust', 'sad']:
            plt.plot(epochs_range, emotion_accs[emotion], label=f'{emotion}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Difficult Emotion Accuracies')
        plt.legend()
        
        # Plot other emotion accuracies
        plt.subplot(2, 2, 4)
        for emotion in ['happy', 'angry', 'surprise', 'neutral']:
            plt.plot(epochs_range, emotion_accs[emotion], label=f'{emotion}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Other Emotion Accuracies')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
        print(f"Training curves saved to {os.path.join(args.save_dir, 'training_curves.png')}")
    except Exception as e:
        print(f"Failed to plot training curves: {e}")

if __name__ == '__main__':
    main() 