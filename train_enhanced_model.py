import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import cv2
import albumentations as A
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

# Import our enhanced model architecture
from enhanced_resemotenet import (
    EnhancedResEmoteNet, 
    ClassBalancedFocalLoss,
    CosineAnnealingWarmupRestarts
)

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define emotion-specific transformations
class EmotionSpecificTransforms:
    """Applies different augmentations based on emotion class"""
    def __init__(self, base_transforms, emotion_specific_transforms=None):
        self.base_transforms = base_transforms
        self.emotion_specific_transforms = emotion_specific_transforms or {}
        
    def __call__(self, image, emotion=None):
        # Apply base transforms first
        image = self.base_transforms(image)
        
        # Apply emotion-specific transforms if available
        if emotion is not None and emotion in self.emotion_specific_transforms:
            image = self.emotion_specific_transforms[emotion](image)
            
        return image

# Custom FER dataset that loads from directories
class FERDirectoryDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, augment_minority=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment_minority = augment_minority
        self.samples = []
        
        # Map of emotion names to indices
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        
        # Load images and labels from directory
        for emotion in EMOTIONS:
            emotion_dir = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_name)
                        emotion_idx = self.emotion_to_idx[emotion]
                        self.samples.append((img_path, emotion_idx))
        
        # Calculate class distribution
        self.labels = [label for _, label in self.samples]
        self.class_counts = Counter(self.labels)
        print(f"{split} set class distribution:", self.class_counts)
        
        # Set emotion-specific transformations for minority classes
        if augment_minority and split == 'train':
            self.emotion_transforms = {}
            avg_count = sum(self.class_counts.values()) / len(self.class_counts)
            max_count = max(self.class_counts.values())
            
            # Apply stronger augmentation to classes that need more help
            # Priority focus on angry (0) class which has 0% accuracy
            problematic_classes = {
                0: 5.0,  # angry - extreme priority
                6: 2.0,  # neutral - high priority
                3: 1.5,  # happy - medium priority
            }
            
            for emotion_idx, count in self.class_counts.items():
                # More aggressive augmentation for minority and problematic classes
                is_problematic = emotion_idx in problematic_classes
                ratio = max_count / count if count < max_count else 1.0
                
                # Extra boost for problematic classes with specific weights
                if is_problematic:
                    ratio *= problematic_classes[emotion_idx]
                
                # Cap strength but make it stronger for problematic classes
                base_strength = 0.5 if is_problematic else 0.4
                strength = min(0.95, base_strength + (ratio - 1) * 0.15)
                
                # Specific ultra-strong augmentation for angry class
                if emotion_idx == 0:  # angry class
                    self.emotion_transforms[emotion_idx] = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.25, rotate_limit=40, p=0.95),
                        A.OneOf([
                            A.GaussNoise(var_limit=(10, 80)),
                            A.GaussianBlur(blur_limit=3),
                            A.MotionBlur(blur_limit=5),
                        ], p=0.8),
                        A.OneOf([
                            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30),
                        ], p=0.8),
                        A.OneOf([
                            A.ElasticTransform(alpha=120, sigma=120 * 0.08, alpha_affine=120 * 0.08),
                            A.GridDistortion(num_steps=5, distort_limit=0.5),
                            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.4),
                        ], p=0.8),
                        # Add special transformations just for angry
                        A.RandomShadow(p=0.3),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
                    ], p=1.0)
                # Even low-count classes should get strong augmentation
                elif ratio > 1.1 or is_problematic:
                    self.emotion_transforms[emotion_idx] = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=30, p=strength),
                        A.OneOf([
                            A.GaussNoise(var_limit=(10, 60)),
                            A.GaussianBlur(blur_limit=3),
                            A.MotionBlur(blur_limit=3),
                        ], p=strength),
                        A.OneOf([
                            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35),
                            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                        ], p=strength),
                        A.OneOf([
                            A.ElasticTransform(alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.05),
                            A.GridDistortion(num_steps=5, distort_limit=0.4),
                            A.OpticalDistortion(distort_limit=0.4, shift_limit=0.3),
                        ], p=strength * 0.7),
                    ], p=1.0)
                    
            # Oversample severely underrepresented or problematic classes
            if split == 'train':
                new_samples = []
                # Special treatment for angry class (extreme oversampling)
                angry_samples = [(path, label) for path, label in self.samples if label == 0]
                if angry_samples:
                    # Make angry samples at least 80% as frequent as the majority class
                    angry_count = self.class_counts.get(0, 0)
                    # Much higher oversampling factor for angry
                    angry_oversampling = max(int(max_count * 0.8 / angry_count), 1) if angry_count > 0 else 5
                    print(f"Extreme oversampling for angry class by factor {angry_oversampling}")
                    new_samples.extend(angry_samples * (angry_oversampling - 1))
                
                # Handle other problematic classes
                for emotion_idx, priority in problematic_classes.items():
                    if emotion_idx != 0 and emotion_idx in self.class_counts:  # Skip angry (already handled)
                        emotion_samples = [(path, label) for path, label in self.samples if label == emotion_idx]
                        # Base oversampling on priority and class count
                        count = self.class_counts[emotion_idx]
                        # Higher oversampling factor based on priority
                        oversampling_factor = min(int(max_count / count * priority * 0.5), 4)
                        if oversampling_factor > 1:
                            print(f"Oversampling class {EMOTIONS[emotion_idx]} by factor {oversampling_factor}")
                            new_samples.extend(emotion_samples * (oversampling_factor - 1))
                
                self.samples.extend(new_samples)
                # Update labels after oversampling
                self.labels = [label for _, label in self.samples]
                self.class_counts = Counter(self.labels)
                print(f"After oversampling, {split} set class distribution:", self.class_counts)
        else:
            self.emotion_transforms = None

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path, emotion_idx = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = np.array(img)
        
        # Resize if needed
        if img.shape[0] != 48 or img.shape[1] != 48:
            img = cv2.resize(img, (48, 48))
        
        # Apply emotion-specific transforms if available
        if self.emotion_transforms is not None and emotion_idx in self.emotion_transforms:
            transformed = self.emotion_transforms[emotion_idx](image=img)
            img = transformed['image']
        
        # Apply general transforms
        if self.transform:
            img = self.transform(img)
        
        return img, emotion_idx

def get_weighted_sampler(dataset):
    """Create a weighted sampler to handle class imbalance"""
    targets = dataset.labels
    class_counts = np.bincount(targets)
    
    # Calculate class weights (inverse frequency)
    class_weights = 1.0 / class_counts
    weights = class_weights[targets]
    
    # Scale weights more aggressively to counter extreme imbalance
    weights = np.power(weights, 1.8)  # More aggressive power for better balancing
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    # Create sampler
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def create_dataloaders(args):
    """Create train and validation data loaders with appropriate augmentations"""
    # Base transformations for all images
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create datasets from directories
    train_dataset = FERDirectoryDataset(
        root_dir=args.train_dir,
        split='train',
        transform=train_transform,
        augment_minority=True
    )
    
    val_dataset = FERDirectoryDataset(
        root_dir=args.test_dir,
        split='val',
        transform=val_transform,
        augment_minority=False
    )
    
    # Create samplers for handling imbalance
    if args.use_weighted_sampling:
        train_sampler = get_weighted_sampler(train_dataset)
        train_shuffle = False  # Don't shuffle when using weighted sampler
    else:
        train_sampler = None
        train_shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

# Fix for the classification report warning about precision
def print_classification_report(targets, predictions, class_names):
    """Print classification report with warning suppression"""
    try:
        report = classification_report(
            targets, predictions, 
            target_names=class_names,
            digits=2,
            zero_division=0  # Set zero_division to 0 to avoid warnings
        )
        print("Classification Report:")
        print(report)
    except Exception as e:
        print(f"Failed to generate classification report: {e}")

# Add a function to extract engineered features from images
def extract_features(image_array, size=(48, 48)):
    """Extract handcrafted features from images for traditional ML models"""
    if image_array.shape != size:
        image_array = cv2.resize(image_array, size)
    
    # Ensure grayscale
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image_array
    
    # Normalize image
    gray_img = gray_img.astype(np.float32) / 255.0
    
    # 1. HOG features
    hog_features = hog(gray_img, orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    
    # 2. LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
    
    # 3. Simple histogram features
    hist, _ = np.histogram(gray_img.ravel(), bins=32, range=(0, 1), density=True)
    
    # 4. Edge features using Canny
    edges = cv2.Canny((gray_img * 255).astype(np.uint8), 100, 200)
    edge_hist, _ = np.histogram(edges.ravel(), bins=16, range=(0, 256), density=True)
    
    # 5. Regions of interest - expressions often concentrate around eyes, mouth
    # Extract features from top half (eyes region) and bottom half (mouth region)
    h, w = gray_img.shape
    top_half = gray_img[:h//2, :]
    bottom_half = gray_img[h//2:, :]
    
    top_hist, _ = np.histogram(top_half.ravel(), bins=16, range=(0, 1), density=True)
    bottom_hist, _ = np.histogram(bottom_half.ravel(), bins=16, range=(0, 1), density=True)
    
    # 6. Simple statistics for each region
    regions = [gray_img, top_half, bottom_half]
    stats = []
    for region in regions:
        stats.extend([
            np.mean(region),
            np.std(region),
            np.max(region) - np.min(region),  # Dynamic range
            np.median(region),
            np.percentile(region, 25),
            np.percentile(region, 75)
        ])
    
    # Combine all features
    combined_features = np.concatenate([
        hog_features,
        lbp_hist,
        hist,
        edge_hist,
        top_hist,
        bottom_hist,
        stats
    ])
    
    return combined_features

# Function to prepare data for traditional ML models
def prepare_traditional_dataset(dataset):
    """Extract features from images for training traditional ML models"""
    features = []
    labels = []
    
    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        img, label = dataset[i]
        
        # Convert PyTorch tensor to numpy array if needed
        if isinstance(img, torch.Tensor):
            # Convert from CHW to HWC format and denormalize
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 0.5) + 0.5  # Assuming normalization was done with [0.5]
            img = (img * 255).astype(np.uint8)
        
        # Extract features
        img_features = extract_features(img)
        features.append(img_features)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def train_traditional_models(args, train_dataset, val_dataset):
    """Train traditional ML models for specific emotion recognition"""
    # Extract features for traditional ML models
    print("Preparing data for traditional ML models...")
    X_train, y_train = prepare_traditional_dataset(train_dataset)
    X_val, y_val = prepare_traditional_dataset(val_dataset)
    
    # Models and results to save
    model_angry = None
    model_neutral = None
    best_angry_model_score = 0
    best_neutral_model_score = 0
    models_path = os.path.join(args.output_dir, 'traditional_models')
    os.makedirs(models_path, exist_ok=True)
    
    # First, train a specialized model for angry class detection
    print("Training specialized model for angry class detection...")
    angry_labels = (y_train == 0).astype(int)  # Binary: is_angry or not
    
    # Try multiple models and parameters for angry detection
    model_candidates = [
        ('rf1', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, class_weight='balanced')),
        ('rf2', RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=1, class_weight='balanced')),
        ('gb1', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, subsample=0.8)),
        ('gb2', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, subsample=0.7))
    ]
    
    for name, model in model_candidates:
        model.fit(X_train, angry_labels)
        # Evaluate with focus on recall (don't miss angry samples)
        y_pred = model.predict(X_val)
        val_angry_labels = (y_val == 0).astype(int)
        recall = np.sum((y_pred == 1) & (val_angry_labels == 1)) / max(np.sum(val_angry_labels == 1), 1)
        precision = np.sum((y_pred == 1) & (val_angry_labels == 1)) / max(np.sum(y_pred == 1), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        # Weight recall more heavily
        score = 0.7 * recall + 0.3 * precision
        
        print(f"Angry model {name}: Recall = {recall:.4f}, Precision = {precision:.4f}, F1 = {f1:.4f}, Score = {score:.4f}")
        
        if score > best_angry_model_score:
            best_angry_model_score = score
            model_angry = model
            joblib.dump(model, os.path.join(models_path, 'angry_classifier.joblib'))
    
    # Second, train a specialized model for neutral class detection
    print("Training specialized model for neutral class detection...")
    neutral_labels = (y_train == 6).astype(int)  # Binary: is_neutral or not
    
    # Try multiple models and parameters for neutral detection
    for name, model in model_candidates:
        model.fit(X_train, neutral_labels)
        # Evaluate
        y_pred = model.predict(X_val)
        val_neutral_labels = (y_val == 6).astype(int)
        recall = np.sum((y_pred == 1) & (val_neutral_labels == 1)) / max(np.sum(val_neutral_labels == 1), 1)
        precision = np.sum((y_pred == 1) & (val_neutral_labels == 1)) / max(np.sum(y_pred == 1), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        score = 0.7 * recall + 0.3 * precision
        
        print(f"Neutral model {name}: Recall = {recall:.4f}, Precision = {precision:.4f}, F1 = {f1:.4f}, Score = {score:.4f}")
        
        if score > best_neutral_model_score:
            best_neutral_model_score = score
            model_neutral = model
            joblib.dump(model, os.path.join(models_path, 'neutral_classifier.joblib'))
    
    return model_angry, model_neutral

# Add function for hybrid prediction at the end of training
def add_hybrid_predict_function(model, model_angry, model_neutral):
    """Attach hybrid prediction function to the model for inference"""
    def hybrid_predict(images, device='cuda:0'):
        """Make predictions using both CNN and traditional ML models"""
        # Make CNN predictions first
        if not isinstance(images, torch.Tensor):
            # Convert to tensor if not already
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            if isinstance(images, list):
                tensor_images = torch.stack([transform(img) for img in images])
            else:
                tensor_images = transform(images).unsqueeze(0)
        else:
            tensor_images = images
            
        tensor_images = tensor_images.to(device)
        
        # Get CNN predictions
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(tensor_images)
            probs = F.softmax(logits, dim=1)
            cnn_preds = probs.argmax(dim=1).cpu().numpy()
            confidence = probs.max(dim=1)[0].cpu().numpy()
        
        # Extract traditional features
        final_preds = []
        for i in range(len(tensor_images)):
            img = tensor_images[i]
            # Convert back to numpy for feature extraction
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 0.5) + 0.5  # Denormalize
            img_np = (img_np * 255).astype(np.uint8)
            
            # Extract features for ML models
            features = extract_features(img_np)
            
            # Get prediction from specialized angry model
            is_angry = model_angry.predict([features])[0] == 1
            angry_prob = model_angry.predict_proba([features])[0][1]
            
            # Get prediction from specialized neutral model
            is_neutral = model_neutral.predict([features])[0] == 1
            neutral_prob = model_neutral.predict_proba([features])[0][1]
            
            # Make final prediction using ensemble logic
            cnn_pred = cnn_preds[i]
            conf = confidence[i]
            
            # Decision rules:
            # 1. If CNN is very confident (>0.9), trust it
            if conf > 0.9:
                final_pred = cnn_pred
            # 2. If angry model is confident, override CNN
            elif is_angry and angry_prob > 0.7:
                final_pred = 0  # angry class
            # 3. If neutral model is confident, override CNN
            elif is_neutral and neutral_prob > 0.7:
                final_pred = 6  # neutral class
            # 4. If CNN predicted not angry/neutral but ML models disagree with high confidence
            elif cnn_pred != 0 and is_angry and angry_prob > 0.85:
                final_pred = 0  # angry class
            elif cnn_pred != 6 and is_neutral and neutral_prob > 0.85:
                final_pred = 6  # neutral class
            # 5. Otherwise trust CNN
            else:
                final_pred = cnn_pred
                
            final_preds.append(final_pred)
            
        return np.array(final_preds)
    
    # Attach the function to the model
    model.hybrid_predict = hybrid_predict
    return model

def train_model(args):
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Set up device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(args)
    
    # Calculate class distribution for loss weighting
    train_class_counts = np.array([train_dataset.class_counts[i] for i in range(len(EMOTIONS))])
    print(f"Train class counts: {train_class_counts}")
    
    # Create CNN model
    model = EnhancedResEmoteNet(
        num_classes=len(EMOTIONS),
        dropout_rate=args.dropout_rate,
        backbone=args.backbone,
        use_fpn=args.use_fpn,
        use_landmarks=args.use_landmarks,
        use_contrastive=args.use_contrastive
    )
    
    # Initialize model weights, move to device
    model = model.to(device)
    
    # Train traditional ML models for specialized binary classification of problematic classes
    model_angry, model_neutral = train_traditional_models(args, train_dataset, val_dataset)
    
    # Create special angry-vs-rest binary dataset for pre-training
    if train_dataset.class_counts.get(0, 0) > 0:  # Only if we have angry samples
        print("Starting specialized angry-class pre-training...")
        
        # Set up binary classifier for angry vs rest
        model.pre_train_angry_classifier = nn.Linear(512, 2).to(device)
        
        # Create binary angry-vs-rest criterion
        angry_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))
        
        # Create angry-specialized optimizer with higher learning rate
        angry_optimizer = optim.Adam([
            {'params': model.stem.parameters(), 'lr': args.lr * 0.1},
            {'params': model.stage1.parameters(), 'lr': args.lr * 0.1},
            {'params': model.stage2.parameters(), 'lr': args.lr * 0.2},
            {'params': model.stage3.parameters(), 'lr': args.lr * 0.5},
            {'params': model.stage4.parameters(), 'lr': args.lr * 1.0},
            {'params': model.pre_train_angry_classifier.parameters(), 'lr': args.lr * 2.0}
        ], lr=args.lr, weight_decay=args.weight_decay)
        
        # Pre-training loop for angry class
        angry_epochs = 15  # Number of epochs for angry pre-training
        for epoch in range(angry_epochs):
            model.train()
            angry_correct = 0
            angry_total = 0
            angry_loss = 0.0
            
            print(f"Angry pre-training epoch {epoch+1}/{angry_epochs}")
            for images, labels in tqdm(train_loader, desc=f"Angry pre-train {epoch+1}"):
                images = images.to(device)
                
                # Convert to binary classification (0=angry, 1=not angry)
                binary_labels = (labels != 0).long().to(device)
                
                # Forward pass using model's feature extractor
                with torch.set_grad_enabled(True):
                    # Extract features
                    x = model.stem(images)
                    x1 = model.stage1(x)
                    x2 = model.stage2(x1)
                    x3 = model.stage3(x2)
                    x4 = model.stage4(x3)
                    
                    # Global pooling
                    x = model.global_pool(x4).view(x4.size(0), -1)
                    
                    # Binary classification
                    binary_logits = model.pre_train_angry_classifier(x)
                    loss = angry_criterion(binary_logits, binary_labels)
                    
                    # Backward and optimize
                    angry_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    angry_optimizer.step()
                    
                    # Compute accuracy
                    _, predicted = binary_logits.max(1)
                    angry_total += binary_labels.size(0)
                    angry_correct += predicted.eq(binary_labels).sum().item()
                    angry_loss += loss.item()
            
            # Calculate epoch metrics
            epoch_acc = 100 * angry_correct / angry_total
            epoch_loss = angry_loss / len(train_loader)
            
            # Evaluate on validation set
            model.eval()
            val_angry_correct = 0
            val_angry_total = 0
            val_angry_loss = 0.0
            
            # Tracking angry class performance specifically
            angry_class_recall = 0
            angry_class_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    binary_labels = (labels != 0).long().to(device)
                    
                    # Extract features
                    x = model.stem(images)
                    x1 = model.stage1(x)
                    x2 = model.stage2(x1)
                    x3 = model.stage3(x2)
                    x4 = model.stage4(x3)
                    
                    # Global pooling
                    x = model.global_pool(x4).view(x4.size(0), -1)
                    
                    # Binary classification
                    binary_logits = model.pre_train_angry_classifier(x)
                    loss = angry_criterion(binary_logits, binary_labels)
                    
                    # Compute accuracy
                    _, predicted = binary_logits.max(1)
                    val_angry_total += binary_labels.size(0)
                    val_angry_correct += predicted.eq(binary_labels).sum().item()
                    val_angry_loss += loss.item()
                    
                    # Check if the model is recognizing angry faces (class 0)
                    angry_mask = (labels == 0).to(device)
                    if angry_mask.any():
                        angry_class_total += angry_mask.sum().item()
                        angry_class_recall += (predicted[angry_mask] == 0).sum().item()
            
            # Calculate validation metrics
            val_epoch_acc = 100 * val_angry_correct / val_angry_total
            val_epoch_loss = val_angry_loss / len(val_loader)
            angry_recall = 100 * angry_class_recall / max(angry_class_total, 1)
            
            print(f"Angry Pre-Train Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
            print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%, Angry Recall: {angry_recall:.2f}%")
        
        print("Angry pre-training complete. Now starting full model training...")
        # Remove the binary classifier after pre-training
        del model.pre_train_angry_classifier
        
        # Reset specific layers for fine-tuning
        # This helps remove any "forgetting" of angry class
        if args.backbone == 'resnet18':
            # Reset final stage to avoid overfitting to binary task
            for m in model.stage4.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    # Create criterion (loss function) for main training
    if args.use_focal_loss:
        criterion = ClassBalancedFocalLoss(
            num_classes=len(EMOTIONS),
            beta=0.99,  # Increase beta for stronger class balancing
            gamma=2.5,  # Increase gamma to focus more on hard examples
            samples_per_class=train_class_counts,
            extra_angry_weight=3.0  # Extra weight for angry class
        )
    else:
        # Use cross-entropy with class weights
        class_weights = 1.0 / torch.tensor(train_class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum() * len(train_class_counts)
        # Give extra weight to angry class
        class_weights[0] = class_weights[0] * 3.0
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer with weight decay
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                             momentum=0.9, nesterov=True, 
                             weight_decay=args.weight_decay)
    
    # Apply different learning rates to different parts of the model to focus on 
    # fine-tuning the final layers while preserving the knowledge in early layers
    param_groups = [
        {'params': model.stem.parameters(), 'lr': args.lr * 0.1},
        {'params': model.stage1.parameters(), 'lr': args.lr * 0.1},
        {'params': model.stage2.parameters(), 'lr': args.lr * 0.3},
        {'params': model.stage3.parameters(), 'lr': args.lr * 0.7},
        {'params': model.stage4.parameters(), 'lr': args.lr * 1.0},
        {'params': model.enhance_s3.parameters(), 'lr': args.lr * 1.5},
        {'params': model.enhance_s4.parameters(), 'lr': args.lr * 1.5},
        {'params': model.attn_s3.parameters(), 'lr': args.lr * 1.5},
        {'params': model.attn_s4.parameters(), 'lr': args.lr * 1.5},
        {'params': model.negative_path.parameters(), 'lr': args.lr * 2.0},  # Higher for emotion paths
        {'params': model.positive_path.parameters(), 'lr': args.lr * 2.0},
        {'params': model.neutral_path.parameters(), 'lr': args.lr * 2.0},
        {'params': model.common_path.parameters(), 'lr': args.lr * 2.0},
        {'params': model.emotion_classifier.parameters(), 'lr': args.lr * 2.0},
    ]
    
    # Create optimizer with parameter groups
    if args.optimizer == 'adam':
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    if args.use_cosine_scheduler:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.epochs // 2,
            cycle_mult=1.0,
            max_lr=args.lr,
            min_lr=args.lr * 0.01,
            warmup_steps=args.warmup_epochs,
            gamma=0.6  # Less aggressive LR reduction to prevent early convergence to local minima
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    # Training stats
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_metric = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, 'best_enhanced_model.pth')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track per-class metrics
    emotion_map = {i: emotion for i, emotion in enumerate(EMOTIONS)}
    
    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_class_correct = {i: 0 for i in range(len(EMOTIONS))}
        train_class_total = {i: 0 for i in range(len(EMOTIONS))}
        
        print(f"Training epoch {epoch+1}/{args.epochs}...")
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Different forward pass based on whether we're using all features
            if args.use_landmarks and args.use_contrastive:
                outputs, valence, arousal, aux_outputs, aux_valence, aux_arousal, landmarks, contrastive_loss = model(images, labels)
            elif args.use_landmarks:
                outputs, valence, arousal, aux_outputs, aux_valence, aux_arousal, landmarks = model(images, labels)
            else:
                outputs, valence, arousal, aux_outputs, aux_valence, aux_arousal = model(images, labels)
            
            # Calculate loss
            main_loss = criterion(outputs, labels)
            aux_loss = criterion(aux_outputs, labels) * 0.4  # Weight auxiliary loss
            
            # Combine all losses
            loss = main_loss + aux_loss
            
            # Add contrastive loss if enabled
            if args.use_contrastive and 'contrastive_loss' in locals():
                loss += contrastive_loss * args.contrastive_weight
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update training statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update per-class accuracy
            for i in range(len(EMOTIONS)):
                mask = (labels == i)
                train_class_correct[i] += predicted[mask].eq(labels[mask]).sum().item()
                train_class_total[i] += mask.sum().item()
            
            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1), 
                'acc': train_acc, 
                'lr': current_lr
            })
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Print per-class training accuracy
        print("Per-class training accuracy:")
        for i in range(len(EMOTIONS)):
            class_acc = 100.0 * train_class_correct[i] / max(train_class_total[i], 1)
            print(f"  {emotion_map[i]}: {class_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class_correct = {i: 0 for i in range(len(EMOTIONS))}
        val_class_total = {i: 0 for i in range(len(EMOTIONS))}
        val_preds = []
        val_targets = []
        class_predictions = {i: 0 for i in range(len(EMOTIONS))}
        
        print("Validating...")
        with torch.no_grad():
            val_pbar = tqdm(val_loader)
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass (no landmarks or contrastive in eval mode)
                outputs, _, _ = model(images)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update per-class accuracy
                for i in range(len(EMOTIONS)):
                    mask = (labels == i)
                    val_class_correct[i] += predicted[mask].eq(labels[mask]).sum().item()
                    val_class_total[i] += mask.sum().item()
                
                # Update prediction distribution
                for i in range(len(EMOTIONS)):
                    class_predictions[i] += (predicted == i).sum().item()
                
                # Save predictions and targets for classification report
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print per-class validation accuracy
        print("Per-class validation accuracy:")
        weighted_val_metric = 0
        for i in range(len(EMOTIONS)):
            class_acc = 100.0 * val_class_correct[i] / max(val_class_total[i], 1)
            print(f"  {emotion_map[i]}: {class_acc:.2f}%")
            
            # Weight minority classes more in the validation metric
            if i == 0:  # angry (priority class)
                weighted_val_metric += class_acc * 0.3
            elif i == 1:  # disgust
                weighted_val_metric += class_acc * 0.15
            elif i in [2, 4, 6]:  # fear, sad, neutral
                weighted_val_metric += class_acc * 0.15
            else:  # happy, surprise
                weighted_val_metric += class_acc * 0.10
        
        # Print prediction distribution
        print("Predictions per class:")
        for i in range(len(EMOTIONS)):
            pred_percent = 100.0 * class_predictions[i] / val_total
            print(f"  {emotion_map[i]}: {class_predictions[i]} ({pred_percent:.1f}%)")
        
        # Print current learning rates
        if hasattr(scheduler, 'get_lr'):
            print(f"Current learning rates: {', '.join([f'{lr:.7f}' for lr in scheduler.get_lr()])}")
        else:
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Print epoch summary
        print(f"Epoch: {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Print classification report using our custom function
        print_classification_report(val_targets, val_preds, EMOTIONS)
        
        # Calculate train-val gap (for overfitting detection)
        train_val_gap = train_acc - val_acc
        print(f"Train-Val Accuracy Gap: {train_val_gap:.2f}%")
        print(f"Weighted validation metric: {weighted_val_metric:.2f}%")
        
        # Update learning rate scheduler
        if args.use_cosine_scheduler:
            scheduler.step()
        else:
            scheduler.step(weighted_val_metric)
        
        # Save best model (using weighted validation metric)
        if weighted_val_metric > best_val_metric:
            best_val_metric = weighted_val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"Saved best model with weighted validation metric: {best_val_metric:.2f}%")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience} (best: {best_val_metric:.2f}% at epoch {epoch+1-patience_counter})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            print(f"Best validation metric: {best_val_metric:.2f}% at epoch {epoch+1-patience_counter}")
            break
    
    # Training completed
    print(f"Training completed! Best validation metric: {best_val_metric:.2f}% at epoch {epoch+1-patience_counter}")
    
    # Try to plot training curves
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    except Exception as e:
        print(f"Failed to plot training curves: {e}")
    
    print(f"[SUCCESS] enhanced model training completed!")
    print(f"Model saved to {best_model_path}")
    
    # After training, attach hybrid prediction function to the model
    print("Creating hybrid ensemble model combining CNN with traditional ML...")
    model = add_hybrid_predict_function(model, model_angry, model_neutral)
    
    # Save the full hybrid model 
    final_model_path = os.path.join(args.output_dir, 'hybrid_model.pth')
    torch.save({
        'cnn_state_dict': model.state_dict(),
        'angry_classifier': os.path.join(args.output_dir, 'traditional_models/angry_classifier.joblib'),
        'neutral_classifier': os.path.join(args.output_dir, 'traditional_models/neutral_classifier.joblib')
    }, final_model_path)
    
    print(f"Hybrid ensemble model saved to {final_model_path}")
    
    return model, best_val_metric

# Add function to load the hybrid model for inference
def load_hybrid_model(model_path, device='cuda:0'):
    """Load the hybrid ensemble model for inference"""
    # Load saved model data
    saved_data = torch.load(model_path, map_location=device)
    
    # Create and load CNN model
    model = EnhancedResEmoteNet(
        num_classes=len(EMOTIONS),
        dropout_rate=0.4,
        backbone='resnet18',
        use_fpn=True,
        use_landmarks=True,
        use_contrastive=True
    ).to(device)
    
    model.load_state_dict(saved_data['cnn_state_dict'])
    
    # Load traditional ML models
    model_angry = joblib.load(saved_data['angry_classifier'])
    model_neutral = joblib.load(saved_data['neutral_classifier'])
    
    # Add hybrid prediction function
    model = add_hybrid_predict_function(model, model_angry, model_neutral)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Emotion Recognition Model')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='./extracted/emotion/train', 
                        help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='./extracted/emotion/test',
                        help='Path to test/validation data directory')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save model and results')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                        help='Backbone architecture')
    parser.add_argument('--dropout_rate', type=float, default=0.65,
                        help='Dropout rate for model')
    parser.add_argument('--use_fpn', action='store_true', default=False,
                        help='Use Feature Pyramid Network')
    parser.add_argument('--use_landmarks', action='store_true', default=False,
                        help='Use facial landmark auxiliary task')
    parser.add_argument('--use_contrastive', action='store_true', default=False,
                        help='Use contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.2,
                        help='Weight for contrastive loss term')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use focal loss for class imbalance')
    parser.add_argument('--use_weighted_sampling', action='store_true', default=True,
                        help='Use weighted sampling for class imbalance')
    parser.add_argument('--use_cosine_scheduler', action='store_true', default=True,
                        help='Use cosine annealing scheduler with warmup')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    # Hardware parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Train the model
    model, best_metric = train_model(args)
    
    print("ResEmoteNet training pipeline completed successfully!")
    print("Model is ready for use with predict_emotion.py")

if __name__ == '__main__':
    main() 