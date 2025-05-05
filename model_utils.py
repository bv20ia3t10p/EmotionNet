from config import *
from utils import *
import torch  # type: ignore
from torchvision import transforms, datasets  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import numpy as np  # type: ignore
import random
import torch.nn.functional as F
import os
from model import ResEmoteNet
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import math
from collections import defaultdict


# Enhanced Mixup augmentation with better parameters
class MixupTransform:
    def __init__(self, alpha=0.4):  # Increased alpha for stronger mixing
        self.alpha = alpha
        
    def __call__(self, batch):
        """Apply mixup to the batch. Expected batch is tuple of (images, labels)"""
        images, labels = batch
        batch_size = len(images)
        
        # Generate mixup coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Create shuffled indices
        indices = torch.randperm(batch_size)
        
        # Mix the images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Return mixed images and pairs of labels with mixup coefficient
        return mixed_images, labels, labels[indices], lam


# CutMix augmentation for better generalization
class CutMixTransform:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        """Apply CutMix to the batch"""
        images, labels = batch
        batch_size = len(images)
        
        # Generate CutMix lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Create shuffled indices
        indices = torch.randperm(batch_size)
        
        # Apply CutMix
        W, H = images.size(2), images.size(3)
        cut_ratio = np.sqrt(1. - lam)
        cut_w = np.int_(W * cut_ratio)
        cut_h = np.int_(H * cut_ratio)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bound
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cuts
        images_copy = images.clone()
        images_copy[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images_copy, labels, labels[indices], lam


def prepare_dataloaders(dataset_path, test_path):
    """Prepare DataLoaders for training and validation using the test dataset for validation."""
    # Training transformation pipeline - with reduced strength for early training stability
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Reduced probability
        transforms.RandomRotation(degrees=DEGREES),
        transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE) if USE_COLOR_JITTER else nn.Identity(),
        transforms.RandomAffine(degrees=DEGREES, translate=TRANSLATE, scale=SCALE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=ERASING_PROB) if USE_RANDOM_ERASING else nn.Identity()
    ])
    
    # Validation transformation pipeline - simple resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=test_path,
        transform=val_transform
    )
    
    # Confirm class indices are the same
    train_classes = train_dataset.classes
    val_classes = val_dataset.classes
    
    assert train_classes == val_classes, f"Train and validation classes don't match: {train_classes} vs {val_classes}"
    
    # Create data loaders with appropriate batch size and worker count
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),  # Use at most 4 workers to avoid memory issues
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,  # Using same batch size for validation for consistent memory usage
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    
    # Print total classes in each dataset
    print(f"Total classes in training dataset: {len(train_dataset.classes)}")
    print(f"Total classes in validation dataset: {len(val_dataset.classes)}")
    
    return train_loader, val_loader


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Enable benchmark mode for faster training
    print(f"🔹 Using Device: {device}\n")
    return device


def find_lr(model, train_loader, optimizer, criterion):
    """Learning rate finder implementation"""
    num_batches = 100
    log_lrs = []
    losses = []
    best_loss = float('inf')
    
    # Set initial learning rate
    lr = 1e-7
    
    # Backup current model parameters
    old_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Check for divergence
        if torch.isnan(loss) or torch.isinf(loss) or loss > 4 * best_loss:
            break
            
        # Update best loss
        if loss < best_loss:
            best_loss = loss.item()
            
        # Record stats
        losses.append(loss.item())
        log_lrs.append(np.log10(lr))
        
        # Compute gradients
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= 1.1
        
    # Restore model parameters
    model.load_state_dict(old_state_dict)
    
    return log_lrs, losses


def resume_model(device, model):
    if RESUME_EPOCH and RESUME_EPOCH != 0:
        resume_model_path = f"{MODEL_PATH}_epoch_{RESUME_EPOCH}.pth"
        try:
            if os.path.exists(resume_model_path):
                print(
                    f"🔹 Resuming model from checkpoint: {resume_model_path}...")
                checkpoint = torch.load(resume_model_path, map_location=device)
                model.load_state_dict(remove_module_prefix(checkpoint))
                print("✅ Model resumed from checkpoint.")
            else:
                print(
                    f"⚠️ Checkpoint for epoch {RESUME_EPOCH} not found. Starting from scratch.")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting from scratch.")
    return model


def save_checkpoint(model, epoch, train_loss, train_acc, val_loss, val_acc):
    # Save checkpoint for the current epoch
    epoch_model_path = f"{MODEL_PATH}_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Epoch {epoch + 1}: TLoss: {train_loss:.4f}, TAcc: {train_acc:.2f}%, VLoss: {val_loss:.4f}, VAcc: {val_acc:.2f}%, Path: {epoch_model_path}")
    log_csv(epoch + 1, train_loss, train_acc, val_loss, val_acc)
    
    # Clean up older checkpoints (keep only last 5 checkpoints to save disk space)
    if epoch > 10:
        old_checkpoint = f"{MODEL_PATH}_epoch_{epoch - 10}.pth"
        if os.path.exists(old_checkpoint):
            try:
                os.remove(old_checkpoint)
            except:
                pass
    
    # Clean up memory
    torch.cuda.empty_cache()


# Implement test-time augmentation for evaluation
def tta_evaluate(model, val_loader, criterion, device, num_augments=TTA_NUM_AUGMENTS):
    """Evaluate model with test-time augmentation for better results"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            
            # Original prediction
            outputs = model(images)
            
            # TTA predictions
            for _ in range(num_augments - 1):
                # Apply simple augmentations
                aug_images = images.clone()
                
                # Random horizontal flip
                if random.random() > 0.5:
                    aug_images = torch.flip(aug_images, dims=[3])
                
                # Small random shifts
                shift_x, shift_y = random.randint(-2, 2), random.randint(-2, 2)
                if shift_x != 0 or shift_y != 0:
                    aug_images = torch.roll(aug_images, shifts=(shift_x, shift_y), dims=(2, 3))
                
                # Add to ensemble
                aug_outputs = model(aug_images)
                outputs += aug_outputs
            
            # Average predictions
            outputs /= num_augments
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_acc = 100. * correct / total
    val_loss = total_loss / (batch_idx + 1)
    
    return val_loss, val_acc


# Implement ensemble evaluation using multiple checkpoints
def ensemble_evaluate(device, val_loader, criterion):
    """Evaluate using an ensemble of best model checkpoints"""
    print("📊 Ensemble Evaluation with Multiple Checkpoints")
    
    # Find the top N model checkpoints by scanning the MODEL_PATH directory
    model_dir = os.path.dirname(MODEL_PATH)
    checkpoints = []
    
    for file in os.listdir(model_dir):
        if file.startswith(os.path.basename(MODEL_PATH) + "_epoch_") and file.endswith(".pth"):
            checkpoint_path = os.path.join(model_dir, file)
            # Extract epoch number for sorting
            try:
                epoch = int(file.split("_")[-1].split(".")[0])
                checkpoints.append((checkpoint_path, epoch))
            except:
                continue
    
    # Sort by epoch (higher is better) and take the top N
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    top_checkpoints = checkpoints[:ENSEMBLE_SIZE]
    
    if not top_checkpoints:
        print("⚠️ No model checkpoints found for ensemble evaluation")
        return
    
    print(f"🔹 Using {len(top_checkpoints)} best model checkpoints for ensemble evaluation")
    models = []
    
    # Import the create_model function to ensure we use the correct model type
    from training import create_model
    
    # Load each model
    for checkpoint_path, epoch in top_checkpoints:
        # Create a new model with the correct architecture
        model = create_model(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded checkpoint from epoch {epoch}")
    
    # Evaluate using the ensemble
    total = 0
    correct = 0
    total_loss = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    print("🔹 Running ensemble evaluation...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            
            # Get predictions from each model
            ensemble_outputs = None
            
            for model in models:
                # Apply test-time augmentation for each model
                model_outputs = None
                
                # Original image
                outputs = model(images)
                
                if model_outputs is None:
                    model_outputs = outputs.clone()
                else:
                    model_outputs += outputs
                
                # Horizontally flipped image
                flipped = torch.flip(images, dims=[3])
                outputs = model(flipped)
                model_outputs += outputs
                
                # Normalized predictions from this model
                model_outputs /= 2.0  # Average the predictions
                
                # Add to ensemble
                if ensemble_outputs is None:
                    ensemble_outputs = model_outputs.clone()
                else:
                    ensemble_outputs += model_outputs
            
            # Average predictions across models
            ensemble_outputs /= len(models)
            
            # Compute loss and accuracy
            loss = criterion(ensemble_outputs, targets)
            total_loss += loss.item()
            
            _, predicted = ensemble_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Track per-class accuracy
            for i in range(batch_size):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processing batch {batch_idx+1}/{len(val_loader)}", end='\r')
    
    # Calculate final metrics
    ensemble_acc = 100.0 * correct / total
    ensemble_loss = total_loss / len(val_loader)
    
    # Calculate per-class accuracies
    class_accuracies = [100.0 * correct_count / max(1, total_count) for correct_count, total_count in zip(class_correct, class_total)]
    class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    
    print(f"\n✅ Ensemble Evaluation Results:")
    print(f"   Loss: {ensemble_loss:.4f}")
    print(f"   Accuracy: {ensemble_acc:.2f}%")
    print("\n   Per-class accuracies:")
    for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
        print(f"   - {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return ensemble_loss, ensemble_acc


# Lookahead optimizer implementation for improved large batch training
class Lookahead(optim.Optimizer):
    """Implements Lookahead optimizer.
    
    It has been proposed in "Lookahead Optimizer: k steps forward, 1 step back"
    (https://arxiv.org/abs/1907.08610)
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        """Initialize Lookahead
        
        Args:
            optimizer: inner optimizer
            k: number of lookahead steps
            alpha: linear interpolation factor
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        # For compatibility with PyTorch schedulers
        self.defaults = optimizer.defaults
        self._optimizer = optimizer
        
        # For compatibility with newer PyTorch versions
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast_p.data)
                param_state["slow_param"].copy_(fast_p.data)
            slow = param_state["slow_param"]
            slow.add_(self.alpha * (fast_p.data - slow))
            fast_p.data.copy_(slow)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] >= self.k:
                self.update(group)
                group["counter"] = 0
        return loss
    
    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state
    
    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
        
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value


# Add large batch normalization handling
def update_bn_for_large_batch(loader, model, device):
    """
    Reset BatchNorm statistics for large batch training
    This improves performance when using large batches by estimating
    better statistics for batch normalization layers.
    """
    if not int(os.environ.get('LARGE_BATCH_BN', 0)):
        return model
        
    print("🔹 Recalibrating BatchNorm statistics for large batch training...")
    model.eval()  # Set to evaluation mode for gathering stats
    
    # Reset all batch norm statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = 0.1  # Default PyTorch momentum
            m.training = True  # Set to training mode to update stats
    
    # Use a subset of the dataset to recalibrate
    num_batches = min(100, len(loader))
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
    print("✅ BatchNorm statistics recalibrated")
    return model
