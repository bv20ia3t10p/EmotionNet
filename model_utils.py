from config import *
from utils import *
import torch  # type: ignore
from torchvision import transforms, datasets  # type: ignore
import numpy as np  # type: ignore
import random
import os
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from config import *
from collections import defaultdict
from tqdm import tqdm # type: ignore


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
    print(f"🔹 Using dataset paths: Train={dataset_path}, Test={test_path}")
    
    # Check if we're using a pre-balanced dataset (based on path or env var)
    # MODEL_BALANCE_DATASET=0 means we're using a physically balanced dataset
    # MODEL_BALANCE_DATASET=1 means we need dataloader balancing
    model_balance_dataset = int(os.environ.get('MODEL_BALANCE_DATASET', '1'))
    using_balanced_dataset = "balanced_" in dataset_path or model_balance_dataset == 0
    
    if using_balanced_dataset:
        print(f"🔹 Using physically balanced dataset (MODEL_BALANCE_DATASET={model_balance_dataset})")
    else:
        print(f"🔹 Using dataloader balancing (MODEL_BALANCE_DATASET={model_balance_dataset})")
    
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
    
    # Apply class balancing through oversampling if enabled and not using a physically balanced dataset
    balance_dataset = int(os.environ.get('BALANCE_DATASET', 0))
    
    if balance_dataset and not using_balanced_dataset:
        print("🔹 Applying class balancing through oversampling...")
        target_samples = int(os.environ.get('TARGET_SAMPLES_PER_CLASS', 5000))
        from utils import oversample_minority_classes
        
        # Get balanced indices
        balanced_indices = oversample_minority_classes(train_dataset, target_samples)
        
        # Create a sampler using the balanced indices
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(balanced_indices)
        shuffle = False  # Don't shuffle when using a sampler
    else:
        if using_balanced_dataset:
            print("🔹 Using physically balanced dataset, no oversampling needed")
        train_sampler = None
        shuffle = True
    
    # Create data loaders with appropriate batch size and worker count
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=train_sampler,
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
    # Check if checkpoints are disabled via environment variable
    disable_checkpoints = os.getenv("DISABLE_CHECKPOINTS", "0") == "1"
    
    if disable_checkpoints:
        print(f"Epoch {epoch + 1}: TLoss: {train_loss:.4f}, TAcc: {train_acc:.2f}%, VLoss: {val_loss:.4f}, VAcc: {val_acc:.2f}% (Checkpoints disabled)")
        # Still log the metrics to CSV
        log_csv(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        return

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


def save_final_model(model, model_path=None):
    """
    Save the final model at the end of training regardless of checkpoint settings.
    This ensures we always have a saved model even when checkpoints are disabled.
    
    Args:
        model: The model to save
        model_path: Optional custom path, defaults to MODEL_PATH from config
    """
    target_path = model_path if model_path else MODEL_PATH
    final_model_path = f"{target_path}_final.pth"
    
    print(f"🔹 Saving final model to {final_model_path}...")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved successfully")
    
    # Also save to the standard path if not already saved
    if not os.path.exists(target_path):
        torch.save(model.state_dict(), target_path)
        print(f"✅ Final model also saved to {target_path}")
    
    return final_model_path


# Implement test-time augmentation for evaluation
def tta_evaluate(model, val_loader, criterion, device, num_augments=TTA_NUM_AUGMENTS):
    """Evaluate model with test-time augmentation for better results"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Check if we should use fast evaluation modes
    disable_checkpoints = os.getenv("DISABLE_CHECKPOINTS", "0") == "1"
    fast_eval = os.getenv("FAST_EVALUATION", "0") == "1" or disable_checkpoints
    ultra_fast = os.getenv("ULTRA_FAST_EVAL", "0") == "1"
    skip_tta = os.getenv("SKIP_TTA", "0") == "1"  # Option to completely skip TTA and just do standard eval
    
    # Use fewer augmentations in fast mode
    if skip_tta or ultra_fast:
        actual_augments = 1  # No augmentation in ultra-fast mode or when TTA is skipped
        if skip_tta:
            print(f"🔹 Skipping test-time augmentation (standard single-pass evaluation)")
        else:
            print(f"🔹 Using ULTRA-FAST evaluation (single pass, subset of data)")
    elif fast_eval:
        actual_augments = min(2, num_augments)  # Use at most 2 augmentations in fast mode
        print(f"🔹 Using fast evaluation with {actual_augments} augmentations")
    else:
        actual_augments = num_augments
    
    with torch.no_grad():
        # Determine how many batches to evaluate
        max_eval_batches = None
        if ultra_fast:
            # In ultra-fast mode, use just a small subset of validation data
            # Generally ~20% or 10 batches, whichever is smaller
            max_eval_batches = min(10, max(5, len(val_loader) // 5))
        
        # Create iterable for evaluation
        if max_eval_batches:
            eval_loader = tqdm(
                list(val_loader)[:max_eval_batches], 
                desc=f"Ultra-fast evaluation ({max_eval_batches}/{len(val_loader)} batches)",
                leave=False
            )
        else:
            eval_loader = tqdm(val_loader, desc="Evaluation", leave=False)
        
        for batch_idx, (images, targets) in enumerate(eval_loader):
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            
            # Original prediction
            outputs = model(images)
            
            # TTA predictions - only if not in ultra-fast mode and TTA not skipped
            if actual_augments > 1:
                for i in range(actual_augments - 1):
                    # Apply simple augmentations
                    aug_images = images.clone()
                    
                    # In fast eval mode, just do horizontal flip
                    if i == 0:  # Always do horizontal flip for first augmentation
                        aug_images = torch.flip(aug_images, dims=[3])
                    # Only add shifts in full evaluation mode
                    elif not fast_eval and i % 2 == 0:
                        shift_x, shift_y = random.randint(-2, 2), random.randint(-2, 2)
                        if shift_x != 0 or shift_y != 0:
                            aug_images = torch.roll(aug_images, shifts=(shift_x, shift_y), dims=(2, 3))
                    
                    # Add to ensemble
                    aug_outputs = model(aug_images)
                    outputs += aug_outputs
                
                # Average predictions
                outputs /= actual_augments
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Early break for ultra-fast evaluation
            if ultra_fast and batch_idx >= max_eval_batches - 1:
                break
    
    val_acc = 100. * correct / total
    val_loss = total_loss / (batch_idx + 1)
    
    return val_loss, val_acc


# Implement ensemble evaluation using multiple checkpoints
def ensemble_evaluate(device, val_loader, criterion):
    """Evaluate using an ensemble of best model checkpoints"""
    print("📊 Ensemble Evaluation with Multiple Checkpoints")
    
    # Find the top N model checkpoints by scanning the MODEL_PATH directory
    model_dir = os.path.dirname(MODEL_PATH)
    
    # Check if checkpoints are disabled
    disable_checkpoints = os.getenv("DISABLE_CHECKPOINTS", "0") == "1"
    if disable_checkpoints:
        print("⚠️ Skipping ensemble evaluation (checkpoints are disabled)")
        return
    
    checkpoints = []
    
    # Only look for checkpoint files if they might exist
    try:
        for file in os.listdir(model_dir):
            if file.startswith(os.path.basename(MODEL_PATH) + "_epoch_") and file.endswith(".pth"):
                checkpoint_path = os.path.join(model_dir, file)
                # Extract epoch number for sorting
                try:
                    epoch = int(file.split("_")[-1].split(".")[0])
                    checkpoints.append((checkpoint_path, epoch))
                except:
                    continue
    except Exception as e:
        print(f"⚠️ Error scanning for checkpoints: {e}")
        return
    
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
        # Use tqdm for a progress bar
        from tqdm import tqdm
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Ensemble Evaluation")):
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
    
    # Clear memory
    for model in models:
        del model
    torch.cuda.empty_cache()
    
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
    # Force enable for large batch sizes
    if BATCH_SIZE < 64:
        print("ℹ️ BatchNorm recalibration skipped for small batch size")
        return model
        
    print("🔹 Recalibrating BatchNorm statistics for large batch training...")
    model.eval()  # Set to evaluation mode for gathering stats
    
    # Reset all batch norm statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = 0.1  # Default PyTorch momentum
            m.training = True  # Set to training mode to update stats
    
    # Use more batches for recalibration with larger batch sizes
    recalibration_batches = 50 if BATCH_SIZE <= 64 else 25
    num_batches = min(recalibration_batches, len(loader))
    print(f"   Using {num_batches} batches for BatchNorm recalibration")
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
    print("✅ BatchNorm statistics recalibrated for large batch training")
    return model


# Add Stochastic Weight Averaging (SWA) for better generalization and convergence
class SWA:
    """Implements Stochastic Weight Averaging for improved generalization.
    
    SWA averages multiple points along the trajectory of SGD, leading to better
    generalization than conventional training. This is particularly effective
    in the final stages of training.
    """
    def __init__(self, model, swa_start=10, swa_freq=5, device=None):
        """Initialize SWA.
        
        Args:
            model: The model to apply SWA to
            swa_start: The epoch to start SWA from
            swa_freq: How frequently to update the SWA model (in epochs)
            device: The device to store the SWA model on
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create a copy of the model for SWA
        self.swa_model = self._create_swa_model()
        self.swa_n = 0  # Counter for number of models added
    
    def _create_swa_model(self):
        """Create a copy of the model for SWA with identical structure."""
        print("🔹 Creating SWA model...")
        
        # For safer initialization, create an exact copy of the model
        import copy
        swa_model = copy.deepcopy(self.model).to(self.device)
        
        # Set to eval mode
        swa_model.eval()
        print("✅ SWA model created successfully")
        return swa_model
    
    def update(self, epoch):
        """Update the SWA model if conditions are met."""
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            # Moving average of parameters
            self.swa_n += 1
            
            # Update each parameter
            for param_q, param_k in zip(self.model.parameters(), self.swa_model.parameters()):
                param_k.data = (param_k.data * self.swa_n + param_q.data) / (self.swa_n + 1)
            
            print(f"✅ SWA model updated at epoch {epoch} (model count: {self.swa_n})")
    
    def finalize(self):
        """Finalize the SWA model by properly updating batch norm statistics."""
        if self.swa_n == 0:
            print("⚠️ SWA model was never updated. Using original model instead.")
            return self.model
            
        print("🔹 Finalizing SWA model with updated batch normalization statistics...")
        
        # Update batch normalization statistics in the SWA model
        self.swa_model.train()
        
        # Create a hook to accumulate batch norm statistics
        def update_bn(model, loader, device):
            # Set up modules for updating statistics
            momenta = {}
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.running_mean = torch.zeros_like(module.running_mean)
                    module.running_var = torch.ones_like(module.running_var)
                    momenta[module] = module.momentum
                    module.momentum = None
                    module.num_batches_tracked *= 0
            
            # Check if fast finalization is enabled
            fast_finalize = os.environ.get('FAST_SWA_FINALIZE', '0') == '1'
            
            # Only use a limited number of batches for efficiency
            max_batches = 10 if fast_finalize else 50  # Even fewer batches in fast mode
            
            print(f"  Using {'fast' if fast_finalize else 'standard'} SWA finalization with {max_batches} batches")
            
            # Update the statistics for each batch
            n = 0
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(loader):
                    # Limit the number of batches processed
                    if batch_idx >= max_batches:
                        break
                        
                    images = images.to(device)
                    b = images.size(0)
                    
                    momentum = b / float(n + b)
                    for module in momenta.keys():
                        module.momentum = momentum
                    
                    model(images)
                    n += b
                    
                    # Progress indicator
                    if batch_idx % 5 == 0:
                        print(f"  Batch {batch_idx+1}/{min(max_batches, len(loader))}", end="\r")
            
            # Restore momentum values
            for module, momentum in momenta.items():
                module.momentum = momentum
                
            print(f"  Used {min(batch_idx+1, max_batches)} batches for BN calibration    ")
        
        # We need a loader to update BN statistics
        try:
            # Try to get the training loader from config
            from config import TRAIN_PATH, TEST_PATH, BATCH_SIZE
            from model_utils import prepare_dataloaders
            
            # Check if we should use a smaller batch size for faster processing
            fast_finalize = os.environ.get('FAST_SWA_FINALIZE', '0') == '1'
            
            # Create a data loader with a subset of the data for faster processing
            train_loader, _ = prepare_dataloaders(TRAIN_PATH, TEST_PATH)
            
            # Use a subset of the training data to update BN statistics
            update_bn(self.swa_model, train_loader, self.device)
        except Exception as e:
            print(f"⚠️ Error updating batch norm statistics: {e}")
            print("⚠️ SWA model may have incorrect batch norm statistics.")
        
        self.swa_model.eval()
        print("✅ SWA model finalized")
        
        # Clear memory explicitly
        torch.cuda.empty_cache() 
        
        return self.swa_model


# Add a function to create learning rate scheduler with cosine annealing and restarts
def create_lr_scheduler(optimizer, train_loader, scheduler_type="cosine", **kwargs):
    """Create a learning rate scheduler based on the specified type.
    
    Args:
        optimizer: The optimizer to schedule
        train_loader: The training data loader
        scheduler_type: The type of scheduler to use
        **kwargs: Additional arguments for specific schedulers
    
    Returns:
        The learning rate scheduler
    """
    from config import NUM_EPOCHS, PATIENCE, FACTOR
    
    if scheduler_type == "cosine":
        # Simple cosine annealing
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS,
            eta_min=1e-6
        )
    elif scheduler_type == "cosine_restart":
        # Cosine annealing with warm restarts
        cycles = kwargs.get("cosine_cycles", 3)
        cycle_length = NUM_EPOCHS // cycles
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cycle_length,  # First restart
            T_mult=1,          # Keep same cycle length
            eta_min=1e-6
        )
    elif scheduler_type == "plateau":
        # Reduce on plateau (default)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=FACTOR, 
            patience=PATIENCE, 
            verbose=True
        )
    elif scheduler_type == "step":
        # Step LR
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    else:
        # Default to ReduceLROnPlateau
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=FACTOR, 
            patience=PATIENCE, 
            verbose=True
        )
