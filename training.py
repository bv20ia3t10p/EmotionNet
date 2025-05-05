import torch  # type: ignore
import torch.optim as optim  # type: ignore
from config import *
from model import *
from model import ACCUMULATION_STEPS
from model_utils import *
from utils import *
import os
import random
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm


def prepare_training_components(model, train_loader):
    print("🔹 Setting Up Training Components ==============================")
    # Add class weights to handle class imbalance
    class_weights = compute_class_weights(TRAIN_PATH)
    print(f"🔹 Using class weights: {[round(w, 2) for w in class_weights]}")
    
    # Use combined loss function for better training
    criterion = CombinedLoss(
        alpha=FOCAL_ALPHA, 
        gamma=FOCAL_GAMMA, 
        class_weights=class_weights, 
        label_smoothing=LABEL_SMOOTHING,
        kl_weight=KL_WEIGHT
    )
    
    # Create optimizer with proper weight decay
    # Don't apply weight decay to batch norm or bias parameters
    decay_parameters = []
    no_decay_parameters = []
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
    
    base_optimizer = optim.AdamW([
        {'params': decay_parameters, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_parameters, 'weight_decay': 0.0}
    ], lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    # Wrap with Lookahead optimizer if enabled (for large batch training)
    if int(os.environ.get('LOOKAHEAD_ENABLED', 0)):
        print("🔹 Using Lookahead optimizer wrapper for improved large batch training")
        from model_utils import Lookahead
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    else:
        optimizer = base_optimizer
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler() if USE_AMP else None
    
    # Use cosine annealing scheduler with warmup
    total_steps = NUM_EPOCHS * len(train_loader) // ACCUMULATION_STEPS
    warmup_steps = WARMUP_EPOCHS * len(train_loader) // ACCUMULATION_STEPS
    
    print(f"🔹 Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Create scheduler on the base optimizer if using Lookahead
    scheduler_optimizer = base_optimizer if int(os.environ.get('LOOKAHEAD_ENABLED', 0)) else optimizer
    
    # Use one cycle policy for better convergence
    one_cycle_scheduler = optim.lr_scheduler.OneCycleLR(
        scheduler_optimizer, 
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Use ReduceLROnPlateau as backup scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        scheduler_optimizer, mode='max', factor=FACTOR, patience=PATIENCE, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    print("✅ Training Components Ready\n")
    return criterion, optimizer, scheduler, one_cycle_scheduler, early_stopping, scaler


def compute_class_weights(dataset_path):
    """Compute class weights inversely proportional to class frequencies."""
    # Get statistics of the dataset
    stats = get_image_stats(dataset_path)
    
    # Calculate weights
    total_samples = sum(stats.values())
    num_classes = len(stats)
    weights = []
    
    for class_name, count in sorted(stats.items()):
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        
    # Normalize weights
    weights = np.array(weights)
    if USE_FOCAL_LOSS:
        # Scale weights for focal loss
        weights = weights / np.sum(weights) * num_classes
    else:
        # Keep original weights for cross-entropy
        weights = weights / np.min(weights)  # Normalize by minimum
    
    return weights.tolist()


def run_training_epoch(model, train_loader, criterion, optimizer, device, mixup_transform, cutmix_transform, one_cycle_scheduler, scaler=None, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    
    # Enable/disable features based on epoch
    if epoch < FREEZE_BACKBONE_EPOCHS and hasattr(model, 'backbone'):
        # Freeze backbone for initial epochs to train only the head
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif epoch == FREEZE_BACKBONE_EPOCHS and hasattr(model, 'backbone'):
        # Unfreeze backbone after freeze epochs
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("✅ Backbone unfrozen at epoch", epoch)
    
    # Gradually enable more complex augmentations as training progresses
    mixup_probability = min(MIXUP_PROB, MIXUP_PROB * epoch / 10) if epoch < 10 else MIXUP_PROB
    cutmix_probability = min(CUTMIX_PROB, CUTMIX_PROB * epoch / 10) if epoch < 10 else CUTMIX_PROB
    
    end = time.time()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_idx, (images, labels) in progress_bar:
        data_time.update(time.time() - end)
        
        images, labels = images.to(device), labels.to(device)
        
        # Apply augmentation based on random selection and epoch-dependent probability
        rand_choice = random.random()
        
        with autocast() if USE_AMP else nullcontext():
            if rand_choice < mixup_probability and epoch >= 1:  # Only apply mixup after first epoch
                mixed_images, labels_a, labels_b, lam = mixup_transform((images, labels))
                images = mixed_images.to(device)
                
                # Convert label indices to one-hot for mixup
                labels_a_one_hot = F.one_hot(labels_a, NUM_CLASSES).float()
                labels_b_one_hot = F.one_hot(labels_b, NUM_CLASSES).float()
                mixed_targets = lam * labels_a_one_hot + (1 - lam) * labels_b_one_hot
                
                # Forward pass
                outputs = model(images)
                
                # For early epochs, use simple cross entropy on the dominant label
                if epoch < 5:
                    dominant_labels = labels_a if lam > 0.5 else labels_b
                    loss = F.cross_entropy(outputs, dominant_labels)
                else:
                    # Compute loss for mixed labels
                    loss = criterion(outputs, mixed_targets)
                
            elif rand_choice < mixup_probability + cutmix_probability and epoch >= 3:  # Only apply cutmix after third epoch
                mixed_images, labels_a, labels_b, lam = cutmix_transform((images, labels))
                images = mixed_images.to(device)
                
                # Convert label indices to one-hot for cutmix
                labels_a_one_hot = F.one_hot(labels_a, NUM_CLASSES).float()
                labels_b_one_hot = F.one_hot(labels_b, NUM_CLASSES).float()
                mixed_targets = lam * labels_a_one_hot + (1 - lam) * labels_b_one_hot
                
                # Forward pass
                outputs = model(images)
                
                # For early epochs, use simple cross entropy on the dominant label
                if epoch < 5:
                    dominant_labels = labels_a if lam > 0.5 else labels_b
                    loss = F.cross_entropy(outputs, dominant_labels)
                else:
                    # Compute loss for mixed labels
                    loss = criterion(outputs, mixed_targets)
                
            else:  # Standard forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        # Scale the loss according to accumulation steps
        loss = loss / ACCUMULATION_STEPS
        
        # Backward and optimize with optional mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Gradient accumulation - only update every ACCUMULATION_STEPS
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                # Optional gradient clipping
                if GRAD_CLIP_VALUE > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if one_cycle_scheduler is not None:
                    one_cycle_scheduler.step()
        else:
            loss.backward()
            
            # Gradient accumulation - only update every ACCUMULATION_STEPS
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                # Optional gradient clipping
                if GRAD_CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                    
                optimizer.step()
                optimizer.zero_grad()
                if one_cycle_scheduler is not None:
                    one_cycle_scheduler.step()
        
        # Compute accuracy
        if (rand_choice < mixup_probability + cutmix_probability) and (epoch >= 1):
            # For mixed batches, use the dominant label for accuracy calculation
            _, predicted = outputs.max(1)
            dominant_labels = labels_a if lam > 0.5 else labels_b
            batch_correct = predicted.eq(dominant_labels).sum().item()
        else:
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            
        # Update metrics
        total_loss += loss.item() * ACCUMULATION_STEPS
        batch_size = images.size(0)
        total += batch_size
        correct += batch_correct
        
        losses.update(loss.item() * ACCUMULATION_STEPS, batch_size)
        accs.update(100.0 * batch_correct / batch_size, batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar every 10 batches
        if batch_idx % 10 == 0:
            progress_bar.set_description(
                f'Batch {batch_idx}/{len(train_loader)}: '
                f'Loss: {losses.val:.4f}, '
                f'Acc: {accs.val:.2f}%, '
                f'Time: {batch_time.val:.2f}s'
            )
    
    # Calculate final metrics
    train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return train_loss, train_acc


def create_model(device):
    """Create and initialize the model based on configuration."""
    
    if MODEL_TYPE == "ResEmoteNet":
        print("🔹 Creating ResEmoteNet model")
        model = ResEmoteNet().to(device)
    elif MODEL_TYPE == "AdvancedEmoteNet":
        print(f"🔹 Creating AdvancedEmoteNet model with {BACKBONE} backbone")
        model = AdvancedEmoteNet(backbone=BACKBONE, pretrained=PRETRAINED).to(device)
    elif MODEL_TYPE == "EmotionViT":
        print(f"🔹 Creating EmotionViT model with {BACKBONE} backbone")
        model = EmotionViT(backbone=BACKBONE, pretrained=PRETRAINED).to(device)
    else:
        print("⚠️ Unknown model type, defaulting to ResEmoteNet")
        model = ResEmoteNet().to(device)
    
    return model


def train_model():
    device = initialize_device()
    print("🔹 Preparing Training Environment")
    
    # Create the appropriate model
    model = create_model(device)
    
    print("🔹 Preparing DataLoaders...")
    train_loader, val_loader = prepare_dataloaders(TRAIN_PATH, TEST_PATH)
    print("✅ DataLoaders Ready")
    print_image_stats(TRAIN_PATH, TEST_PATH)
    
    criterion, optimizer, scheduler, one_cycle_scheduler, early_stopping, scaler = prepare_training_components(model, train_loader)
    start_epoch = RESUME_EPOCH if RESUME_EPOCH else 0
    model = resume_model(device, model)
    
    # Apply large batch normalization handling if enabled
    if int(os.environ.get('LARGE_BATCH_BN', 0)) and BATCH_SIZE >= 256:
        from model_utils import update_bn_for_large_batch
        model = update_bn_for_large_batch(train_loader, model, device)
    
    # Enhanced transforms with stronger augmentation
    mixup_transform = MixupTransform(alpha=MIXUP_ALPHA)
    cutmix_transform = CutMixTransform(alpha=CUTMIX_ALPHA)
    
    if ENABLE_LR_FINDER:
        print("🔹 Running Learning Rate Finder...")
        log_lrs, losses = find_lr(model, train_loader, optimizer, criterion)
        plot_lr_finder(log_lrs, losses)
        save_lr_finder_plot(log_lrs, losses, os.path.join(DATASET_PATH, "lr_finder_plot.png"))
        print("✅ Learning Rate Finder Complete. Adjust learning rate based on the plot.\n")
    
    print("🔹 Begin training ==============================")
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)
        
        # Training phase
        train_loss, train_acc = run_training_epoch(
            model, train_loader, criterion, optimizer, device, 
            mixup_transform, cutmix_transform, one_cycle_scheduler, scaler, epoch)
        
        # Validation phase with test-time augmentation
        val_loss, val_acc = tta_evaluate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ New best model saved with validation accuracy: {val_acc:.2f}%")
        
        save_checkpoint(model, epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Early stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"⚠️ Early stopping triggered at epoch {epoch + 1}")
            break
    
    print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✅ Final Model Saved at {MODEL_PATH}\n")
    
    # Final evaluation with ensemble of best checkpoints
    print("🔹 Performing final ensemble evaluation...")
    ensemble_evaluate(device, val_loader, criterion)


# Context manager for conditional execution
class nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
