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
import numpy as np  # type: ignore
from torch.cuda.amp import autocast, GradScaler  # type: ignore
import torch.nn.functional as F  # type: ignore
from config import *
from tqdm import tqdm  # type: ignore
from torchvision import transforms
import torch.nn as nn
from torchvision import datasets


def prepare_training_components(model, train_loader):
    print("🔹 Setting Up Training Components ==============================")
    
    # Calculate sample counts per class for class-balanced loss
    samples_per_class = get_samples_per_class(TRAIN_PATH)
    print(f"🔹 Class distribution: {samples_per_class}")
    
    # Add class weights to handle class imbalance
    class_weights = compute_class_weights(TRAIN_PATH)
    print(f"🔹 Using class weights: {[round(w, 2) for w in class_weights]}")

    # Use improved combined loss function for better training
    if os.environ.get('USE_CLASS_BALANCED_LOSS', '1') == '1':
        print("🔹 Using Class-Balanced Loss for extreme imbalance handling")
        criterion = CombinedLoss(
            class_weights=class_weights,
            samples_per_class=samples_per_class,
            beta=float(os.environ.get('CB_BETA', 0.9999)),
            gamma=FOCAL_GAMMA,
            label_smoothing=LABEL_SMOOTHING,
            kl_weight=KL_WEIGHT,
            center_weight=float(os.environ.get('CENTER_WEIGHT', 0.005))
        )
    else:
        print("🔹 Using standard Focal Loss with class weights")
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            class_weights=class_weights,
            label_smoothing=LABEL_SMOOTHING,
            scale_pos_weight=float(os.environ.get('SCALE_POS_WEIGHT', 1.2))
        )

    # Create optimizer with proper weight decay and parameter grouping
    # Group parameters for different weight decay application
    decay_parameters = []
    no_decay_parameters = []
    head_parameters = []  # Higher learning rate for classifier head

    for name, param in model.named_parameters():
        if 'classifier' in name or 'head' in name:
            head_parameters.append(param)
        elif 'bias' in name or 'bn' in name or 'norm' in name or 'layernorm' in name:
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
    
    # Get head learning rate multiplier
    head_lr_multiplier = float(os.environ.get('HEAD_LR_MULTIPLIER', 10.0))
    print(f"🔹 Using {head_lr_multiplier}x learning rate for classifier head")

    # Create optimizer with parameter groups
    base_optimizer = optim.AdamW([
        {'params': decay_parameters, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_parameters, 'weight_decay': 0.0},
        {'params': head_parameters, 'weight_decay': WEIGHT_DECAY, 'lr': LEARNING_RATE * head_lr_multiplier}
    ], lr=LEARNING_RATE, betas=(0.9, 0.999))

    # Wrap with Lookahead optimizer if enabled (for large batch training)
    if int(os.environ.get('LOOKAHEAD_ENABLED', 0)):
        print("🔹 Using Lookahead optimizer wrapper for improved large batch training")
        from model_utils import Lookahead
        optimizer = Lookahead(
            base_optimizer,
            k=int(os.environ.get('LOOKAHEAD_K', 5)),
            alpha=float(os.environ.get('LOOKAHEAD_ALPHA', 0.5))
        )
    else:
        optimizer = base_optimizer

    # Set up gradient scaler for mixed precision training
    scaler = GradScaler() if USE_AMP else None

    # Use appropriate scheduler based on configuration
    scheduler_type = os.environ.get('SCHEDULER_TYPE', 'cosine')
    scheduler_optimizer = base_optimizer if int(
        os.environ.get('LOOKAHEAD_ENABLED', 0)) else optimizer
    print(f"🔹 Using {scheduler_type} learning rate scheduler")

    # Create scheduler based on type
    if scheduler_type == "cosine" or scheduler_type == "cosine_restart":
        # Use the create_lr_scheduler function from model_utils
        cosine_cycles = int(os.environ.get('COSINE_CYCLES', 3))
        scheduler = create_lr_scheduler(
            scheduler_optimizer,
            train_loader,
            scheduler_type=scheduler_type,
            cosine_cycles=cosine_cycles
        )

        # Placeholder for ReduceLROnPlateau as backup
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            scheduler_optimizer, mode='max', factor=FACTOR, patience=PATIENCE, verbose=True
        )
    else:
        # Use ReduceLROnPlateau as the primary scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            scheduler_optimizer, mode='max', factor=FACTOR, patience=PATIENCE, verbose=True
        )
        plateau_scheduler = scheduler  # Same scheduler

    # Initialize Stochastic Weight Averaging if enabled
    swa_enabled = int(os.environ.get('SWA_ENABLED', 0))
    if swa_enabled:
        swa_start = int(os.environ.get('SWA_START_EPOCH', 150))
        swa_freq = int(os.environ.get('SWA_FREQ', 5))
        print(
            f"🔹 Stochastic Weight Averaging (SWA) enabled, starting at epoch {swa_start}")
        from model_utils import SWA
        swa = SWA(model, swa_start=swa_start, swa_freq=swa_freq)
    else:
        swa = None

    # Setup enhanced distillation loss if enabled
    distillation_enabled = int(os.environ.get('SELF_DISTILLATION_ENABLED', 0))
    if distillation_enabled:
        distill_start = int(os.environ.get('SELF_DISTILLATION_START', 100))
        distill_temp = float(os.environ.get('SELF_DISTILLATION_TEMP', 2.0))
        distill_alpha = float(os.environ.get('SELF_DISTILLATION_ALPHA', 0.3))
        attention_beta = float(os.environ.get('ATTENTION_BETA', 0.1))
        print(
            f"🔹 Enhanced self-distillation enabled, starting at epoch {distill_start}")
        print(f"   - Temperature: {distill_temp}, Alpha: {distill_alpha}, Attention Beta: {attention_beta}")
        distill_criterion = DistillationLoss(
            alpha=distill_alpha, 
            temperature=distill_temp,
            attention_beta=attention_beta
        )
        teacher_model = None  # Will be loaded at the appropriate epoch
    else:
        distill_criterion = None
        teacher_model = None
        distill_start = float('inf')  # Never start

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    print("✅ Training Components Ready\n")

    return criterion, optimizer, scheduler, plateau_scheduler, early_stopping, scaler, swa, distill_criterion, teacher_model, distill_start


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


def get_samples_per_class(dataset_path):
    """Count the number of samples per class in the dataset."""
    stats = get_image_stats(dataset_path)
    class_names = sorted(stats.keys())
    return [stats[cls] for cls in class_names]


def run_training_epoch(model, train_loader, criterion, optimizer, device, mixup_transform, cutmix_transform, scheduler, scaler=None, epoch=0, teacher_model=None, distill_criterion=None, distill_start=float('inf')):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # Enable/disable self-distillation based on epoch
    use_distillation = teacher_model is not None and epoch >= distill_start
    if use_distillation:
        print(f"🔹 Using self-distillation at epoch {epoch}")
        teacher_model.eval()  # Teacher should be in eval mode

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

    # Progressive augmentation phases
    progressive_aug = int(os.environ.get('PROGRESSIVE_AUGMENTATION', 0))
    if progressive_aug:
        phase_1 = int(os.environ.get('PHASE_1_EPOCHS', 15))
        phase_2 = int(os.environ.get('PHASE_2_EPOCHS', 40))
        phase_3 = int(os.environ.get('PHASE_3_EPOCHS', 70))
        phase_4 = int(os.environ.get('PHASE_4_EPOCHS', 150))

        # Set augmentation strength based on phase
        if epoch < phase_1:
            phase = 1
            mixup_probability = MIXUP_PROB * 0.3  # Reduced in phase 1
            cutmix_probability = CUTMIX_PROB * 0.3  # Reduced in phase 1
        elif epoch < phase_2:
            phase = 2
            mixup_probability = MIXUP_PROB * 0.6  # Moderate in phase 2
            cutmix_probability = CUTMIX_PROB * 0.6  # Moderate in phase 2
        elif epoch < phase_3:
            phase = 3
            mixup_probability = MIXUP_PROB * 0.8  # Stronger in phase 3
            cutmix_probability = CUTMIX_PROB * 0.8  # Stronger in phase 3
        elif epoch < phase_4:
            phase = 4
            mixup_probability = MIXUP_PROB  # Full strength
            cutmix_probability = CUTMIX_PROB  # Full strength
        else:
            phase = 5
            mixup_probability = MIXUP_PROB * 1.1  # Even stronger for final phase
            cutmix_probability = CUTMIX_PROB * 1.1  # Even stronger for final phase

        print(f"🔹 Progressive augmentation phase {phase} (epoch {epoch})")
    else:
        # Standard augmentation approach (gradually increasing)
        mixup_probability = min(
            MIXUP_PROB, MIXUP_PROB * epoch / 10) if epoch < 10 else MIXUP_PROB
        cutmix_probability = min(
            CUTMIX_PROB, CUTMIX_PROB * epoch / 10) if epoch < 10 else CUTMIX_PROB

    end = time.time()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_idx, (images, labels) in progress_bar:
        try:
            data_time.update(time.time() - end)

            # Ensure proper data types and dimensions
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Check for NaN values in images
            if torch.isnan(images).any():
                print("⚠️ NaN values detected in input images. Skipping batch.")
                continue
                
            # Store variables needed for accuracy calculation outside the if-blocks
            is_mixup_or_cutmix = False
            mixup_labels_a = None
            mixup_labels_b = None
            mixup_lam = 0.5

            with autocast() if USE_AMP else nullcontext():
                # Extract features for attention transfer in distillation if needed
                features = None
                teacher_features = None

                # Apply data augmentation based on random selection
                if rand_choice := random.random() < mixup_probability and epoch >= 1:
                    # Apply Mixup
                    try:
                        mixed_images, labels_a, labels_b, lam = mixup_transform((images, labels))
                        images = mixed_images.to(device, non_blocking=True)

                        # Store for accuracy calculation later
                        is_mixup_or_cutmix = True
                        mixup_labels_a = labels_a
                        mixup_labels_b = labels_b
                        mixup_lam = lam

                        # Convert label indices to one-hot for mixup
                        labels_a_one_hot = F.one_hot(labels_a, NUM_CLASSES).float()
                        labels_b_one_hot = F.one_hot(labels_b, NUM_CLASSES).float()
                        mixed_targets = lam * labels_a_one_hot + (1 - lam) * labels_b_one_hot

                        # Forward pass
                        if use_distillation and hasattr(model, 'extract_features'):
                            outputs, features = model.extract_features(images, return_features=True)
                        else:
                            outputs = model(images)

                        # Apply distillation if enabled
                        if use_distillation:
                            with torch.no_grad():
                                if hasattr(teacher_model, 'extract_features'):
                                    teacher_outputs, teacher_features = teacher_model.extract_features(
                                        images, return_features=True)
                                else:
                                    teacher_outputs = teacher_model(images)
                                    
                            loss = distill_criterion(
                                outputs, teacher_outputs, mixed_targets, criterion,
                                features, teacher_features)
                        else:
                            # For early epochs, use simple cross entropy on the dominant label
                            if epoch < 5:
                                dominant_labels = labels_a if lam > 0.5 else labels_b
                                loss = F.cross_entropy(outputs, dominant_labels)
                            else:
                                # Compute loss for mixed labels
                                loss = criterion(outputs, mixed_targets)
                    except Exception as e:
                        print(f"⚠️ Error applying Mixup: {e}. Reverting to standard forward pass.")
                        # Fallback to standard forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        is_mixup_or_cutmix = False

                elif random.random() < cutmix_probability + mixup_probability and epoch >= 3:
                    # Apply CutMix
                    try:
                        mixed_images, labels_a, labels_b, lam = cutmix_transform((images, labels))
                        images = mixed_images.to(device, non_blocking=True)

                        # Store for accuracy calculation later
                        is_mixup_or_cutmix = True
                        mixup_labels_a = labels_a
                        mixup_labels_b = labels_b
                        mixup_lam = lam

                        # Convert label indices to one-hot for cutmix
                        labels_a_one_hot = F.one_hot(labels_a, NUM_CLASSES).float()
                        labels_b_one_hot = F.one_hot(labels_b, NUM_CLASSES).float()
                        mixed_targets = lam * labels_a_one_hot + (1 - lam) * labels_b_one_hot

                        # Forward pass
                        if use_distillation and hasattr(model, 'extract_features'):
                            outputs, features = model.extract_features(images, return_features=True)
                        else:
                            outputs = model(images)

                        # Apply distillation if enabled
                        if use_distillation:
                            with torch.no_grad():
                                if hasattr(teacher_model, 'extract_features'):
                                    teacher_outputs, teacher_features = teacher_model.extract_features(
                                        images, return_features=True)
                                else:
                                    teacher_outputs = teacher_model(images)
                                    
                            loss = distill_criterion(
                                outputs, teacher_outputs, mixed_targets, criterion,
                                features, teacher_features)
                        else:
                            # For early epochs, use simple cross entropy on the dominant label
                            if epoch < 5:
                                dominant_labels = labels_a if lam > 0.5 else labels_b
                                loss = F.cross_entropy(outputs, dominant_labels)
                            else:
                                # Compute loss for mixed labels
                                loss = criterion(outputs, mixed_targets)
                    except Exception as e:
                        print(f"⚠️ Error applying CutMix: {e}. Reverting to standard forward pass.")
                        # Fallback to standard forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        is_mixup_or_cutmix = False

                else:  # Standard forward pass
                    # Forward pass with potential feature extraction for distillation
                    if use_distillation and hasattr(model, 'extract_features'):
                        outputs, features = model.extract_features(images, return_features=True)
                    else:
                        outputs = model(images)

                    # Apply distillation if enabled
                    if use_distillation:
                        with torch.no_grad():
                            if hasattr(teacher_model, 'extract_features'):
                                teacher_outputs, teacher_features = teacher_model.extract_features(
                                    images, return_features=True)
                            else:
                                teacher_outputs = teacher_model(images)
                                
                        loss = distill_criterion(
                            outputs, teacher_outputs, labels, criterion,
                            features, teacher_features)
                    else:
                        loss = criterion(outputs, labels)

            # Check for NaN loss and skip if found
            if torch.isnan(loss).any():
                print("⚠️ NaN loss detected. Skipping batch update.")
                continue

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
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), GRAD_CLIP_VALUE)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Step scheduler if it's a per-step scheduler
                    if hasattr(scheduler, 'step_count'):  # OneCycleLR or CosineAnnealing*
                        scheduler.step()
            else:
                loss.backward()

                # Gradient accumulation - only update every ACCUMULATION_STEPS
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # Optional gradient clipping
                    if GRAD_CLIP_VALUE > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), GRAD_CLIP_VALUE)

                    optimizer.step()
                    optimizer.zero_grad()

                    # Step scheduler if it's a per-step scheduler
                    if hasattr(scheduler, 'step_count'):  # OneCycleLR or CosineAnnealing*
                        scheduler.step()

            # Compute accuracy - using the stored mixup/cutmix variables
            try:
                # Get predictions (handle potential shape issues)
                if len(outputs.shape) > 1:
                    _, predicted = outputs.max(1)
                else:
                    predicted = (outputs > 0).long()  # Binary case fallback
                
                if is_mixup_or_cutmix and epoch >= 1:
                    # If we used mixup/cutmix, use dominant label for accuracy
                    dominant_labels = mixup_labels_a if mixup_lam > 0.5 else mixup_labels_b
                    batch_correct = predicted.eq(dominant_labels).sum().item()
                else:
                    # Standard accuracy calculation
                    batch_correct = predicted.eq(labels).sum().item()

                # Update metrics
                total_loss += loss.item() * ACCUMULATION_STEPS
                batch_size = images.size(0)
                total += batch_size
                correct += batch_correct

                losses.update(loss.item() * ACCUMULATION_STEPS, batch_size)
                accs.update(100.0 * batch_correct / batch_size, batch_size)
            except Exception as e:
                print(f"⚠️ Error calculating accuracy: {e}")
                # Continue without updating metrics

            batch_time.update(time.time() - end)
            end = time.time()

            # Update progress bar every few batches
            if batch_idx % 10 == 0:
                progress_bar.set_description(
                    f'Batch {batch_idx}/{len(train_loader)}: '
                    f'Loss: {losses.val:.4f}, '
                    f'Acc: {accs.val:.2f}%, '
                    f'Time: {batch_time.val:.2f}s'
                )
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️ CUDA out of memory in batch {batch_idx}. Skipping batch and clearing cache.")
                torch.cuda.empty_cache()
                continue
            elif 'shape mismatch' in str(e) or 'dimension' in str(e) or 'size' in str(e):
                print(f"⚠️ Dimension error in batch {batch_idx}: {e}. Skipping batch.")
                continue
            else:
                print(f"⚠️ Runtime error in batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            print(f"⚠️ Unexpected error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate final metrics
    train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    train_acc = 100.0 * correct / total if total > 0 else 0.0

    # Release memory
    torch.cuda.empty_cache()

    return train_loss, train_acc


def create_model(device):
    """Create and initialize the model based on configuration."""

    if MODEL_TYPE == "EmotionViT":
        print(f"🔹 Creating EmotionViT model with {BACKBONE} backbone")
        model = EmotionViT(backbone=BACKBONE, pretrained=PRETRAINED).to(device)
    elif MODEL_TYPE == "ConvNeXtEmoteNet":
        print(f"🔹 Creating ConvNeXtEmoteNet model with {BACKBONE} backbone")
        model = ConvNeXtEmoteNet(
            backbone=BACKBONE, pretrained=PRETRAINED).to(device)
    else:
        print("⚠️ Unknown model type, defaulting to ConvNeXtEmoteNet")
        model = ConvNeXtEmoteNet(
            backbone=BACKBONE, pretrained=PRETRAINED).to(device)

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

    criterion, optimizer, scheduler, plateau_scheduler, early_stopping, scaler, swa, distill_criterion, teacher_model, distill_start = prepare_training_components(
        model, train_loader)

    start_epoch = RESUME_EPOCH if RESUME_EPOCH else 0
    model = resume_model(device, model)

    # Apply large batch normalization handling - force enable for batch size >= 64
    # This is critical with batch size ~200 to ensure proper batch norm statistics
    print(
        f"🔹 Using large batch size ({BATCH_SIZE}), applying BatchNorm recalibration...")
    from model_utils import update_bn_for_large_batch
    model = update_bn_for_large_batch(train_loader, model, device)

    # Enhanced transforms with stronger augmentation to prevent overfitting
    mixup_transform = MixupTransform(alpha=MIXUP_ALPHA)
    cutmix_transform = CutMixTransform(alpha=CUTMIX_ALPHA)

    # Anti-overfitting message
    print("🔹 Anti-overfitting measures:")
    print(f"   - Weight decay: {WEIGHT_DECAY}")
    print(
        f"   - Dropout rates: Head {HEAD_DROPOUT}, Feature {FEATURE_DROPOUT}")
    print(
        f"   - Augmentation: MixUp {MIXUP_PROB}/{MIXUP_ALPHA}, CutMix {CUTMIX_PROB}/{CUTMIX_ALPHA}")
    print(f"   - Label smoothing: {LABEL_SMOOTHING}")
    print(f"   - Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(
        f"   - SWA starting at epoch: {os.environ.get('SWA_START_EPOCH', 'disabled')}")

    if ENABLE_LR_FINDER:
        print("🔹 Running Learning Rate Finder...")
        log_lrs, losses = find_lr(model, train_loader, optimizer, criterion)
        plot_lr_finder(log_lrs, losses)
        save_lr_finder_plot(log_lrs, losses, os.path.join(
            DATASET_PATH, "lr_finder_plot.png"))
        print("✅ Learning Rate Finder Complete. Adjust learning rate based on the plot.\n")

    print("🔹 Begin training ==============================")
    best_val_acc = 0.0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)

        # Load teacher model for self-distillation if enabled and it's time
        if distill_criterion is not None and epoch == distill_start and teacher_model is None:
            print("🔹 Loading teacher model for self-distillation...")

            # Try to load a checkpoint from ~30% back in training
            if epoch >= 30:
                teacher_epoch = max(1, epoch - 30)
                teacher_path = f"{MODEL_PATH}_epoch_{teacher_epoch}.pth"

                if os.path.exists(teacher_path):
                    teacher_model = create_model(device)
                    teacher_model.load_state_dict(
                        torch.load(teacher_path, map_location=device))
                    teacher_model.eval()  # Set to evaluation mode
                    print(f"✅ Loaded teacher model from epoch {teacher_epoch}")
                else:
                    print(
                        f"⚠️ Teacher checkpoint not found at {teacher_path}. Disabling distillation.")

        # Training phase
        train_loss, train_acc = run_training_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_transform, cutmix_transform, scheduler, scaler, epoch,
            teacher_model, distill_criterion, distill_start
        )

        # Determine if we should skip validation this epoch to speed up training
        # With ultra-fast enabled, only validate every 5 epochs except for specific checkpoints
        skip_validation = False
        ultra_fast = os.getenv("ULTRA_FAST_EVAL", "0") == "1"
        if ultra_fast and epoch > 0:
            # Always validate at important milestones:
            # - First and last epochs
            # - At SWA start epoch
            # - After freeze backbone epochs
            # - When early stopping might be checked
            important_epochs = [
                0,  # First epoch
                FREEZE_BACKBONE_EPOCHS,  # After freezing
                NUM_EPOCHS - 1,  # Last epoch
            ]

            if swa is not None:
                important_epochs.append(swa.swa_start)  # SWA start epoch

            # Only validate every 5 epochs or at important milestones
            skip_validation = (epoch % 5 != 0) and (epoch not in important_epochs) and (
                epoch < NUM_EPOCHS - EARLY_STOPPING_PATIENCE)

        if skip_validation:
            # Use previous validation metrics for logging
            print(
                f"🔹 Skipping validation at epoch {epoch+1} for speed (using previous metrics)")
            # Don't modify val_loss or val_acc
        else:
            # Validation phase with test-time augmentation
            val_loss, val_acc = tta_evaluate(
                model, val_loader, criterion, device)

            # Update learning rate if using ReduceLROnPlateau
            if not hasattr(scheduler, 'step_count'):
                plateau_scheduler.step(val_acc)

            # Update best model
            if os.getenv('DISABLE_CHECKPOINTS') == '0' and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(
                    f"✅ New best model saved with validation accuracy: {val_acc:.2f}%")

            # Early stopping check
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"⚠️ Early stopping triggered at epoch {epoch + 1}")
                break

        # Save checkpoint regardless of validation status
        save_checkpoint(model, epoch, train_loss, train_acc, val_loss, val_acc)

        # Update SWA model if enabled
        if swa is not None:
            swa.update(epoch)

    # Finalize SWA model if enabled
    if swa is not None and swa.swa_n > 0:
        print("🔹 Finalizing SWA model...")
        swa_model = swa.finalize()

        # Only perform SWA evaluation if checkpoints aren't disabled
        disable_checkpoints = os.getenv("DISABLE_CHECKPOINTS", "0") == "1"

        if not disable_checkpoints:
            # Evaluate SWA model
            print("🔹 Evaluating SWA model...")
            swa_val_loss, swa_val_acc = tta_evaluate(
                swa_model, val_loader, criterion, device)
            print(
                f"✅ SWA model accuracy: {swa_val_acc:.2f}% (vs best: {best_val_acc:.2f}%)")

            # Save SWA model if it's better
            if swa_val_acc > best_val_acc:
                swa_model_path = f"{MODEL_PATH}_swa.pth"
                torch.save(swa_model.state_dict(), swa_model_path)
                print(
                    f"✅ SWA model saved with improved accuracy: {swa_val_acc:.2f}%")

                # Update best model if SWA is better
                torch.save(swa_model.state_dict(), MODEL_PATH)
                best_val_acc = swa_val_acc
                # Use SWA model as final model
                from model_utils import save_final_model
                save_final_model(swa_model)
            else:
                # Save the regular model as final
                from model_utils import save_final_model
                save_final_model(model)
        else:
            # When checkpoints are disabled, just directly save SWA as the final model
            print("ℹ️ Skipping SWA evaluation (checkpoints disabled)")
            from model_utils import save_final_model
            save_final_model(swa_model)
            # Also save as the standard model
            torch.save(swa_model.state_dict(), MODEL_PATH)
    else:
        # No SWA, save the regular model as final
        from model_utils import save_final_model
        save_final_model(model)

    print(
        f"✅ Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✅ Final Model Saved at {MODEL_PATH}\n")

    # Only perform ensemble evaluation if checkpoints were enabled
    disable_checkpoints = os.getenv("DISABLE_CHECKPOINTS", "0") == "1"
    if not disable_checkpoints:
        print("🔹 Performing final ensemble evaluation...")
        ensemble_evaluate(device, val_loader, criterion)
    else:
        print("ℹ️ Skipping ensemble evaluation (checkpoints were disabled)")


def prepare_dataloaders(train_path, test_path):
    """Prepare the train and validation dataloaders with enhanced augmentations"""
    print(f"🔹 Using dataset paths: Train={train_path}, Test={test_path}")

    # Define emotion-specific advanced augmentations
    class RandomGaussianBlur(nn.Module):
        """Apply Gaussian blur with random sigma"""
        def __init__(self, kernel_size=3, sigma_range=(0.1, 1.5), p=0.3):
            super().__init__()
            self.kernel_size = kernel_size
            self.sigma_range = sigma_range
            self.p = p
        
        def forward(self, img):
            if random.random() < self.p:
                sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
                return transforms.functional.gaussian_blur(img, self.kernel_size, [sigma, sigma])
            return img
    
    class RandomGrayScale(nn.Module):
        """Apply grayscale with probability p"""
        def __init__(self, p=0.2):
            super().__init__()
            self.p = p
            self.grayscale = transforms.Grayscale(3)
        
        def forward(self, img):
            if random.random() < self.p:
                return self.grayscale(img)
            return img
    
    class RandomSharpen(nn.Module):
        """Apply random sharpening"""
        def __init__(self, p=0.3):
            super().__init__()
            self.p = p
        
        def forward(self, img):
            if random.random() < self.p:
                # Convert tensor to PIL for sharpening
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    img_pil = transforms.ToPILImage()(img)
                else:
                    img_pil = img
                
                # Apply sharpening filter
                enhancer = transforms.functional.adjust_sharpness
                factor = random.uniform(1.0, 3.0)  # Adjust sharpness factor
                return enhancer(img_pil, factor)
            return img

    # Enhanced training transformation pipeline with progressive complexity
    transform_train = transforms.Compose([
        # Resize with a slight random variation for better generalization
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        
        # Spatial augmentations
        transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally
        transforms.RandomRotation(degrees=DEGREES, fill=0),
        transforms.RandomAffine(degrees=0, translate=TRANSLATE, scale=SCALE, fill=0),
        
        # Color augmentations
        transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST,
                               saturation=SATURATION, hue=HUE) if USE_COLOR_JITTER else nn.Identity(),
        RandomGrayScale(p=0.15),  # Occasional grayscale for robustness
        RandomGaussianBlur(p=0.25),  # Occasional blur for robustness
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Additional augmentations after normalization
        transforms.RandomErasing(p=ERASING_PROB, scale=(0.02, 0.15), ratio=(0.3, 3.3)) 
            if USE_RANDOM_ERASING else nn.Identity()
    ])

    # Multi-scale validation transformation for test-time augmentation
    transform_val = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),  # Slightly larger
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # Then center crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets using standard ImageFolder
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transform_train
    )

    # Create validation dataset
    val_dataset = datasets.ImageFolder(
        root=test_path,
        transform=transform_val
    )
    
    # Print dataset statistics
    print(f"Total classes in training dataset: {len(train_dataset.classes)}")
    print(f"Total classes in validation dataset: {len(val_dataset.classes)}")
    print(f"Total samples in validation dataset: {len(val_dataset)}")
    
    # Create DataLoaders with optimal settings for large batch training
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 2),  # Optimize workers based on CPU
        pin_memory=True,
        persistent_workers=True if torch.cuda.is_available() else False,  # Keep workers alive between epochs
        drop_last=True,  # Avoid small last batch
        prefetch_factor=2  # Prefetch batches
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader
