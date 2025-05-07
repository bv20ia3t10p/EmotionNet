import os
import csv
import torch  # type: ignore
from datetime import datetime
from config import *
from PIL import Image, ImageOps, ImageFilter, ImageEnhance  # type: ignore
import random
import matplotlib.pyplot as plt  # type: ignore
import math
from torch.cuda.amp import autocast, GradScaler # type: ignore
from collections import Counter
import shutil
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# Utility for tracking averages of values during training
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def remove_module_prefix(state_dict):
    """Remove the '_module.' prefix from keys in the state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_module.", "")  # Remove the '_module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

# Utility Functions


def log_configuration():
    """Log configuration constants to a file with a unique name."""
    print("🔹 Logging Configuration Constants ==============================")

    config_data = {
        "ROOT": ROOT,
        "DATASET_PATH": DATASET_PATH,
        "MODEL_PATH": MODEL_PATH,
        "LOG_CSV_PATH": LOG_CSV_PATH,
        "RESUME_EPOCH": RESUME_EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_EPOCHS": NUM_EPOCHS
    }

    for key, value in config_data.items():
        print(f"   🔹 {key}: {value}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_config_log_path = f"{ROOT}/config_log_{timestamp}.txt"

    with open(unique_config_log_path, 'w') as f:
        f.write("Configuration Constants:\n")
        for key, value in config_data.items():
            f.write(f"{key}: {value}\n")

    print(f"✅ Configuration logged to {unique_config_log_path}\n")

# Utility Functions


def log_csv(epoch, train_loss, train_acc, val_loss, val_acc):
    file_exists = os.path.isfile(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Train Acc",
                            "Val Loss", "Val Acc", "Timestamp"])
        writer.writerow([epoch, train_loss, train_acc, val_loss,
                        val_acc, datetime.now().isoformat()])


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a given dataset."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), 100 * correct / total


def ensure_images_per_class(dataset_path, target_num_images=7500):
    """Ensure each class in the dataset has the target number of images."""
    def augment_image(image):
        """Apply random augmentations to an image."""
        augmentations = [
            lambda x: x.rotate(random.randint(-30, 30)),  # Random rotation
            lambda x: ImageOps.mirror(x),  # Horizontal flip
            lambda x: ImageOps.crop(x, border=random.randint(0, 10)),  # Random crop
        ]
        augmentation = random.choice(augmentations)
        return augmentation(image)

    # Iterate through each class folder
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(
            class_path) if f.endswith(('.jpg', '.png'))]
        num_images = len(images)

        if num_images < target_num_images:
            for i in range(target_num_images - num_images):
                original_image_path = os.path.join(
                    class_path, images[i % num_images])
                with Image.open(original_image_path) as img:
                    augmented_image = augment_image(img)
                    new_image_name = f'aug_{num_images + i + 1}.jpg'
                    augmented_image.save(os.path.join(
                        class_path, new_image_name))

    print("✅ Ensured each class has the target number of images.")


def get_image_stats(dataset_path):
    """Get the number of images per class in the dataset."""
    stats = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(
                class_path) if f.endswith(('.jpg', '.png'))]
            stats[class_name] = len(images)
    return stats


def print_image_stats(train_path, val_path):
    """Print the number of images per class for train and validation datasets."""
    print("🔹 Train Dataset Stats ==============================")
    train_stats = get_image_stats(train_path)
    for class_name, count in train_stats.items():
        print(f"   🔹 {class_name}: {count} images")

    print("\n🔹 Validation Dataset Stats:")
    val_stats = get_image_stats(val_path)
    for class_name, count in val_stats.items():
        print(f"   🔹 {class_name}: {count} images")

    print("✅ Image statistics printed successfully\n")


class EarlyStopping:
    """Early stopping to terminate training when validation accuracy stops improving."""

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def find_lr(model, train_loader, optimizer, criterion, init_value=1e-8, final_value=1.0, beta=0.98):
    device = next(model.parameters()).device
    num = min(len(train_loader) - 1, 100)  # Limit to 100 batches
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    
    # Set LR for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Initialize mixed precision training
    scaler = GradScaler() if USE_AMP else None
    
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    
    # Save original model state to restore later
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for i, data in enumerate(train_loader):
        if i >= num:
            break
            
        batch_num += 1
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if USE_AMP and scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Check if loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            # Restore original model state
            model.load_state_dict(original_state)
            return log_lrs, losses

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Record loss and learning rate
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Update learning rate
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Print progress
        if batch_num % 10 == 0:
            print(f"Batch {batch_num}/{num}, LR: {lr:.8f}, Loss: {smoothed_loss:.4f}", end='\r')

    # Restore original model state
    model.load_state_dict(original_state)
    return log_lrs, losses


def plot_lr_finder(log_lrs, losses):
    plt.plot(log_lrs, losses)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()


def save_lr_finder_plot(log_lrs, losses, filename="lr_finder_plot.png"):
    plt.plot(log_lrs, losses)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.savefig(filename)
    print(f"Learning rate finder plot saved as {filename}")


class nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass


def oversample_minority_classes(dataset, target_samples_per_class=None):
    """
    Oversample minority classes to reduce class imbalance.
    
    Args:
        dataset: ImageFolder dataset with samples attribute
        target_samples_per_class: Target number of samples per class. If None,
                                 uses the count of the majority class
    
    Returns:
        List of indices to use for balanced sampling
    """
    # Get class distribution
    class_counts = Counter([target for _, target in dataset.samples])
    print(f"🔹 Original class distribution: {dict(class_counts)}")
    
    # Set target count to the majority class if not specified
    if target_samples_per_class is None:
        target_samples_per_class = max(class_counts.values())
    
    # Create indices by class
    class_indices = {class_idx: [] for class_idx in range(len(dataset.classes))}
    
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_indices[class_idx].append(idx)
    
    # Create balanced sample indices with oversampling
    balanced_indices = []
    
    for class_idx, indices in class_indices.items():
        # Calculate how many times to repeat each sample
        if len(indices) == 0:
            continue
            
        # Repeat samples to reach target count
        current_count = len(indices)
        repeat_factor = target_samples_per_class / current_count
        
        # Full repeats
        full_copies = int(repeat_factor)
        for _ in range(full_copies):
            balanced_indices.extend(indices)
        
        # Partial repeat for the remainder
        remainder = int((repeat_factor - full_copies) * current_count)
        if remainder > 0:
            balanced_indices.extend(random.sample(indices, remainder))
    
    print(f"🔹 After oversampling: {len(balanced_indices)} samples")
    
    return balanced_indices


def balance_dataset_advanced(dataset_path, target_samples=5000):
    """
    Advanced dataset balancing based on recent research:
    1. For majority classes: Apply instance hardness threshold (keep harder examples)
    2. For minority classes: Use synthetic augmentation + original samples
    """
    try:
        print(f"👉 Starting advanced dataset balancing process...")
        
        # Create output directory for balanced dataset
        balanced_dir = os.path.join(os.path.dirname(dataset_path), "balanced_" + os.path.basename(dataset_path))
        
        # Clean up any existing balanced directory
        if os.path.exists(balanced_dir):
            print(f"👉 Removing existing balanced directory: {balanced_dir}")
            shutil.rmtree(balanced_dir)
            print(f"✅ Removed existing balanced directory")
            
        # Create fresh directory
        os.makedirs(balanced_dir, exist_ok=True)
        print(f"👉 Created fresh balanced output directory: {balanced_dir}")
        
        # Get class distribution
        class_counts = {}
        class_dirs = {}
        
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                class_counts[class_name] = len(image_files)
                class_dirs[class_name] = class_path
                
                # Create corresponding directory in balanced dataset
                os.makedirs(os.path.join(balanced_dir, class_name), exist_ok=True)
        
        print(f"👉 Original class distribution: {class_counts}")
        
        # Load the model for difficulty estimation (use our emotion recognition model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"👉 Using device: {device} for processing")
        
        # Advanced augmentation for minority classes
        minority_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2),
                                   fill=0),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
        
        # Process each class
        for class_name, count in class_counts.items():
            source_dir = class_dirs[class_name]
            target_dir = os.path.join(balanced_dir, class_name)
            
            # Strategy for majority classes (count > target_samples)
            if count > target_samples:
                print(f"👉 Reducing majority class {class_name} from {count} to {target_samples}")
                
                # Copy all samples first but will select subset later
                image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                # Using uniform random sampling for simplicity
                # In production, would use instance hardness with model predictions
                selected_files = np.random.choice(image_files, target_samples, replace=False)
                
                # Copy selected files
                for file in selected_files:
                    shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))
                    
            # Strategy for minority classes (count < target_samples)
            elif count < target_samples:
                print(f"👉 Augmenting minority class {class_name} from {count} to {target_samples}")
                
                # Copy all original samples
                image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for file in image_files:
                    shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))
                
                # Generate synthetic samples through advanced augmentation
                samples_to_generate = target_samples - count
                original_indices = list(range(count))
                
                # Use batched approach for large augmentations
                batch_size = 100
                for batch_start in range(0, samples_to_generate, batch_size):
                    batch_end = min(batch_start + batch_size, samples_to_generate)
                    batch_size_actual = batch_end - batch_start
                    print(f"👉 Generating batch {batch_start}-{batch_end} of {samples_to_generate} augmented samples for {class_name}")
                    
                    for i in range(batch_start, batch_end):
                        # Select a random original image
                        idx = np.random.choice(original_indices)
                        orig_file = image_files[idx]
                        
                        try:
                            # Open and apply augmentation
                            with Image.open(os.path.join(source_dir, orig_file)) as img:
                                # Apply multiple augmentations sequentially for more diversity
                                augmented = img.copy()
                                for _ in range(3):  # Apply 3 random augmentations
                                    augmented = minority_transform(augmented)
                                
                                # Save the augmented image
                                new_filename = f"aug_{class_name}_{i}.jpg"
                                augmented.save(os.path.join(target_dir, new_filename))
                        except Exception as e:
                            print(f"⚠️ Error processing image {orig_file}: {str(e)}")
                            # Try a different file as fallback
                            fallback_idx = np.random.choice(original_indices)
                            fallback_file = image_files[fallback_idx]
                            try:
                                with Image.open(os.path.join(source_dir, fallback_file)) as img:
                                    augmented = img.copy()
                                    augmented = minority_transform(augmented)
                                    new_filename = f"aug_{class_name}_{i}.jpg"
                                    augmented.save(os.path.join(target_dir, new_filename))
                            except Exception as e2:
                                print(f"❌ Failed with fallback image too: {str(e2)}")
                                # Just copy original as last resort
                                shutil.copy(
                                    os.path.join(source_dir, fallback_file),
                                    os.path.join(target_dir, f"copy_{class_name}_{i}.jpg")
                                )
            
            # For classes that match the target count, just copy all
            else:
                print(f"👉 Class {class_name} already has {count} samples (target: {target_samples})")
                image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for file in image_files:
                    shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))
        
        # Verify final distribution
        final_counts = {}
        for class_name in os.listdir(balanced_dir):
            class_path = os.path.join(balanced_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                final_counts[class_name] = len(image_files)
        
        print(f"👉 Final balanced distribution: {final_counts}")
        print(f"✅ Successfully created balanced dataset at: {balanced_dir}")
        return balanced_dir
        
    except Exception as e:
        import traceback
        print(f"❌ Error in balance_dataset_advanced: {str(e)}")
        traceback.print_exc()
        return None
