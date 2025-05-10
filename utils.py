import os
import csv
from datetime import datetime
from config import *
import matplotlib.pyplot as plt  # type: ignore
import math
from torch.cuda.amp import autocast, GradScaler  # type: ignore


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


def get_image_stats(dataset_path):
    """Get the number of images per class in the dataset."""
    stats = {}

    # Check if dataset_path contains debugging output or newlines
    if '\n' in dataset_path:
        # Extract the actual path from the string
        import re
        match = re.search(r'TRAIN_PATH: (.*?)(?:\n|$)', dataset_path)
        if match:
            dataset_path = match.group(1).strip()
            print(f"⚠️ Fixed dataset path: {dataset_path}")
        else:
            print(f"⚠️ Warning: Invalid dataset path format: {dataset_path}")
            # Default to the path from config
            from config import TRAIN_PATH as CONFIG_TRAIN_PATH
            dataset_path = CONFIG_TRAIN_PATH
            print(f"⚠️ Using default path from config: {dataset_path}")

    # Ensure path exists
    if not os.path.exists(dataset_path):
        print(f"⚠️ Error: Dataset path does not exist: {dataset_path}")
        return {}

    print(f"🔹 Getting image stats for: {dataset_path}")

    try:
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(
                    class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                stats[class_name] = len(images)
    except Exception as e:
        print(f"⚠️ Error when getting image stats: {str(e)}")
        return {}

    return stats


def print_image_stats(train_path, val_path):
    """Print the number of images per class for train and validation datasets."""
    print("🔹 Train Dataset Stats ==============================")
    train_stats = get_image_stats(train_path)

    if not train_stats:
        print("⚠️ No training data found or error accessing training directory")
    else:
        for class_name, count in train_stats.items():
            print(f"   🔹 {class_name}: {count} images")

    print("\n🔹 Validation Dataset Stats:")
    val_stats = get_image_stats(val_path)

    if not val_stats:
        print("⚠️ No validation data found or error accessing validation directory")
    else:
        for class_name, count in val_stats.items():
            print(f"   🔹 {class_name}: {count} images")

    print("✅ Image statistics process completed\n")


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
            print(
                f"Batch {batch_num}/{num}, LR: {lr:.8f}, Loss: {smoothed_loss:.4f}", end='\r')

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
