import os
import argparse
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

from model import get_model
from dataset import get_dataloaders, EMOTION_MAP

# Mixup Implementation
class Mixup:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def __call__(self, batch):
        """
        Apply mixup to the batch. Returns mixed inputs, pairs of targets, and lambda
        """
        inputs, targets = batch
        batch_size = inputs.size(0)
        
        # Generate mixup coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Convert to tensor with gradient tracking
        lam = torch.tensor(lam, device=inputs.device, dtype=inputs.dtype, requires_grad=False)
            
        # Create shuffled indices
        index = torch.randperm(batch_size).to(inputs.device)
        
        # Mix the inputs - keep gradients attached
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        
        # Return mixed inputs, pairs of targets, and lambda
        return mixed_inputs, targets, targets[index], lam

# Add Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
        if self.weight is not None:
            # Apply weights to the loss - ensure weight is on same device
            self.weight = self.weight.to(pred.device)
            loss = -(true_dist * pred).sum(dim=self.dim)
            weights_for_batch = self.weight[target]
            return (loss * weights_for_batch).mean()
        else:
            return -(true_dist * pred).sum(dim=self.dim).mean()

# Add Focal Loss to better handle extreme class imbalance
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation for imbalanced classification
        
        Args:
            weight (torch.Tensor): Class weights
            gamma (float): Focusing parameter, higher values give more weight to hard examples
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        # Ensure weight is on the same device
        if self.weight is not None:
            self.weight = self.weight.to(input.device)
            
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Combined Focal and Label Smoothing Loss
class FocalLabelSmoothingLoss(nn.Module):
    def __init__(self, classes, weight=None, gamma=2.0, smoothing=0.1):
        super(FocalLabelSmoothingLoss, self).__init__()
        self.focal = FocalLoss(weight=weight, gamma=gamma, reduction='none')
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight
        self.gamma = gamma
        
    def forward(self, pred, target):
        # Apply label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate focal loss with smoothed labels
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)
        pt = torch.exp(-loss)
        focal_loss = (1 - pt) ** self.gamma * loss
        
        if self.weight is not None:
            # Make sure weight is on the same device
            self.weight = self.weight.to(pred.device)
            weights_for_batch = self.weight[target]
            return (focal_loss * weights_for_batch).mean()
        else:
            return focal_loss.mean()

# For reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Generate HTML report
def generate_html_report(metrics, output_dir, epoch):
    """
    Generate HTML report with training metrics and plots
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report - Epoch {epoch}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .metrics {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
            .metric-box {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; width: 200px; text-align: center; }}
            .charts {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .chart {{ margin: 15px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>EmotionNet Training Report</h1>
                <h2>Epoch {epoch}</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <h3>Train Accuracy</h3>
                    <p>{metrics['train_accuracy'][-1]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Validation Accuracy</h3>
                    <p>{metrics['val_accuracy'][-1]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Train Loss</h3>
                    <p>{metrics['train_loss'][-1]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Validation Loss</h3>
                    <p>{metrics['val_loss'][-1]:.4f}</p>
                </div>
            </div>
            
            <div class="charts">
                <div class="chart">
                    <h3>Accuracy</h3>
                    <img src="accuracy_plot_epoch_{epoch}.png" alt="Accuracy Plot" width="500">
                </div>
                <div class="chart">
                    <h3>Loss</h3>
                    <img src="loss_plot_epoch_{epoch}.png" alt="Loss Plot" width="500">
                </div>
                <div class="chart">
                    <h3>Confusion Matrix (Validation)</h3>
                    <img src="confusion_matrix_epoch_{epoch}.png" alt="Confusion Matrix" width="500">
                </div>
                <div class="chart">
                    <h3>F1 Score</h3>
                    <img src="f1_plot_epoch_{epoch}.png" alt="F1 Score Plot" width="500">
                </div>
            </div>
            
            <h3>Training History</h3>
            <table>
                <tr>
                    <th>Epoch</th>
                    <th>Train Loss</th>
                    <th>Val Loss</th>
                    <th>Train Acc</th>
                    <th>Val Acc</th>
                    <th>Train F1</th>
                    <th>Val F1</th>
                </tr>
    """
    
    for i in range(len(metrics['train_loss'])):
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{metrics['train_loss'][i]:.4f}</td>
                    <td>{metrics['val_loss'][i]:.4f}</td>
                    <td>{metrics['train_accuracy'][i]:.4f}</td>
                    <td>{metrics['val_accuracy'][i]:.4f}</td>
                    <td>{metrics['train_f1'][i]:.4f}</td>
                    <td>{metrics['val_f1'][i]:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    report_path = os.path.join(output_dir, f'training_report_epoch_{epoch}.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Create latest report symlink
    latest_report_path = os.path.join(output_dir, 'latest_training_report.html')
    if os.path.exists(latest_report_path):
        os.remove(latest_report_path)
    
    # Write to the latest report file directly instead of symlinking
    with open(latest_report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

# Plot and save metrics
def plot_metrics(metrics, output_dir, epoch):
    """
    Plot and save training metrics
    """
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(range(1, epoch+1), metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_plot_epoch_{epoch}.png'))
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), metrics['train_loss'], label='Train Loss')
    plt.plot(range(1, epoch+1), metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'loss_plot_epoch_{epoch}.png'))
    plt.close()
    
    # F1 Score plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), metrics['train_f1'], label='Train F1')
    plt.plot(range(1, epoch+1), metrics['val_f1'], label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'f1_plot_epoch_{epoch}.png'))
    plt.close()
    
    # Save latest versions of all plots
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(range(1, epoch+1), metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latest_accuracy_plot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), metrics['train_loss'], label='Train Loss')
    plt.plot(range(1, epoch+1), metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latest_loss_plot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), metrics['train_f1'], label='Train F1')
    plt.plot(range(1, epoch+1), metrics['val_f1'], label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latest_f1_plot.png'))
    plt.close()

# Create confusion matrix with emotion names
def plot_confusion_matrix(cm, output_path, title="Confusion Matrix"):
    """
    Plot confusion matrix with emotion names as labels
    """
    plt.figure(figsize=(12, 10))
    
    # Create labels for the plot
    emotion_labels = [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))]
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, 
                yticklabels=emotion_labels)
    
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Calculate class weights based on class distribution
def calculate_class_weights(labels):
    """
    Calculate class weights based on the inverse of class frequency
    with custom weight adjustments for certain classes
    """
    class_counts = np.bincount(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    # Initialize weights using inverse class frequency
    weights = torch.zeros(n_classes)
    for i in range(n_classes):
        if class_counts[i] > 0:
            weights[i] = n_samples / (n_classes * class_counts[i])
        else:
            weights[i] = 0
    
    # Manual adjustments for specific classes
    # Balance Disgust weight to encourage some predictions but not too many
    weights[1] = 0.7  # Disgust - increased from 0.1 to 0.7
    # Increase Fear weight as it's underperforming
    weights[2] = 1.2  # Fear
    
    return weights

# Get weighted sampler for handling class imbalance
def get_sampler(dataset):
    """
    Create a weighted sampler to handle class imbalance
    """
    labels = dataset.data['emotion'].values
    class_counts = np.bincount(labels)
    
    # Calculate weights inversely proportional to class frequencies
    weights = 1. / class_counts
    sample_weights = weights[labels]
    
    # Create a weighted random sampler
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler

# Define the training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, output_dir, num_epochs=25, mixed_precision=True, enable_dynamic_weights=True):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1': [],
        'val_f1': [],
        'confusion_matrices': [],
        'class_f1_scores': []
    }
    
    # Initialize gradient scaler for mixed precision training
    mixed_precision = False  # Force disable mixed precision for debugging
    scaler = GradScaler() if mixed_precision else None
    
    # Early stopping parameters with increased patience for better convergence
    patience = 15  # Increased from 7
    counter = 0
    best_val_loss = float('inf')
    
    # Learning rate warmup parameters
    warmup_epochs = 5
    warmup_factor = 10  # Initial LR will be LR/warmup_factor
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Print model parameters to confirm they require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"WARNING: Parameter {name} does not require gradients!")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Implement learning rate warmup
        if epoch < warmup_epochs:
            # Gradually increase LR from initial_lr/warmup_factor to initial_lr
            warmup_lr = initial_lr / warmup_factor + (initial_lr - initial_lr / warmup_factor) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.6f}")
        
        epoch_metrics = {}
        epoch_class_f1 = {}
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            all_preds = []
            all_labels = []
            
            # Create progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()}')
            
            # Iterate over data
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)  # More efficient zeroing
                
                # Track history if only in train phase - SIMPLIFIED APPROACH
                if phase == 'train':
                    # Simple forward pass without any fancy features
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # DEBUG: Check if loss requires gradients
                    if not loss.requires_grad:
                        print("WARNING: Loss does not require gradients!")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Add gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # DEBUG: Check for NaN gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"WARNING: NaN gradients in {name}")
                    
                    # Update weights
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                
                # Get predictions and calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct = torch.sum(preds == labels.data).item()
                running_corrects += correct
                batch_acc = correct / batch_size
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar with metrics
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            # Set zero_division=0 to avoid warnings
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Calculate per-class F1 scores
            class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            
            # Count class predictions to check balance
            class_predictions = np.bincount(all_preds, minlength=len(EMOTION_MAP))
            total_preds = sum(class_predictions)
            
            # Print prediction distribution
            if phase == 'train':
                print("\nPrediction distribution:")
                prediction_distribution = []
                for class_idx, count in enumerate(class_predictions):
                    emotion_name = EMOTION_MAP[class_idx]
                    percentage = (count / total_preds) * 100 if total_preds > 0 else 0
                    prediction_distribution.append(percentage / 100.0)  # Store as decimal
                    print(f"  {emotion_name}: {count} predictions ({percentage:.2f}%)")
            
            # Check if predictions are too biased toward Disgust
            disgust_percentage = (class_predictions[1] / total_preds) * 100 if total_preds > 0 else 0
            if disgust_percentage > 30 and phase == 'train':
                print("\n⚠️ WARNING: Model is predicting Disgust too frequently!")
                print(f"   Disgust predictions: {disgust_percentage:.2f}% of all predictions")
            elif disgust_percentage == 0 and phase == 'train':
                print("\n⚠️ WARNING: Model is not predicting any Disgust samples!")
            
            # Check if the model predicts any Disgust class samples
            disgust_predicted = 1 in all_preds
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(len(EMOTION_MAP)))
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
            
            # Print per-class F1 scores
            print(f'{phase} Per-class F1 scores:')
            
            for class_idx in range(len(EMOTION_MAP)):
                emotion_name = EMOTION_MAP[class_idx]
                if str(class_idx) in class_report:
                    class_f1 = class_report[str(class_idx)]['f1-score']
                    print(f"  {emotion_name}: {class_f1:.4f}")
                    # Store per-class F1 for this epoch
                    epoch_class_f1[emotion_name] = class_f1
            
            # Store metrics for this phase
            if phase == 'train':
                metrics['train_loss'].append(epoch_loss)
                metrics['train_accuracy'].append(epoch_acc)
                metrics['train_f1'].append(epoch_f1)
                epoch_metrics['train_loss'] = epoch_loss
                epoch_metrics['train_accuracy'] = epoch_acc
                epoch_metrics['train_f1'] = epoch_f1
            else:
                metrics['val_loss'].append(epoch_loss)
                metrics['val_accuracy'].append(epoch_acc)
                metrics['val_f1'].append(epoch_f1)
                epoch_metrics['val_loss'] = epoch_loss
                epoch_metrics['val_accuracy'] = epoch_acc
                epoch_metrics['val_f1'] = epoch_f1
                
                # Early stopping check
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Load best model weights and return
                        model.load_state_dict(best_model_wts)
                        return model, best_acc, best_f1, metrics
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                
        # Store per-class F1 scores for this epoch
        metrics['class_f1_scores'].append(epoch_class_f1)
        epoch_metrics['class_f1_scores'] = epoch_class_f1
                
        # Save metrics as JSON after each epoch
        with open(os.path.join(output_dir, f'metrics_epoch_{epoch+1}.json'), 'w') as f:
            json.dump(epoch_metrics, f, indent=4)
            
        # Save all metrics to date
        with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Plot and save metrics
        plot_metrics(metrics, output_dir, epoch+1)
        
        # Generate HTML report
        report_path = generate_html_report(metrics, output_dir, epoch+1)
        print(f"Generated HTML report: {report_path}")
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}, F1: {best_f1:.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Final training curves saved in the last epoch reports
    
    return model, best_acc, best_f1, metrics

def evaluate_model(model, test_loader, device, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Create progress bar for evaluation
    progress_bar = tqdm(test_loader, desc='Evaluating')
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Calculate batch accuracy for progress bar
            batch_acc = torch.sum(preds == labels.to(device)).item() / inputs.size(0)
            progress_bar.set_postfix({'batch_acc': f'{batch_acc:.4f}'})
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Get per-class metrics
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    # Count class predictions to check balance
    class_predictions = np.bincount(all_preds, minlength=len(EMOTION_MAP))
    total_preds = sum(class_predictions)
    
    # Print prediction distribution
    print("\nTest Prediction distribution:")
    for class_idx, count in enumerate(class_predictions):
        emotion_name = EMOTION_MAP[class_idx]
        percentage = (count / total_preds) * 100 if total_preds > 0 else 0
        print(f"  {emotion_name}: {count} predictions ({percentage:.2f}%)")
    
    # Check if the model predicts any Disgust class samples
    disgust_predicted = 1 in all_preds
    if not disgust_predicted:
        print("\n⚠️ WARNING: Model is not predicting any Disgust class samples in the test set!")
    
    # Create confusion matrix with emotion names
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm=cm, 
        output_path=os.path.join(output_dir, 'final_confusion_matrix.png'),
        title='Final Test Confusion Matrix'
    )
    
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    
    # Print per-class F1 scores
    print("Test Per-class F1 scores:")
    class_f1_dict = {}
    for class_idx in range(len(EMOTION_MAP)):
        emotion_name = EMOTION_MAP[class_idx]
        if str(class_idx) in class_report:
            class_f1 = class_report[str(class_idx)]['f1-score']
            class_f1_dict[emotion_name] = class_f1
            print(f"  {emotion_name}: {class_f1:.4f}")
    
    # Save test metrics as JSON
    test_metrics = {
        'accuracy': test_acc,
        'f1_score': test_f1,
        'confusion_matrix': cm.tolist(),
        'class_report': class_report,
        'class_f1_scores': class_f1_dict
    }
    
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    return test_acc, test_f1

# Add function to generate final report (was missing)
def generate_final_report(train_metrics, test_metrics, output_dir):
    """
    Generate a comprehensive final report for the training run
    """
    # Create a report combining training and test metrics
    report_path = os.path.join(output_dir, 'final_report.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EmotionNet - Final Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .metrics {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
            .metric-box {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; width: 200px; text-align: center; }}
            .charts {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .chart {{ margin: 15px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>EmotionNet Final Training Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <h3>Final Training Accuracy</h3>
                    <p>{train_metrics['train_accuracy'][-1]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Final Validation Accuracy</h3>
                    <p>{train_metrics['val_accuracy'][-1]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Test Accuracy</h3>
                    <p>{test_metrics[0]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Test F1 Score</h3>
                    <p>{test_metrics[1]:.4f}</p>
                </div>
            </div>
            
            <div class="charts">
                <div class="chart">
                    <h3>Accuracy</h3>
                    <img src="latest_accuracy_plot.png" alt="Accuracy Plot" width="500">
                </div>
                <div class="chart">
                    <h3>Loss</h3>
                    <img src="latest_loss_plot.png" alt="Loss Plot" width="500">
                </div>
                <div class="chart">
                    <h3>F1 Score</h3>
                    <img src="latest_f1_plot.png" alt="F1 Score Plot" width="500">
                </div>
                <div class="chart">
                    <h3>Confusion Matrix (Test)</h3>
                    <img src="final_confusion_matrix.png" alt="Confusion Matrix" width="500">
                </div>
            </div>
            
            <div class="summary">
                <h3>Training Summary</h3>
                <p>The model was trained for {len(train_metrics['train_loss'])} epochs with early stopping.</p>
                <p>Best validation accuracy: {max(train_metrics['val_accuracy']):.4f}</p>
                <p>Best validation F1 score: {max(train_metrics['val_f1']):.4f}</p>
                <p>Final test accuracy: {test_metrics[0]:.4f}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Train a model on FER2013 dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/fer2013', help='Data directory')
    parser.add_argument('--csv_file', type=str, default='dataset/fer2013/icml_face_data.csv', help='CSV file with data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')  # Increased batch size
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')  # Increased epochs
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')  # Adjusted learning rate
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--balance_classes', action='store_true', help='Use techniques to handle class imbalance')
    parser.add_argument('--focal_loss', action='store_true', help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--disgust_weight_multiplier', type=float, default=0.7, 
                       help='Multiplier for Disgust class weight (to prevent overprediction)')
    parser.add_argument('--mixup', action='store_true', help='Use mixup augmentation')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save run parameters with timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_params = vars(args)
    run_params['timestamp'] = run_timestamp
    run_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save parameters to JSON file
    params_file = os.path.join(args.output_dir, f'run_params_{run_timestamp}.json')
    with open(params_file, 'w') as f:
        json.dump(run_params, f, indent=4)
    print(f"Saved run parameters to {params_file}")
    
    # Also save to a "latest" file
    latest_params_file = os.path.join(args.output_dir, 'latest_run_params.json')
    with open(latest_params_file, 'w') as f:
        json.dump(run_params, f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get dataloaders with class balancing if requested
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_file=args.csv_file,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_classes=args.balance_classes
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Read the dataset to get label distribution for loss setup
    import pandas as pd
    data = pd.read_csv(args.csv_file)
    train_data = data[data[' Usage'] == 'Training']
    train_labels = train_data['emotion'].values
    
    # Calculate class weights with special handling for certain classes
    class_weights = calculate_class_weights(train_labels)
    
    # Normalize the weights
    class_weights = class_weights / class_weights.sum() * len(EMOTION_MAP)
    
    # Move weights to device
    class_weights = class_weights.to(device)
    
    # Print class distribution and weights
    class_counts = np.bincount(train_labels)
    print("\nClass distribution in training set:")
    print("-" * 50)
    print(f"{'Emotion':<10} {'Count':<8} {'Percentage':<12} {'Weight'}")
    print("-" * 50)
    
    for class_idx, count in enumerate(class_counts):
        emotion = EMOTION_MAP[class_idx]
        percentage = count / len(train_labels) * 100
        weight = class_weights[class_idx].item()
        print(f"{emotion:<10} {count:<8} {percentage:>6.2f}%       {weight:.4f}")
    
    print("-" * 50)
    
    # Use CrossEntropyLoss with label smoothing and class weights
    print(f"Using CrossEntropyLoss with label smoothing={args.label_smoothing} and class weights")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    
    # Get model
    model = get_model(num_classes=7)
    model = model.to(device)
    
    # Verify all parameters require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"WARNING: Parameter {name} does not require gradients. Setting requires_grad=True")
            param.requires_grad = True
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        [
            {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
            {'params': model.channel_attention.parameters()},
            {'params': model.self_attention.parameters()},
            {'params': model.transformer_encoder.parameters()},
            {'params': model.fusion.parameters()},
            {'params': model.fc1.parameters()},
            {'params': model.bn1.parameters()},
            {'params': model.fc2.parameters()},
            {'params': model.bn2.parameters()},
            {'params': model.fc3.parameters()},
        ],
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Use OneCycleLR scheduler for better convergence
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr, args.lr, args.lr, args.lr, args.lr, args.lr, args.lr, args.lr, args.lr],
        steps_per_epoch=steps_per_epoch,
        epochs=args.num_epochs,
        pct_start=0.2,  # Use 20% of training for warmup
        div_factor=10,  # Initial learning rate will be max_lr/10
        final_div_factor=100,  # Final learning rate will be max_lr/100
    )
    
    # Train the model
    model, best_acc, best_f1, metrics = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        mixed_precision=False  # Force disable mixed precision
    )
    
    # Test the model on the test set
    test_metrics = evaluate_model(model, test_loader, device, args.output_dir)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
    
    # Save test metrics
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Generate final report
    final_report_path = generate_final_report(metrics, test_metrics, args.output_dir)
    print(f"Generated final report: {final_report_path}")

if __name__ == '__main__':
    main() 