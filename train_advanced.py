import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import json

from models import create_model
from dataset import get_data_loaders, MixUp, CutMix
from generate_report import generate_html_report

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        smooth_loss = loss / n
        eps_i = self.epsilon / n
        loss = (1 - self.epsilon) * nll + eps_i * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AdvancedTrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Initialize loss functions
        self._setup_losses()
        
        # Initialize optimizer
        self._setup_optimizer()
        
        # Initialize scheduler
        self._setup_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.patience_counter = 0
        
    def _setup_losses(self):
        """Setup loss functions based on configuration"""
        loss_type = self.config.get('loss_type', 'focal')
        
        if loss_type == 'focal':
            # Calculate class weights for focal loss
            class_weights = self.config.get('class_weights', None)
            if class_weights is not None:
                alpha = torch.tensor(class_weights, dtype=torch.float32)
            else:
                # Default balanced weights
                alpha = torch.ones(7) * 0.25
                alpha[1] = 2.0  # Boost Disgust class
                alpha[2] = 1.5  # Boost Fear class
            
            self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
            
        elif loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # MixUp/CutMix loss handling
        self.mixup_fn = MixUp(alpha=self.config.get('mixup_alpha', 0.2)) if self.config.get('use_mixup', False) else None
        self.cutmix_fn = CutMix(alpha=self.config.get('cutmix_alpha', 1.0)) if self.config.get('use_cutmix', False) else None
        
    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates"""
        # Separate parameters into backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Different learning rates for backbone and head
        lr = self.config.get('lr', 1e-3)
        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': lr}
        ]
        
        # Choose optimizer
        opt_type = self.config.get('optimizer', 'adamw')
        if opt_type == 'adamw':
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.config.get('weight_decay', 0.01))
        elif opt_type == 'sgd':
            self.optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True, weight_decay=self.config.get('weight_decay', 0.0001))
        else:
            self.optimizer = optim.Adam(param_groups)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine_warm_restarts')
        
        if scheduler_type == 'cosine_warm_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.config.get('T_0', 10), 
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 50),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
    
    def _mixup_criterion(self, pred, y_a, y_b, lam):
        """Compute loss for MixUp/CutMix"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        
        # Gradient accumulation setup
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Apply MixUp or CutMix
            if self.training and np.random.rand() < 0.5:
                if self.mixup_fn and np.random.rand() < 0.5:
                    inputs, labels_a, labels_b, lam = self.mixup_fn(inputs, labels)
                    mixed = True
                elif self.cutmix_fn:
                    inputs, labels_a, labels_b, lam = self.cutmix_fn(inputs, labels)
                    mixed = True
                else:
                    mixed = False
            else:
                mixed = False
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    if mixed:
                        loss = self._mixup_criterion(outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                    loss = loss / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad', 1.0))
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                if mixed:
                    loss = self._mixup_criterion(outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad', 1.0))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * accumulation_steps * inputs.size(0)
            
            if not mixed:
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        # Calculate metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Calculate F1 score
        if all_preds:
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            epoch_f1 = 0.0
        
        return epoch_loss, epoch_acc.item(), epoch_f1
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Test-time augmentation
                if self.config.get('use_tta', True):
                    # Original + flipped
                    outputs1 = self.model(inputs)
                    outputs2 = self.model(torch.flip(inputs, dims=[3]))
                    outputs = (outputs1 + outputs2) / 2
                else:
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc.item(), epoch_f1, all_preds, all_labels
    
    def train(self, train_loader, val_loader, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch + 1)
            
            # Validation phase
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log results
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{self.config["epochs"]}')
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'config': self.config
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f'Saved best model with val_acc: {val_acc:.4f}')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            # Generate confusion matrix periodically
            if (epoch + 1) % 5 == 0:
                self._plot_confusion_matrix(val_labels, val_preds, epoch + 1, save_dir)
        
        return self.history
    
    def _plot_confusion_matrix(self, true_labels, pred_labels, epoch, save_dir):
        """Plot and save confusion matrix"""
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_names, 
                    yticklabels=emotion_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()

def main():
    # Configuration
    config = {
        'backbone': 'convnext_xlarge',
        'batch_size': 16,
        'img_size': 224,
        'epochs': 30,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine_warm_restarts',
        'T_0': 5,
        'T_mult': 2,
        'loss_type': 'focal',
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'use_amp': True,
        'gradient_accumulation_steps': 2,
        'clip_grad': 1.0,
        'use_tta': True,
        'patience': 10,
        'dropout_rate': 0.5
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = create_model(
        model_type='single',
        backbone_name=config['backbone'],
        num_classes=7,
        dropout_rate=config['dropout_rate'],
        use_attention=True,
        pretrained=True
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        test_csv='data/test.csv',
        img_dir='data/images',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=4,
        use_weighted_sampler=True,
        use_mixup=config['use_mixup'],
        use_cutmix=config['use_cutmix']
    )
    
    # Create trainer
    trainer = AdvancedTrainer(model, device, config)
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Load best model
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = trainer.validate(test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    
    # Generate final report
    class_report = classification_report(test_labels, test_preds, 
                                       target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                                       output_dict=True)
    
    print("\nPer-class performance:")
    for emotion, metrics in class_report.items():
        if emotion not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{emotion}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

if __name__ == '__main__':
    main() 