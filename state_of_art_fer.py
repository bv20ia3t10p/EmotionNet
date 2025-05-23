import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from models import create_model, EmotionClassifier
from dataset import get_data_loaders, MixUp, CutMix
import timm
from torchvision import transforms
import cv2

class AdaBelief(optim.Optimizer):
    """AdaBelief Optimizer - Adapting Stepsizes by the Belief in Observed Gradients"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['v_hat'] = torch.zeros_like(p.data)
                        
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                grad_residual = grad - m
                v.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                
                if group['amsgrad']:
                    v_hat = state['v_hat']
                    torch.max(v_hat, v, out=v_hat)
                    denom = (v_hat.sqrt() / bias_correction2.sqrt()).add_(group['eps'])
                else:
                    denom = (v.sqrt() / bias_correction2.sqrt()).add_(group['eps'])
                    
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                    
                # Update parameters
                p.data.add_(m / bias_correction1, alpha=-group['lr'] / denom)
                
        return loss

class PolyLoss(nn.Module):
    """PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"""
    def __init__(self, epsilon=2.0, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # probability of true class
        poly_loss = ce + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss

class SelfAttentionPooling(nn.Module):
    """Self-attention pooling layer for better feature aggregation"""
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size = x.size(0)
        channels = x.size(1)
        
        # Reshape to (batch, channels, height*width)
        x_flat = x.view(batch_size, channels, -1)
        
        # Transpose to (batch, height*width, channels)
        x_flat = x_flat.transpose(1, 2)
        
        # Calculate attention scores
        att_scores = self.W(x_flat).squeeze(-1)
        att_scores = F.softmax(att_scores, dim=1)
        
        # Apply attention
        x_att = torch.bmm(att_scores.unsqueeze(1), x_flat).squeeze(1)
        
        return x_att

class StateOfArtFER(nn.Module):
    """State-of-the-art FER model combining best practices from recent research"""
    def __init__(self, num_classes=7, backbone='convnext_base', dropout_rate=0.5):
        super().__init__()
        
        # Create backbone with specific modifications for FER
        if 'convnext' in backbone:
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            feature_dim = self.backbone.num_features
        else:
            base_model = EmotionClassifier(
                num_classes=num_classes,
                backbone_name=backbone,
                dropout_rate=dropout_rate,
                use_attention=True,
                pretrained=True
            )
            self.backbone = base_model.backbone
            feature_dim = base_model.feature_size
        
        # Multi-scale feature extraction
        self.multi_scale = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((3, 3))
        ])
        
        # Calculate total feature dimension after multi-scale
        total_features = feature_dim * (1 + 4 + 9)  # 1x1 + 2x2 + 3x3
        
        # Self-attention pooling
        self.attention_pool = SelfAttentionPooling(feature_dim)
        
        # Feature fusion with batch normalization
        self.fusion = nn.Sequential(
            nn.Linear(total_features + feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-head classification (inspired by multi-task learning)
        self.classifier_main = nn.Linear(1024, num_classes)
        self.classifier_aux = nn.Linear(1024, num_classes)  # Auxiliary classifier
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone(x)
        
        # Multi-scale pooling
        multi_scale_features = []
        for pool in self.multi_scale:
            pooled = pool(features)
            pooled = pooled.view(pooled.size(0), -1)
            multi_scale_features.append(pooled)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(multi_scale_features, dim=1)
        
        # Self-attention pooling
        att_features = self.attention_pool(features)
        
        # Combine all features
        combined_features = torch.cat([multi_scale_features, att_features], dim=1)
        
        # Feature fusion
        fused_features = self.fusion(combined_features)
        
        if return_features:
            return fused_features
        
        # Main and auxiliary predictions
        logits_main = self.classifier_main(fused_features)
        logits_aux = self.classifier_aux(fused_features)
        
        # Temperature scaling
        logits_main = logits_main / self.temperature
        
        return logits_main, logits_aux

class AdvancedAugmentation:
    """Advanced augmentation techniques for FER"""
    def __init__(self, img_size=224):
        self.img_size = img_size
        
        # Base augmentations
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img):
        return self.transform(img)

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for model ensemble learning"""
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Knowledge distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

class StateOfArtTrainer:
    """Advanced trainer implementing state-of-the-art techniques"""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Setup training components
        self._setup_optimizer()
        self._setup_losses()
        self._setup_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Exponential Moving Average for model weights
        self.ema = self._setup_ema() if config.get('use_ema', True) else None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'lr': []
        }
        
        self.best_val_acc = 0
        self.best_val_f1 = 0
        
    def _setup_optimizer(self):
        """Setup advanced optimizer with layer-wise learning rates"""
        # Layer-wise learning rate decay
        params = []
        lr = self.config['lr']
        lr_decay = self.config.get('lr_decay', 0.95)
        
        # Backbone layers (lower learning rate)
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                layer_depth = name.count('.')
                layer_lr = lr * (lr_decay ** layer_depth)
                params.append({'params': param, 'lr': layer_lr})
            else:
                params.append({'params': param, 'lr': lr})
        
        # Choose optimizer
        opt_name = self.config.get('optimizer', 'adabelief')
        if opt_name == 'adabelief':
            self.optimizer = AdaBelief(params, lr=lr, weight_decay=self.config.get('weight_decay', 0.01))
        elif opt_name == 'adamw':
            self.optimizer = optim.AdamW(params, weight_decay=self.config.get('weight_decay', 0.01))
        else:
            self.optimizer = optim.SGD(params, momentum=0.9, nesterov=True, weight_decay=1e-4)
            
    def _setup_losses(self):
        """Setup advanced loss functions"""
        loss_type = self.config.get('loss_type', 'poly')
        
        if loss_type == 'poly':
            self.criterion = PolyLoss(epsilon=2.0)
        else:
            # Focal loss with class weights
            from train_advanced import FocalLoss
            class_weights = torch.tensor([1.0, 5.0, 2.5, 0.8, 1.2, 1.0, 1.0])
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            
        # Auxiliary loss weight
        self.aux_weight = self.config.get('aux_weight', 0.3)
        
        # MixUp and CutMix
        self.mixup = MixUp(alpha=0.3) if self.config.get('use_mixup', True) else None
        self.cutmix = CutMix(alpha=1.0) if self.config.get('use_cutmix', True) else None
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'onecycle')
        
        if scheduler_type == 'onecycle':
            steps_per_epoch = self.config['steps_per_epoch']
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['lr'],
                epochs=self.config['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000
            )
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-6
            )
            
    def _setup_ema(self):
        """Setup Exponential Moving Average"""
        class EMA:
            def __init__(self, model, decay=0.999):
                self.model = model
                self.decay = decay
                self.shadow = {}
                self.backup = {}
                
            def register(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()
                        
            def update(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        new_val = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                        self.shadow[name] = new_val.clone()
                        
            def apply_shadow(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.backup[name] = param.data
                        param.data = self.shadow[name]
                        
            def restore(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data = self.backup[name]
                        
        ema = EMA(self.model, decay=0.999)
        ema.register()
        return ema
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Apply MixUp or CutMix
            mixed = False
            if self.training and np.random.rand() < 0.5:
                if self.mixup and np.random.rand() < 0.5:
                    inputs, labels_a, labels_b, lam = self.mixup(inputs, labels)
                    mixed = True
                elif self.cutmix:
                    inputs, labels_a, labels_b, lam = self.cutmix(inputs, labels)
                    mixed = True
                    
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    logits_main, logits_aux = self.model(inputs)
                    
                    if mixed:
                        loss_main = lam * self.criterion(logits_main, labels_a) + (1 - lam) * self.criterion(logits_main, labels_b)
                        loss_aux = lam * self.criterion(logits_aux, labels_a) + (1 - lam) * self.criterion(logits_aux, labels_b)
                    else:
                        loss_main = self.criterion(logits_main, labels)
                        loss_aux = self.criterion(logits_aux, labels)
                        
                    loss = loss_main + self.aux_weight * loss_aux
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits_main, logits_aux = self.model(inputs)
                
                if mixed:
                    loss_main = lam * self.criterion(logits_main, labels_a) + (1 - lam) * self.criterion(logits_main, labels_b)
                    loss_aux = lam * self.criterion(logits_aux, labels_a) + (1 - lam) * self.criterion(logits_aux, labels_b)
                else:
                    loss_main = self.criterion(logits_main, labels)
                    loss_aux = self.criterion(logits_aux, labels)
                    
                loss = loss_main + self.aux_weight * loss_aux
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
            # Update EMA
            if self.ema:
                self.ema.update()
                
            # Update scheduler
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
                
            # Statistics
            _, preds = torch.max(logits_main, 1)
            running_loss += loss.item() * inputs.size(0)
            
            if not mixed:
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.6f}'})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
        
    def validate(self, val_loader, use_tta=True):
        """Validation with Test Time Augmentation"""
        self.model.eval()
        
        # Apply EMA weights for validation
        if self.ema:
            self.ema.apply_shadow()
            
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if use_tta and self.config.get('use_tta', True):
                    # Test Time Augmentation
                    batch_size = inputs.size(0)
                    
                    # Original
                    logits1, _ = self.model(inputs)
                    
                    # Flipped
                    inputs_flip = torch.flip(inputs, dims=[3])
                    logits2, _ = self.model(inputs_flip)
                    
                    # Average predictions
                    logits = (logits1 + logits2) / 2
                else:
                    logits, _ = self.model(inputs)
                    
                loss = self.criterion(logits, labels)
                
                _, preds = torch.max(logits, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Restore original weights
        if self.ema:
            self.ema.restore()
            
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc.item(), epoch_f1, all_preds, all_labels
        
    def train(self, train_loader, val_loader, save_dir='checkpoints_sota'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch + 1)
            
            # Validation
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(val_loader)
            
            # Update scheduler (for cosine annealing)
            if not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
                
            # Log results
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'config': self.config
                }
                
                if self.ema:
                    self.ema.apply_shadow()
                    checkpoint['ema_state_dict'] = self.model.state_dict()
                    self.ema.restore()
                    
                torch.save(checkpoint, os.path.join(save_dir, 'best_model_sota.pth'))
                print(f'Saved best model with val_acc: {val_acc:.4f}')
                
            # Generate confusion matrix every 10 epochs
            if (epoch + 1) % 10 == 0:
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
    # Configuration for state-of-the-art training
    config = {
        # Model
        'backbone': 'convnext_base',  # Best performing backbone
        'num_classes': 7,
        'dropout_rate': 0.5,
        
        # Training
        'batch_size': 32,
        'epochs': 80,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'optimizer': 'adabelief',  # Advanced optimizer
        'scheduler': 'onecycle',
        'loss_type': 'poly',  # PolyLoss
        'aux_weight': 0.3,
        
        # Augmentation
        'img_size': 224,
        'use_mixup': True,
        'use_cutmix': True,
        'use_tta': True,
        
        # Advanced techniques
        'use_amp': True,
        'use_ema': True,
        
        # Data
        'num_workers': 4
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = StateOfArtFER(
        num_classes=config['num_classes'],
        backbone=config['backbone'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        test_csv='data/test.csv',
        img_dir='data',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        use_weighted_sampler=True,
        use_mixup=config['use_mixup'],
        use_cutmix=config['use_cutmix']
    )
    
    # Calculate steps per epoch for OneCycle scheduler
    config['steps_per_epoch'] = len(train_loader)
    
    # Create trainer
    trainer = StateOfArtTrainer(model, device, config)
    
    # Train
    print("\nStarting state-of-the-art training...")
    print("="*50)
    history = trainer.train(train_loader, val_loader)
    
    # Save training history
    with open('sota_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    # Load best model for final evaluation
    checkpoint = torch.load('checkpoints_sota/best_model_sota.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = trainer.validate(test_loader)
    
    print(f'\nFinal Results:')
    print(f'Best Validation Accuracy: {trainer.best_val_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    
    # Detailed classification report
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=emotion_names))
    
    # Save final results
    results = {
        'best_val_acc': float(trainer.best_val_acc),
        'best_val_f1': float(trainer.best_val_f1),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'config': config,
        'classification_report': classification_report(test_labels, test_preds,
                                                     target_names=emotion_names,
                                                     output_dict=True)
    }
    
    with open('sota_final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nTraining completed successfully!")
    print("Results saved to checkpoints_sota/")

if __name__ == '__main__':
    main() 