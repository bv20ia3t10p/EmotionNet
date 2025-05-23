import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time
import json
from typing import Dict, List, Tuple, Optional, Union
import torchvision.models as models

# Import our custom modules
from models import create_model, EmotionClassifier
from augmentation import DynamicAugmentation, MixAugmentation, get_transform
from class_strategies import DynamicClassWeighting, FocalLossWithDynamicGamma, ClassAnalyzer

class ImprovedDynamicTrainer:
    """
    Trainer that dynamically adapts strategies based on performance.
    Uses a single strong model instead of an ensemble.
    """
    def __init__(
        self,
        num_classes: int = 7,
        class_names: List[str] = None,
        backbone_name: str = 'efficientnet_b2',
        img_size: int = 224,
        batch_size: int = 64,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.5,
        focal_gamma: float = 2.0,
        use_weighted_sampler: bool = True,
        use_mixup: bool = True,
        use_class_weighting: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'outputs'
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.backbone_name = backbone_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.focal_gamma = focal_gamma
        self.use_weighted_sampler = use_weighted_sampler
        self.use_mixup = use_mixup
        self.use_class_weighting = use_class_weighting
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'class_weights': [],
            'focal_gammas': [],
            'learning_rates': []
        }
    
    def _init_components(self):
        """Initialize all components."""
        # Initialize mixup augmentation
        if self.use_mixup:
            self.mixup = MixAugmentation()
        else:
            self.mixup = None
        
        # Initialize class weighting
        if self.use_class_weighting:
            self.class_weighting = DynamicClassWeighting(num_classes=self.num_classes)
            initial_weights = torch.ones(self.num_classes)
            self.focal_loss = FocalLossWithDynamicGamma(
                num_classes=self.num_classes,
                alpha=initial_weights,
                gamma=torch.ones(self.num_classes) * self.focal_gamma
            )
        else:
            self.class_weighting = None
            self.focal_loss = nn.CrossEntropyLoss()
        
        # Initialize class analyzer
        self.analyzer = ClassAnalyzer(num_classes=self.num_classes, class_names=self.class_names)
        
        # Initialize model - using a single strong model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
    
    def _create_model(self):
        """Create a single strong model."""
        # Create a model with strong backbone
        if self.backbone_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate, inplace=True),
                nn.Linear(num_ftrs, self.num_classes)
            )
        elif self.backbone_name == 'convnext_small':
            model = models.convnext_small(pretrained=True)
            num_ftrs = model.classifier[2].in_features
            model.classifier = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LayerNorm(num_ftrs),
                nn.Linear(num_ftrs, self.num_classes)
            )
        else:
            # Use our custom model creator for other backbones
            model = create_model(
                model_type='single',
                backbone_name=self.backbone_name,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate
            )
        
        return model.to(self.device)
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler with differential learning rates."""
        # Separate parameters for differential learning rates
        backbone_params = []
        head_params = []
        
        # For EfficientNet and ConvNext models
        if self.backbone_name.startswith('efficientnet'):
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        elif self.backbone_name.startswith('convnext'):
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        else:
            # For custom models
            for name, param in self.model.named_parameters():
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        # Create optimizer with differential learning rates
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},
            {'params': head_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Scheduler will be initialized later with correct number of epochs
        self.scheduler = None
    
    def prepare_data(self, train_dataset, val_dataset, num_epochs=30):
        """
        Prepare data loaders with optional weighted sampling.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs (for scheduler)
            
        Returns:
            Train and validation data loaders
        """
        # Create weighted sampler for handling class imbalance
        if self.use_weighted_sampler and hasattr(train_dataset, 'targets'):
            # Get class counts
            class_counts = {}
            for t in train_dataset.targets:
                if isinstance(t, torch.Tensor):
                    t = t.item()
                class_counts[t] = class_counts.get(t, 0) + 1
            
            # Calculate weights
            weights = 1.0 / torch.tensor([class_counts.get(i, 1) for i in range(self.num_classes)], dtype=torch.float)
            
            # Normalize weights
            weights = weights / weights.sum() * self.num_classes
            
            # Create sample weights for each instance
            sample_weights = torch.tensor([weights[t] for t in train_dataset.targets])
            
            # Print weights for each class
            print("Class weights used for sampling:")
            for i, w in enumerate(weights):
                print(f"  {self.class_names[i]}: {w:.4f}")
            
            # Create sampler
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            # Create train loader with sampler
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            # Create regular train loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        
        # Create validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize scheduler with correct number of steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.learning_rate * 0.1, self.learning_rate],
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.3,  # Warm up for first 30% of training
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary of training metrics
        """
        # Training mode
        self.model.train()
        
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        
        # Progress bar
        print(f"Epoch {epoch+1}/{total_epochs}")
        print("----------")
        
        # Update mixup probabilities if using mixup
        if self.use_mixup:
            self.mixup.update_probabilities(epoch, total_epochs)
        
        # Training loop
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply mixup/cutmix if enabled
            if self.use_mixup:
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets, epoch, total_epochs)
                mixed_targets = True
            else:
                mixed_targets = False
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(inputs)
            
            # Calculate loss
            if mixed_targets:
                # For mixed targets, compute loss for both target sets
                loss = lam * self.focal_loss(outputs, targets_a) + (1 - lam) * self.focal_loss(outputs, targets_b)
            else:
                loss = self.focal_loss(outputs, targets)
            
            # Backward + optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            
            # For accuracy in mixup, we use the original targets
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for F1 calculation
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
        
        # Calculate metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        class_f1 = f1_score(all_targets, all_preds, average=None)
        
        # Calculate class distribution
        class_counts = np.bincount(all_preds, minlength=self.num_classes)
        class_percentages = class_counts / len(all_preds) * 100
        
        # Print prediction distribution
        print("Prediction distribution:")
        for i in range(self.num_classes):
            print(f"  {self.class_names[i]}: {class_counts[i]} predictions ({class_percentages[i]:.2f}%)")
        
        # Print metrics
        print(f"train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
        print("train Per-class F1 scores:")
        for i, f1 in enumerate(class_f1):
            print(f"  {self.class_names[i]}: {f1:.4f}")
        
        # Update class weights if enabled
        if self.use_class_weighting:
            # Create confusion matrix
            cm = confusion_matrix(all_targets, all_preds, labels=range(self.num_classes))
            
            # Update weights
            new_weights = self.class_weighting.update_weights(class_f1, cm)
            new_gammas = self.class_weighting.get_focal_gamma(class_f1)
            
            # Update focal loss parameters
            self.focal_loss.update_parameters(
                alpha=new_weights,
                gamma=torch.tensor(new_gammas)
            )
            
            # Save weights and gammas
            self.history['class_weights'].append(new_weights.tolist())
            self.history['focal_gammas'].append(new_gammas)
        
        # Update class analyzer
        self.analyzer.update_metrics(
            f1_scores=class_f1,
            confusion_mat=confusion_matrix(all_targets, all_preds, labels=range(self.num_classes))
        )
        
        # Save current learning rates
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.history['learning_rates'].append(current_lrs)
        
        return {
            'loss': epoch_loss,
            'acc': epoch_acc,
            'f1': epoch_f1,
            'class_f1': class_f1.tolist(),
            'class_distribution': class_percentages.tolist()
        }
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        # Evaluation mode
        self.model.eval()
        
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        
        # Validation loop
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                loss = self.focal_loss(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store predictions and targets for F1 calculation
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        class_f1 = f1_score(all_targets, all_preds, average=None)
        
        # Print metrics
        print(f"val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
        print("val Per-class F1 scores:")
        for i, f1 in enumerate(class_f1):
            print(f"  {self.class_names[i]}: {f1:.4f}")
        
        return {
            'loss': epoch_loss,
            'acc': epoch_acc,
            'f1': epoch_f1,
            'class_f1': class_f1.tolist()
        }
    
    def train(self, train_dataset, val_dataset, num_epochs=30):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        # Prepare data
        train_loader, val_loader = self.prepare_data(train_dataset, val_dataset, num_epochs)
        
        # Train
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = -1
        
        # For each epoch
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Check if best model
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                
                # Save model
                self.save_model(os.path.join(self.output_dir, 'models', 'best_model.pt'))
            
            # Generate report
            self.generate_epoch_report(epoch, train_metrics, val_metrics)
            
            # Print class analysis
            self.analyzer.print_analysis()
        
        # Print final results
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best validation F1 score: {best_val_f1:.4f}")
        print(f"Best epoch: {best_epoch+1}")
        
        # Save history
        self.save_history()
        
        # Generate final plots
        self.generate_plots()
        
        return self.history
    
    def save_model(self, filename):
        """
        Save the model.
        
        Args:
            filename: Output filename
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'backbone_name': self.backbone_name,
            'img_size': self.img_size,
            'dropout_rate': self.dropout_rate
        }, filename)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a saved model.
        
        Args:
            filename: Input filename
        """
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Model file {filename} does not exist.")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=self.device)
        
        # Update parameters
        self.backbone_name = checkpoint.get('backbone_name', self.backbone_name)
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.dropout_rate = checkpoint.get('dropout_rate', self.dropout_rate)
        self.class_names = checkpoint.get('class_names', self.class_names)
        
        # Re-initialize model
        self.model = self._create_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Re-initialize optimizer
        self._init_optimizer()
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {filename}")
        return True
    
    def save_history(self):
        """Save training history."""
        filename = os.path.join(self.output_dir, 'history.json')
        
        # Convert tensors to lists
        history_copy = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                history_copy[key] = [v.tolist() for v in value]
            else:
                history_copy[key] = value
        
        with open(filename, 'w') as f:
            json.dump(history_copy, f)
        
        print(f"History saved to {filename}")
    
    def generate_epoch_report(self, epoch, train_metrics, val_metrics):
        """
        Generate a report for the current epoch.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Create report directory if it doesn't exist
        report_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Create plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training curves
        epochs = list(range(epoch+1))
        axs[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss')
        axs[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(epochs, self.history['train_acc'], label='Train Acc')
        axs[0, 1].plot(epochs, self.history['val_acc'], label='Val Acc')
        axs[0, 1].set_title('Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot class F1 scores
        for i, f1 in enumerate(val_metrics['class_f1']):
            axs[1, 0].plot(epochs[-1], f1, 'o', label=self.class_names[i])
        
        axs[1, 0].set_title('Class F1 Scores')
        axs[1, 0].set_xlabel('Class')
        axs[1, 0].set_ylabel('F1 Score')
        axs[1, 0].set_xticks(range(len(self.class_names)))
        axs[1, 0].set_xticklabels(self.class_names, rotation=45)
        axs[1, 0].grid(True)
        
        # Plot class weights if available
        if self.use_class_weighting and len(self.history['class_weights']) > 0:
            weights = self.history['class_weights'][-1]
            axs[1, 1].bar(range(len(weights)), weights)
            axs[1, 1].set_title('Class Weights')
            axs[1, 1].set_xlabel('Class')
            axs[1, 1].set_ylabel('Weight')
            axs[1, 1].set_xticks(range(len(self.class_names)))
            axs[1, 1].set_xticklabels(self.class_names, rotation=45)
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(report_dir, f'epoch_{epoch+1}.png'))
        plt.close()
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Training Report - Epoch {epoch+1}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th {{ background-color: #4CAF50; color: white; }}
                .plot {{ width: 100%; max-width: 800px; }}
            </style>
        </head>
        <body>
            <h1>Training Report - Epoch {epoch+1}</h1>
            
            <h2>Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Training</th>
                    <th>Validation</th>
                </tr>
                <tr>
                    <td>Loss</td>
                    <td>{train_metrics['loss']:.4f}</td>
                    <td>{val_metrics['loss']:.4f}</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>{train_metrics['acc']:.4f}</td>
                    <td>{val_metrics['acc']:.4f}</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td>{train_metrics['f1']:.4f}</td>
                    <td>{val_metrics['f1']:.4f}</td>
                </tr>
            </table>
            
            <h2>Class F1 Scores</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Training F1</th>
                    <th>Validation F1</th>
                </tr>
        """
        
        for i in range(self.num_classes):
            html += f"""
                <tr>
                    <td>{self.class_names[i]}</td>
                    <td>{train_metrics['class_f1'][i]:.4f}</td>
                    <td>{val_metrics['class_f1'][i]:.4f}</td>
                </tr>
            """
        
        html += f"""
            </table>
            
            <h2>Plots</h2>
            <img src='epoch_{epoch+1}.png' class='plot'>
            
            <h2>Class Analysis</h2>
            <pre>{self.get_class_analysis()}</pre>
            
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(report_dir, f'epoch_{epoch+1}.html'), 'w') as f:
            f.write(html)
        
        print(f"Generated HTML report: {os.path.join(report_dir, f'epoch_{epoch+1}.html')}")
    
    def get_class_analysis(self):
        """Get class analysis as a string."""
        import io
        import sys
        
        # Capture print output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        # Print analysis
        self.analyzer.print_analysis()
        
        # Get output
        output = new_stdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        return output
    
    def generate_plots(self):
        """Generate final plots."""
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_curves.png'))
        plt.close()
        
        # Plot class F1 scores over time
        if len(self.history['val_f1']) > 0:
            # Extract class F1 scores
            class_f1_history = []
            for epoch in range(len(self.history['val_f1'])):
                if epoch < len(self.history['val_f1']) and epoch < len(self.analyzer.f1_history):
                    val_metrics = {
                        'class_f1': self.analyzer.f1_history[epoch]
                    }
                    class_f1_history.append(val_metrics['class_f1'])
            
            # Plot
            plt.figure(figsize=(12, 6))
            for i in range(self.num_classes):
                f1_scores = [epoch_f1[i] for epoch_f1 in class_f1_history]
                plt.plot(f1_scores, label=self.class_names[i])
            
            plt.title('Class F1 Scores Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'class_f1_history.png'))
            plt.close()
        
        # Plot class weights over time
        if self.use_class_weighting and len(self.history['class_weights']) > 0:
            plt.figure(figsize=(12, 6))
            for i in range(self.num_classes):
                weights = [epoch_weights[i] for epoch_weights in self.history['class_weights']]
                plt.plot(weights, label=self.class_names[i])
            
            plt.title('Class Weights Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'class_weights_history.png'))
            plt.close()
        
        # Plot confusion matrix
        if len(self.analyzer.confusion_matrices) > 0:
            self.class_weighting.plot_confusion_matrix(
                epoch=-1,  # Last epoch
                class_names=self.class_names,
                save_path=os.path.join(plots_dir, 'confusion_matrix.png')
            )
            
        # Plot class weight history
        if self.use_class_weighting:
            self.class_weighting.plot_history(
                save_path=os.path.join(plots_dir, 'class_weight_history.png')
            )
        
        print(f"Generated plots in {plots_dir}")

# Example usage
if __name__ == "__main__":
    import torch
    from torchvision import transforms, datasets
    import os
    
    # Set random seed
    torch.manual_seed(42)
    
    # Define class names
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Create trainer
    trainer = ImprovedDynamicTrainer(
        num_classes=7,
        class_names=class_names,
        backbone_name='efficientnet_b2',  # Using stronger backbone
        img_size=224,
        batch_size=64,
        learning_rate=5e-4,  # Adjusted learning rate
        weight_decay=1e-4,
        dropout_rate=0.5,
        focal_gamma=2.0,
        use_weighted_sampler=True,  # Explicitly handle class imbalance
        use_mixup=True,
        use_class_weighting=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir='outputs'
    )
    
    # Load datasets
    # This is just an example, replace with your actual datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Replace with actual dataset paths
    data_dir = 'data/emotion_dataset'
    if os.path.exists(data_dir):
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
        
        # Train model
        history = trainer.train(train_dataset, val_dataset, num_epochs=30)
        
        # Save model
        trainer.save_model('outputs/models/final_model.pt')
    else:
        print(f"Dataset directory {data_dir} not found. Please update with your actual dataset path.") 