import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Union
from models import create_model, EmotionClassifier, ModelEnsemble

class EnsembleManager:
    """
    Manages the training and evaluation of an ensemble of models.
    """
    def __init__(
        self,
        ensemble_size: int = 3,
        backbone_names: Optional[List[str]] = None,
        num_classes: int = 7,
        dropout_rates: Optional[List[float]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'ensemble_models'
    ):
        self.ensemble_size = ensemble_size
        
        # Set default backbones if not provided
        if backbone_names is None:
            self.backbone_names = [
                'resnet34',
                'efficientnet_b0',
                'mobilenet_v3_large',
                'convnext_tiny',
                'vit_small'
            ][:ensemble_size]  # Take the first n backbones
        else:
            if len(backbone_names) < ensemble_size:
                raise ValueError(f"Expected at least {ensemble_size} backbone names, got {len(backbone_names)}")
            self.backbone_names = backbone_names[:ensemble_size]
        
        # Set default dropout rates if not provided
        if dropout_rates is None:
            self.dropout_rates = [0.5] * ensemble_size
        else:
            if len(dropout_rates) < ensemble_size:
                raise ValueError(f"Expected at least {ensemble_size} dropout rates, got {len(dropout_rates)}")
            self.dropout_rates = dropout_rates[:ensemble_size]
        
        self.num_classes = num_classes
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize models
        self.models = []
        for i in range(ensemble_size):
            model = create_model(
                model_type='single',
                backbone_name=self.backbone_names[i],
                num_classes=num_classes,
                dropout_rate=self.dropout_rates[i]
            )
            self.models.append(model)
        
        # Create ensemble model
        self.ensemble = ModelEnsemble(self.models)
        
        # Validation metrics for each model
        self.val_metrics = {i: [] for i in range(ensemble_size)}
        
        # Best validation metrics for each model
        self.best_val_metrics = {i: {'epoch': -1, 'accuracy': 0.0, 'f1': 0.0} for i in range(ensemble_size)}
        
        # Ensemble validation metrics
        self.ensemble_val_metrics = []
        self.best_ensemble_metrics = {'epoch': -1, 'accuracy': 0.0, 'f1': 0.0}
    
    def train_single_model(
        self,
        model_idx: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 10,
        device: Optional[str] = None
    ) -> Dict:
        """
        Train a single model in the ensemble.
        
        Args:
            model_idx: Index of the model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss criterion
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs to train
            device: Device to train on
            
        Returns:
            Dictionary of training metrics
        """
        if device is None:
            device = self.device
        
        model = self.models[model_idx]
        model.to(device)
        
        best_val_accuracy = 0.0
        best_val_f1 = 0.0
        best_epoch = -1
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs} - Model {model_idx+1}/{self.ensemble_size}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_predictions = []
            train_targets = []
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
                
                # Store predictions and targets for F1 calculation
                train_predictions.extend(predicted.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
            
            # Calculate F1 score for training data
            train_f1 = self._calculate_f1(train_predictions, train_targets)
            
            # Step the scheduler if it exists
            if scheduler is not None:
                scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)
                    
                    # Store predictions and targets for F1 calculation
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / train_total
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / val_total
            val_f1 = self._calculate_f1(val_predictions, val_targets)
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_f1 = val_f1
                best_epoch = epoch
                
                # Save model
                self._save_model(model_idx, model, epoch, val_accuracy, val_f1)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        # Update best validation metrics
        self.best_val_metrics[model_idx] = {
            'epoch': best_epoch,
            'accuracy': best_val_accuracy,
            'f1': best_val_f1
        }
        
        return history
    
    def train_ensemble(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]] = None,
        num_epochs: int = 10,
        device: Optional[str] = None
    ) -> Dict:
        """
        Train all models in the ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss criterion
            optimizers: List of optimizers for each model
            schedulers: List of schedulers for each model
            num_epochs: Number of epochs to train
            device: Device to train on
            
        Returns:
            Dictionary of training metrics
        """
        if device is None:
            device = self.device
        
        histories = []
        
        # Train each model individually
        for i in range(self.ensemble_size):
            print(f"\nTraining model {i+1}/{self.ensemble_size}")
            
            history = self.train_single_model(
                model_idx=i,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizers[i],
                scheduler=schedulers[i] if schedulers else None,
                num_epochs=num_epochs,
                device=device
            )
            
            histories.append(history)
        
        # Load best models
        self.load_best_models()
        
        # Evaluate ensemble on validation set
        ensemble_metrics = self.evaluate_ensemble(val_loader, criterion, device)
        
        # Update ensemble weights based on validation accuracies
        val_accuracies = [self.best_val_metrics[i]['accuracy'] for i in range(self.ensemble_size)]
        self.ensemble.update_weights(val_accuracies)
        
        return {
            'model_histories': histories,
            'ensemble_metrics': ensemble_metrics
        }
    
    def evaluate_ensemble(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: Optional[str] = None
    ) -> Dict:
        """
        Evaluate the ensemble on a dataset.
        
        Args:
            data_loader: Data loader
            criterion: Loss criterion
            device: Device to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if device is None:
            device = self.device
        
        # Move ensemble to device
        self.ensemble.to(device)
        
        # Evaluation mode
        self.ensemble.eval()
        
        # Initialize metrics
        ensemble_loss = 0.0
        ensemble_correct = 0
        ensemble_total = 0
        ensemble_predictions = []
        ensemble_targets = []
        
        # Evaluate each model individually for comparison
        individual_metrics = []
        for i in range(self.ensemble_size):
            self.models[i].to(device)
            self.models[i].eval()
            
            model_correct = 0
            model_total = 0
            model_predictions = []
            model_targets = []
            
            with torch.no_grad():
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = self.models[i](inputs)
                    
                    # Statistics
                    _, predicted = torch.max(outputs, 1)
                    model_correct += (predicted == targets).sum().item()
                    model_total += targets.size(0)
                    
                    # Store predictions and targets for F1 calculation
                    model_predictions.extend(predicted.cpu().numpy())
                    model_targets.extend(targets.cpu().numpy())
            
            # Calculate metrics
            model_accuracy = model_correct / model_total
            model_f1 = self._calculate_f1(model_predictions, model_targets)
            
            individual_metrics.append({
                'accuracy': model_accuracy,
                'f1': model_f1
            })
        
        # Evaluate ensemble
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = self.ensemble(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                ensemble_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                ensemble_correct += (predicted == targets).sum().item()
                ensemble_total += targets.size(0)
                
                # Store predictions and targets for F1 calculation
                ensemble_predictions.extend(predicted.cpu().numpy())
                ensemble_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        ensemble_loss /= len(data_loader.dataset)
        ensemble_accuracy = ensemble_correct / ensemble_total
        ensemble_f1 = self._calculate_f1(ensemble_predictions, ensemble_targets)
        
        # Update best ensemble metrics
        if ensemble_accuracy > self.best_ensemble_metrics['accuracy']:
            self.best_ensemble_metrics = {
                'epoch': len(self.ensemble_val_metrics),
                'accuracy': ensemble_accuracy,
                'f1': ensemble_f1
            }
            
            # Save ensemble model
            self._save_ensemble()
        
        # Store ensemble metrics
        self.ensemble_val_metrics.append({
            'loss': ensemble_loss,
            'accuracy': ensemble_accuracy,
            'f1': ensemble_f1
        })
        
        # Print results
        print("\nEnsemble Evaluation Results:")
        print(f"Ensemble Loss: {ensemble_loss:.4f}, Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
        print("\nIndividual Model Results:")
        for i, metrics in enumerate(individual_metrics):
            print(f"Model {i+1}: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return {
            'ensemble': {
                'loss': ensemble_loss,
                'accuracy': ensemble_accuracy,
                'f1': ensemble_f1
            },
            'individual': individual_metrics
        }
    
    def _calculate_f1(self, predictions: List[int], targets: List[int]) -> float:
        """
        Calculate macro F1 score.
        
        Args:
            predictions: List of predicted labels
            targets: List of target labels
            
        Returns:
            Macro F1 score
        """
        from sklearn.metrics import f1_score
        return f1_score(targets, predictions, average='macro')
    
    def _save_model(self, model_idx: int, model: nn.Module, epoch: int, accuracy: float, f1: float):
        """
        Save a model checkpoint.
        
        Args:
            model_idx: Index of the model
            model: The model to save
            epoch: Current epoch
            accuracy: Validation accuracy
            f1: Validation F1 score
        """
        filename = os.path.join(self.save_dir, f"model_{model_idx}_best.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'f1': f1,
            'backbone_name': self.backbone_names[model_idx],
            'dropout_rate': self.dropout_rates[model_idx]
        }, filename)
        
        print(f"Saved model {model_idx} to {filename}")
    
    def _save_ensemble(self):
        """
        Save the ensemble model.
        """
        filename = os.path.join(self.save_dir, "ensemble_best.pt")
        
        torch.save({
            'model_state_dict': self.ensemble.state_dict(),
            'weights': self.ensemble.weights,
            'backbone_names': self.backbone_names,
            'dropout_rates': self.dropout_rates
        }, filename)
        
        print(f"Saved ensemble to {filename}")
    
    def load_best_models(self):
        """
        Load the best model for each position in the ensemble.
        """
        for i in range(self.ensemble_size):
            filename = os.path.join(self.save_dir, f"model_{i}_best.pt")
            
            if os.path.exists(filename):
                checkpoint = torch.load(filename, map_location=self.device)
                
                # Create a new model with the same architecture
                model = create_model(
                    model_type='single',
                    backbone_name=checkpoint['backbone_name'],
                    num_classes=self.num_classes,
                    dropout_rate=checkpoint['dropout_rate']
                )
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Update model in ensemble
                self.models[i] = model
                self.ensemble.models[i] = model
                
                print(f"Loaded model {i} from {filename}")
            else:
                print(f"No checkpoint found at {filename}")
    
    def load_ensemble(self, filename: Optional[str] = None):
        """
        Load the ensemble model.
        
        Args:
            filename: Path to ensemble checkpoint
        """
        if filename is None:
            filename = os.path.join(self.save_dir, "ensemble_best.pt")
        
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            
            # Create models with same architecture
            backbone_names = checkpoint.get('backbone_names', self.backbone_names)
            dropout_rates = checkpoint.get('dropout_rates', self.dropout_rates)
            
            models = []
            for i in range(len(backbone_names)):
                model = create_model(
                    model_type='single',
                    backbone_name=backbone_names[i],
                    num_classes=self.num_classes,
                    dropout_rate=dropout_rates[i]
                )
                models.append(model)
            
            # Create ensemble
            ensemble = ModelEnsemble(models)
            
            # Load ensemble weights
            ensemble.load_state_dict(checkpoint['model_state_dict'])
            
            # Update weights if available
            if 'weights' in checkpoint:
                ensemble.weights = checkpoint['weights']
            
            # Update instance variables
            self.models = models
            self.ensemble = ensemble
            self.backbone_names = backbone_names
            self.dropout_rates = dropout_rates
            
            print(f"Loaded ensemble from {filename}")
        else:
            print(f"No checkpoint found at {filename}")
    
    def predict(self, inputs: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        """
        Make predictions with the ensemble.
        
        Args:
            inputs: Input tensor
            device: Device to use
            
        Returns:
            Predicted classes
        """
        if device is None:
            device = self.device
        
        # Move ensemble to device
        self.ensemble.to(device)
        
        # Evaluation mode
        self.ensemble.eval()
        
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.ensemble(inputs)
            _, predicted = torch.max(outputs, 1)
        
        return predicted 