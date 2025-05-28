"""
Utility Functions and Constants for EmotionNet
Contains helper functions, constants, and utility classes
"""

import torch
import numpy as np
import random
import os
from typing import Dict, Any

# Local constants (moved from deleted constants.py)
EMOTION_MAP = {i: label for i, label in enumerate(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])}
EMOTION_MAP_REVERSE = {label: i for label, i in EMOTION_MAP.items()}
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = 7

# Emotion colors for visualization (if needed)
EMOTION_COLORS = {
    'Angry': '#FF0000',      # Red
    'Disgust': '#800080',    # Purple
    'Fear': '#FFA500',       # Orange
    'Happy': '#00FF00',      # Green
    'Sad': '#0000FF',        # Blue
    'Surprise': '#FFFF00',   # Yellow
    'Neutral': '#808080'     # Gray
}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 1, 64, 64)) -> None:
    """Print a summary of the model architecture"""
    param_info = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input Size: {input_size}")
    print(f"Total Parameters: {param_info['total_params']:,}")
    print(f"Trainable Parameters: {param_info['trainable_params']:,}")
    print(f"Non-trainable Parameters: {param_info['non_trainable_params']:,}")
    
    # Calculate model size in MB
    param_size = param_info['total_params'] * 4  # Assuming float32
    buffer_size = sum(p.numel() for p in model.buffers()) * 4
    model_size = (param_size + buffer_size) / (1024 * 1024)
    print(f"Model Size: {model_size:.2f} MB")
    print("="*60)


def save_training_info(config: Dict[str, Any], save_dir: str) -> None:
    """Save training information and configuration"""
    import json
    import datetime
    
    training_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'environment': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    info_file = os.path.join(save_dir, 'training_info.json')
    
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Training info saved to: {info_file}")


class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


def calculate_class_weights(class_counts: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """Calculate class weights for handling class imbalance"""
    if method == 'balanced':
        # Standard sklearn balanced approach
        n_samples = np.sum(class_counts)
        n_classes = len(class_counts)
        weights = n_samples / (n_classes * class_counts)
    elif method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / class_counts
        weights = weights / np.sum(weights) * len(weights)
    elif method == 'sqrt':
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / np.sqrt(class_counts)
        weights = weights / np.sum(weights) * len(weights)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights


def print_class_distribution(class_counts: np.ndarray, class_names: list = None) -> None:
    """Print class distribution statistics"""
    if class_names is None:
        class_names = EMOTION_LABELS
    
    total = np.sum(class_counts)
    
    print("\nClass Distribution:")
    print("-" * 40)
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / total) * 100
        print(f"{name:>10}: {count:>6} ({percentage:>5.1f}%)")
    print("-" * 40)
    print(f"{'Total':>10}: {total:>6} (100.0%)")
    
    # Calculate imbalance ratio
    max_count = np.max(class_counts)
    min_count = np.min(class_counts)
    imbalance_ratio = max_count / min_count
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0 