"""
Metrics calculation for EmotionNet training.
Handles accuracy, F1, confusion matrix, and classification reports.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, List, Any

# Define emotion labels for both datasets
FER2013_EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FERPLUS_EMOTION_LABELS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']


def calculate_metrics(all_targets: List[int], all_preds: List[int], loss: float) -> Dict[str, Any]:
    """Calculate comprehensive metrics for training/validation."""
    if len(all_preds) == 0 or len(all_targets) == 0:
        return get_empty_metrics(loss)
    
    # Basic metrics
    accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Detect the number of classes based on predictions and targets
    num_classes = max(max(all_targets) + 1, max(all_preds) + 1)
    
    # Use the appropriate emotion labels based on class count
    if num_classes == 8:
        emotion_labels = FERPLUS_EMOTION_LABELS
    else:
        emotion_labels = FER2013_EMOTION_LABELS
    
    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    class_report = classification_report(
        all_targets, all_preds,
        target_names=emotion_labels,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'class_report': class_report,
        'num_classes': num_classes
    }


def get_empty_metrics(loss: float, num_classes=7) -> Dict[str, Any]:
    """Return empty metrics when no valid predictions exist."""
    return {
        'loss': loss,
        'accuracy': 0.0,
        'f1_score': 0.0,
        'confusion_matrix': [[0] * num_classes for _ in range(num_classes)],
        'class_report': {},
        'num_classes': num_classes
    }


def print_epoch_results(*args) -> None:
    """
    Print formatted epoch results.
    Supports both old and new function signatures:
    - Old: (epoch, total_epochs, train_metrics, val_metrics, lr, epoch_time)
    - New: (train_metrics, val_metrics)
    """
    if len(args) == 6:
        # Old function signature with 6 arguments
        epoch, total_epochs, train_metrics, val_metrics, lr, epoch_time = args
        print(f'\nEpoch {epoch}/{total_epochs} - Time: {epoch_time:.2f}s')
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%, F1: {train_metrics["f1_score"]:.4f}')
        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2f}%, F1: {val_metrics["f1_score"]:.4f}')
        print(f'Learning Rate: {lr:.6f}')
    elif len(args) == 2:
        # New function signature with just train_metrics and val_metrics
        train_metrics, val_metrics = args
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%, F1: {train_metrics["f1_score"]:.4f}')
        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2f}%, F1: {val_metrics["f1_score"]:.4f}')
    else:
        raise ValueError(f"Invalid number of arguments: {len(args)}")


def get_class_weights_tensor(class_weights_dict: Dict[str, float], device, num_classes=7) -> 'torch.Tensor':
    """Convert class weights dict to tensor."""
    import torch
    
    if num_classes == 8:
        emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    else:
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    weights = [class_weights_dict.get(emotion, 1.0) for emotion in emotion_names]
    return torch.FloatTensor(weights).to(device) 