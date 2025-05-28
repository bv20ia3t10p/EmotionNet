"""
Metrics calculation for EmotionNet training.
Handles accuracy, F1, confusion matrix, and classification reports.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, List, Any

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def calculate_metrics(all_targets: List[int], all_preds: List[int], loss: float) -> Dict[str, Any]:
    """Calculate comprehensive metrics for training/validation."""
    if len(all_preds) == 0 or len(all_targets) == 0:
        return get_empty_metrics(loss)
    
    # Basic metrics
    accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(
        all_targets, all_preds,
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'class_report': class_report
    }


def get_empty_metrics(loss: float) -> Dict[str, Any]:
    """Return empty metrics when no valid predictions exist."""
    return {
        'loss': loss,
        'accuracy': 0.0,
        'f1_score': 0.0,
        'confusion_matrix': [[0] * 7 for _ in range(7)],
        'class_report': {}
    }


def print_epoch_results(epoch: int, total_epochs: int, train_metrics: Dict[str, Any], 
                       val_metrics: Dict[str, Any], lr: float, epoch_time: float) -> None:
    """Print formatted epoch results."""
    print(f'\nEpoch {epoch}/{total_epochs} - Time: {epoch_time:.2f}s')
    print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%, F1: {train_metrics["f1_score"]:.4f}')
    print(f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2f}%, F1: {val_metrics["f1_score"]:.4f}')
    print(f'Learning Rate: {lr:.6f}')


def get_class_weights_tensor(class_weights_dict: Dict[str, float], device) -> 'torch.Tensor':
    """Convert class weights dict to tensor."""
    import torch
    
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    weights = [class_weights_dict.get(emotion, 1.0) for emotion in emotion_names]
    return torch.FloatTensor(weights).to(device) 