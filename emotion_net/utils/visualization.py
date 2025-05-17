"""Visualization utilities for the emotion recognition model."""

import numpy as np
import matplotlib.pyplot as plt
from emotion_net.config.constants import EMOTIONS

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(EMOTIONS))
    plt.xticks(tick_marks, list(EMOTIONS.values()), rotation=45)
    plt.yticks(tick_marks, list(EMOTIONS.values()))
    
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig(save_path)
    plt.close() 