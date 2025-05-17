import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any

class ModelEvaluator:
    """Class for model evaluation and visualization."""
    
    def __init__(self, model: tf.keras.Model, class_names: List[str]):
        self.model = model
        self.class_names = class_names
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on test data."""
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Get model predictions."""
        return self.model.predict(X_test)
    
    def compute_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Compute classification report."""
        # Convert one-hot encoded labels to integers if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
            
        return classification_report(y_true, y_pred, target_names=self.class_names)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             figsize: Tuple[int, int] = (10, 8),
                             normalize: bool = False) -> None:
        """Plot confusion matrix."""
        # Convert one-hot encoded labels to integers if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        # Create figure
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
    def plot_training_history(self, history: Dict[str, Any]) -> None:
        """Plot training history."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy subplot
        ax1.plot(history['accuracy'])
        ax1.plot(history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        
        # Loss subplot
        ax2.plot(history['loss'])
        ax2.plot(history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        
    def visualize_predictions(self, X_test: np.ndarray, y_true: np.ndarray, 
                             num_samples: int = 10) -> None:
        """Visualize model predictions on sample images."""
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Convert to label indices
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
            
        # Create figure
        fig = plt.figure(figsize=(15, num_samples * 2))
        
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(X_test))
            img = X_test[idx]
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            # Display image and labels
            ax = fig.add_subplot(num_samples, 1, i+1)
            
            # Convert RGB to grayscale for display if needed
            if img.shape[-1] == 3:
                img_display = img[:, :, 0]
            else:
                img_display = img[:, :, 0]
                
            ax.imshow(img_display, cmap='gray')
            ax.set_title(f"True: {true_label}, Predicted: {pred_label}")
            ax.axis('off')
            
        plt.tight_layout()
