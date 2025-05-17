import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import TrainingConfig

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """Custom callback to plot confusion matrix after each epoch."""
    
    def __init__(self, validation_data, train_data=None, class_names=None):
        super().__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        self.class_names = class_names or [str(i) for i in range(7)]
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculate and plot validation confusion matrix
        x_val, y_val = self.validation_data
        y_val_true = np.argmax(y_val, axis=1)
        y_val_pred = np.argmax(self.model.predict(x_val, verbose=0), axis=1)
        cm_val = confusion_matrix(y_val_true, y_val_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
        plt.show()
        
        # Calculate and plot training confusion matrix if available
        if self.train_data is not None:
            x_train, y_train = self.train_data
            # Use a subset for training to speed things up
            indices = np.random.choice(len(x_train), min(1000, len(x_train)), replace=False)
            x_train_subset = x_train[indices]
            y_train_subset = y_train[indices]
            
            y_train_true = np.argmax(y_train_subset, axis=1)
            y_train_pred = np.argmax(self.model.predict(x_train_subset, verbose=0), axis=1)
            cm_train = confusion_matrix(y_train_true, y_train_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Training Confusion Matrix - Epoch {epoch+1}')
            plt.show()

class ModelTrainer:
    """Class for model training and evaluation."""
    
    def __init__(self, model: tf.keras.Model, config: TrainingConfig):
        self.model = model
        self.config = config
        
    def compile_model(self) -> None:
        """Compile model with optimizer and loss function."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def create_callbacks(self, validation_data=None, train_data=None, class_names=None, checkpoint_path: str = 'best_model.h5') -> List[tf.keras.callbacks.Callback]:
        """Create callbacks for training."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Confusion matrix callback
        if validation_data is not None:
            cm_callback = ConfusionMatrixCallback(
                validation_data=validation_data,
                train_data=train_data,
                class_names=class_names
            )
            callbacks.append(cm_callback)
        
        return callbacks
    
    def train(self, 
              train_generator: tf.keras.preprocessing.image.DirectoryIterator,
              val_generator: tf.keras.preprocessing.image.DirectoryIterator,
              train_steps: int,
              val_steps: int,
              class_weights: Dict[int, float] = None,
              validation_data=None,
              train_data=None,
              class_names=None,
              checkpoint_path: str = 'best_model.h5') -> tf.keras.callbacks.History:
        """Train model using generators."""
        callbacks = self.create_callbacks(
            validation_data=validation_data,
            train_data=train_data,
            class_names=class_names,
            checkpoint_path=checkpoint_path
        )
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=self.config.epochs,
            validation_data=val_generator,
            validation_steps=val_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1  # Show progress bar
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on test data."""
        return self.model.evaluate(X_test, y_test, verbose=1)  # Show progress bar
