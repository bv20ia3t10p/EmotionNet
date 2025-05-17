import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras_preprocessing.image import ImageDataGenerator
import os

from src.config import DataConfig, TrainingConfig, AugmentationConfig

class FER2013Processor:
    """Processor for the FER2013 dataset."""
    
    def __init__(self, data_config: DataConfig, train_config: TrainingConfig, 
                 aug_config: AugmentationConfig):
        self.data_config = data_config
        self.train_config = train_config
        self.aug_config = aug_config
        
    def load_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load FER2013 dataset from CSV, respecting the original dataset splits."""
        # Check if cached preprocessed data exists
        if self.data_config.cache_preprocessed and os.path.exists(self.data_config.cached_data_path):
            print(f"Loading cached preprocessed data from {self.data_config.cached_data_path}")
            cached_data = np.load(self.data_config.cached_data_path, allow_pickle=True)
            return {
                'train': (cached_data['X_train'], cached_data['y_train']),
                'val': (cached_data['X_val'], cached_data['y_val']),
                'test': (cached_data['X_test'], cached_data['y_test'])
            }
        
        print(f"Loading data from {self.data_config.data_path}")
        
        # Parse train data path to get directory
        data_dir = os.path.dirname(self.data_config.data_path)
        
        # Assume structure with train.csv, val.csv, test.csv
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'test.csv')  # Using test.csv as validation
        
        # Load train data
        train_data = pd.read_csv(train_path)
        train_pixels = train_data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32'))
        X_train = np.stack(train_pixels.values)
        X_train = X_train.reshape(-1, 48, 48, 1)
        y_train = train_data['emotion'].values
        
        # Load validation/test data if available
        if os.path.exists(val_path):
            val_data = pd.read_csv(val_path)
            val_pixels = val_data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32'))
            X_val = np.stack(val_pixels.values)
            X_val = X_val.reshape(-1, 48, 48, 1)
            y_val = val_data['emotion'].values
        else:
            # If no separate validation file, split train data
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.train_config.validation_split,
                random_state=self.train_config.random_state
            )
        
        # Store dataset splits
        data_dict = {
            'train': (X_train, y_train),
            'val': (X_val, y_val)
        }
        
        return data_dict
    
    def preprocess(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply preprocessing to images and labels."""
        processed_dict = {}
        
        for split_name, (X, y) in data_dict.items():
            # Normalize images
            if self.data_config.normalize:
                X = X / 255.0
            
            # Apply histogram equalization for better contrast
            if self.data_config.apply_hist_eq:
                for i in range(X.shape[0]):
                    img = X[i, :, :, 0]
                    if self.data_config.normalize:
                        img = (img * 255).astype(np.uint8)
                    img = tf.image.adjust_contrast(img, self.data_config.contrast_factor)
                    if self.data_config.normalize:
                        X[i, :, :, 0] = img / 255.0
                    else:
                        X[i, :, :, 0] = img
            
            # One-hot encode labels
            y_categorical = np.zeros((len(y), 7))
            y_categorical[np.arange(len(y)), y] = 1
            
            # Expand dimensions for transfer learning models if configured
            if self.data_config.expand_dims:
                X_expanded = np.repeat(X, 3, axis=3)
            else:
                X_expanded = X
                
            processed_dict[split_name] = (X_expanded, y_categorical)
        
        # Cache preprocessed data if configured
        if self.data_config.cache_preprocessed:
            np.savez(
                self.data_config.cached_data_path,
                X_train=processed_dict['train'][0],
                y_train=processed_dict['train'][1],
                X_val=processed_dict['val'][0],
                y_val=processed_dict['val'][1]
            )
            
        return processed_dict
    
    def create_generators(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Create data generators for training and validation."""
        train_datagen = ImageDataGenerator(
            rotation_range=self.aug_config.rotation_range,
            width_shift_range=self.aug_config.width_shift_range,
            height_shift_range=self.aug_config.height_shift_range,
            shear_range=self.aug_config.shear_range,
            zoom_range=self.aug_config.zoom_range,
            horizontal_flip=self.aug_config.horizontal_flip,
            fill_mode=self.aug_config.fill_mode
        )
        
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            data_dict['train'][0], 
            data_dict['train'][1],
            batch_size=self.train_config.batch_size
        )
        
        val_generator = val_datagen.flow(
            data_dict['val'][0], 
            data_dict['val'][1],
            batch_size=self.train_config.batch_size
        )
        
        return {
            'train_generator': train_generator,
            'val_generator': val_generator,
            'train_steps': len(data_dict['train'][0]) // self.train_config.batch_size,
            'val_steps': len(data_dict['val'][0]) // self.train_config.batch_size,
            'train_data': data_dict['train'],
            'val_data': data_dict['val']
        }
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Compute class weights to handle imbalanced classes."""
        # Convert one-hot encoded labels back to integers
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_integers = np.argmax(y_train, axis=1)
        else:
            y_integers = y_train
            
        class_weights = class_weight.compute_class_weight(
            class_weight=self.train_config.class_weight_mode, 
            classes=np.unique(y_integers), 
            y=y_integers
        )
        
        return {i: class_weights[i] for i in range(len(class_weights))} 