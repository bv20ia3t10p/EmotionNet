import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from typing import List, Dict, Any

from src.config import AugmentationConfig

class TestTimeAugmentation:
    """Class for test-time augmentation."""
    
    def __init__(self, model: tf.keras.Model, config: AugmentationConfig):
        self.model = model
        self.config = config
        
    def create_augmentations(self, image: np.ndarray, num_augmentations: int = None) -> List[np.ndarray]:
        """Create augmented versions of an image."""
        if num_augmentations is None:
            num_augmentations = self.config.tta_augmentations
            
        datagen = ImageDataGenerator(
            rotation_range=self.config.tta_rotation_range,
            width_shift_range=self.config.tta_width_shift_range,
            height_shift_range=self.config.tta_height_shift_range,
            zoom_range=self.config.tta_zoom_range,
            horizontal_flip=self.config.tta_horizontal_flip,
            fill_mode='nearest'
        )
        
        # Expand dimensions for batch
        image_batch = np.expand_dims(image, axis=0)
        augmented_images = [image]
        
        # Generate augmented images
        for _ in range(num_augmentations - 1):
            for batch in datagen.flow(image_batch, batch_size=1):
                augmented_images.append(batch[0])
                break
                
        return augmented_images
        
    def predict_with_tta(self, image: np.ndarray, num_augmentations: int = None) -> np.ndarray:
        """Predict with test-time augmentation."""
        if not self.config.tta_enabled:
            return self.model.predict(np.expand_dims(image, axis=0))[0]
            
        # Create augmented versions
        augmented_images = self.create_augmentations(image, num_augmentations)
        
        # Make predictions on each augmented image
        predictions = []
        for aug_image in augmented_images:
            pred = self.model.predict(np.expand_dims(aug_image, axis=0))[0]
            predictions.append(pred)
            
        # Average predictions
        return np.mean(predictions, axis=0)
        
    def predict_batch_with_tta(self, images: np.ndarray) -> np.ndarray:
        """Predict with test-time augmentation for a batch of images."""
        if not self.config.tta_enabled:
            return self.model.predict(images)
            
        # Apply TTA to each image in the batch
        all_predictions = []
        for i in range(images.shape[0]):
            image = images[i]
            prediction = self.predict_with_tta(image)
            all_predictions.append(prediction)
            
        return np.array(all_predictions)
