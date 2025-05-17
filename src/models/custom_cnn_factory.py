import tensorflow as tf
from keras import layers

from src.config import ModelConfig
from src.models.base_model_factory import BaseModelFactory

class CustomCNNFactory(BaseModelFactory):
    """Factory for creating custom CNN models."""
    
    def create_model(self, input_tensor: tf.Tensor, config: ModelConfig) -> tf.Tensor:
        """Create custom CNN model."""
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        return x 