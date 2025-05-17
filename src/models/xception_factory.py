import tensorflow as tf
from keras import applications

from src.config import ModelConfig
from src.models.base_model_factory import BaseModelFactory

class XceptionFactory(BaseModelFactory):
    """Factory for creating Xception models."""
    
    def create_model(self, input_tensor: tf.Tensor, config: ModelConfig) -> tf.Tensor:
        """Create Xception model."""
        xception = applications.Xception(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=(48, 48, 3)
        )
        
        # Freeze early layers if specified
        if "Xception" in config.freeze_layers:
            for layer in xception.layers[:config.freeze_layers["Xception"]]:
                layer.trainable = False
                
        return xception.output 