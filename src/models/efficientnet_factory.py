import tensorflow as tf
from keras import applications

from src.config import ModelConfig
from src.models.base_model_factory import BaseModelFactory

class EfficientNetFactory(BaseModelFactory):
    """Factory for creating EfficientNet models."""
    
    def create_model(self, input_tensor: tf.Tensor, config: ModelConfig) -> tf.Tensor:
        """Create EfficientNetB0 model."""
        effnet = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=(48, 48, 3)
        )
        
        # Freeze early layers if specified
        if "EfficientNetB0" in config.freeze_layers:
            for layer in effnet.layers[:config.freeze_layers["EfficientNetB0"]]:
                layer.trainable = False
                
        return effnet.output 