import tensorflow as tf
from keras import layers, models
from typing import Dict

from src.config import ModelConfig
from src.models.attention_module import AttentionModule
from src.models.cbam_attention import CBAMAttention
from src.models.base_model_factory import BaseModelFactory
from src.models.efficientnet_factory import EfficientNetFactory
from src.models.xception_factory import XceptionFactory
from src.models.custom_cnn_factory import CustomCNNFactory

class ModelBuilder:
    """Builder for creating ensemble model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.attention_module = CBAMAttention(attention_ratio=config.attention_ratio)
        self.model_factories = {
            "EfficientNetB0": EfficientNetFactory(),
            "Xception": XceptionFactory(),
            "CustomCNN": CustomCNNFactory()
        }
        
    def build(self) -> models.Model:
        """Build ensemble model."""
        input_shape = self.config.input_shape
        inputs = layers.Input(shape=input_shape)
        
        # Convert grayscale to RGB for transfer learning models if needed
        if input_shape[-1] == 1:
            x = layers.Concatenate()([inputs, inputs, inputs])
        else:
            x = inputs
        
        # Create base models and apply attention
        model_outputs = []
        
        for model_name in self.config.base_models:
            if model_name not in self.model_factories:
                raise ValueError(f"Unknown model type: {model_name}")
                
            factory = self.model_factories[model_name]
            output = factory.create_model(x, self.config)
            
            # Apply attention
            output = self.attention_module.apply(output)
            
            # Global pooling
            output = layers.GlobalAveragePooling2D()(output)
            model_outputs.append(output)
        
        # Combine outputs if multiple models
        if len(model_outputs) > 1:
            combined = layers.Concatenate()(model_outputs)
        else:
            combined = model_outputs[0]
        
        # Add dense layers
        x = combined
        for units in self.config.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
            
        # Output layer
        outputs = layers.Dense(self.config.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
