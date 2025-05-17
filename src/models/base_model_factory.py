import tensorflow as tf
from abc import ABC, abstractmethod

from src.config import ModelConfig

class BaseModelFactory(ABC):
    """Factory for creating base models."""
    
    @abstractmethod
    def create_model(self, input_tensor: tf.Tensor, config: ModelConfig) -> tf.Tensor:
        """Create base model using input tensor."""
        pass 