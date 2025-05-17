import tensorflow as tf
from abc import ABC, abstractmethod

class AttentionModule(ABC):
    """Abstract base class for attention mechanisms."""
    
    @abstractmethod
    def apply(self, input_feature: tf.Tensor) -> tf.Tensor:
        """Apply attention to input feature."""
        pass 