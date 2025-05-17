import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw data."""
        pass
        
    @abstractmethod
    def preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data."""
        pass
    
    @abstractmethod
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data into train, validation, and test sets."""
        pass
    
    @abstractmethod
    def create_generators(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create data generators for training and validation."""
        pass
    
    @abstractmethod
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        pass 