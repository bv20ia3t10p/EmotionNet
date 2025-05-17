from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_shape: Tuple[int, int, int] = (48, 48, 1)
    num_classes: int = 7
    base_models: List[str] = None
    dropout_rate: float = 0.5
    attention_ratio: int = 8
    dense_units: List[int] = None
    freeze_layers: Dict[str, int] = None
    
    def __post_init__(self):
        if self.base_models is None:
            self.base_models = ["EfficientNetB0", "Xception", "CustomCNN"]
        if self.dense_units is None:
            self.dense_units = [512, 256]
        if self.freeze_layers is None:
            self.freeze_layers = {"EfficientNetB0": 100, "Xception": 70} 