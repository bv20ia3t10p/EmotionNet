from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.0001
    validation_split: float = 0.1
    test_split: float = 0.1
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    min_lr: float = 1e-6
    class_weight_mode: str = "balanced"
    random_state: int = 42 