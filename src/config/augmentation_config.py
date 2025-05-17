from dataclasses import dataclass

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    fill_mode: str = "nearest"
    
    # Test-time augmentation config
    tta_enabled: bool = True
    tta_augmentations: int = 10
    tta_rotation_range: int = 10
    tta_width_shift_range: float = 0.1
    tta_height_shift_range: float = 0.1
    tta_zoom_range: float = 0.1
    tta_horizontal_flip: bool = True 