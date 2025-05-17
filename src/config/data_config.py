from dataclasses import dataclass

@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    data_path: str = "fer2013.csv"
    apply_hist_eq: bool = True
    contrast_factor: float = 2.0
    normalize: bool = True
    expand_dims: bool = True
    cache_preprocessed: bool = True
    cached_data_path: str = "preprocessed_data.npz" 