import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

class DataManager:
    def create_datasets(self, image_size=224, validation_split=0.1):
        """Create training and validation datasets."""
        # ... existing code ...
        
        # Calculate class weights for sampling
        class_counts = np.bincount(np.array(self.train_labels))
        # Ensure all classes have at least 1 sample
        class_counts = np.maximum(class_counts, 1)
        
        # Calculate weights (inverse frequency with stronger emphasis)
        # Using square root to make the weights less extreme but still effective
        weights = 1.0 / np.sqrt(class_counts)
        
        # Special handling for disgust class (index 1) which is heavily underrepresented
        weights[1] *= 2.0  # Extra boost for disgust class
        
        # Compute sample weights
        sample_weights = weights[self.train_labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        # Create weighted sampler
        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.train_labels),
            replacement=True
        )
        
        # Return the datasets and sampler
        return self.train_dataset, self.val_dataset, self.train_sampler

    def create_loaders(self, batch_size=32, num_workers=4):
        """Create data loaders."""
        # ... existing code ...
        
        # Use the sampler for training loader if available
        if hasattr(self, 'train_sampler'):
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=batch_size,
                sampler=self.train_sampler,  # Use weighted sampler
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        
        # ... rest of the existing code ... 