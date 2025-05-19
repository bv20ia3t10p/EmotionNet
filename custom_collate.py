"""
Custom collate function to handle inconsistent tensor shapes in the DataLoader
"""
import torch
import numpy as np

def custom_collate_fn(batch):
    """
    Custom collate function that ensures all tensors have the same shape (1, 48, 48)
    before stacking them in a batch
    
    Args:
        batch: A list of tuples (image, label)
        
    Returns:
        A tuple (images, labels) where images and labels are batched tensors
    """
    images = []
    labels = []
    
    for item in batch:
        image, label = item
        
        # Check if image is a tensor with expected dimensions
        if isinstance(image, torch.Tensor):
            # Ensure it's a 3D tensor
            if image.dim() == 2:
                image = image.unsqueeze(0)
                
            # Check shape and fix if needed
            if image.shape != (1, 48, 48):
                if image.shape[0] == 3 and image.shape[1] == 48 and image.shape[2] == 48:
                    # Convert RGB to grayscale by taking mean
                    image = image.mean(dim=0, keepdim=True)
                elif image.shape[0] == 1 and image.shape[1] == 1 and image.shape[2] == 48:
                    # Expand [1, 1, 48] to [1, 48, 48]
                    image = image.expand(1, 48, 48)
                elif image.shape[0] == 1:
                    # Try to reshape
                    try:
                        image = image.reshape(1, 48, 48)
                    except:
                        print(f"Warning: Could not reshape tensor of shape {image.shape}, using zeros")
                        image = torch.zeros((1, 48, 48))
                else:
                    # Replace with zeros
                    print(f"Warning: Tensor has incompatible shape {image.shape}, using zeros")
                    image = torch.zeros((1, 48, 48))
        else:
            # If not a tensor, create a zero tensor
            print(f"Warning: Image is not a tensor ({type(image)}), using zeros")
            image = torch.zeros((1, 48, 48))
        
        # Check the label
        if isinstance(label, torch.Tensor):
            # Convert to Python scalar
            label = label.item()
        
        # Append to lists
        images.append(image)
        labels.append(label)
    
    # Stack images and convert labels to tensor
    try:
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
    except:
        # Debug information if stacking fails
        print("Error stacking tensors. Tensor shapes:")
        for i, img in enumerate(images):
            print(f"Image {i} shape: {img.shape}")
        # Create empty tensors as fallback
        batch_size = len(images)
        images = torch.zeros((batch_size, 1, 48, 48))
        labels = torch.tensor(labels)
        
    return images, labels 