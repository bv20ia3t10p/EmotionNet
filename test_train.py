"""Simple test script to verify model loading."""

import os
import torch
from emotion_net.models import create_model

def main():
    """Test model creation and basic operations."""
    print("Creating model...")
    model = create_model('sota_resemote_medium', num_classes=7, pretrained=True)
    print(f"Model created: {type(model).__name__}")
    
    # Test forward pass with random data
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # Set model to eval mode
        model.eval()
        output = model(dummy_input)
    
    # Check output
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    # Set PYTHONPATH to include current directory
    os.environ['PYTHONPATH'] = os.getcwd()
    main() 