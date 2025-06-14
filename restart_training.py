#!/usr/bin/env python3
"""
Restart Enhanced GReFEL Training with Fixed Parameters
"""

import subprocess
import sys
import os

def main():
    print("🔧 Restarting Enhanced GReFEL Training with Fixed Parameters")
    print("=" * 60)
    
    # Fixed training command with stable hyperparameters
    cmd = [
        sys.executable, "train.py",
        "--data_dir", "./FERPlus",
        "--use_soft_labels",  # Use probability distributions
        "--batch_size", "16",  # Small batch for stability
        "--epochs", "150", 
        "--lr", "2e-4",  # Good learning rate
        "--weight_decay", "0.01",
        "--warmup_epochs", "0",  # No warmup for simplicity
        "--label_smoothing", "0.11",
        "--mixup_alpha", "0.0",  # Disable mixup initially for stability
        "--feature_dim", "768",
        "--num_anchors", "10",
        "--drop_rate", "0.1",  # Reduced dropout
        "--grad_clip", "0.5",  # Smaller gradient clipping
        "--ema_decay", "0.0",  # Disable EMA initially
        "--num_workers", "2",  # Reduced workers
        "--save_freq", "10"
    ]
    
    print("Fixed Parameters:")
    print("  - Batch size: 16 (reduced for stability)")
    print("  - Learning rate: 2e-4 (increased from 1e-6)")
    print("  - Mixup: Disabled initially")
    print("  - EMA: Disabled initially") 
    print("  - Gradient clip: 0.5 (reduced)")
    print("  - Dropout: 0.1 (reduced)")
    print("  - Geometry loss: Much smaller weights")
    print("  - Loss clamping: Reduced to 5.0")
    print()
    
    # Check if data directory exists
    if not os.path.exists("./FERPlus"):
        print("❌ Error: ./FERPlus directory not found!")
        print("Please run preprocessing first:")
        print("  python datasets/preprocess_ferplus.py")
        return 1
    
    try:
        # Run training
        print("🚀 Starting training...")
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 