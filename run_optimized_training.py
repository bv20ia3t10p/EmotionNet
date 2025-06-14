#!/usr/bin/env python3
"""
Optimized GReFEL Training Script
Runs training with paper-specified hyperparameters and soft labels by default.
"""

import subprocess
import sys
import os

def main():
    # Default optimized training command with conservative hyperparameters
    cmd = [
        sys.executable, "train.py",
        "--data_dir", "./FERPlus",
        "--use_soft_labels",  # Use probability distributions (default)
        "--batch_size", "32",  # Further reduced for stability
        "--epochs", "150", 
        "--lr", "1e-4",  # Proper learning rate for stable training
        "--weight_decay", "0.01",  # Reduced weight decay
        "--warmup_epochs", "10",
        "--label_smoothing", "0.11",
        "--mixup_alpha", "0.2",
        "--feature_dim", "768",
        "--num_anchors", "10",
        "--drop_rate", "0.15",
        "--grad_clip", "1.0",
        "--ema_decay", "0.9999",
        "--num_workers", "4",  # Reduced for stability
        "--save_freq", "10"
    ]
    
    
    # Check if data directory exists
    if not os.path.exists("./FERPlus"):
        print("❌ Error: ./FERPlus directory not found!")
        print("Please run preprocessing first:")
        print("  python datasets/preprocess_ferplus.py")
        return 1
    
    # Check if processed data exists
    if not os.path.exists("./FERPlus/processed"):
        print("❌ Error: Processed data not found!")
        print("Please run preprocessing first:")
        print("  python datasets/preprocess_ferplus.py")
        return 1
    
    try:
        # Run training
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