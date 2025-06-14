#!/usr/bin/env python3
"""
Convenient Test Evaluation Runner
Runs test set evaluation with optimal settings matching training configuration.
"""

import subprocess
import sys
import os

def main():
    # Test evaluation command with optimized settings
    cmd = [
        sys.executable, "evaluate_test_set.py",
        "--data_dir", "./FERPlus",
        "--checkpoint_dir", "checkpoints",
        "--batch_size", "32",  # Smaller batch for stable evaluation
        "--num_workers", "4",
        "--device", "cuda",
        "--output_dir", "test_results",
        "--use_soft_labels"  # Match training configuration
    ]
    
    # Check if data directory exists
    if not os.path.exists("./FERPlus"):
        print("❌ Error: ./FERPlus directory not found!")
        print("Please ensure FERPlus dataset is available.")
        return 1
    
    # Check if checkpoints exist
    if not os.path.exists("checkpoints"):
        print("❌ Error: checkpoints directory not found!")
        print("Please run training first to generate model checkpoints.")
        return 1
    
    # Check for specific checkpoint files
    best_checkpoint = os.path.exists("checkpoints/best_model.pth")
    ema_checkpoint = os.path.exists("checkpoints/best_ema_model.pth")
    
    if not best_checkpoint and not ema_checkpoint:
        print("❌ Error: No model checkpoints found!")
        print("Please run training first to generate checkpoints:")
        print("  python run_optimized_training.py")
        return 1
    
    print("🚀 Starting test set evaluation...")
    if best_checkpoint:
        print("  ✅ Found best model checkpoint")
    if ema_checkpoint:
        print("  ✅ Found EMA model checkpoint")
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, check=True)
        print("✅ Test evaluation completed successfully!")
        print("\nResults saved to: ./test_results/")
        print("📊 Check test_evaluation_summary.json for quick overview")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Test evaluation failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Test evaluation interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 