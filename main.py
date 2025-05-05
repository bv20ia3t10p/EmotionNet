from utils import *
from training import *
import warnings
import argparse
import sys

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Using a non-full backward hook*")
warnings.filterwarnings(
    "ignore", message=".*UserWarning: Secure RNG turned off.*?")

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Recognition Training/Testing")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test (default: train)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for testing (default: use MODEL_PATH from config)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("🔹 Starting Emotion Recognition Script ==============================")
    log_configuration()
    
    if args.mode == 'train':
        print("🔹 Running in TRAINING mode")
        # Log class distribution to identify imbalance
        train_stats = get_image_stats(TRAIN_PATH)
        
        # Ensure balanced classes for training
        ensure_images_per_class(TRAIN_PATH, 7500)
        
        # Start training
        train_model()
    
    elif args.mode == 'test':
        print("🔹 Running in TESTING mode")
        
        # Override MODEL_PATH if specified
        if args.model_path:
            print(f"🔹 Using model: {args.model_path}")
            MODEL_PATH = args.model_path
        
        # Import and run test function
        from test_model import test_model
        test_model()
    
    else:
        print(f"⚠️ Unknown mode: {args.mode}")
        sys.exit(1)
