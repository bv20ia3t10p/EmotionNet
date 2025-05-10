from utils import *
from training import *
import warnings
import argparse
import sys
import os
import re

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

def sanitize_path(path):
    """Sanitize path that might contain debug output or newlines"""
    if not path or not isinstance(path, str):
        return path
        
    # Extract the actual path if it contains debug output
    if '\n' in path:
        match = re.search(r'TRAIN_PATH: (.*?)(?:\n|$)', path)
        if match:
            path = match.group(1).strip()
            print(f"⚠️ Sanitized path from debug output: {path}")
    
    return path

if __name__ == "__main__":
    args = parse_args()
    
    print("🔹 Starting Emotion Recognition Script ==============================")
    log_configuration()
    
    # Ensure environment variables are correctly set
    from config import TRAIN_PATH as CONFIG_TRAIN_PATH, TEST_PATH as CONFIG_TEST_PATH
    
    # Sanitize paths that might have debug output
    TRAIN_PATH = sanitize_path(CONFIG_TRAIN_PATH)
    TEST_PATH = sanitize_path(CONFIG_TEST_PATH)
    
    # Set cleaned environment variables for downstream code
    os.environ['TRAIN_PATH'] = TRAIN_PATH
    os.environ['TEST_PATH'] = TEST_PATH
    os.environ['MODEL_BALANCE_DATASET'] = '0'
    
    # Print the actual paths that will be used
    print(f"🔹 FINAL TRAINING PATH: {TRAIN_PATH}")
    print(f"🔹 Using original dataset (MODEL_BALANCE_DATASET=0)")
    
    # Verify paths exist
    if not os.path.exists(TRAIN_PATH):
        print(f"⚠️ Warning: Training path does not exist: {TRAIN_PATH}")
        # Try to use a fixed path
        fixed_path = "./extracted/emotion/train"
        if os.path.exists(fixed_path):
            print(f"⚠️ Using fixed training path: {fixed_path}")
            TRAIN_PATH = fixed_path
            os.environ['TRAIN_PATH'] = TRAIN_PATH
            
    if not os.path.exists(TEST_PATH):
        print(f"⚠️ Warning: Testing path does not exist: {TEST_PATH}")
        # Try to use a fixed path
        fixed_path = "./extracted/emotion/test"
        if os.path.exists(fixed_path):
            print(f"⚠️ Using fixed testing path: {fixed_path}")
            TEST_PATH = fixed_path
            os.environ['TEST_PATH'] = TEST_PATH
    
    if args.mode == 'train':
        print("🔹 Running in TRAINING mode")
        # Log class distribution to identify imbalance
        train_stats = get_image_stats(TRAIN_PATH)
        
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
