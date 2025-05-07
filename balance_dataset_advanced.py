#!/usr/bin/env python
"""
Standalone script for advanced dataset balancing.
This script creates a balanced dataset with target number of samples per class,
and directly updates the environment variables by creating env_vars.py.

Usage:
    python balance_dataset_advanced.py <dataset_path> <target_samples>

Example:
    python balance_dataset_advanced.py ./extracted/emotion/train 5000
"""

import os
import sys
import shutil
import random
import numpy as np
import traceback
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path


def balance_dataset_standalone(dataset_path, target_samples=5000):
    """
    Advanced dataset balancing that creates a physically balanced dataset:
    1. For majority classes: Apply random selection (keeping the harder examples)
    2. For minority classes: Use synthetic augmentation + original samples
    
    Args:
        dataset_path: Path to the root directory of the dataset
        target_samples: Target number of samples per class
        
    Returns:
        Path to the balanced dataset
    """
    try:
        print(f"🔶 Starting advanced dataset balancing process...")
        
        # Create output directory for balanced dataset
        parent_dir = os.path.dirname(dataset_path)
        base_name = os.path.basename(dataset_path)
        balanced_dir = os.path.join(parent_dir, f"balanced_{base_name}")
        
        # Check if balanced directory already exists and remove it
        if os.path.exists(balanced_dir):
            print(f"🔶 Removing existing balanced directory: {balanced_dir}")
            shutil.rmtree(balanced_dir)
            print(f"✅ Removed existing balanced directory")
        
        # Create fresh balanced directory
        os.makedirs(balanced_dir, exist_ok=True)
        print(f"✅ Created fresh balanced output directory: {balanced_dir}")
        
        # Get class distribution
        class_counts = {}
        class_dirs = {}
        
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                class_counts[class_name] = len(image_files)
                class_dirs[class_name] = class_path
                
                # Create corresponding directory in balanced dataset
                os.makedirs(os.path.join(balanced_dir, class_name), exist_ok=True)
        
        print(f"📊 Original class distribution: {class_counts}")
        
        # Advanced augmentation for minority classes
        minority_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
        
        # Process each class
        for class_idx, (class_name, count) in enumerate(class_counts.items()):
            source_dir = class_dirs[class_name]
            target_dir = os.path.join(balanced_dir, class_name)
            print(f"🔶 Processing class {class_idx+1}/{len(class_counts)}: {class_name}")
            
            # Strategy for majority classes (count > target_samples)
            if count > target_samples:
                print(f"  ↓ Reducing majority class {class_name} from {count} to {target_samples}")
                
                # Get list of image files
                image_files = [f for f in os.listdir(source_dir) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                # Using uniform random sampling
                selected_files = np.random.choice(image_files, target_samples, replace=False)
                
                # Copy selected files
                for i, file in enumerate(selected_files):
                    if i % 500 == 0:
                        print(f"  ↳ Copied {i}/{target_samples} images...")
                    
                    src_path = os.path.join(source_dir, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.copy(src_path, dst_path)
                    
            # Strategy for minority classes (count < target_samples)
            elif count < target_samples:
                print(f"  ↑ Augmenting minority class {class_name} from {count} to {target_samples}")
                
                # Copy all original samples
                image_files = [f for f in os.listdir(source_dir) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                print(f"  ↳ Copying {len(image_files)} original images...")
                for file in image_files:
                    src_path = os.path.join(source_dir, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.copy(src_path, dst_path)
                
                # Generate synthetic samples through advanced augmentation
                samples_to_generate = target_samples - count
                original_indices = list(range(len(image_files)))
                
                # Use batched approach for large augmentations
                batch_size = 100
                for batch_start in range(0, samples_to_generate, batch_size):
                    batch_end = min(batch_start + batch_size, samples_to_generate)
                    print(f"  ↳ Generating batch {batch_start+1}-{batch_end} of {samples_to_generate} augmented samples")
                    
                    for i in range(batch_start, batch_end):
                        # Select a random original image
                        idx = np.random.choice(original_indices)
                        orig_file = image_files[idx]
                        
                        try:
                            # Open and apply augmentation
                            with Image.open(os.path.join(source_dir, orig_file)) as img:
                                # Apply multiple augmentations for diversity
                                augmented = img.copy()
                                for _ in range(3):  # Apply 3 random augmentations
                                    augmented = minority_transform(augmented)
                                
                                # Save the augmented image
                                new_filename = f"aug_{class_name}_{i}.jpg"
                                augmented.save(os.path.join(target_dir, new_filename))
                        except Exception as e:
                            print(f"  ⚠️ Error processing image {orig_file}: {str(e)}")
                            # Try a different file as fallback
                            fallback_idx = np.random.choice(original_indices)
                            fallback_file = image_files[fallback_idx]
                            try:
                                with Image.open(os.path.join(source_dir, fallback_file)) as img:
                                    augmented = img.copy()
                                    augmented = minority_transform(augmented)
                                    new_filename = f"aug_{class_name}_{i}.jpg"
                                    augmented.save(os.path.join(target_dir, new_filename))
                            except Exception as e2:
                                print(f"  ❌ Failed with fallback image too: {str(e2)}")
                                # Just copy original as last resort
                                shutil.copy(
                                    os.path.join(source_dir, fallback_file),
                                    os.path.join(target_dir, f"copy_{class_name}_{i}.jpg")
                                )
            
            # For classes that match the target count, just copy all
            else:
                print(f"  ↔ Class {class_name} already has {count} samples (target: {target_samples})")
                image_files = [f for f in os.listdir(source_dir) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                for file in image_files:
                    src_path = os.path.join(source_dir, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.copy(src_path, dst_path)
            
            # Verify this class was processed correctly
            processed_files = [f for f in os.listdir(target_dir) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"  ✓ Class {class_name}: {len(processed_files)}/{target_samples} images")
        
        # Verify final distribution
        final_counts = {}
        for class_name in os.listdir(balanced_dir):
            class_path = os.path.join(balanced_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                final_counts[class_name] = len(image_files)
        
        print(f"📊 Final balanced distribution: {final_counts}")
        print(f"🎉 Successfully created balanced dataset at: {balanced_dir}")
        return balanced_dir
        
    except Exception as e:
        print(f"❌ Error in dataset balancing: {str(e)}")
        print("❌ Detailed traceback:")
        traceback.print_exc()
        return None


def update_env_variables(balanced_path):
    """
    Create env_vars.py file to update environment variables for training scripts
    
    Args:
        balanced_path: Path to the balanced dataset
    """
    try:
        # Create or overwrite env_vars.py
        with open('env_vars.py', 'w') as f:
            f.write("""import os
import sys

# Dataset path override by advanced balancer
BALANCED_PATH = \"\"\"{}\"\"\"
os.environ['TRAIN_PATH'] = BALANCED_PATH.strip()
os.environ['MODEL_BALANCE_DATASET'] = "0"

# When run directly, only print the path with no other output
if __name__ == "__main__":
    # For shell script extraction - print only the path
    if len(sys.argv) > 1 and sys.argv[1] == "--path-only":
        print(os.environ['TRAIN_PATH'])
        sys.exit(0)
    
    # Normal output for interactive use
    print(f"🔹 Loaded environment variables from balance_dataset_advanced.py:")
    print(f"   TRAIN_PATH: {{os.environ['TRAIN_PATH']}}")
    print(f"   MODEL_BALANCE_DATASET: {{os.environ['MODEL_BALANCE_DATASET']}}")
""".format(balanced_path))
        
        print(f"✅ Created env_vars.py with updated environment variables")
        return True
    except Exception as e:
        print(f"❌ Error updating environment variables: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    target_samples = int(sys.argv[2])
    
    print(f"🔶 Balancing dataset at {dataset_path} with target {target_samples} samples per class")
    
    # First validate the dataset path
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist")
        sys.exit(1)
        
    # Check if we have subdirectories (classes)
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not class_dirs:
        print(f"❌ Error: No class subdirectories found in '{dataset_path}'")
        sys.exit(1)
        
    print(f"✅ Found {len(class_dirs)} class directories: {', '.join(class_dirs)}")
    
    # Check if each class has images
    total_images = 0
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        print(f"   - {class_dir}: {len(images)} images")
    
    print(f"✅ Total images: {total_images}")
    
    # Run the balancing process
    balanced_path = balance_dataset_standalone(dataset_path, target_samples)
    
    if balanced_path:
        # Update environment variables
        if update_env_variables(balanced_path):
            print(f"✅ Successfully updated environment variables in env_vars.py")
        else:
            print(f"⚠️ Failed to update environment variables, but balancing was successful")
        
        print(balanced_path)  # Print the path as last line for easier capture
        sys.exit(0)
    else:
        sys.exit(1) 