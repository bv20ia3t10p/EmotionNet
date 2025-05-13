"""Data Manager specific to FER2013 dataset."""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import tempfile
import shutil
import glob
import atexit

from .parsers import parse_fer2013
from .dataset import BaseEmotionDataset
from emotion_net.config.constants import EMOTIONS

class FER2013DataManager:
    """Handles loading, parsing, and splitting of FER2013 data."""
    def __init__(self, data_dir, test_dir=None, image_size=224, val_split_ratio=0.1, seed=42):
        self.data_dir = os.path.abspath(data_dir)
        if test_dir:
            self.test_dir = os.path.abspath(test_dir)
        else:
            self.test_dir = None
        self.image_size = image_size
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.all_classes = list(EMOTIONS.values())
        self.temp_dir = None  # Store the temporary directory path
        # Add FER2013 specific mean and std for standardization
        self.fer_mean = 0.5086  # Empirical mean from FER2013 dataset
        self.fer_std = 0.2532   # Empirical std from FER2013 dataset
        
        # Clean up any existing temporary directories from previous runs
        self._cleanup_old_temp_dirs()
        
        # Register cleanup on exit to ensure temp directories are always removed
        atexit.register(self.cleanup_temp_dir)
        
        print(f"Initializing FER2013DataManager with data_dir: {data_dir}")

    def _cleanup_old_temp_dirs(self):
        """Clean up any existing FER2013 temporary directories from previous runs."""
        try:
            # Look for any temp directories that match our prefix pattern
            temp_pattern = os.path.join(tempfile.gettempdir(), "fer2013_*")
            old_temp_dirs = glob.glob(temp_pattern)
            
            if old_temp_dirs:
                print(f"Found {len(old_temp_dirs)} old temporary directories, cleaning up...")
                for old_dir in old_temp_dirs:
                    try:
                        if os.path.exists(old_dir) and os.path.isdir(old_dir):
                            print(f"Removing old temporary directory: {old_dir}")
                            shutil.rmtree(old_dir, ignore_errors=True)
                    except Exception as e:
                        print(f"Error removing directory {old_dir}: {e}")
        except Exception as e:
            print(f"Error during old temp directory cleanup: {e}")

    def get_datasets(self):
        """Loads, splits, and returns train, val, test datasets and train labels."""
        # Always try CSV loading first for FER2013 at ./dataset/fer2013
        if self._is_csv_dataset():
            print(f"Found CSV files in {self.data_dir}, creating directory structure from CSV")
            return self._create_dir_from_csv_and_load()
        else:
            # Only fall back to directory structure if CSV not found
            print(f"No CSV files found in {self.data_dir}, trying directory structure")
            return self._load_from_directory()
    
    def _is_csv_dataset(self):
        """Check if we're dealing with CSV files instead of directory structure."""
        # Look for standard CSV filenames
        train_csv = os.path.join(self.data_dir, 'train.csv')
        test_csv = os.path.join(self.data_dir, 'test.csv')
        full_csv = os.path.join(self.data_dir, 'icml_face_data.csv')
        
        # Print available files in the directory for debugging
        print(f"Files in {self.data_dir}:")
        try:
            for file in os.listdir(self.data_dir):
                print(f"  {file}")
        except Exception as e:
            print(f"Error listing directory: {e}")
        
        has_csv = os.path.exists(train_csv) or os.path.exists(full_csv)
        print(f"CSV files found: {has_csv}")
        return has_csv
    
    def _create_dir_from_csv_and_load(self):
        """Create directory structure from CSV and then load directly from those files."""
        # Cleanup any existing temporary directory first
        self.cleanup_temp_dir()
        
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="fer2013_")
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Create subdirectories for emotions
        train_dir = os.path.join(self.temp_dir, 'train')
        test_dir = os.path.join(self.temp_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Create emotion subdirectories
        for emotion_name in self.all_classes:
            os.makedirs(os.path.join(train_dir, emotion_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, emotion_name), exist_ok=True)
        
        # Load CSV files
        train_csv = os.path.join(self.data_dir, 'train.csv')
        test_csv = os.path.join(self.data_dir, 'test.csv')
        full_csv = os.path.join(self.data_dir, 'icml_face_data.csv')
        
        # Initialize dataframes
        train_df = None
        test_df = None
        val_df = None
        
        # Process CSVs based on what's available
        if os.path.exists(full_csv):
            print(f"Processing full dataset from {full_csv}")
            full_df = pd.read_csv(full_csv)
            full_df.columns = full_df.columns.str.strip()
            
            # Split by Usage column if it exists - use all Training for train, PublicTest for validation, PrivateTest for test
            if 'Usage' in full_df.columns:
                train_df = full_df[full_df['Usage'] == 'Training'].copy()
                val_df = full_df[full_df['Usage'] == 'PublicTest'].copy()
                test_df = full_df[full_df['Usage'] == 'PrivateTest'].copy()
                print(f"Split by Usage: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
            else:
                # No Usage column - manual split 70/15/15
                total_samples = len(full_df)
                indices = np.random.permutation(total_samples)
                train_size = int(total_samples * 0.7)
                val_size = int(total_samples * 0.15)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size+val_size]
                test_indices = indices[train_size+val_size:]
                
                train_df = full_df.iloc[train_indices].copy()
                val_df = full_df.iloc[val_indices].copy()
                test_df = full_df.iloc[test_indices].copy()
                print(f"Manual split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
        else:
            # Use separate train.csv and test.csv files
            print(f"Using separate CSV files")
            
            if os.path.exists(train_csv):
                print(f"Loading training data from {train_csv}")
                train_df = pd.read_csv(train_csv)
                train_df.columns = train_df.columns.str.strip()
                print(f"Loaded {len(train_df)} training samples from {train_csv}")
                
                # If we need to split train into train/val
                if self.val_split_ratio > 0:
                    # Get indices for train/val split
                    indices = np.random.permutation(len(train_df))
                    split_idx = int(len(indices) * (1 - self.val_split_ratio))
                    train_indices = indices[:split_idx]
                    val_indices = indices[split_idx:]
                    
                    # Create val_df from part of train_df
                    val_df = train_df.iloc[val_indices].copy()
                    # Update train_df to exclude validation samples
                    train_df = train_df.iloc[train_indices].copy()
                    print(f"Split training data: {len(train_df)} for training, {len(val_df)} for validation")
            
            if os.path.exists(test_csv):
                print(f"Loading test data from {test_csv}")
                test_df = pd.read_csv(test_csv)
                test_df.columns = test_df.columns.str.strip()
                print(f"Loaded {len(test_df)} test samples from {test_csv}")
                
                # If we don't have validation data yet, use test data for validation
                if val_df is None:
                    val_df = test_df.copy()
                    print(f"Using test data as validation")
        
        # Error check - if we don't have training data, can't proceed
        if train_df is None or len(train_df) == 0:
            print(f"Error: Could not load sufficient training data from CSV files")
            return None, None, None, []
        
        # Process training data - save images to disk
        if train_df is not None and len(train_df) > 0:
            print(f"Saving {len(train_df)} training images...")
            self._csv_to_images(train_df, train_dir, is_train=True)
        
        # Process validation data
        if val_df is not None and len(val_df) > 0:
            print(f"Saving {len(val_df)} validation images...")
            self._csv_to_images(val_df, test_dir, is_train=False)
        
        # Process test data (if separate from validation)
        test_output_dir = os.path.join(self.temp_dir, 'actual_test')
        if test_df is not None and len(test_df) > 0 and val_df is not test_df:
            os.makedirs(test_output_dir, exist_ok=True)
            for emotion_name in self.all_classes:
                os.makedirs(os.path.join(test_output_dir, emotion_name), exist_ok=True)
            
            print(f"Saving {len(test_df)} test images...")
            self._csv_to_images(test_df, test_output_dir, is_train=False)
        
        # Now we'll load datasets from the saved image files, using absolute paths
        train_paths = []
        train_labels = []
        val_paths = []
        val_labels = []
        test_paths = []
        test_labels = []
        
        # Get training images
        for emotion_idx, emotion_name in EMOTIONS.items():
            emotion_dir = os.path.join(train_dir, emotion_name)
            if not os.path.exists(emotion_dir):
                continue
                
            # List all images in this emotion directory
            image_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(image_files)} training images for {emotion_name}")
            
            # Add to training data
            for img_file in image_files:
                img_path = os.path.join(emotion_dir, img_file)
                img_path = os.path.abspath(img_path)  # Ensure absolute path
                train_paths.append(img_path)
                train_labels.append(emotion_idx)
        
        # Get validation images
        for emotion_idx, emotion_name in EMOTIONS.items():
            emotion_dir = os.path.join(test_dir, emotion_name)
            if not os.path.exists(emotion_dir):
                continue
                
            # List all images in this emotion directory
            image_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(image_files)} validation images for {emotion_name}")
            
            # Add to validation data
            for img_file in image_files:
                img_path = os.path.join(emotion_dir, img_file)
                img_path = os.path.abspath(img_path)  # Ensure absolute path
                val_paths.append(img_path)
                val_labels.append(emotion_idx)
        
        # Get test images (if separate directory exists)
        if os.path.exists(test_output_dir):
            for emotion_idx, emotion_name in EMOTIONS.items():
                emotion_dir = os.path.join(test_output_dir, emotion_name)
                if not os.path.exists(emotion_dir):
                    continue
                    
                # List all images in this emotion directory
                image_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Found {len(image_files)} test images for {emotion_name}")
                
                # Add to test data
                for img_file in image_files:
                    img_path = os.path.join(emotion_dir, img_file)
                    img_path = os.path.abspath(img_path)  # Ensure absolute path
                    test_paths.append(img_path)
                    test_labels.append(emotion_idx)
        
        # Create datasets 
        train_dataset = None
        val_dataset = None
        test_dataset = None
        
        if train_paths:
            train_dataset = BaseFER2013Dataset(
                train_paths, train_labels, 
                self.all_classes, mode='train', 
                image_size=self.image_size,
                dataset_name='fer2013',
                fer_mean=self.fer_mean,
                fer_std=self.fer_std
            )
            
        if val_paths:
            val_dataset = BaseFER2013Dataset(
                val_paths, val_labels, 
                self.all_classes, mode='val', 
                image_size=self.image_size,
                dataset_name='fer2013',
                fer_mean=self.fer_mean,
                fer_std=self.fer_std
            )
        
        # Create test dataset if we have test paths
        if test_paths:
            test_dataset = BaseFER2013Dataset(
                test_paths, test_labels, 
                self.all_classes, mode='test', 
                image_size=self.image_size,
                dataset_name='fer2013',
                fer_mean=self.fer_mean,
                fer_std=self.fer_std
            )
        else:
            # If no separate test data, use validation data as test
            print("No separate test data found, using validation dataset for testing")
            test_dataset = val_dataset
        
        # Print summary
        print("-- FER2013DataManager: Dataset Loading Summary --")
        print(f"Train: {len(train_dataset) if train_dataset else 0} samples")
        print(f"Val:   {len(val_dataset) if val_dataset else 0} samples")
        print(f"Test:  {len(test_dataset) if test_dataset else 0} samples")
        print(f"Class names: {self.all_classes}")
        print(f"Training labels for sampler: {len(train_labels)} samples")
        print("-------------------------------------------------")
            
        # Store directories for future reference
        self.train_dir = train_dir
        self.test_dir = test_dir
        
        # Return: train_dataset, val_dataset, test_dataset, train_labels
        return train_dataset, val_dataset, test_dataset, train_labels
    
    def _csv_to_images(self, df, output_dir, is_train=True):
        """Extract images from DataFrame and save to output directory with enhanced preprocessing."""
        from PIL import Image, ImageOps, ImageEnhance
        import numpy as np
        import os
        
        # Ensure output_dir is absolute
        output_dir = os.path.abspath(output_dir)
        print(f"Extracting {len(df)} images to {output_dir}")
        
        # Find column names
        pixel_col = None
        emotion_col = None
        
        if 'pixels' in df.columns:
            pixel_col = 'pixels'
        elif 'Pixels' in df.columns:
            pixel_col = 'Pixels'
        else:
            print(f"Error: No pixel column found in DataFrame. Columns: {df.columns.tolist()}")
            return
            
        if 'emotion' in df.columns:
            emotion_col = 'emotion'
        elif 'Emotion' in df.columns:
            emotion_col = 'Emotion'
        else:
            print(f"Warning: No emotion column found. Using neutral class for all images.")
        
        # Calculate class weights for augmentation (training only)
        class_counts = {}
        max_count = 0
        if emotion_col and is_train:
            for e in df[emotion_col]:
                e = int(e)
                if e not in class_counts:
                    class_counts[e] = 0
                class_counts[e] += 1
                max_count = max(max_count, class_counts[e])
        
        # Calculate augmentation ratios based on class imbalance
        augment_emotions = {}
        if class_counts and max_count > 0:
            for e, count in class_counts.items():
                # Calculate ratio relative to most common class
                ratio = max_count / count
                # Add 5x more augmentations for disgust class (always rare)
                if e == 1:  # disgust
                    ratio *= 5
                
                # Set upper/lower bounds on augmentation
                if ratio > 10:
                    ratio = 10  # Cap at 10x
                elif ratio < 1:
                    ratio = 1  # No augmentation for already common classes
                
                if ratio > 1.5:  # Only augment if there's significant imbalance
                    augment_emotions[e] = int(ratio)
            
            print(f"Class-based augmentation ratios: {augment_emotions}")
        else:
            # Fallback to previous fixed ratios if we couldn't compute class weights
            augment_emotions = {1: 10, 2: 3, 5: 2}
        
        # Process rows
        count = 0
        
        for idx, row in df.iterrows():
            try:
                # Get emotion label
                if emotion_col:
                    emotion_idx = int(row[emotion_col])
                    if 0 <= emotion_idx < len(self.all_classes):
                        emotion_name = self.all_classes[emotion_idx]
                    else:
                        emotion_name = 'neutral'  # Default if out of range
                else:
                    emotion_name = 'neutral'  # Default if no emotion column
                    emotion_idx = 6  # Neutral index
                
                # Parse pixels
                pixel_str = row[pixel_col]
                if isinstance(pixel_str, str):
                    if ' ' in pixel_str:
                        pixels = np.array([int(p) for p in pixel_str.split()], dtype=np.float32)
                    else:
                        pixels = np.array([int(p) for p in pixel_str.split(',')], dtype=np.float32)
                else:
                    pixels = np.array(pixel_str, dtype=np.float32)
                
                # Check pixel count
                expected_size = 48 * 48
                if len(pixels) != expected_size:
                    if len(pixels) > expected_size:
                        pixels = pixels[:expected_size]
                    else:
                        continue  # Skip if not enough pixels
                
                # Normalize pixel values to improve learning
                pixels = pixels / 255.0
                
                # Apply advanced image processing
                # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                pixels_img = (pixels * 255).astype(np.uint8).reshape(48, 48)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                pixels_img = clahe.apply(pixels_img)
                pixels = pixels_img.astype(np.float32) / 255.0
                
                # 2. Apply improved contrast normalization
                pixels_mean = np.mean(pixels)
                pixels_std = np.std(pixels)
                if pixels_std > 0:
                    pixels = (pixels - pixels_mean) / (pixels_std + 1e-7)
                    # Scale back to 0-1 range approximately
                    pixels = (pixels * 0.25) + 0.5
                    # Clip to ensure we're in 0-1 range
                    pixels = np.clip(pixels, 0, 1)
                
                # Standardize using FER2013 dataset statistics
                pixels = (pixels - self.fer_mean) / (self.fer_std + 1e-7)
                pixels = np.clip(pixels, -2.5, 2.5)  # Clip outliers
                
                # Rescale to 0-1 for saving as image
                pixels_normalized = (pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-7)
                
                # Convert to 0-255 range for saving
                pixels_to_save = (pixels_normalized * 255).astype(np.uint8)
                
                # Reshape and create image
                img = pixels_to_save.reshape(48, 48)
                pil_img = Image.fromarray(img)
                
                # Save to appropriate directory
                emotion_dir = os.path.join(output_dir, emotion_name)
                os.makedirs(emotion_dir, exist_ok=True)
                img_path = os.path.join(emotion_dir, f"{idx}.png")
                
                # Ensure the image path is absolute
                img_path = os.path.abspath(img_path)
                
                pil_img.save(img_path)
                
                # Create augmented versions for minority classes during training
                if is_train and emotion_idx in augment_emotions:
                    num_augmentations = augment_emotions[emotion_idx]
                    
                    # Different augmentation strategies for different emotions
                    for aug_idx in range(num_augmentations):
                        # Create augmented versions with different transformations
                        aug_img = pil_img.copy()
                        
                        # Special treatment per emotion
                        if emotion_idx == 1:  # disgust - needs more intense augmentation
                            if aug_idx % 5 == 0:  # Rotate more
                                angle = np.random.uniform(-20, 20)
                                aug_img = aug_img.rotate(angle, resample=Image.BILINEAR)
                            elif aug_idx % 5 == 1:  # Higher contrast
                                factor = np.random.uniform(1.1, 1.4)
                                aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)
                            elif aug_idx % 5 == 2:  # Random shifts
                                aug_img = ImageOps.expand(aug_img, border=(
                                    np.random.randint(0, 7),  # left
                                    np.random.randint(0, 7),  # top
                                    np.random.randint(0, 7),  # right
                                    np.random.randint(0, 7)   # bottom
                                ), fill=0)
                                aug_img = aug_img.resize((48, 48), Image.BILINEAR)
                            elif aug_idx % 5 == 3:  # Brightness
                                factor = np.random.uniform(0.8, 1.2)
                                aug_img = ImageEnhance.Brightness(aug_img).enhance(factor)
                            else:  # Slight zoom
                                zoom = np.random.uniform(0.9, 1.1)
                                w, h = aug_img.size
                                new_w, new_h = int(w * zoom), int(h * zoom)
                                aug_img = aug_img.resize((new_w, new_h), Image.BILINEAR)
                                # Center crop to original size
                                left = max(0, (new_w - w) // 2)
                                top = max(0, (new_h - h) // 2)
                                right = min(new_w, left + w)
                                bottom = min(new_h, top + h)
                                aug_img = aug_img.crop((left, top, right, bottom))
                                aug_img = aug_img.resize((48, 48), Image.BILINEAR)
                        
                        elif emotion_idx == 2:  # fear - moderate augmentation
                            if aug_idx % 3 == 0:  # Rotate
                                angle = np.random.uniform(-15, 15)
                                aug_img = aug_img.rotate(angle, resample=Image.BILINEAR)
                            elif aug_idx % 3 == 1:  # Adjust contrast
                                factor = np.random.uniform(0.9, 1.3)
                                aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)
                            else:  # Slight shift
                                aug_img = ImageOps.expand(aug_img, border=(
                                    np.random.randint(0, 5),  # left
                                    np.random.randint(0, 5),  # top
                                    np.random.randint(0, 5),  # right
                                    np.random.randint(0, 5)   # bottom
                                ), fill=0)
                                aug_img = aug_img.resize((48, 48), Image.BILINEAR)
                        
                        else:  # other emotions - basic augmentation
                            if aug_idx % 3 == 0:  # Rotate
                                angle = np.random.uniform(-10, 10)
                                aug_img = aug_img.rotate(angle, resample=Image.BILINEAR)
                            elif aug_idx % 3 == 1:  # Adjust contrast
                                factor = np.random.uniform(0.8, 1.2)
                                aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)
                            else:  # Slight shift
                                aug_img = ImageOps.expand(aug_img, border=(
                                    np.random.randint(0, 3),  # left
                                    np.random.randint(0, 3),  # top
                                    np.random.randint(0, 3),  # right
                                    np.random.randint(0, 3)   # bottom
                                ), fill=0)
                                aug_img = aug_img.resize((48, 48), Image.BILINEAR)
                        
                        # Save augmented image
                        aug_path = os.path.join(emotion_dir, f"{idx}_aug{aug_idx}.png")
                        aug_path = os.path.abspath(aug_path)
                        aug_img.save(aug_path)
                
                # Verify the image was saved correctly
                if not os.path.exists(img_path):
                    print(f"Warning: Failed to save image at {img_path}")
                
                count += 1
                if count % 1000 == 0:
                    print(f"Saved {count} images")
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        print(f"Successfully saved {count} images to {output_dir}")
        
    def _load_from_directory(self):
        """Load FER2013 dataset from directory structure with image files."""
        train_dataset, val_dataset, test_dataset = None, None, None
        train_labels_list = []

        # --- Train/Val Data ---
        train_data_dir = self.data_dir # Assume data_dir points to the train subdirectory
        full_train_paths, full_train_labels = parse_fer2013(train_data_dir)

        if not full_train_paths: # Handle case where parsing failed
             print(f"Error: Failed to parse training data from {train_data_dir}. Cannot proceed.")
             return None, None, None, []
             
        # Verify that paths exist
        for i, path in enumerate(full_train_paths[:10]):
            if not os.path.exists(path):
                print(f"Warning: File path does not exist: {path}")
                # Try making it absolute
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    print(f"  Found as absolute path: {abs_path}")
                    # Update all paths to use absolute paths
                    full_train_paths = [os.path.abspath(p) for p in full_train_paths]
                    break

        # Split the loaded training data indices into train/val
        print(f"Splitting FER2013 training data (ratio: {self.val_split_ratio})...")
        np.random.seed(self.seed)
        indices = np.arange(len(full_train_paths))

        if self.val_split_ratio > 0 and self.val_split_ratio < 1:
            try:
                train_indices, val_indices = train_test_split(
                    indices, test_size=self.val_split_ratio, random_state=self.seed, stratify=full_train_labels
                )
            except ValueError as e:
                 print(f"Stratify failed (maybe too few samples in a class?): {e}. Splitting without stratify.")
                 train_indices, val_indices = train_test_split(
                    indices, test_size=self.val_split_ratio, random_state=self.seed
                 )
            # Create train dataset
            train_paths = [full_train_paths[i] for i in train_indices]
            train_labels_list = [full_train_labels[i] for i in train_indices]
            
            # Create dataset using BaseEmotionDataset instead of FER2013Dataset for file paths
            train_dataset = BaseEmotionDataset(
                train_paths, train_labels_list, 
                self.all_classes, mode='train', 
                image_size=self.image_size,
                dataset_name='fer2013'
            )

            # Create val dataset
            val_paths = [full_train_paths[i] for i in val_indices]
            val_labels = [full_train_labels[i] for i in val_indices]
            
            # Create dataset using BaseEmotionDataset instead of FER2013Dataset for file paths
            val_dataset = BaseEmotionDataset(
                val_paths, val_labels, 
                self.all_classes, mode='val', 
                image_size=self.image_size,
                dataset_name='fer2013'
            )
        else:
             print("Validation split ratio is <= 0 or >= 1, using full training set for training.")
             
             # Create dataset using BaseEmotionDataset instead of FER2013Dataset for file paths
             train_dataset = BaseEmotionDataset(
                 full_train_paths, full_train_labels, 
                 self.all_classes, mode='train', 
                 image_size=self.image_size,
                 dataset_name='fer2013'
             )
             train_labels_list = full_train_labels
             val_dataset = None # No validation set

        # --- Test Data ---
        # Infer test dir relative to train_dir if not provided explicitly
        potential_test_dir = os.path.join(os.path.dirname(os.path.abspath(self.data_dir)), 'test')
        test_data_dir = self.test_dir if self.test_dir else potential_test_dir

        if os.path.exists(test_data_dir):
            test_paths, test_labels = parse_fer2013(test_data_dir)
            if test_paths:
                # Verify test paths exist
                for i, path in enumerate(test_paths[:10]):
                    if not os.path.exists(path):
                        print(f"Warning: Test file path does not exist: {path}")
                        # Try making them absolute
                        test_paths = [os.path.abspath(p) for p in test_paths]
                        break
                        
                # Create dataset using BaseEmotionDataset instead of FER2013Dataset for file paths
                test_dataset = BaseEmotionDataset(
                    test_paths, test_labels, 
                    self.all_classes, mode='test', 
                    image_size=self.image_size,
                    dataset_name='fer2013'
                )
        else:
            print(f"Warning: Test directory '{test_data_dir}' not found.")

        # --- Final Summary & Return ---
        print("-- FER2013DataManager: Dataset Loading Summary --")
        print(f"Train: {len(train_dataset) if train_dataset else 0} samples")
        print(f"Val:   {len(val_dataset) if val_dataset else 0} samples")
        print(f"Test:  {len(test_dataset) if test_dataset else 0} samples")
        print(f"Class names: {self.all_classes}")
        print(f"Training labels for sampler: {len(train_labels_list)} samples")
        print("-------------------------------------------------")

        # Ensure train_labels_list corresponds to the actual train_dataset created
        if train_dataset and len(train_labels_list) != len(train_dataset.labels):
            print(f"Warning: Mismatch between train_labels_list ({len(train_labels_list)}) and train_dataset labels ({len(train_dataset.labels)}). Re-assigning labels for sampler.")
            train_labels_list = train_dataset.labels # Use labels directly from created dataset

        return train_dataset, val_dataset, test_dataset, train_labels_list

    def get_temp_dir(self):
        """Returns the path to the temporary directory if it exists."""
        return self.temp_dir
        
    def get_train_dir(self):
        """Returns the path to the train directory in the temporary structure."""
        if self.temp_dir:
            return os.path.join(self.temp_dir, 'train')
        return None
        
    def get_test_dir(self):
        """Returns the path to the test directory in the temporary structure."""
        if self.temp_dir:
            return os.path.join(self.temp_dir, 'test')
        return None
        
    def cleanup_temp_dir(self):
        """Clean up the temporary directory if it exists."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                print(f"Cleaning up temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory: {e}")
            finally:
                self.temp_dir = None
                
    def __del__(self):
        """Destructor to ensure temp directory is cleaned up."""
        self.cleanup_temp_dir()

# Enhanced FER2013 dataset class with improved transforms
class BaseFER2013Dataset(BaseEmotionDataset):
    """Enhanced dataset for FER2013 with improved augmentations and normalization."""
    def __init__(self, images, labels, classes, mode='train', image_size=224, 
                 dataset_name='fer2013', fer_mean=0.5086, fer_std=0.2532):
        # Store the paths/images and labels
        self.classes = classes
        self.mode = mode
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.fer_mean = fer_mean
        self.fer_std = fer_std
        
        # Determine if we're dealing with PIL images or file paths
        self.is_in_memory = isinstance(images[0], Image.Image) if images else False
        
        if self.is_in_memory:
            # Store the PIL images directly
            self.images = images
            self.labels = labels
            print(f"  BaseFER2013Dataset: Using in-memory PIL Images ({len(images)} images)")
        else:
            # We're dealing with file paths
            self.image_paths = images
            self.labels = labels
            print(f"  BaseFER2013Dataset: Using file paths ({len(images)} paths)")
            
            # Validate file paths exist (check first 10)
            for i, path in enumerate(self.image_paths[:10]):
                if not os.path.exists(path):
                    print(f"  Warning: File path does not exist: {path}")
                else:
                    print(f"  Path exists: {path}")
        
        # Get enhanced transforms
        self.transform = self._get_enhanced_transforms(mode, image_size)
        
        print(f"  BaseFER2013Dataset created for mode '{mode}' with {len(self)} samples.")
        
        # Print class distribution
        if len(self) > 0:
            self._print_class_distribution()
    
    def _get_enhanced_transforms(self, mode, image_size):
        """Get enhanced transforms specific for FER2013 dataset."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Standard normalization using FER2013 statistics
        normalize = A.Normalize(
            mean=[self.fer_mean, self.fer_mean, self.fer_mean], 
            std=[self.fer_std, self.fer_std, self.fer_std],
            max_pixel_value=1.0
        )
        
        if mode == 'train':
            # Enhanced training transforms with focus on discriminative features
            return A.Compose([
                # Spatial augmentations
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.7),
                
                # Facial feature augmentations
                A.OneOf([
                    # Mouth region augmentation (helps distinguish happy/sad)
                    A.GridDistortion(distort_limit=0.15, p=0.5),
                    # Eye region augmentation (helps with anger/fear/sadness)
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.5),
                    # General facial structure preservation with noise
                    A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
                ], p=0.6),
                
                # Advanced intensity augmentations
                A.OneOf([
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
                    A.Equalize(p=0.5),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                ], p=0.6),
                
                # Facial occlusion simulation
                A.OneOf([
                    # Dropout for occlusion robustness
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, p=0.5),
                    # Random shadows (simulates lighting variations)
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.5),
                ], p=0.5),
                
                # Add small amount of blur (simulates real-world capture conditions)
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                
                # Region-specific transforms for important facial areas 
                A.Lambda(image=self._apply_region_specific_transforms, p=0.5),
                
                # Ensure grayscale is converted to RGB for the model
                A.ToGray(p=1.0),
                A.Resize(height=image_size, width=image_size),
                
                # Cutout specific regions that might confuse similar emotions
                A.Lambda(image=self._apply_emotion_specific_cutout, p=0.3),
                
                normalize,
                ToTensorV2(),
            ])
        elif mode == 'val':
            # Validation transforms - basic normalization with light augmentation
            return A.Compose([
                A.ToGray(p=1.0),
                A.Resize(height=image_size, width=image_size),
                normalize,
                ToTensorV2(),
            ])
        else:
            # Test transforms with test-time augmentation
            return A.Compose([
                A.ToGray(p=1.0),
                A.Resize(height=image_size, width=image_size),
                normalize,
                ToTensorV2(),
            ])
    
    def _apply_emotion_specific_cutout(self, image, **kwargs):
        """Apply emotion-specific cutout augmentation to help with commonly confused emotions."""
        import numpy as np
        
        # Create a copy of the input image
        img = image.copy()
        h, w = img.shape[:2]
        
        # Randomly choose an augmentation type focused on confused emotions
        aug_type = np.random.choice(['angry_sad', 'fear_sad', 'neutral_sad', 'surprise_fear', 'none'], 
                                    p=[0.25, 0.25, 0.2, 0.15, 0.15])
        
        if aug_type == 'angry_sad':
            # Focus on mouth and eyebrows (main difference between angry/sad)
            # Emphasize eyebrows by leaving them intact but occluding other areas
            eye_region_y = int(h * 0.4)
            mouth_region_y = int(h * 0.7)
            
            # Randomly choose to emphasize eyebrows or mouth
            if np.random.random() > 0.5:
                # Apply cutout to mouth region to emphasize eyebrows
                mouth_height = int(h * 0.2)
                x1 = np.random.randint(0, w - int(w * 0.5))
                y1 = mouth_region_y
                cutout_width = int(w * 0.5)
                img[y1:y1+mouth_height, x1:x1+cutout_width] = 0
            else:
                # Apply cutout to upper face to emphasize mouth
                eye_height = int(h * 0.3)
                x1 = np.random.randint(0, w - int(w * 0.7))
                cutout_width = int(w * 0.7)
                img[eye_region_y-eye_height:eye_region_y, x1:x1+cutout_width] = 0
                
        elif aug_type == 'fear_sad':
            # Focus on eyes (wider in fear, normal in sadness)
            eye_region_y = int(h * 0.35)
            eye_height = int(h * 0.15)
            
            # Cut out either left or right eye
            if np.random.random() > 0.5:
                # Left eye
                x1 = int(w * 0.2)
                cutout_width = int(w * 0.25)
            else:
                # Right eye
                x1 = int(w * 0.55)
                cutout_width = int(w * 0.25)
                
            img[eye_region_y:eye_region_y+eye_height, x1:x1+cutout_width] = 0
            
        elif aug_type == 'neutral_sad':
            # Focus on mouth corners (down in sad, straight in neutral)
            mouth_region_y = int(h * 0.7)
            mouth_height = int(h * 0.15)
            
            # Cut out central mouth area
            x1 = int(w * 0.35)
            cutout_width = int(w * 0.3)
            img[mouth_region_y:mouth_region_y+mouth_height, x1:x1+cutout_width] = 0
            
        elif aug_type == 'surprise_fear':
            # Focus on mouth shape (O in surprise, more tense in fear)
            if np.random.random() > 0.5:
                # Emphasize mouth
                mouth_region_y = int(h * 0.65)
                mouth_height = int(h * 0.25)
                x1 = int(w * 0.3)
                cutout_width = int(w * 0.4)
                
                # Create border around mouth and cut out the rest
                border = 5
                temp_mask = np.zeros_like(img)
                temp_mask[mouth_region_y:mouth_region_y+mouth_height, x1:x1+cutout_width] = 1
                temp_mask = temp_mask.astype(bool)
                
                # Invert the mask
                img[~temp_mask] = np.mean(img)
            else:
                # Emphasize eyes
                eye_region_y = int(h * 0.3)
                eye_height = int(h * 0.2)
                img[:eye_region_y, :] = np.mean(img)
                img[eye_region_y+eye_height:, :] = np.mean(img)
        
        return img
        
    def _apply_region_specific_transforms(self, image, **kwargs):
        """Apply transforms specific to facial regions."""
        import numpy as np
        import cv2
        
        # Create a copy of the input image
        img = image.copy()
        h, w = img.shape[:2]
        
        # Define facial regions (approximate for 48x48 or similar sized images)
        eye_region = [int(h * 0.2), int(h * 0.45), int(w * 0.1), int(w * 0.9)]  # y1, y2, x1, x2
        mouth_region = [int(h * 0.6), int(h * 0.9), int(w * 0.25), int(w * 0.75)]  # y1, y2, x1, x2
        
        # Enhance contrast in eyes (helps with fear/surprise/anger discrimination)
        eyes = img[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]].copy()
        if np.random.random() > 0.5 and eyes.size > 0:
            # Apply CLAHE to eye region
            if len(eyes.shape) == 3:
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_RGB2GRAY) if eyes.shape[2] == 3 else eyes[:,:,0]
            else:
                eyes_gray = eyes
                
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            eyes_enhanced = clahe.apply((eyes_gray * 255).astype(np.uint8))
            eyes_enhanced = eyes_enhanced.astype(np.float32) / 255.0
            
            if len(eyes.shape) == 3:
                for c in range(eyes.shape[2]):
                    eyes[:,:,c] = eyes_enhanced
            else:
                eyes = eyes_enhanced
                
            img[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]] = eyes
        
        # Enhance mouth region (helps with happy/sad/disgust discrimination)
        mouth = img[mouth_region[0]:mouth_region[1], mouth_region[2]:mouth_region[3]].copy()
        if np.random.random() > 0.5 and mouth.size > 0:
            # Apply slight sharpening to mouth region
            if len(mouth.shape) == 3:
                mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY) if mouth.shape[2] == 3 else mouth[:,:,0]
            else:
                mouth_gray = mouth
                
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1], 
                              [-1, 9, -1], 
                              [-1, -1, -1]]) * 0.5
            
            mouth_sharp = cv2.filter2D((mouth_gray * 255).astype(np.uint8), -1, kernel)
            mouth_sharp = mouth_sharp.astype(np.float32) / 255.0
            
            if len(mouth.shape) == 3:
                for c in range(mouth.shape[2]):
                    mouth[:,:,c] = mouth_sharp
            else:
                mouth = mouth_sharp
                
            img[mouth_region[0]:mouth_region[1], mouth_region[2]:mouth_region[3]] = mouth
            
        return img
    
    def __len__(self):
        return len(self.images) if self.is_in_memory else len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get label
        label = self.labels[idx]
        
        try:
            if self.is_in_memory:
                # Using PIL images from memory
                img = self.images[idx]
        # Convert PIL image to numpy array
                img = np.array(img)
            else:
                # Using file paths
                img_path = self.image_paths[idx]
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"File not found: {img_path}")
                
                # Open the image file directly with PIL first
                try:
                    pil_img = Image.open(img_path).convert('RGB')  # Always load as RGB for the model
                    img = np.array(pil_img)
                except Exception as e_pil:
                    # Fallback to OpenCV if PIL fails
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"OpenCV couldn't load image: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] for augmentation
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
        
            # Apply transforms
            if self.transform:
                    # Apply transform directly (images are already in correct format)
                transformed = self.transform(image=img)
                img = transformed["image"]
            
            return img, torch.tensor(label, dtype=torch.long) 
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder tensor
            img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return img, torch.tensor(label, dtype=torch.long)
    
    def _print_class_distribution(self):
        """Print class distribution in this dataset."""
        if not hasattr(self, 'labels'):
            return
        
        label_counts = {}
        for l in self.labels:
            if l not in label_counts:
                label_counts[l] = 0
            label_counts[l] += 1
        
        print(f"  Class distribution in {self.mode} dataset:")
        total = len(self.labels)
        for cls_idx, cls_name in enumerate(self.classes):
            count = label_counts.get(cls_idx, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"    {cls_name}: {count} samples ({percentage:.1f}%)")

# Keep original FER2013Dataset for backward compatibility
class FER2013Dataset(BaseFER2013Dataset):
    """Legacy class for backward compatibility."""
    pass 