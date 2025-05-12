"""Data Manager specific to FER2013 dataset."""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

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
        print(f"Initializing FER2013DataManager with data_dir: {data_dir}")

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
        import tempfile
        import shutil
        from PIL import Image
        import os
        
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
        
        # Process CSVs based on what's available
        if os.path.exists(full_csv):
            print(f"Processing full dataset from {full_csv}")
            full_df = pd.read_csv(full_csv)
            full_df.columns = full_df.columns.str.strip()
            
            # Split by Usage column if it exists - use all Training for train, PublicTest for validation
            if 'Usage' in full_df.columns:
                train_df = full_df[full_df['Usage'] == 'Training'].copy()
                test_df = full_df[full_df['Usage'] == 'PublicTest'].copy()
                print(f"Split by Usage: {len(train_df)} training, {len(test_df)} test/validation samples")
            else:
                # No Usage column - manual split 80/20
                total_samples = len(full_df)
                train_size = int(total_samples * 0.8)
                
                train_df = full_df.iloc[:train_size].copy()
                test_df = full_df.iloc[train_size:].copy()
                print(f"Manual split: {len(train_df)} training, {len(test_df)} test/validation samples")
        else:
            # Use separate train.csv and test.csv files
            print(f"Using separate CSV files")
            
            if os.path.exists(train_csv):
                print(f"Loading training data from {train_csv}")
                train_df = pd.read_csv(train_csv)
                train_df.columns = train_df.columns.str.strip()
                print(f"Loaded {len(train_df)} training samples from {train_csv}")
            
            if os.path.exists(test_csv):
                print(f"Loading test data from {test_csv}")
                test_df = pd.read_csv(test_csv)
                test_df.columns = test_df.columns.str.strip()
                print(f"Loaded {len(test_df)} test/validation samples from {test_csv}")
        
        # Error check - if we don't have training data, can't proceed
        if train_df is None or len(train_df) == 0:
            print(f"Error: Could not load sufficient training data from CSV files")
            return None, None, None, []
        
        # Process training data - save images to disk
        if train_df is not None and len(train_df) > 0:
            print(f"Saving {len(train_df)} training images...")
            self._csv_to_images(train_df, train_dir)
        
        # Process test data (to be used as validation)
        if test_df is not None and len(test_df) > 0:
            print(f"Saving {len(test_df)} test/validation images...")
            self._csv_to_images(test_df, test_dir)
        
        # Now we'll load datasets from the saved image files, using absolute paths
        train_paths = []
        train_labels = []
        val_paths = []  # This will be loaded from test_dir (no splitting)
        val_labels = []
        
        # Get training images - use all of them for training (no splitting)
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
        
        # Get validation images from test directory
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
        
        # Create datasets using BaseEmotionDataset for file paths
        train_dataset = None
        val_dataset = None
        
        if train_paths:
            train_dataset = BaseEmotionDataset(
                train_paths, train_labels, 
                self.all_classes, mode='train', 
                image_size=self.image_size,
                dataset_name='fer2013'
            )
            
        if val_paths:
            val_dataset = BaseEmotionDataset(
                val_paths, val_labels, 
                self.all_classes, mode='val', 
                image_size=self.image_size,
                dataset_name='fer2013'
            )
            
        # Print summary
        print("-- FER2013DataManager: Dataset Loading Summary --")
        print(f"Train: {len(train_dataset) if train_dataset else 0} samples")
        print(f"Val:   {len(val_dataset) if val_dataset else 0} samples")
        print(f"Class names: {self.all_classes}")
        print(f"Training labels for sampler: {len(train_labels)} samples")
        print("-------------------------------------------------")
            
        # Print summary of created files by directory
        print(f"Created directory structure with images:")
        for emotion in self.all_classes:
            train_emotion_dir = os.path.join(train_dir, emotion)
            test_emotion_dir = os.path.join(test_dir, emotion)
            
            train_count = 0
            if os.path.exists(train_emotion_dir):
                train_count = len([f for f in os.listdir(train_emotion_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            val_count = 0
            if os.path.exists(test_emotion_dir):
                val_count = len([f for f in os.listdir(test_emotion_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
            print(f"  {emotion}: {train_count} training, {val_count} validation images")
        
        # Validate that images can be loaded
        print("Validating that images can be loaded...")
        if train_paths:
            for i in range(min(5, len(train_paths))):
                path = train_paths[i]
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            pil_img = Image.open(f).convert('RGB')
                            print(f"  Successfully loaded image {i}: {path}")
                    except Exception as e:
                        print(f"  Error loading image {i}: {path} - {e}")
                else:
                    print(f"  Image path {i} does not exist: {path}")
        
        # Store directories for future reference
        self.train_dir = train_dir
        self.test_dir = test_dir
        
        # Return: train_dataset, val_dataset (from test dir), None (no separate test), train_labels
        return train_dataset, val_dataset, None, train_labels
    
    def _csv_to_images(self, df, output_dir):
        """Extract images from DataFrame and save to output directory."""
        from PIL import Image
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
                
                # Parse pixels
                pixel_str = row[pixel_col]
                if isinstance(pixel_str, str):
                    if ' ' in pixel_str:
                        pixels = np.array([int(p) for p in pixel_str.split()], dtype=np.uint8)
                    else:
                        pixels = np.array([int(p) for p in pixel_str.split(',')], dtype=np.uint8)
                else:
                    pixels = np.array(pixel_str, dtype=np.uint8)
                
                # Check pixel count
                expected_size = 48 * 48
                if len(pixels) != expected_size:
                    if len(pixels) > expected_size:
                        pixels = pixels[:expected_size]
                    else:
                        continue  # Skip if not enough pixels
                
                # Reshape and create image
                img = pixels.reshape(48, 48)
                pil_img = Image.fromarray(img)
                
                # Save to appropriate directory
                emotion_dir = os.path.join(output_dir, emotion_name)
                os.makedirs(emotion_dir, exist_ok=True)
                img_path = os.path.join(emotion_dir, f"{idx}.png")
                
                # Ensure the image path is absolute
                img_path = os.path.abspath(img_path)
                
                pil_img.save(img_path)
                
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
            import shutil
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

class FER2013Dataset(BaseEmotionDataset):
    """Dataset for FER2013 data loaded either from memory or files."""
    def __init__(self, images, labels, classes, mode='train', image_size=224, dataset_name='fer2013'):
        # Store the paths/images and labels
        self.classes = classes
        self.mode = mode
        self.image_size = image_size
        self.dataset_name = dataset_name
        
        # Determine if we're dealing with PIL images or file paths
        self.is_in_memory = isinstance(images[0], Image.Image) if images else False
        
        if self.is_in_memory:
            # Store the PIL images directly
            self.images = images
            self.labels = labels
            print(f"  FER2013Dataset: Using in-memory PIL Images ({len(images)} images)")
        else:
            # We're dealing with file paths
            self.image_paths = images
            self.labels = labels
            print(f"  FER2013Dataset: Using file paths ({len(images)} paths)")
            
            # Validate file paths exist (check first 10)
            for i, path in enumerate(self.image_paths[:10]):
                if not os.path.exists(path):
                    print(f"  Warning: File path does not exist: {path}")
                else:
                    print(f"  Path exists: {path}")
        
        # Get transforms
        from .augmentations import get_transforms
        self.transform = get_transforms(mode, image_size, dataset_name)
        
        print(f"  FER2013Dataset created for mode '{mode}' with {len(self)} samples.")
        
        # Print class distribution
        if len(self) > 0:
            self._print_class_distribution()
    
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
                    pil_img = Image.open(img_path).convert('RGB')
                    img = np.array(pil_img)
                except Exception as e_pil:
                    # Fallback to OpenCV if PIL fails
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"OpenCV couldn't load image: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ensure grayscale is converted to RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
            
            return img, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder tensor
            img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return img, torch.tensor(label, dtype=torch.long) 