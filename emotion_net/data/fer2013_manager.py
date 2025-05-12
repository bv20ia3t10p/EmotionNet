"""Data Manager specific to FER2013 dataset."""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .parsers import parse_fer2013
from .dataset import BaseEmotionDataset
from emotion_net.config.constants import EMOTIONS

class FER2013DataManager:
    """Handles loading, parsing, and splitting of FER2013 data."""
    def __init__(self, data_dir, test_dir=None, image_size=224, val_split_ratio=0.1, seed=42):
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.all_classes = list(EMOTIONS.values())
        print(f"Initializing FER2013DataManager with train_dir: {data_dir}, test_dir: {test_dir}")

    def get_datasets(self):
        """Loads, splits, and returns train, val, test datasets and train labels.
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, train_labels_list)
                   Datasets are instances of BaseEmotionDataset (or None).
        """
        train_dataset, val_dataset, test_dataset = None, None, None
        train_labels_list = []

        # Check if we're dealing with directory structure or CSV files
        if self._is_csv_dataset():
            return self._load_from_csv()
        else:
            return self._load_from_directory()
    
    def _is_csv_dataset(self):
        """Check if we're dealing with CSV files instead of directory structure."""
        # Look for standard CSV filenames
        train_csv = os.path.join(self.data_dir, 'train.csv')
        test_csv = os.path.join(self.data_dir, 'test.csv')
        full_csv = os.path.join(self.data_dir, 'icml_face_data.csv')
        
        return os.path.exists(train_csv) or os.path.exists(full_csv)
    
    def _load_from_csv(self):
        """Load FER2013 dataset from CSV files."""
        print("Loading FER2013 dataset from CSV files...")
        
        # Try to load train.csv or fall back to the full dataset
        train_csv = os.path.join(self.data_dir, 'train.csv')
        test_csv = os.path.join(self.data_dir, 'test.csv')
        full_csv = os.path.join(self.data_dir, 'icml_face_data.csv')
        
        train_df = None
        test_df = None
        
        # Load training data
        if os.path.exists(train_csv):
            print(f"Loading training data from {train_csv}")
            train_df = pd.read_csv(train_csv)
        elif os.path.exists(full_csv):
            print(f"Loading from full dataset {full_csv}")
            full_df = pd.read_csv(full_csv)
            # Filter by Usage column if it exists
            if 'Usage' in full_df.columns:
                train_df = full_df[full_df['Usage'] == 'Training']
                test_df = full_df[full_df['Usage'] == 'PublicTest']
            else:
                # If no Usage column, use a fixed split
                print("No Usage column found, using fixed train/test split")
                train_size = int(len(full_df) * 0.8)
                train_df = full_df.iloc[:train_size]
                test_df = full_df.iloc[train_size:]
        
        if train_df is None:
            print("Error: Could not find training data CSV")
            return None, None, None, []
        
        # Load test data if not already loaded
        if test_df is None and os.path.exists(test_csv):
            print(f"Loading test data from {test_csv}")
            test_df = pd.read_csv(test_csv)
        
        # Process the dataframes into datasets
        train_dataset, val_dataset, train_labels_list = self._process_train_df(train_df)
        
        # Process test data if available
        test_dataset = None
        if test_df is not None:
            test_dataset = self._process_test_df(test_df)
        
        # Print summary
        print("-- FER2013DataManager: Dataset Loading Summary --")
        print(f"Train: {len(train_dataset) if train_dataset else 0} samples")
        print(f"Val:   {len(val_dataset) if val_dataset else 0} samples")
        print(f"Test:  {len(test_dataset) if test_dataset else 0} samples")
        print(f"Class names: {self.all_classes}")
        print(f"Training labels for sampler: {len(train_labels_list)} samples")
        print("-------------------------------------------------")
        
        return train_dataset, val_dataset, test_dataset, train_labels_list
    
    def _process_train_df(self, df):
        """Process training dataframe into datasets."""
        from PIL import Image
        import io
        import base64
        
        # Extract features and labels
        X = []
        y = []
        
        if 'pixels' in df.columns:
            # Standard FER2013 format with pixels column
            for _, row in df.iterrows():
                try:
                    # Extract pixel values and convert to numpy array
                    pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                    # Reshape to 48x48 image
                    img = pixels.reshape(48, 48)
                    # Convert to PIL Image
                    img = Image.fromarray(img)
                    # Store the image
                    X.append(img)
                    # Get emotion label
                    if 'emotion' in df.columns:
                        y.append(int(row['emotion']))
                    else:
                        # Default to neutral if no emotion column
                        y.append(6)  # 6 = neutral in FER2013
                except Exception as e:
                    print(f"Error processing row: {e}")
        else:
            print("Error: Could not find 'pixels' column in CSV")
            return None, None, []
        
        # Convert to numpy arrays
        y = np.array(y)
        
        # Split into train and validation sets
        if self.val_split_ratio > 0 and len(X) > 0:
            try:
                # Split indices
                indices = np.arange(len(X))
                train_indices, val_indices = train_test_split(
                    indices, test_size=self.val_split_ratio, random_state=self.seed, stratify=y
                )
                
                # Create datasets
                train_images = [X[i] for i in train_indices]
                train_labels = [y[i] for i in train_indices]
                val_images = [X[i] for i in val_indices]
                val_labels = [y[i] for i in val_indices]
                
                # Create PyTorch datasets
                train_dataset = FER2013Dataset(train_images, train_labels, self.all_classes, mode='train', image_size=self.image_size)
                val_dataset = FER2013Dataset(val_images, val_labels, self.all_classes, mode='val', image_size=self.image_size)
                
                return train_dataset, val_dataset, train_labels
            except Exception as e:
                print(f"Error splitting dataset: {e}")
                return None, None, []
        else:
            # Use all data for training
            train_dataset = FER2013Dataset(X, y, self.all_classes, mode='train', image_size=self.image_size)
            return train_dataset, None, y.tolist()
    
    def _process_test_df(self, df):
        """Process test dataframe into dataset."""
        # Similar to _process_train_df but for test data
        from PIL import Image
        
        # Extract features and labels
        X = []
        y = []
        
        if 'pixels' in df.columns:
            for _, row in df.iterrows():
                try:
                    pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                    img = pixels.reshape(48, 48)
                    img = Image.fromarray(img)
                    X.append(img)
                    if 'emotion' in df.columns:
                        y.append(int(row['emotion']))
                    else:
                        y.append(6)  # Neutral
                except Exception as e:
                    print(f"Error processing test row: {e}")
        else:
            print("Error: Could not find 'pixels' column in test CSV")
            return None
        
        # Create test dataset
        test_dataset = FER2013Dataset(X, y, self.all_classes, mode='test', image_size=self.image_size)
        return test_dataset
        
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
            train_dataset = BaseEmotionDataset(train_paths, train_labels_list, self.all_classes, mode='train', image_size=self.image_size)

            # Create val dataset
            val_paths = [full_train_paths[i] for i in val_indices]
            val_labels = [full_train_labels[i] for i in val_indices]
            val_dataset = BaseEmotionDataset(val_paths, val_labels, self.all_classes, mode='val', image_size=self.image_size)
        else:
             print("Validation split ratio is <= 0 or >= 1, using full training set for training.")
             train_dataset = BaseEmotionDataset(full_train_paths, full_train_labels, self.all_classes, mode='train', image_size=self.image_size)
             train_labels_list = full_train_labels
             val_dataset = None # No validation set

        # --- Test Data ---
        # Infer test dir relative to train_dir if not provided explicitly
        potential_test_dir = os.path.join(os.path.dirname(os.path.abspath(self.data_dir)), 'test')
        test_data_dir = self.test_dir if self.test_dir else potential_test_dir

        if os.path.exists(test_data_dir):
            test_paths, test_labels = parse_fer2013(test_data_dir)
            if test_paths:
                test_dataset = BaseEmotionDataset(test_paths, test_labels, self.all_classes, mode='test', image_size=self.image_size)
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


class FER2013Dataset(BaseEmotionDataset):
    """Dataset for FER2013 data loaded directly from memory (not files)."""
    def __init__(self, images, labels, classes, mode='train', image_size=224, dataset_name='fer2013'):
        # Skip the parent __init__ but initialize the same attributes
        self.classes = classes
        self.mode = mode
        self.image_size = image_size
        self.dataset_name = dataset_name
        
        # Store the PIL images directly
        self.images = images
        self.labels = labels
        
        # Get transforms
        from .augmentations import get_transforms
        self.transform = get_transforms(mode, image_size, dataset_name)
        
        print(f"  FER2013Dataset created for mode '{mode}' with {len(self)} samples.")
        
        # Print class distribution
        if len(self) > 0:
            self._print_class_distribution()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and label
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert PIL image to numpy array
        img = np.array(img)
        
        # Make sure it's grayscale -> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        
        return img, torch.tensor(label, dtype=torch.long) 