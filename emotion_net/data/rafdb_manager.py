"""Data Manager specific to RAF-DB dataset."""

import os

from .parsers import parse_rafdb, RAFDB_EMOTIONS_MAP
from .dataset import BaseEmotionDataset

class RAFDBDataManager:
    """Handles loading and parsing of RAF-DB data."""
    def __init__(self, data_dir, test_dir=None, image_size=224):
        # test_dir is ignored for RAF-DB standard protocol
        self.data_dir = data_dir
        self.image_size = image_size
        self.all_classes = list(RAFDB_EMOTIONS_MAP.values())
        print(f"Initializing RAFDBDataManager with data_dir: {data_dir}")

    def get_datasets(self):
        """Loads train and test (used as val) datasets and train labels.
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, train_labels_list)
                   val_dataset is the official RAF-DB test set.
                   test_dataset is typically None for RAF-DB standard protocol.
                   Datasets are instances of BaseEmotionDataset (or None).
        """
        train_dataset, val_dataset, test_dataset = None, None, None
        train_labels_list = []
        
        # Try to find RAF-DB dataset structure
        # First check original expected path
        rafdb_base_dir = os.path.join(self.data_dir, 'basic')
        
        # If not found, check for DATASET structure in our repository
        if not os.path.exists(rafdb_base_dir):
            dataset_dir = os.path.join(self.data_dir, 'DATASET')
            if os.path.exists(dataset_dir):
                print(f"Found RAF-DB in DATASET directory structure at {dataset_dir}")
                # Use custom parser for the DATASET format
                return self._load_from_dataset_structure(dataset_dir)
            else:
                print(f"Warning: RAF-DB base directory expected at '{rafdb_base_dir}' but not found.")
                print(f"DATASET directory not found at '{dataset_dir}' either.")
                print(f"Trying '{self.data_dir}' directly.")
                rafdb_base_dir = self.data_dir  # Fallback

        # Original parsing logic for standard RAF-DB format
        train_paths, train_labels_list = parse_rafdb(rafdb_base_dir, mode='train')
        if train_paths:
            train_dataset = BaseEmotionDataset(
                train_paths, 
                train_labels_list, 
                self.all_classes, 
                mode='train', 
                image_size=self.image_size,
                dataset_name='rafdb'
            )
        else:  # Handle parsing failure
            print(f"Error: Failed to parse training data from {rafdb_base_dir}. Cannot proceed.")
            return None, None, None, []

        # Parse test data (used as validation set)
        val_paths, val_labels = parse_rafdb(rafdb_base_dir, mode='test')
        if val_paths:
            # Use 'val' mode for transforms, although currently same as 'test'
            val_dataset = BaseEmotionDataset(
                val_paths, 
                val_labels, 
                self.all_classes, 
                mode='val', 
                image_size=self.image_size,
                dataset_name='rafdb'
            )

        # No separate final test set in standard RAF-DB protocol
        test_dataset = None

        # --- Final Summary & Return ---
        print("-- RAFDBDataManager: Dataset Loading Summary --")
        print(f"Train: {len(train_dataset) if train_dataset else 0} samples")
        print(f"Val (Official Test Split): {len(val_dataset) if val_dataset else 0} samples")
        print(f"Test:  {len(test_dataset) if test_dataset else 0} samples (Expected None)")
        print(f"Class names: {self.all_classes}")
        print(f"Training labels for sampler: {len(train_labels_list)} samples")
        print("--------------------------------------------")
        
        # Ensure train_labels_list corresponds to the actual train_dataset created
        if train_dataset and len(train_labels_list) != len(train_dataset.labels):
            print(f"Warning: Mismatch between train_labels_list ({len(train_labels_list)}) and train_dataset labels ({len(train_dataset.labels)}). Re-assigning labels for sampler.")
            train_labels_list = train_dataset.labels # Use labels directly from created dataset

        return train_dataset, val_dataset, test_dataset, train_labels_list
    
    def _load_from_dataset_structure(self, dataset_dir):
        """Load RAF-DB data from the DATASET directory structure used in this repo.
        
        Args:
            dataset_dir: Path to the DATASET directory containing train/ and test/
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, train_labels_list)
        """
        train_paths = []
        train_labels = []
        val_paths = []
        val_labels = []
        
        # Process train directory (contains subdirectories for each class)
        train_dir = os.path.join(dataset_dir, 'train')
        if os.path.exists(train_dir):
            print(f"Loading RAF-DB training data from {train_dir}")
            # Each subdirectory is a class (1-7)
            for class_idx in range(1, 8):  # RAF-DB uses 1-7 for class indices
                class_dir = os.path.join(train_dir, str(class_idx))
                if not os.path.exists(class_dir):
                    print(f"Warning: Class directory {class_dir} not found.")
                    continue
                    
                count = 0
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_file)
                        train_paths.append(img_path)
                        # Convert from 1-based to 0-based indices
                        train_labels.append(class_idx - 1)
                        count += 1
                        
                print(f"  Found {count} images for class '{class_idx}' ({RAFDB_EMOTIONS_MAP[class_idx]})")
        
        # Process test directory
        test_dir = os.path.join(dataset_dir, 'test')
        if os.path.exists(test_dir):
            print(f"Loading RAF-DB test data from {test_dir}")
            # Each subdirectory is a class (1-7)
            for class_idx in range(1, 8):  # RAF-DB uses 1-7 for class indices
                class_dir = os.path.join(test_dir, str(class_idx))
                if not os.path.exists(class_dir):
                    print(f"Warning: Class directory {class_dir} not found.")
                    continue
                    
                count = 0
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_file)
                        val_paths.append(img_path)
                        # Convert from 1-based to 0-based indices
                        val_labels.append(class_idx - 1)
                        count += 1
                        
                print(f"  Found {count} test images for class '{class_idx}' ({RAFDB_EMOTIONS_MAP[class_idx]})")
        
        # Create datasets
        train_dataset = None
        val_dataset = None
        test_dataset = None  # No separate test set in standard RAF-DB protocol
        
        if train_paths:
            train_dataset = BaseEmotionDataset(
                train_paths, 
                train_labels, 
                self.all_classes, 
                mode='train', 
                image_size=self.image_size,
                dataset_name='rafdb'
            )
        
        if val_paths:
            val_dataset = BaseEmotionDataset(
                val_paths, 
                val_labels, 
                self.all_classes, 
                mode='val', 
                image_size=self.image_size,
                dataset_name='rafdb'
            )
        
        # Summary
        print("-- RAFDBDataManager: Dataset Loading Summary --")
        print(f"Train: {len(train_dataset) if train_dataset else 0} samples")
        print(f"Val (Official Test Split): {len(val_dataset) if val_dataset else 0} samples")
        print(f"Test: None (Expected for RAF-DB)")
        print(f"Class names: {self.all_classes}")
        print(f"Training labels for sampler: {len(train_labels)} samples")
        print("--------------------------------------------")
        
        return train_dataset, val_dataset, test_dataset, train_labels 