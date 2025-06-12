import os
import pandas as pd
import shutil
from PIL import Image
import numpy as np

def preprocess_ferplus(data_dir):
    """
    Preprocess FERPlus dataset into the required format.
    
    Args:
        data_dir: Path to the FERPlus dataset root directory containing fer2013new.csv
    """
    # Create required directories
    os.makedirs(os.path.join(data_dir, 'processed', 'images'), exist_ok=True)
    
    # Read the original annotation file
    try:
        # First try reading fer2013new.csv (FERPlus format)
        df = pd.read_csv(os.path.join(data_dir, 'fer2013new.csv'))
        print("Found FERPlus format data")
        
        # Convert voting to majority label
        emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        df['emotion'] = df[emotion_columns].idxmax(axis=1)
        df['label'] = pd.Categorical(df['emotion'], categories=emotion_columns).codes
        
        # Get pixels from original FER2013 file
        fer2013_df = pd.read_csv(os.path.join(os.path.dirname(data_dir), 'fer2013.csv'))
        df['pixels'] = fer2013_df['pixels']
        
    except (FileNotFoundError, KeyError):
        print("Could not find or process FERPlus format, trying FER2013 format...")
        try:
            # Try reading fer2013.csv (original FER2013 format)
            df = pd.read_csv(os.path.join(os.path.dirname(data_dir), 'fer2013.csv'))
            print("Found FER2013 format data")
            df['label'] = df['emotion']  # FER2013 already has numeric labels
        except FileNotFoundError:
            raise FileNotFoundError("Could not find either fer2013new.csv or fer2013.csv")
    
    # Split data based on Usage
    train_df = df[df['Usage'] == 'Training'].copy()
    val_df = df[df['Usage'] == 'PublicTest'].copy()
    test_df = df[df['Usage'] == 'PrivateTest'].copy()
    
    # Process images and create new CSVs
    for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"Processing {split} split...")
        # Create image files
        for idx, row in split_df.iterrows():
            pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(pixels)
            img_path = f'{split}_{idx:05d}.png'
            img.save(os.path.join(data_dir, 'processed', 'images', img_path))
            split_df.loc[idx, 'image'] = img_path
        
        # Save CSV with image paths and labels
        output_df = split_df[['image', 'label']]
        output_path = os.path.join(data_dir, 'processed', f'{split}_ferplus.csv')
        output_df.to_csv(output_path, index=False)
        print(f"Saved {split} split to {output_path}")
    
    print("FERPlus dataset preprocessing completed!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to FERPlus dataset root directory')
    args = parser.parse_args()
    
    preprocess_ferplus(args.data_dir) 