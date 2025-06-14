import os
import pandas as pd
import shutil
from PIL import Image
import numpy as np

def preprocess_ferplus(data_dir, use_soft_labels=True):
    """
    Preprocess FERPlus dataset into the required format.
    
    Args:
        data_dir: Path to the FERPlus dataset root directory containing fer2013new.csv
        use_soft_labels: If True, use probability distribution from votes. If False, use majority vote.
    """
    # Create required directories
    os.makedirs(os.path.join(data_dir, 'processed', 'images'), exist_ok=True)
    
    # Read the original annotation file
    try:
        # First try reading fer2013new.csv (FERPlus format)
        df = pd.read_csv(os.path.join(data_dir, 'fer2013new.csv'))
        print("Found FERPlus format data")
        
        emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        
        if use_soft_labels:
            # Use probability distribution from votes
            print("Using soft labels (probability distribution)")
            # Normalize vote counts to probabilities
            vote_sums = df[emotion_columns].sum(axis=1)
            for col in emotion_columns:
                df[f'{col}_prob'] = df[col] / vote_sums
            
            # Also keep the hard label for evaluation
            df['emotion'] = df[emotion_columns].idxmax(axis=1)
            df['label'] = pd.Categorical(df['emotion'], categories=emotion_columns).codes
        else:
            # Use majority vote (original implementation)
            print("Using hard labels (majority vote)")
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
            use_soft_labels = False  # FER2013 doesn't have vote data
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
        if use_soft_labels and 'neutral_prob' in split_df.columns:
            # Include both hard labels and soft label probabilities
            prob_columns = [f'{col}_prob' for col in emotion_columns]
            output_df = split_df[['image', 'label'] + prob_columns]
        else:
            output_df = split_df[['image', 'label']]
            
        output_path = os.path.join(data_dir, 'processed', f'{split}_ferplus.csv')
        output_df.to_csv(output_path, index=False)
        print(f"Saved {split} split to {output_path}")
    
    print(f"FERPlus dataset preprocessing completed with {'soft' if use_soft_labels else 'hard'} labels!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./FERPlus', help='Path to FERPlus dataset root directory')
    parser.add_argument('--use_soft_labels', action='store_true', default=True, help='Use probability distribution from votes (default)')
    parser.add_argument('--use_hard_labels', dest='use_soft_labels', action='store_false', help='Use majority vote')
    args = parser.parse_args()
    
    preprocess_ferplus(args.data_dir, args.use_soft_labels) 