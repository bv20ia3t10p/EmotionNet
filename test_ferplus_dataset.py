#!/usr/bin/env python3
"""
Simple test script to verify FERPlus dataset loading
"""

import pandas as pd
import os
from PIL import Image
import numpy as np

def test_ferplus_structure():
    """Test if FERPlus dataset is properly structured"""
    
    print("🔍 Testing FERPlus Dataset Structure")
    print("=" * 50)
    
    # Check if main CSV exists
    csv_file = 'FERPlus-master/fer2013new.csv'
    if not os.path.exists(csv_file):
        print(f"❌ Main CSV not found: {csv_file}")
        return False
    
    # Load CSV and check structure
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ CSV loaded successfully")
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check splits
        usage_counts = df['Usage'].value_counts()
        print(f"   Split distribution:")
        for split, count in usage_counts.items():
            print(f"     {split}: {count}")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Test loading a few images
    print(f"\n📸 Testing Image Loading")
    train_df = df[df['Usage'] == 'Training'].head(5)
    
    for idx, row in train_df.iterrows():
        img_path = os.path.join('FERPlus-master', 'data', 'Training', row['Image name'])
        
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                print(f"✅ Loaded {row['Image name']}: {image.size} {image.mode}")
                
                # Check emotion labels
                emotions = ['neutral', 'happiness', 'surprise', 'sadness', 
                           'anger', 'disgust', 'fear', 'contempt']
                vote_counts = [row[emotion] for emotion in emotions]
                dominant_emotion = np.argmax(vote_counts)
                print(f"   Emotion: {emotions[dominant_emotion]} (votes: {vote_counts})")
                
            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")
                return False
        else:
            print(f"❌ Image not found: {img_path}")
            return False
    
    print(f"\n✅ FERPlus dataset structure verified!")
    return True

if __name__ == "__main__":
    test_ferplus_structure() 