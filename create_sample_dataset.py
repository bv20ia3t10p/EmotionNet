import pandas as pd
import numpy as np
import os
from PIL import Image

def create_sample_fer_dataset():
    """Create a sample FER dataset for demonstration"""
    
    print("Creating sample FER dataset...")
    
    # Create directories
    os.makedirs('sample_dataset/images/Training', exist_ok=True)
    os.makedirs('sample_dataset/images/PublicTest', exist_ok=True)
    
    # Emotion labels
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    
    # Generate sample data
    train_data = []
    test_data = []
    
    # Training data (200 samples total, 25 per class)
    for emotion_id, emotion_name in enumerate(emotions):
        for i in range(25):
            # Generate random grayscale image (48x48 pixels)
            pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
            
            # Create PIL image and save
            img = Image.fromarray(pixels, mode='L')
            img_filename = f"{emotion_name}_{i:03d}.png"
            img_path = f"sample_dataset/images/Training/{img_filename}"
            img.save(img_path)
            
            # Add to CSV data
            train_data.append({
                'emotion': emotion_id,
                'image': f"Training/{img_filename}",
                'Usage': 'Training'
            })
    
    # Test data (80 samples total, 10 per class)
    for emotion_id, emotion_name in enumerate(emotions):
        for i in range(10):
            # Generate random grayscale image
            pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
            
            # Create PIL image and save
            img = Image.fromarray(pixels, mode='L')
            img_filename = f"{emotion_name}_test_{i:03d}.png"
            img_path = f"sample_dataset/images/PublicTest/{img_filename}"
            img.save(img_path)
            
            # Add to CSV data
            test_data.append({
                'emotion': emotion_id,
                'image': f"PublicTest/{img_filename}",
                'Usage': 'PublicTest'
            })
    
    # Combine and shuffle
    all_data = train_data + test_data
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    # Save CSV
    df.to_csv('sample_dataset/fer_sample.csv', index=False)
    
    print(f"Sample dataset created:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Classes: {len(emotions)}")
    print(f"  CSV file: sample_dataset/fer_sample.csv")
    print(f"  Images directory: sample_dataset/images/")

if __name__ == "__main__":
    create_sample_fer_dataset() 