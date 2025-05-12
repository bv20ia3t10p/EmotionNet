"""Data parsing functions for different emotion datasets."""

import os
from emotion_net.config.constants import EMOTIONS # FER2013 specific

# Define RAF-DB specific constants (assuming 7 basic emotions)
# Note: These could also live in constants.py if preferred
RAFDB_EMOTIONS_MAP = {
    1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness',
    5: 'Sadness', 6: 'Anger', 7: 'Neutral'
}
RAFDB_LABEL_MAP = {i: i - 1 for i in range(1, 8)} # Map 1-7 to 0-6

def parse_fer2013(data_dir):
    """Parses FER2013 data organized in subdirectories by emotion name.
    
    Args:
        data_dir (str): Path to the directory containing emotion subdirectories.

    Returns:
        tuple: (list_of_image_paths, list_of_labels)
    """
    paths = []
    labels = []
    print(f"Parsing FER2013 data from: {data_dir}")
    
    # Ensure data_dir is an absolute path
    data_dir = os.path.abspath(data_dir)
    
    # Use the global FER2013 EMOTIONS map {0: 'angry', ...}
    for emotion_idx, emotion_name in EMOTIONS.items():
        emotion_dir = os.path.join(data_dir, emotion_name)
        if not os.path.exists(emotion_dir):
            print(f"Warning: FER2013 directory {emotion_dir} not found.")
            continue
        count = 0
        for img_file in os.listdir(emotion_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(emotion_dir, img_file)
                
                # Ensure path is absolute
                img_path = os.path.abspath(img_path)
                
                # Verify file exists
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}")
                    continue
                    
                paths.append(img_path)
                labels.append(emotion_idx) # Use index 0-6
                count += 1
        print(f"  Found {count} images for class '{emotion_name}'")
    if not paths:
        print(f"Error: No images found in {data_dir} or its subdirectories following FER2013 structure.")
    return paths, labels

def parse_rafdb(rafdb_base_dir, mode):
    """Parses RAF-DB basic dataset annotations.

    Args:
        rafdb_base_dir (str): Path to the RAF-DB 'basic' directory.
        mode (str): 'train' or 'test'.

    Returns:
        tuple: (list_of_image_paths, list_of_labels)
    """
    paths = []
    labels = []
    image_dir = os.path.join(rafdb_base_dir, 'Image', 'original')
    annotation_path = os.path.join(rafdb_base_dir, 'Annotation', 'list_patition_label.txt')
    print(f"Parsing RAF-DB data for mode '{mode}' from: {annotation_path}")

    if not os.path.exists(image_dir):
         print(f"Error: RAF-DB image directory not found: {image_dir}")
         return paths, labels
    
    try:
        count = 0
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                image_name = parts[0]
                label = int(parts[1]) # 1-7 index

                # Check if image belongs to the current mode
                image_prefix = image_name.split('_')[0] # train or test
                if image_prefix == mode:
                    img_path = os.path.join(image_dir, image_name)
                    if os.path.exists(img_path):
                        paths.append(img_path)
                        labels.append(RAFDB_LABEL_MAP[label]) # Map to 0-6 index
                        count += 1
                    # else:
                    #      print(f"Warning: RAF-DB image not found: {img_path}") # Can be noisy
        print(f"  Found {count} images for mode '{mode}'")

    except FileNotFoundError:
        print(f"Error: RAF-DB annotation file not found at {annotation_path}")
    except Exception as e:
        print(f"Error reading RAF-DB annotations: {e}")

    if not paths:
        print(f"Error: No {mode} images/labels found based on {annotation_path}")
    return paths, labels 