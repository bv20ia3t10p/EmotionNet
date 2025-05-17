"""
Simplified face preprocessing for FER2013 dataset without dlib dependency.
This uses OpenCV's built-in face detection which should be easier to install.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import glob

def detect_and_crop_face(image, face_detector, desired_size=224):
    """Detect face and crop it to focus on facial features"""
    # Convert to grayscale if needed
    if isinstance(image, np.ndarray):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    else:
        return None
    
    # Ensure we have uint8 image for face detection
    img_gray = img_gray if img_gray.dtype == np.uint8 else (img_gray * 255).astype(np.uint8)
    
    # Detect faces using OpenCV's built-in Haar Cascade
    faces = face_detector.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        # No face found, return resized original image
        return cv2.resize(image, (desired_size, desired_size))
    
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add padding (20%)
    padding = int(0.2 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    # Crop face with padding
    face_img = image[y1:y2, x1:x2]
    
    # Apply contrast and brightness normalization
    if len(face_img.shape) == 3:
        # Color image
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_img = clahe.apply(face_img)
    
    # Resize to desired size
    return cv2.resize(face_img, (desired_size, desired_size))

def process_directory(input_dir, output_dir, desired_size=224):
    """Process all images in a directory and save cropped faces"""
    # Load OpenCV face detector
    print("Loading face detection model...")
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(face_cascade_path):
        print(f"ERROR: OpenCV face cascade not found at {face_cascade_path}")
        # Attempt to find it in alternative locations
        alt_paths = [
            'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                face_cascade_path = path
                print(f"Found cascade file at: {path}")
                break
        else:
            print("Could not find face cascade file. Please download it manually.")
            return False
    
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_types = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for img_type in image_types:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', img_type), recursive=True))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files):
        # Get relative path to preserve directory structure
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                failed += 1
                continue
                
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process face
            processed = detect_and_crop_face(img_rgb, face_detector, desired_size)
            
            if processed is None:
                print(f"Warning: Could not process {img_path}")
                # Copy original image
                cv2.imwrite(out_path, img)
                failed += 1
                continue
            
            # Save the processed face
            cv2.imwrite(out_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            successful += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed += 1
    
    print(f"Processed {successful} images successfully, {failed} images failed")
    return True

def main():
    parser = argparse.ArgumentParser(description='Process faces in dataset without dlib')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed faces')
    parser.add_argument('--size', type=int, default=224, help='Desired output size (default: 224)')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.size)

if __name__ == "__main__":
    main() 