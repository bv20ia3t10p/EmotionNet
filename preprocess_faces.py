"""
Face alignment preprocessing for FER2013 dataset.
This significantly improves accuracy by ensuring facial features are in consistent positions.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import glob
from PIL import Image
import dlib

def align_face(image, face_detector, landmark_predictor, desired_size=224):
    """Detect face and align it based on facial landmarks"""
    # Detect faces
    if isinstance(image, np.ndarray):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    else:
        return None
    
    # Convert image to dlib format if needed
    img_dlib = img_gray if img_gray.dtype == np.uint8 else (img_gray * 255).astype(np.uint8)
    
    # Detect faces
    faces = face_detector(img_dlib, 1)
    if len(faces) == 0:
        # No face found, return original image
        return cv2.resize(image, (desired_size, desired_size))
    
    # Get the largest face
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Get facial landmarks
    landmarks = landmark_predictor(img_dlib, face)
    
    # Convert landmarks to numpy array
    landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # Get eyes coordinates
    left_eye = landmarks_np[36:42].mean(axis=0).astype(int)
    right_eye = landmarks_np[42:48].mean(axis=0).astype(int)
    
    # Get angle between eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Calculate face center
    center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Get new image dimensions
    h, w = image.shape[:2]
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Calculate new dimensions after rotation
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix for translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (nW, nH))
    
    # Re-detect face after rotation
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY) if len(rotated.shape) == 3 else rotated
    faces = face_detector(rotated_gray, 1)
    
    if len(faces) == 0:
        # If face detection fails after rotation, resize original image
        return cv2.resize(rotated, (desired_size, desired_size))
    
    # Get the largest face
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Add padding around the face
    padding_factor = 0.2  # Add 20% padding
    width = face.width()
    height = face.height()
    
    # Calculate padding
    left = max(0, face.left() - int(width * padding_factor))
    top = max(0, face.top() - int(height * padding_factor))
    right = min(rotated.shape[1], face.right() + int(width * padding_factor))
    bottom = min(rotated.shape[0], face.bottom() + int(height * padding_factor))
    
    # Extract face with padding
    face_img = rotated[top:bottom, left:right]
    
    # Resize to desired size
    aligned_face = cv2.resize(face_img, (desired_size, desired_size))
    
    return aligned_face

def process_directory(input_dir, output_dir, desired_size=224):
    """Process all images in a directory and save aligned faces"""
    # Load dlib face detector and shape predictor
    print("Loading face detection models...")
    face_detector = dlib.get_frontal_face_detector()
    
    # Use pre-trained model for landmarks
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print(f"ERROR: Please download the shape predictor model from:")
        print(f"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(f"Extract it and place it in the current directory.")
        return False
    
    landmark_predictor = dlib.shape_predictor(model_path)
    
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
            
            # Align face
            aligned = align_face(img_rgb, face_detector, landmark_predictor, desired_size)
            
            if aligned is None:
                print(f"Warning: No face detected in {img_path}")
                # Copy original image
                cv2.imwrite(out_path, img)
                failed += 1
                continue
            
            # Save the aligned face
            cv2.imwrite(out_path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            successful += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed += 1
    
    print(f"Processed {successful} images successfully, {failed} images failed")
    return True

def main():
    parser = argparse.ArgumentParser(description='Align faces in dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for aligned faces')
    parser.add_argument('--size', type=int, default=224, help='Desired output size (default: 224)')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.size)

if __name__ == "__main__":
    main() 