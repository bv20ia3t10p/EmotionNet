#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

from enhanced_resemotenet import EnhancedResEmoteNet
from train_enhanced_model import extract_features, load_hybrid_model

# Define emotion labels
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear', 
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Color map for visualizing different emotions
EMOTION_COLORS = {
    'angry': (0, 0, 255),       # Red
    'disgust': (0, 140, 255),   # Orange
    'fear': (0, 255, 255),      # Yellow
    'happy': (0, 255, 0),       # Green
    'sad': (255, 0, 0),         # Blue
    'surprise': (255, 0, 255),  # Purple
    'neutral': (255, 255, 255)  # White
}

def preprocess_image(image_path, target_size=(48, 48)):
    """Load and preprocess image for the model"""
    # Check if it's a file path or numpy array
    if isinstance(image_path, str):
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
    else:
        # Assume it's already a numpy array
        img = image_path
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Convert to PIL Image first
    pil_img = Image.fromarray(resized)
    tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    
    return tensor, img

def detect_faces(image_path, face_cascade_path="haarcascade_frontalface_default.xml"):
    """Detect faces in an image and return face regions"""
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path.copy()
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return img, faces

def predict_emotion(model, image_tensor, device, use_hybrid=True):
    """Predict emotion from preprocessed image tensor"""
    # Set model to evaluation mode
    model.eval()
    
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Get bias parameters from model or use defaults
    angry_bias = getattr(model, 'angry_bias', 4.0)
    neutral_bias = getattr(model, 'neutral_bias', 2.0)
    
    # Make prediction (using hybrid model if available)
    with torch.no_grad():
        if use_hybrid and hasattr(model, 'hybrid_predict'):
            # Use hybrid prediction
            pred_idx = model.hybrid_predict(image_tensor, device=device)[0]
            emotion = EMOTIONS[pred_idx]
            
            # Still need to get CNN probabilities for visualization
            logits, valence, arousal = model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]
            
            # Apply probability bias for underrepresented classes
            # Artificial boosting of angry (0) and neutral (6) probabilities
            probs_np = probs.cpu().numpy()
            
            # Apply boosts using model parameters
            probs_np[0] *= angry_bias  # Angry
            probs_np[6] *= neutral_bias  # Neutral
            
            # Re-normalize
            probs_np = probs_np / np.sum(probs_np)
            
            # Determine if we should force prediction of angry or neutral
            # based on certain visual cues in the image (simplified approach)
            img_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 0.5) + 0.5  # Denormalize
            
            # Extract HOG features to detect anger patterns
            features = extract_features(img_np)
            
            # Get probabilities from original prediction 
            confidence = probs_np[pred_idx] * 100
            all_confidences = {EMOTIONS[i]: probs_np[i] * 100 for i in range(len(EMOTIONS))}
            
            # If the hybrid model ignored angry/neutral despite visual cues,
            # override with more aggressive bias
            angry_threshold = 0.15 + (angry_bias / 20.0)  # Adjusts threshold based on bias
            neutral_threshold = 0.20 + (neutral_bias / 20.0)  # Adjusts threshold based on bias
            
            if pred_idx != 0 and probs_np[0] > angry_threshold:  # If angry probability is significant
                pred_idx = 0  # Force angry
                emotion = EMOTIONS[pred_idx]
                confidence = probs_np[pred_idx] * 100
            elif pred_idx != 6 and probs_np[6] > neutral_threshold:  # If neutral probability is significant
                pred_idx = 6  # Force neutral
                emotion = EMOTIONS[pred_idx]
                confidence = probs_np[pred_idx] * 100
                
            valence_value = valence.item()
            arousal_value = arousal.item()
        else:
            # Use standard CNN prediction with probability bias
            logits, valence, arousal = model(image_tensor)
            
            # Get predicted class
            probs = F.softmax(logits, dim=1)[0]
            
            # Apply probability bias for underrepresented classes
            probs_np = probs.cpu().numpy()
            
            # Boost angry and neutral class probabilities using model parameters
            # Standard model gets even higher boost
            probs_np[0] *= angry_bias * 1.25  # Angry with extra boost for standard model
            probs_np[6] *= neutral_bias * 1.5  # Neutral with extra boost for standard model
            
            # Re-normalize
            probs_np = probs_np / np.sum(probs_np)
            
            # Get new predicted class
            emotion_idx = np.argmax(probs_np)
            emotion = EMOTIONS[emotion_idx]
            confidence = probs_np[emotion_idx] * 100
            
            # Get all class confidences
            all_confidences = {EMOTIONS[i]: probs_np[i] * 100 for i in range(len(EMOTIONS))}
            
            # Get valence-arousal values
            valence_value = valence.item()
            arousal_value = arousal.item()
    
    return emotion, confidence, all_confidences, (valence_value, arousal_value)

def visualize_prediction(image, emotion, confidence, all_confidences, face_coords=None):
    """Visualize prediction on the image"""
    # Create a copy of the image
    result = image.copy()
    
    # If face coordinates are provided, draw a rectangle around the face
    if face_coords is not None:
        x, y, w, h = face_coords
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
    
    # Add emotion label with confidence
    label = f"{emotion}: {confidence:.1f}%"
    if face_coords is not None:
        x, y, w, h = face_coords
        cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                   EMOTION_COLORS.get(emotion, (255, 255, 255)), 2)
    else:
        cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                   EMOTION_COLORS.get(emotion, (255, 255, 255)), 2)
    
    # Create a bar chart showing all emotion confidences
    plt.figure(figsize=(10, 6))
    emotions = list(all_confidences.keys())
    confidences = list(all_confidences.values())
    
    # Create bars with different colors
    bars = plt.bar(emotions, confidences)
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.plasma(i / len(emotions)))
    
    plt.ylabel('Confidence (%)')
    plt.title('Emotion Prediction Confidence')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    return result

def process_single_image(image_path, model, device, args):
    """Process a single image for emotion recognition"""
    print(f"Processing {image_path}...")
    
    # Detect faces in the image
    try:
        img, faces = detect_faces(image_path, args.face_cascade)
        
        if len(faces) == 0:
            print("No faces detected in the image, processing entire image")
            # Process the entire image if no faces are detected
            tensor, img = preprocess_image(image_path)
            emotion, confidence, all_confidences, va_values = predict_emotion(model, tensor, device, use_hybrid=args.use_hybrid)
            
            # Visualize prediction
            result = visualize_prediction(img, emotion, confidence, all_confidences)
            
            # Save output
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            output_path = f"{base_name}_output.jpg"
            plot_path = f"{base_name}_plot.png"
            
            cv2.imwrite(output_path, result)
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Prediction: {emotion} ({confidence:.1f}%)")
            print(f"Output saved to {output_path}")
            print(f"Plot saved to {plot_path}")
        else:
            print(f"Detected {len(faces)} faces")
            result = img.copy()
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face ROI and preprocess
                face_roi = img[y:y+h, x:x+w]
                tensor, _ = preprocess_image(face_roi)
                
                # Predict emotion
                emotion, confidence, all_confidences, va_values = predict_emotion(model, tensor, device, use_hybrid=args.use_hybrid)
                
                # Draw on the result image
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
                
                # Add emotion label with confidence
                label = f"{emotion}: {confidence:.1f}%"
                cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                print(f"Face {i+1}: {emotion} ({confidence:.1f}%)")
                
                # Save individual face plot
                base_name = os.path.basename(image_path).rsplit('.', 1)[0]
                plot_path = f"{base_name}_face{i+1}_plot.png"
                
                # Create individual face confidence plot
                plt.figure(figsize=(10, 6))
                emotions = list(all_confidences.keys())
                confidences = list(all_confidences.values())
                
                bars = plt.bar(emotions, confidences)
                for j, bar in enumerate(bars):
                    bar.set_color(plt.cm.plasma(j / len(emotions)))
                
                plt.ylabel('Confidence (%)')
                plt.title(f'Face {i+1} Emotion Prediction')
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            
            # Save the result
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            output_path = f"{base_name}_output.jpg"
            cv2.imwrite(output_path, result)
            
            print(f"Output saved to {output_path}")
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from an image using Enhanced Emotion Recognition Model')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the input image')
    parser.add_argument('--model', type=str, default='./models/hybrid_model.pth',
                        help='Path to the model weights (use hybrid_model.pth for ensemble model)')
    parser.add_argument('--face_cascade', type=str, default='haarcascade_frontalface_default.xml',
                        help='Path to the face cascade XML file')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Path to save the output image')
    parser.add_argument('--plot', type=str, default='emotion_confidence.png',
                        help='Path to save the confidence plot')
    parser.add_argument('--use_hybrid', action='store_true', default=True,
                        help='Use hybrid ensemble model for better angry/neutral recognition')
    parser.add_argument('--angry_bias', type=float, default=4.0,
                        help='Bias multiplier for angry class (higher values = more angry predictions)')
    parser.add_argument('--neutral_bias', type=float, default=2.0,
                        help='Bias multiplier for neutral class (higher values = more neutral predictions)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                        help='Backbone architecture')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if args.use_hybrid and args.model.endswith('hybrid_model.pth'):
        # Load hybrid ensemble model
        print("Loading hybrid ensemble model...")
        model = load_hybrid_model(args.model, device)
    else:
        # Load regular CNN model
        print("Loading standard CNN model...")
        model = EnhancedResEmoteNet(
            num_classes=7, 
            dropout_rate=0.4,
            backbone=args.backbone,
            use_fpn=True,
            use_landmarks=True,
            use_contrastive=True
        )
        
        # Load model weights
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.to(device)
    
    # Add bias parameters to the model for prediction
    model.angry_bias = args.angry_bias
    model.neutral_bias = args.neutral_bias
    
    model.eval()
    
    # Check if it's a single image or a directory
    if os.path.isdir(args.image):
        # Process all images in the directory
        image_files = [f for f in os.listdir(args.image) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(args.image, img_file)
            process_single_image(img_path, model, device, args)
    else:
        # Process a single image
        process_single_image(args.image, model, device, args)

if __name__ == '__main__':
    main() 