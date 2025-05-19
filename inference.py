"""
Inference script for using the trained ensemble FER model
Supports both image files and webcam input
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import dlib
import matplotlib.pyplot as plt
from PIL import Image
from ensemble_train import AdvancedModel, preprocess_face, EMOTIONS, DEVICE, IMAGE_SIZE, HAS_FACE_ALIGNMENT

# Constants
MODEL_TYPES = [
    'efficientnet_b0',
    'resnet18',
    'mobilenetv3_small',
    'convnext_tiny',
    'vit_tiny_patch16',
    'swin_tiny',
    'resnest50d'
]

def load_ensemble_models(models_dir='.'):
    """Load all trained models in the ensemble"""
    models = []
    weights = []
    
    # Check if model files exist
    model_files = [f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.pth')]
    
    if not model_files:
        print(f"No model files found in {models_dir}. Please train the models first.")
        return None, None
    
    # Try to load ensemble weights if available
    try:
        weights = np.load('ensemble_weights.npy')
        print(f"Loaded ensemble weights: {weights}")
    except:
        # Set equal weights if not available
        weights = np.ones(len(MODEL_TYPES)) / len(MODEL_TYPES)
        print("Using equal weights for ensemble.")
    
    # Load each model
    for i, model_type in enumerate(MODEL_TYPES):
        model_path = os.path.join(models_dir, f'model_{i+1}.pth')
        
        if os.path.exists(model_path):
            try:
                model = AdvancedModel(model_type).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                models.append(model)
                print(f"Loaded model {i+1}: {model_type}")
            except Exception as e:
                print(f"Error loading model {model_type}: {str(e)}")
        else:
            print(f"Model file not found: {model_path}")
    
    if not models:
        print("No models could be loaded. Please train the models first.")
        return None, None
    
    # Adjust weights if needed
    if len(weights) != len(models):
        weights = np.ones(len(models)) / len(models)
    
    return models, weights

def process_image(image_path, models, weights, show_result=True):
    """Process a single image file"""
    # Read image
    try:
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # If input is already an image array
            if len(image_path.shape) == 3:
                image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
            else:
                image = image_path
    except Exception as e:
        print(f"Error reading image: {str(e)}")
        return None
    
    # Apply face detection and alignment
    if HAS_FACE_ALIGNMENT:
        processed_image = preprocess_face(image)
    else:
        # Just resize if no face alignment
        processed_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Apply transformations
    transform = A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=processed_image)
    tensor_image = transformed["image"].unsqueeze(0).to(DEVICE)
    
    # Run prediction
    with torch.no_grad():
        all_probs = []
        
        for model in models:
            outputs = model(tensor_image)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
        
        # Apply weights to each model's prediction
        weighted_probs = np.zeros((1, len(EMOTIONS)))
        for i, weight in enumerate(weights):
            weighted_probs += weight * all_probs[i]
        
        # Get prediction
        pred_idx = np.argmax(weighted_probs, axis=1)[0]
        emotion = EMOTIONS[pred_idx]
        confidence = weighted_probs[0, pred_idx]
    
    # Display result
    if show_result and isinstance(image_path, str):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB))
        plt.title(f"Prediction: {emotion} ({confidence:.2f})")
        
        plt.tight_layout()
        plt.show()
    
    # Return prediction
    return {
        'emotion': emotion,
        'confidence': confidence,
        'probabilities': {EMOTIONS[i]: float(weighted_probs[0, i]) for i in range(len(EMOTIONS))},
        'processed_image': processed_image
    }

def run_webcam(models, weights):
    """Run real-time emotion recognition on webcam feed"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Process frame
        result = process_image(frame, models, weights, show_result=False)
        
        if result:
            emotion = result['emotion']
            confidence = result['confidence']
            processed_image = result['processed_image']
            
            # Display original frame with prediction
            cv2.putText(frame, f"{emotion}: {confidence:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display processed face
            processed_display = cv2.resize(processed_image, (IMAGE_SIZE*2, IMAGE_SIZE*2))
            frame[10:10+processed_display.shape[0], frame.shape[1]-processed_display.shape[1]-10:frame.shape[1]-10] = processed_display
            
            # Show frame
            cv2.imshow('FER Webcam', frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release and cleanup
    cap.release()
    cv2.destroyAllWindows()

def main(args):
    # Load models
    models, weights = load_ensemble_models(args.models_dir)
    
    if models is None or weights is None:
        return
    
    # Process based on mode
    if args.webcam:
        run_webcam(models, weights)
    elif args.image:
        process_image(args.image, models, weights)
    else:
        print("Please specify either --image or --webcam")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FER2013 Inference")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time prediction")
    parser.add_argument("--models_dir", type=str, default=".", help="Directory containing model files")
    
    args = parser.parse_args()
    
    if not args.image and not args.webcam:
        parser.print_help()
        print("\nPlease specify either --image or --webcam")
        sys.exit(1)
    
    main(args) 