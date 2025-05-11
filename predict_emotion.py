import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import models from separate files
from resemotenet import ResEmoteNet
from enhanced_resemotenet import EnhancedResEmoteNet

# Define emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def parse_args():
    parser = argparse.ArgumentParser(description='Predict facial emotions using trained models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a single image to predict')
    parser.add_argument('--test_dir', type=str, default=None, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--use_camera', action='store_true', help='Use webcam for real-time predictions')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--model_type', type=str, choices=['original', 'enhanced'], 
                        help='Model type (if not specified in the checkpoint)')
    return parser.parse_args()

def load_model(model_path, device, model_type=None):
    """Load the trained model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model type from checkpoint if not specified
    if model_type is None:
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
            # Default to original for backward compatibility
            model_type = 'original'
            print("Model type not found in checkpoint, defaulting to 'original'")
    
    # Initialize correct model architecture
    if model_type == 'original':
        model = ResEmoteNet(num_classes=len(EMOTIONS)).to(device)
    else:  # enhanced
        model = EnhancedResEmoteNet(num_classes=len(EMOTIONS)).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Model type: {model_type}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    model.eval()
    return model, model_type

def preprocess_image(image_path):
    """Preprocess a single image for prediction."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_single_image(model, image_path, device, model_type='original'):
    """Make prediction on a single image."""
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        # Both models have the same inference interface
        logits, valence, arousal = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probs).item()
    
    return {
        'emotion': EMOTIONS[prediction],
        'confidence': probs[prediction].item() * 100,
        'probabilities': {EMOTIONS[i]: probs[i].item() * 100 for i in range(len(EMOTIONS))},
        'valence': valence.item(),
        'arousal': arousal.item()
    }

def visualize_prediction(image_path, result, output_path=None):
    """Visualize the prediction results."""
    # Load and resize image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display the image
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {result['emotion']} ({result['confidence']:.2f}%)")
    ax1.axis('off')
    
    # Create bar chart of probabilities
    emotions = list(result['probabilities'].keys())
    probabilities = list(result['probabilities'].values())
    
    # Sort by probability for better visualization
    sorted_indices = np.argsort(probabilities)
    emotions = [emotions[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    
    y_pos = np.arange(len(emotions))
    ax2.barh(y_pos, probabilities, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(emotions)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Emotion Probabilities')
    
    # Add valence-arousal information
    plt.suptitle(f"Valence: {result['valence']:.2f}, Arousal: {result['arousal']:.2f}", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def evaluate_model(model, test_dir, device, output_dir, model_type='original'):
    """Evaluate the model on a test directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transforms for test images
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    all_predictions = []
    all_labels = []
    
    # Process each emotion directory
    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found: {emotion_dir}")
            continue
        
        print(f"Processing {emotion} images...")
        
        # Process each image in the emotion directory
        for img_name in tqdm(os.listdir(emotion_dir)):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(emotion_dir, img_name)
                
                # Load and preprocess image
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    logits, _, _ = model(image_tensor)
                    prediction = torch.argmax(logits, dim=1).item()
                
                all_predictions.append(prediction)
                all_labels.append(emotion_idx)
    
    # Calculate metrics
    accuracy = 100 * sum(1 for p, l in zip(all_predictions, all_labels) if p == l) / len(all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=EMOTIONS)
    
    print(f"Evaluation complete. Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    norm_cm_path = os.path.join(output_dir, f'normalized_confusion_matrix_{model_type}.png')
    plt.savefig(norm_cm_path)
    plt.close()
    
    # Save classification report
    report_path = os.path.join(output_dir, f'classification_report_{model_type}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Results saved to {output_dir}")
    return accuracy, cm, report

def webcam_inference(model, device, model_type='original'):
    """Run real-time inference using webcam."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Define transformation for detected faces
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create a window with model type in the title
    window_name = f'Facial Emotion Recognition - {model_type.capitalize()} Model'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Starting webcam feed. Press 'q' to quit.")
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_roi = frame[y:y+h, x:x+w]
            face_tensor = transform(face_roi).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                logits, valence, arousal = model(face_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                prediction = torch.argmax(probs).item()
                confidence = probs[prediction].item() * 100
            
            # Draw bounding box and display prediction
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{EMOTIONS[prediction]}: {confidence:.1f}%"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display valence-arousal values
            va_text = f"V: {valence.item():.2f}, A: {arousal.item():.2f}"
            cv2.putText(frame, va_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show top-3 emotions with probabilities
            top3_indices = torch.topk(probs, 3)[1].cpu().numpy()
            for i, idx in enumerate(top3_indices):
                top_emotion = f"{EMOTIONS[idx]}: {probs[idx].item()*100:.1f}%"
                cv2.putText(frame, top_emotion, (x, y+h+45+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display the model type at the top of the frame
        cv2.putText(frame, f"Model: {model_type.capitalize()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    if not any([args.image_path, args.test_dir, args.use_camera]):
        print("Error: Please specify either --image_path, --test_dir, or --use_camera")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_type = load_model(args.model_path, device, args.model_type)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process according to specified mode
    if args.image_path:
        # Single image prediction
        result = predict_single_image(model, args.image_path, device, model_type)
        
        print(f"\nPrediction Results (Model: {model_type}):")
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Valence: {result['valence']:.2f}")
        print(f"Arousal: {result['arousal']:.2f}\n")
        
        # Print all probabilities
        print("Probabilities for each emotion:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"{emotion}: {prob:.2f}%")
        
        # Visualize the prediction
        output_path = os.path.join(args.output_dir, f'prediction_result_{model_type}.png')
        visualize_prediction(args.image_path, result, output_path)
    
    elif args.test_dir:
        # Evaluate on test directory
        evaluate_model(model, args.test_dir, device, args.output_dir, model_type)
    
    elif args.use_camera:
        # Real-time webcam inference
        webcam_inference(model, device, model_type)

if __name__ == "__main__":
    main() 