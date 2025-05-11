"""Prediction script for the emotion recognition model."""

import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from emotion_net.models.ensemble import EnsembleModel
from emotion_net.config.constants import EMOTIONS, IMAGE_MEAN, IMAGE_STD

def predict(model, image_path, device, image_size=224):
    """Make a prediction on a single image."""
    # Load and preprocess the image
    transform = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs, _ = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Return emotion and confidence
    emotion = EMOTIONS[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    return emotion, confidence, probabilities.cpu().numpy()[0]

def main():
    """Main function for making predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Make predictions with the emotion recognition model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model weights")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the image to predict")
    parser.add_argument("--backbones", type=str, nargs="+", 
                        default=["efficientnet_b0", "resnet18"],
                        help="Backbone architectures to use")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = EnsembleModel(num_classes=len(EMOTIONS), backbones=args.backbones)
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Make prediction
    try:
        emotion, confidence, probabilities = predict(model, args.image_path, device, args.image_size)
        
        print(f"\nPredicted emotion: {emotion} with confidence: {confidence:.2f}")
        print("\nAll probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{EMOTIONS[i]}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main() 