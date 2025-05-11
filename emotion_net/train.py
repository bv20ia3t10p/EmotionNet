"""Main training script for the emotion recognition model."""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd

from emotion_net.config.constants import (
    EMOTIONS, DEFAULT_IMAGE_SIZE, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_PATIENCE,
    DEFAULT_BACKBONES
)
from emotion_net.data.dataset import load_data, AdvancedEmotionDataset
from emotion_net.models.ensemble import EnsembleModel
from emotion_net.training.trainer import train

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train the emotion recognition model")
    parser.add_argument("--data_dir", type=str, default="./extracted/emotion/train", 
                        help="Path to training data directory")
    parser.add_argument("--test_dir", type=str, default="./extracted/emotion/test", 
                        help="Path to test data directory")
    parser.add_argument("--model_dir", type=str, default="./models", 
                        help="Directory to save models")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to pre-trained model to resume from")
    parser.add_argument("--backbones", type=str, nargs="+", 
                        default=DEFAULT_BACKBONES,
                        help="Backbone architectures to use")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, 
                        help="Size to resize images to")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, 
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, 
                        help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, 
                        help="Base learning rate")
    parser.add_argument("--no_amp", action="store_true", 
                        help="Disable automatic mixed precision")
    parser.add_argument("--no_ema", action="store_true", 
                        help="Disable exponential moving average model")
    parser.add_argument("--save_path", type=str, default="models/high_accuracy_model.pth",
                        help="Path to save the best model")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data
    train_paths, train_labels = load_data(args.data_dir, EMOTIONS)
    test_paths, test_labels = load_data(args.test_dir, EMOTIONS)
    
    # Split training data into train and validation sets
    indices = np.arange(len(train_paths))
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    
    train_indices = indices[:split]
    valid_indices = indices[split:]
    
    train_dataset = AdvancedEmotionDataset(
        [train_paths[i] for i in train_indices],
        [train_labels[i] for i in train_indices],
        mode='train',
        image_size=args.image_size
    )
    
    valid_dataset = AdvancedEmotionDataset(
        [train_paths[i] for i in valid_indices],
        [train_labels[i] for i in valid_indices],
        mode='val',
        image_size=args.image_size
    )
    
    test_dataset = AdvancedEmotionDataset(
        test_paths, test_labels, mode='test', image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=min(os.cpu_count(), 4), pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(os.cpu_count(), 4), pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(os.cpu_count(), 4), pin_memory=True
    )
    
    # Create model
    model = EnsembleModel(num_classes=len(EMOTIONS), backbones=args.backbones)
    
    # Load pre-trained weights if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model = model.to(device)
    
    # Print model summary
    print(f"Model created with backbones: {args.backbones}")
    
    # Train model
    best_acc = train(
        model, train_loader, valid_loader,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=device,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        save_path=args.save_path
    )
    
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    # Test best model
    print("Testing best model...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate per-class accuracies
    class_correct = [0] * len(EMOTIONS)
    class_total = [0] * len(EMOTIONS)
    
    for pred, target in zip(all_preds, all_targets):
        class_correct[target] += int(pred == target)
        class_total[target] += 1
    
    # Print test results
    test_acc = 100 * test_correct / test_total
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Print per-class accuracies
    for i in range(len(EMOTIONS)):
        if class_total[i] > 0:
            print(f"Class {EMOTIONS[i]}: {100*class_correct[i]/class_total[i]:.2f}%")
        else:
            print(f"Class {EMOTIONS[i]}: N/A (no samples)")
    
    # Generate and save classification report
    report = classification_report(all_targets, all_preds, 
                                  target_names=list(EMOTIONS.values()),
                                  output_dict=True)
    
    # Convert to DataFrame and save as CSV
    report_df = pd.DataFrame(report).transpose()
    report_path = args.save_path.replace(".pth", "_classification_report.csv")
    report_df.to_csv(report_path)
    
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    main() 