import torch # type: ignore
import os
import sys
from model import *
from model_utils import *
from config import *
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore


def load_model(model_path, device):
    """Load the trained model."""
    print(f"🔹 Loading model from {model_path}")
    
    # Create model based on MODEL_TYPE environment variable
    if MODEL_TYPE == "EmotionViT":
        model = EmotionViT(backbone=BACKBONE, pretrained=False).to(device)
    elif MODEL_TYPE == "ConvNeXtEmoteNet":
        model = ConvNeXtEmoteNet(backbone=BACKBONE, pretrained=False).to(device)
    else:
        print(f"⚠️ Unknown model type: {MODEL_TYPE}, defaulting to ConvNeXtEmoteNet")
        model = ConvNeXtEmoteNet(backbone=BACKBONE, pretrained=False).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        sys.exit(1)
        
    model.eval()
    return model


def test_model():
    """Test the trained model on validation dataset with detailed metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Load validation dataset
    _, val_loader = prepare_dataloaders(TRAIN_PATH, TEST_PATH)
    
    # Initialize metrics
    total = 0
    correct = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    all_preds = []
    all_targets = []
    all_probs = []  # Store class probabilities for ROC-AUC calculation
    
    # Advanced TTA (Test Time Augmentation)
    print(f"🔹 Evaluating model with Advanced TTA ({TTA_NUM_AUGMENTS} augmentations)...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            
            # Original prediction
            outputs = model(images)
            
            # TTA predictions
            for aug_idx in range(TTA_NUM_AUGMENTS - 1):
                # Apply diverse augmentations based on aug_idx
                aug_images = images.clone()
                
                # Horizontal flip for odd indices
                if aug_idx % 2 == 1:
                    aug_images = torch.flip(aug_images, dims=[3])
                
                # Small shifts based on modulo 3
                shift_type = aug_idx % 3
                if shift_type == 0:  # Horizontal shift
                    shift_amount = 4 if aug_idx < 3 else 2
                    aug_images = torch.roll(aug_images, shifts=(shift_amount, 0), dims=(2, 3))
                elif shift_type == 1:  # Vertical shift
                    shift_amount = 4 if aug_idx < 3 else 2
                    aug_images = torch.roll(aug_images, shifts=(0, shift_amount), dims=(2, 3))
                elif shift_type == 2:  # Both directions
                    shift_amount = 3 if aug_idx < 3 else 2
                    aug_images = torch.roll(aug_images, shifts=(shift_amount, shift_amount), dims=(2, 3))
                
                # Center crop with padding for odd augmentation indices
                if aug_idx % 2 == 1:
                    pad_size = 4
                    padded = torch.zeros_like(aug_images)
                    padded[:, :, pad_size:-pad_size, pad_size:-pad_size] = aug_images[:, :, pad_size:-pad_size, pad_size:-pad_size]
                    aug_images = padded
                
                # Color jitter for high indices
                if aug_idx > 3:
                    # Simple brightness adjustment (multiply by scalar)
                    brightness_factor = 0.9 + (aug_idx % 3) * 0.1  # 0.9, 1.0, or 1.1
                    aug_images = aug_images * brightness_factor
                
                # Add to ensemble
                aug_outputs = model(aug_images)
                outputs += aug_outputs
            
            # Average predictions
            outputs /= TTA_NUM_AUGMENTS
            
            # Get predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Track per-class accuracy
            for i in range(batch_size):
                label = targets[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                    
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracies
    class_names = [d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))]
    if len(class_names) != NUM_CLASSES:
        class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
        
    class_accuracies = [100.0 * c / max(1, t) for c, t in zip(class_correct, class_total)]
    
    # Print results
    print(f"\n✅ Test Results:")
    print(f"   🔹 Overall Accuracy: {accuracy:.2f}%")
    print("\n   🔹 Per-class accuracies:")
    for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
        print(f"      - {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Calculate advanced metrics if sklearn is available
    try:
        from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score # type: ignore
        import seaborn as sns # type: ignore
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, 'confusion_matrix.png'))
        print(f"\n✅ Confusion matrix saved to {os.path.join(ROOT, 'confusion_matrix.png')}")
        
        # Classification report (precision, recall, f1-score)
        report = classification_report(all_targets, all_preds, target_names=class_names)
        print("\n🔹 Classification Report:")
        print(report)
        
        # ROC-AUC for multi-class
        all_probs = np.array(all_probs)
        all_targets_one_hot = np.zeros((len(all_targets), NUM_CLASSES))
        for i, target in enumerate(all_targets):
            all_targets_one_hot[i, target] = 1
            
        # One-vs-rest ROC-AUC
        try:
            roc_auc = roc_auc_score(all_targets_one_hot, all_probs, multi_class='ovr')
            print(f"\n🔹 ROC-AUC Score (One-vs-Rest): {roc_auc:.4f}")
        except ValueError as e:
            print(f"\n⚠️ Could not calculate ROC-AUC: {e}")
            
    except ImportError:
        print("\n⚠️ sklearn or seaborn not installed, skipping advanced metrics visualization")
    
    return accuracy


if __name__ == "__main__":
    test_model() 