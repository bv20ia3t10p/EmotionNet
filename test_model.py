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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import argparse
from emotion_post_processor import apply_post_processing, evaluate_with_post_processing


def parse_args():
    parser = argparse.ArgumentParser(description='Test emotion recognition model')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, 
                        help='Path to model weights')
    parser.add_argument('--data_path', type=str, default=TEST_PATH, 
                        help='Path to test data')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE,
                        help='Model type (ConvNeXtEmoteNet or EmotionViT)')
    parser.add_argument('--backbone', type=str, default=BACKBONE,
                        help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for testing')
    parser.add_argument('--tta', action='store_true', 
                        help='Enable test-time augmentation')
    parser.add_argument('--tta_num', type=int, default=TTA_NUM_AUGMENTS,
                        help='Number of TTA augmentations')
    parser.add_argument('--ensemble', action='store_true',
                        help='Enable model ensemble')
    parser.add_argument('--ensemble_size', type=int, default=ENSEMBLE_SIZE,
                        help='Number of models to ensemble')
    parser.add_argument('--post_process', action='store_true',
                        help='Enable post-processing')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualizations')
    return parser.parse_args()


def get_test_loader(data_path, batch_size=32):
    # Define transforms
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    test_dataset = datasets.ImageFolder(
        root=data_path,
        transform=test_transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return test_loader, test_dataset.classes


def get_tta_transforms(num_augments=5):
    """Get test-time augmentation transforms"""
    transforms_list = []
    
    # Base transform (center crop, normalize)
    transforms_list.append(
        transforms.Compose([
            transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Horizontal flip
    transforms_list.append(
        transforms.Compose([
            transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Brightness adjustments
    if num_augments > 2:
        for brightness in [-0.1, 0.1]:
            transforms_list.append(
                transforms.Compose([
                    transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
                    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ColorJitter(brightness=brightness),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
    
    # Small rotations
    if num_augments > 4:
        for angle in [-5, 5]:
            transforms_list.append(
                transforms.Compose([
                    transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
                    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.RandomRotation(degrees=(angle, angle)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
    
    return transforms_list[:num_augments]


def load_model(model_path, model_type, backbone, device):
    """Load model from checkpoint"""
    if model_type == "ConvNeXtEmoteNet":
        model = ConvNeXtEmoteNet(backbone=backbone).to(device)
    elif model_type == "EmotionViT":
        model = EmotionViT(backbone=backbone).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if not os.path.exists(model_path):
        # Try alternate paths
        alt_paths = [
            f"{model_path}_final.pth",
            f"{model_path}_swa.pth",
            MODEL_PATH,
            f"{MODEL_PATH}_final.pth",
            f"{MODEL_PATH}_swa.pth"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative model path: {model_path}")
                break
        else:
            raise FileNotFoundError(f"Model not found at {model_path} or any alternative paths")
    
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel wrapped state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_ensemble_models(model_path, model_type, backbone, device, ensemble_size=5):
    """Load multiple models for ensembling"""
    models = []
    
    # Get directory and prefix
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # Look for checkpoints
    checkpoints = []
    for file in os.listdir(model_dir):
        if file.startswith(model_name) and "epoch_" in file:
            try:
                epoch = int(file.split('epoch_')[1].split('.')[0])
                checkpoints.append((epoch, os.path.join(model_dir, file)))
            except:
                continue
    
    # Sort by epoch (descending)
    checkpoints.sort(reverse=True)
    
    # Use SWA model if available
    swa_path = f"{model_path}_swa.pth"
    if os.path.exists(swa_path):
        # Create and load SWA model
        swa_model = load_model(swa_path, model_type, backbone, device)
        models.append(swa_model)
        print(f"Loaded SWA model from {swa_path}")
    
    # Add most recent checkpoints
    for epoch, ckpt_path in checkpoints[:ensemble_size-len(models)]:
        model = None
        try:
            model = load_model(ckpt_path, model_type, backbone, device)
            models.append(model)
            print(f"Loaded checkpoint from epoch {epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint {ckpt_path}: {str(e)}")
            if model is not None:
                del model
            continue
    
    return models


def evaluate(model, test_loader, device, use_tta=False, tta_num=5, use_post_process=False, ensemble_models=None):
    """Evaluate model on test set"""
    if use_tta and use_post_process and ensemble_models:
        # Use specialized evaluation function for full pipeline
        return evaluate_with_post_processing(model, test_loader, device, ensemble_models)
    
    # Standard evaluation
    model.eval()
    if ensemble_models:
        for m in ensemble_models:
            m.eval()
    
    total = 0
    correct = 0
    class_correct = {}
    class_total = {}
    all_preds = []
    all_targets = []
    
    # Initialize per-class tracking
    classes = test_loader.dataset.classes
    for i in range(len(classes)):
        class_correct[i] = 0
        class_total[i] = 0
    
    # Create TTA transforms if needed
    tta_transforms = get_tta_transforms(tta_num) if use_tta else None
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluation", leave=False):
            batch_size = images.size(0)
            images, targets = images.to(device), targets.to(device)
            
            if use_tta:
                # Perform TTA
                all_outputs = []
                # Original prediction
                outputs = model(images)
                all_outputs.append(outputs)
                
                # TTA predictions
                for t_idx in range(1, len(tta_transforms)):
                    # Apply transform
                    transformed_images = torch.stack([
                        tta_transforms[t_idx](img.cpu()) for img in images
                    ]).to(device)
                    
                    # Get predictions
                    tta_outputs = model(transformed_images)
                    all_outputs.append(tta_outputs)
                
                # Average predictions
                if use_post_process and ensemble_models:
                    # Add ensemble model outputs
                    for ens_model in ensemble_models:
                        ens_outputs = ens_model(images)
                        all_outputs.append(ens_outputs)
                    
                    # Apply post-processing
                    _, predictions, _ = apply_post_processing(
                        all_outputs, ensemble=True, refine=True
                    )
                else:
                    # Simple averaging
                    outputs = torch.stack(all_outputs).mean(dim=0)
                    _, predictions = torch.max(outputs, 1)
            else:
                if use_post_process and ensemble_models:
                    # Get predictions from all models
                    all_outputs = [model(images)]
                    for ens_model in ensemble_models:
                        all_outputs.append(ens_model(images))
                    
                    # Apply post-processing
                    _, predictions, _ = apply_post_processing(
                        all_outputs, ensemble=True, refine=True
                    )
                else:
                    # Standard prediction
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
            
            # Compute accuracy
            total += batch_size
            correct += (predictions == targets).sum().item()
            
            # Compute per-class accuracy
            for i in range(len(classes)):
                idx = (targets == i)
                class_total[i] += idx.sum().item()
                class_correct[i] += ((predictions == i) & idx).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_accuracies[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0
    
    return accuracy, class_accuracies, all_preds, all_targets


def visualize_predictions(model, test_dataset, device, save_path=None, num_samples=10):
    """Visualize model predictions on random samples"""
    model.eval()
    
    # Get class names
    class_names = test_dataset.classes
    
    # Select random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and label
            image, label = test_dataset[idx]
            true_class = class_names[label]
            
            # Get prediction
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = class_names[pred.item()]
            
            # Get confidence
            prob = F.softmax(output, dim=1)[0][pred.item()].item()
            
            # Get original image
            img_path = test_dataset.samples[idx][0]
            orig_img = Image.open(img_path).convert('RGB')
            
            # Display image
            axes[i].imshow(orig_img)
            color = 'green' if pred.item() == label else 'red'
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {prob:.2f}", color=color)
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    test_loader, class_names = get_test_loader(args.data_path, args.batch_size)
    print(f"Test dataset classes: {class_names}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type, args.backbone, device)
    print("Model loaded successfully.")
    
    # Load ensemble models if enabled
    ensemble_models = None
    if args.ensemble:
        print(f"Loading {args.ensemble_size} models for ensemble...")
        ensemble_models = load_ensemble_models(
            args.model_path, args.model_type, args.backbone, device, args.ensemble_size
        )
        print(f"Loaded {len(ensemble_models)} ensemble models.")
    
    # Evaluate model
    print("Evaluating model performance...")
    features = (
        f"TTA{'(' + str(args.tta_num) + ')' if args.tta else ''}"
        f"{' + Ensemble(' + str(len(ensemble_models or [])) + ')' if args.ensemble else ''}"
        f"{' + PostProcess' if args.post_process else ''}"
    )
    print(f"Using: {features}")
    
    accuracy, class_accuracies, all_preds, all_targets = evaluate(
        model, test_loader, device, 
        use_tta=args.tta, 
        tta_num=args.tta_num,
        use_post_process=args.post_process,
        ensemble_models=ensemble_models
    )
    
    # Print results
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print("\nPer-class Accuracies:")
    for i, class_name in enumerate(class_names):
        print(f"  - Class {i} ({class_name}): {class_accuracies[i]:.2f}%")
    
    # Visualize predictions if requested
    if args.visualize:
        print("\nVisualizing predictions...")
        save_path = args.save_path if args.save_path else None
        visualize_predictions(model, test_loader.dataset, device, save_path)
    

if __name__ == "__main__":
    main() 