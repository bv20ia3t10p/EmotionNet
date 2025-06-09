#!/usr/bin/env python3
"""
Diagnostic script to identify why EffectiveFER model collapsed
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# Import model
try:
    from train_upgraded_fixed import EffectiveFER, FERPlusDataset
    print("✅ Successfully imported EffectiveFER model")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

def load_model(checkpoint_path, device):
    """Load model for diagnosis"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = EffectiveFER(
        img_size=64,
        patch_size=16,
        num_classes=8,
        embed_dim=512,
        depth=8,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def diagnose_model_outputs(model, dataloader, device, num_batches=5):
    """Analyze raw model outputs"""
    print("\n" + "="*60)
    print("DIAGNOSING MODEL OUTPUTS")
    print("="*60)
    
    all_logits = []
    all_probs = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            
            # Get raw logits
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
    
    # Combine all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"Analyzed {len(all_logits)} samples")
    
    # 1. Check logit statistics
    print(f"\nLogit Statistics:")
    print(f"  Mean: {all_logits.mean():.4f}")
    print(f"  Std:  {all_logits.std():.4f}")
    print(f"  Min:  {all_logits.min():.4f}")
    print(f"  Max:  {all_logits.max():.4f}")
    
    # 2. Check if logits are saturated
    print(f"\nLogit Range Analysis:")
    for i in range(8):
        class_logits = all_logits[:, i]
        print(f"  Class {i}: mean={class_logits.mean():.3f}, "
              f"std={class_logits.std():.3f}, "
              f"range=[{class_logits.min():.3f}, {class_logits.max():.3f}]")
    
    # 3. Check prediction distribution
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    print(f"\nPrediction Distribution:")
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    for pred, count in zip(unique_preds, pred_counts):
        percentage = count / len(all_predictions) * 100
        print(f"  {emotion_names[pred]}: {count} samples ({percentage:.1f}%)")
    
    # 4. Check probability statistics
    print(f"\nProbability Statistics:")
    max_probs = all_probs.max(axis=1)
    print(f"  Max prob mean: {max_probs.mean():.4f}")
    print(f"  Max prob std:  {max_probs.std():.4f}")
    print(f"  Confidence > 0.9: {(max_probs > 0.9).mean()*100:.1f}%")
    print(f"  Confidence > 0.99: {(max_probs > 0.99).mean()*100:.1f}%")
    
    return all_logits, all_probs, all_predictions

def diagnose_model_weights(model):
    """Analyze model weights for issues"""
    print("\n" + "="*60)
    print("DIAGNOSING MODEL WEIGHTS")
    print("="*60)
    
    # Check final classifier layer
    if hasattr(model, 'head'):
        classifier = model.head
    elif hasattr(model, 'classifier'):
        classifier = model.classifier
    else:
        print("❌ Could not find classifier layer")
        return
    
    print(f"Classifier layer: {type(classifier)}")
    
    with torch.no_grad():
        if hasattr(classifier, 'weight'):
            weight = classifier.weight
            bias = classifier.bias if classifier.bias is not None else None
            
            print(f"\nWeight Analysis:")
            print(f"  Shape: {weight.shape}")
            print(f"  Mean: {weight.mean().item():.6f}")
            print(f"  Std:  {weight.std().item():.6f}")
            print(f"  Range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            
            # Check for extreme weights
            extreme_weights = (torch.abs(weight) > 10).float().mean()
            print(f"  Extreme weights (|w| > 10): {extreme_weights.item()*100:.1f}%")
            
            if bias is not None:
                print(f"\nBias Analysis:")
                print(f"  Shape: {bias.shape}")
                print(f"  Values: {bias.cpu().numpy()}")
                
                # Check for extreme bias towards one class
                max_bias_idx = bias.argmax().item()
                min_bias_idx = bias.argmin().item()
                bias_range = bias.max().item() - bias.min().item()
                
                emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                               'anger', 'disgust', 'fear', 'contempt']
                
                print(f"  Highest bias: {emotion_names[max_bias_idx]} ({bias[max_bias_idx].item():.3f})")
                print(f"  Lowest bias:  {emotion_names[min_bias_idx]} ({bias[min_bias_idx].item():.3f})")
                print(f"  Bias range: {bias_range:.3f}")
                
                if bias_range > 5:
                    print(f"  ⚠️  EXTREME BIAS DETECTED! Range > 5")

def plot_logit_distribution(logits, save_path="logit_distribution.png"):
    """Plot logit distribution"""
    emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 
                    'anger', 'disgust', 'fear', 'contempt']
    
    plt.figure(figsize=(15, 10))
    
    # Plot logit distributions for each class
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.hist(logits[:, i], bins=30, alpha=0.7, density=True)
        plt.title(f'{emotion_names[i]}')
        plt.xlabel('Logit Value')
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Logit distribution plot saved to {save_path}")

def diagnose_gradient_flow(model, dataloader, device):
    """Check if gradients are flowing properly"""
    print("\n" + "="*60)
    print("DIAGNOSING GRADIENT FLOW")
    print("="*60)
    
    model.train()
    
    # Get one batch
    images, labels, _ = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if param_norm.item() < 1e-7:
                zero_grad_count += 1
                
    total_norm = total_norm ** (1. / 2)
    
    print(f"Gradient Analysis:")
    print(f"  Total gradient norm: {total_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")
    print(f"  Parameters with near-zero gradients: {zero_grad_count}")
    print(f"  Percentage near-zero: {zero_grad_count/param_count*100:.1f}%")
    
    if total_norm > 100:
        print("  ⚠️  EXPLODING GRADIENTS DETECTED!")
    elif total_norm < 1e-5:
        print("  ⚠️  VANISHING GRADIENTS DETECTED!")
    
    model.eval()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'best_effective_fer.pth'
    model = load_model(model_path, device)
    
    # Create test dataset
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FERPlusDataset(
        csv_file='./FERPlus-master/fer2013new.csv',
        fer2013_csv_file='./fer2013.csv',
        split='PrivateTest',
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Run diagnostics
    print("Starting comprehensive model diagnosis...")
    
    # 1. Analyze outputs
    logits, probs, predictions = diagnose_model_outputs(model, test_loader, device)
    
    # 2. Analyze weights
    diagnose_model_weights(model)
    
    # 3. Check gradient flow
    diagnose_gradient_flow(model, test_loader, device)
    
    # 4. Plot distributions
    plot_logit_distribution(logits)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("Check the output above for:")
    print("1. Extreme logit values (saturation)")
    print("2. Biased classifier weights/bias")
    print("3. Gradient flow issues")
    print("4. Prediction collapse to single class")

if __name__ == "__main__":
    main() 