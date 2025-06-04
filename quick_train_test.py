#!/usr/bin/env python3
"""
Quick training test for GReFEL implementation
This script runs a short training session to verify everything works
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm

from grefel_implementation import GReFELModel, GReFELLoss
from train_grefel import FacialExpressionDataset, DataAugmentation

def quick_training_test():
    """Run a quick training test to verify the implementation"""
    
    print("🚀 Starting GReFEL Quick Training Test")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_classes': 8,
        'batch_size': 8,
        'num_epochs': 3,  # Just a few epochs for testing
        'embed_dim': 256,  # Smaller for faster testing
        'num_heads': 8,
        'depth': 3,  # Fewer layers for testing
        'num_anchors_per_class': 5,  # Fewer anchors for testing
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Device: {config['device']}")
    print(f"Configuration: {config}")
    
    # Check if sample dataset exists
    csv_path = 'sample_dataset/fer_sample.csv'
    img_dir = 'sample_dataset/images'
    
    if not os.path.exists(csv_path):
        print(f"❌ Sample dataset not found at {csv_path}")
        print("Please run: python create_sample_dataset.py")
        return False
    
    # Load dataset
    print("\n📁 Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Split into train and test
    train_df = df[df['Usage'] == 'Training'].copy()
    test_df = df[df['Usage'] == 'PublicTest'].copy()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save temporary splits
    train_df.to_csv('temp_train.csv', index=False)
    test_df.to_csv('temp_test.csv', index=False)
    
    # Create datasets
    print("\n🏗️ Creating datasets...")
    train_dataset = FacialExpressionDataset(
        csv_file='temp_train.csv',
        img_dir=img_dir,
        transform=DataAugmentation.get_train_transforms()
    )
    
    test_dataset = FacialExpressionDataset(
        csv_file='temp_test.csv',
        img_dir=img_dir,
        transform=DataAugmentation.get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    print("\n🧠 Creating GReFEL model...")
    model = GReFELModel(
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        depth=config['depth'],
        num_anchors_per_class=config['num_anchors_per_class']
    ).to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Loss and optimizer
    criterion = GReFELLoss(lambda_cls=1.0, lambda_anchor=0.1, lambda_center=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Training loop
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    
    model.train()
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(config['device'])
                landmarks = batch['landmarks'].to(config['device'])
                labels = batch['label'].to(config['device'])
                
                # Forward pass
                outputs = model(images, landmarks)
                losses = criterion(outputs, labels, model)
                
                # Backward pass
                optimizer.zero_grad()
                losses['total_loss'].backward()
                optimizer.step()
                
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Cls': f"{losses['cls_loss'].item():.4f}"
                })
                
                # Break early for quick test
                if batch_idx >= 5:  # Only process a few batches
                    break
                    
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                return False
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
    
    # Quick evaluation
    print("\n📊 Running quick evaluation...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                images = batch['image'].to(config['device'])
                landmarks = batch['landmarks'].to(config['device'])
                labels = batch['label'].to(config['device'])
                
                outputs = model(images, landmarks)
                predictions = torch.argmax(outputs['final_probs'], dim=-1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
                # Break early for quick test
                if batch_idx >= 2:
                    break
                    
            except Exception as e:
                print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                return False
    
    accuracy = correct / total if total > 0 else 0
    print(f"Quick test accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save model
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'accuracy': accuracy
    }, 'grefel_quick_test.pth')
    
    # Cleanup
    if os.path.exists('temp_train.csv'):
        os.remove('temp_train.csv')
    if os.path.exists('temp_test.csv'):
        os.remove('temp_test.csv')
    
    print("\n✅ Quick training test completed successfully!")
    print("=" * 60)
    print("🎉 GReFEL implementation is working correctly!")
    
    return True

def test_single_forward_pass():
    """Test a single forward pass with dummy data"""
    print("\n🔍 Testing single forward pass...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GReFELModel(num_classes=8, embed_dim=256).to(device)
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    landmarks = torch.randn(batch_size, 68, 2).to(device)
    labels = torch.randint(0, 8, (batch_size,)).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images, landmarks)
    
    print(f"Forward pass successful!")
    print(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    return True

if __name__ == "__main__":
    print("🎭 GReFEL Implementation Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Single forward pass
        if not test_single_forward_pass():
            exit(1)
        
        # Test 2: Quick training
        if not quick_training_test():
            exit(1)
            
        print("\n🏆 All tests passed! GReFEL is ready for use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 