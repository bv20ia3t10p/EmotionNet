#!/usr/bin/env python3
"""
Test script for Enhanced GReFEL model
"""

import torch
import numpy as np
from models.grefel import GReFEL, EnhancedGReFELLoss

def test_model_forward():
    """Test model forward pass"""
    print("🧪 Testing Enhanced GReFEL forward pass...")
    
    # Create model
    model = GReFEL(num_classes=8, feature_dim=768)
    model.eval()
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    try:
        with torch.no_grad():
            outputs = model(x)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Reliability shape: {outputs['reliability'].shape}")
        print(f"   Reliability score shape: {outputs['reliability_score'].shape}")
        print(f"   Geometry loss: {outputs['geo_loss'].item():.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_loss_function():
    """Test enhanced loss function"""
    print("\n🧪 Testing Enhanced GReFEL Loss...")
    
    # Create loss function
    criterion = EnhancedGReFELLoss(num_classes=8, label_smoothing=0.11)
    
    # Create dummy outputs
    batch_size = 4
    outputs = {
        'logits': torch.randn(batch_size, 8),
        'reliability': torch.randn(batch_size, 768),
        'reliability_score': torch.sigmoid(torch.randn(batch_size)),
        'geo_loss': torch.tensor(0.01),
        'features': torch.randn(batch_size, 768)
    }
    
    # Test with hard labels
    print("   Testing with hard labels...")
    hard_targets = torch.randint(0, 8, (batch_size,))
    
    try:
        loss_dict = criterion(outputs, hard_targets, is_soft_labels=False)
        print(f"   ✅ Hard labels: total_loss={loss_dict['total_loss'].item():.4f}")
        print(f"      Classification: {loss_dict['classification_loss'].item():.4f}")
        print(f"      Geometry: {loss_dict['geometry_loss'].item():.6f}")
        print(f"      Reliability: {loss_dict['reliability_loss'].item():.4f}")
    except Exception as e:
        print(f"   ❌ Hard labels failed: {e}")
        return False
    
    # Test with soft labels
    print("   Testing with soft labels...")
    soft_targets = torch.softmax(torch.randn(batch_size, 8), dim=1)
    
    try:
        loss_dict = criterion(outputs, soft_targets, is_soft_labels=True)
        print(f"   ✅ Soft labels: total_loss={loss_dict['total_loss'].item():.4f}")
        print(f"      Classification: {loss_dict['classification_loss'].item():.4f}")
        print(f"      Geometry: {loss_dict['geometry_loss'].item():.6f}")
        print(f"      Reliability: {loss_dict['reliability_loss'].item():.4f}")
    except Exception as e:
        print(f"   ❌ Soft labels failed: {e}")
        return False
    
    return True

def test_training_step():
    """Test a complete training step"""
    print("\n🧪 Testing complete training step...")
    
    # Create model and loss
    model = GReFEL(num_classes=8, feature_dim=768)
    criterion = EnhancedGReFELLoss(num_classes=8, label_smoothing=0.11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy data
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    targets = torch.softmax(torch.randn(batch_size, 8), dim=1)  # Soft labels
    
    try:
        # Forward pass
        model.train()
        outputs = model(images)
        
        # Loss computation
        loss_dict = criterion(outputs, targets, is_soft_labels=True)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if not has_grad:
            print("   ❌ No gradients computed")
            return False
        
        # Optimizer step
        optimizer.step()
        
        print(f"   ✅ Training step successful!")
        print(f"      Loss: {loss.item():.4f}")
        print(f"      Gradients: Present")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced GReFEL Test Suite")
    print("=" * 50)
    
    tests = [
        test_model_forward,
        test_loss_function,
        test_training_step
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced GReFEL is ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 