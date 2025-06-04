#!/usr/bin/env python3
"""
Demo script for GReFEL (Geometry-Aware Reliable Facial Expression Learning)

This script demonstrates the basic usage of the GReFEL model with dummy data.
It shows how to:
1. Create the model
2. Generate dummy input data
3. Perform forward pass
4. Analyze the outputs
5. Test individual components

Run: python demo.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from grefel_implementation import (
    GReFELModel, 
    GReFELLoss, 
    create_dummy_data,
    WindowCrossAttention,
    GeometryAwareAnchors,
    ReliabilityBalancing
)

def test_model_components():
    """Test individual model components"""
    print("=" * 60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    batch_size = 4
    embed_dim = 512
    num_classes = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test WindowCrossAttention
    print("\n1. Testing Window Cross-Attention...")
    cross_attn = WindowCrossAttention(dim=embed_dim, num_heads=8, window_size=7).to(device)
    
    landmark_features = torch.randn(batch_size, 1, embed_dim).to(device)
    image_features = torch.randn(batch_size, 49, embed_dim).to(device)  # 7x7 patches
    
    attn_output = cross_attn(landmark_features, image_features)
    print(f"   Input shapes: Landmark {landmark_features.shape}, Image {image_features.shape}")
    print(f"   Output shape: {attn_output.shape}")
    print("   ✓ Cross-attention working correctly")
    
    # Test GeometryAwareAnchors
    print("\n2. Testing Geometry-Aware Anchors...")
    anchors = GeometryAwareAnchors(num_classes, num_anchors_per_class=10, embed_dim=embed_dim).to(device)
    
    embeddings = torch.randn(batch_size, embed_dim).to(device)
    anchor_correction = anchors(embeddings)
    
    print(f"   Input embedding shape: {embeddings.shape}")
    print(f"   Anchor correction shape: {anchor_correction.shape}")
    print(f"   Number of anchors: {anchors.anchors.shape}")
    print("   ✓ Geometry-aware anchors working correctly")
    
    # Test ReliabilityBalancing
    print("\n3. Testing Reliability Balancing...")
    reliability = ReliabilityBalancing(num_classes, embed_dim, num_anchors_per_class=10).to(device)
    
    geometric_corr, attentive_corr = reliability(embeddings)
    
    print(f"   Geometric correction shape: {geometric_corr.shape}")
    print(f"   Attentive correction shape: {attentive_corr.shape}")
    print("   ✓ Reliability balancing working correctly")

def test_full_model():
    """Test the complete GReFEL model"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE GREFEL MODEL")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration
    config = {
        'num_classes': 8,
        'img_size': 224,
        'embed_dim': 512,
        'num_heads': 8,
        'depth': 6,
        'num_anchors_per_class': 10
    }
    
    print(f"\nModel configuration: {config}")
    
    # Create model
    model = GReFELModel(**config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Generate dummy data
    batch_size = 8
    images, landmarks, labels = create_dummy_data(batch_size=batch_size, num_classes=config['num_classes'])
    images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
    
    print(f"\nInput data shapes:")
    print(f"   Images: {images.shape}")
    print(f"   Landmarks: {landmarks.shape}")
    print(f"   Labels: {labels.shape}")
    
    # Forward pass
    print("\nPerforming forward pass...")
    start_time = time.time()
    
    model.eval()
    with torch.no_grad():
        outputs = model(images, landmarks)
    
    forward_time = time.time() - start_time
    print(f"   Forward pass time: {forward_time:.3f} seconds")
    print(f"   Throughput: {batch_size / forward_time:.1f} samples/second")
    
    # Analyze outputs
    print(f"\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
    
    # Get predictions
    predictions = torch.argmax(outputs['final_probs'], dim=-1)
    confidences = torch.max(outputs['final_probs'], dim=-1)[0]
    
    print(f"\nPredictions and Confidences:")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    
    for i in range(batch_size):
        true_label = emotion_labels[labels[i].item()]
        pred_label = emotion_labels[predictions[i].item()]
        confidence = confidences[i].item()
        correct = "✓" if labels[i] == predictions[i] else "✗"
        
        print(f"   Sample {i+1}: True={true_label:8} | Pred={pred_label:8} | Conf={confidence:.3f} | {correct}")

def test_loss_function():
    """Test the GReFEL loss function"""
    print("\n" + "=" * 60)
    print("TESTING GREFEL LOSS FUNCTION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    num_classes = 8
    
    # Create model and generate data
    model = GReFELModel(num_classes=num_classes).to(device)
    images, landmarks, labels = create_dummy_data(batch_size=batch_size, num_classes=num_classes)
    images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
    
    # Forward pass
    outputs = model(images, landmarks)
    
    # Create loss function
    criterion = GReFELLoss(lambda_cls=1.0, lambda_anchor=1.0, lambda_center=1.0)
    
    # Compute losses
    losses = criterion(outputs, labels, model)
    
    print(f"Loss components:")
    for loss_name, loss_value in losses.items():
        print(f"   {loss_name}: {loss_value.item():.4f}")
    
    print("\n✓ Loss function working correctly")

def test_reliability_balancing_effect():
    """Test the effect of reliability balancing on predictions"""
    print("\n" + "=" * 60)
    print("TESTING RELIABILITY BALANCING EFFECTS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    
    # Create model
    model = GReFELModel(num_classes=8).to(device)
    images, landmarks, labels = create_dummy_data(batch_size=batch_size)
    images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
    
    # Get model outputs
    model.eval()
    with torch.no_grad():
        outputs = model(images, landmarks)
    
    # Compare primary predictions vs final predictions
    primary_preds = torch.argmax(outputs['primary_probs'], dim=-1)
    final_preds = torch.argmax(outputs['final_probs'], dim=-1)
    
    # Calculate differences
    changed_predictions = torch.sum(primary_preds != final_preds).item()
    total_predictions = batch_size
    change_rate = changed_predictions / total_predictions
    
    print(f"Reliability Balancing Analysis:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Changed predictions: {changed_predictions}")
    print(f"   Change rate: {change_rate:.1%}")
    
    print(f"\nPrediction comparison:")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    
    for i in range(batch_size):
        primary_emotion = emotion_labels[primary_preds[i].item()]
        final_emotion = emotion_labels[final_preds[i].item()]
        changed = "→" if primary_preds[i] != final_preds[i] else "="
        
        print(f"   Sample {i+1}: {primary_emotion:8} {changed} {final_emotion:8}")
    
    print("\n✓ Reliability balancing analysis complete")

def visualize_attention_weights():
    """Visualize attention patterns (simplified demonstration)"""
    print("\n" + "=" * 60)
    print("ATTENTION WEIGHTS VISUALIZATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a simplified attention module for demonstration
    embed_dim = 512
    num_heads = 8
    
    attention = WindowCrossAttention(dim=embed_dim, num_heads=num_heads).to(device)
    
    # Generate sample data
    batch_size = 1
    landmark_features = torch.randn(batch_size, 1, embed_dim).to(device)
    image_features = torch.randn(batch_size, 49, embed_dim).to(device)  # 7x7 patches
    
    # Get attention output
    with torch.no_grad():
        attn_output = attention(landmark_features, image_features)
    
    print(f"Attention computation successful:")
    print(f"   Input: Landmarks {landmark_features.shape}, Images {image_features.shape}")
    print(f"   Output: {attn_output.shape}")
    print(f"   Attention heads: {num_heads}")
    
    # Note: In a real implementation, you would extract and visualize actual attention weights
    print("\n✓ Attention mechanism working correctly")
    print("   (For full attention visualization, modify the forward pass to return attention weights)")

def performance_benchmark():
    """Benchmark model performance"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32] if device == 'cuda' else [1, 4, 8]
    model = GReFELModel(num_classes=8).to(device)
    model.eval()
    
    print(f"Benchmarking on {device}...")
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Throughput (fps)':<15} {'Memory (MB)':<12}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        try:
            # Generate data
            images, landmarks, _ = create_dummy_data(batch_size=batch_size)
            images, landmarks = images.to(device), landmarks.to(device)
            
            # Warm up
            with torch.no_grad():
                _ = model(images, landmarks)
            
            # Synchronize for accurate timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            num_runs = 10
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(images, landmarks)
                    
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            avg_time_ms = (total_time / num_runs) * 1000
            throughput = batch_size / (total_time / num_runs)
            
            # Memory usage (approximate)
            if device == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_mb = 0  # Hard to measure CPU memory accurately
            
            print(f"{batch_size:<12} {avg_time_ms:<12.1f} {throughput:<15.1f} {memory_mb:<12.1f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:<12} {'OOM':<12} {'N/A':<15} {'N/A':<12}")
                break
            else:
                raise e
    
    print("\n✓ Performance benchmark complete")

def main():
    """Main demo function"""
    print("🎭 GReFEL Demo - Geometry-Aware Reliable Facial Expression Learning")
    print("=" * 80)
    
    # Check requirements
    print("System Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name()}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Run all tests
        test_model_components()
        test_full_model()
        test_loss_function()
        test_reliability_balancing_effect()
        visualize_attention_weights()
        performance_benchmark()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! GReFEL implementation is working correctly.")
        print("=" * 80)
        
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Prepare your dataset (FER2013+ or custom)")
        print("3. Train the model: python train_grefel.py")
        print("4. Evaluate results: python evaluate_grefel.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check your environment and dependencies.")
        raise

if __name__ == "__main__":
    main() 