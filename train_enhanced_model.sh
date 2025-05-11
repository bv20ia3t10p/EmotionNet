#!/bin/bash
# Train Enhanced Emotion Recognition Model with focus on angry class recognition

# Make script stop on first error
set -e

# Set GPU ID (modify as needed)
GPU_ID=0

# Ensure output directory exists
mkdir -p ./models

echo "Starting enhanced angry-focused ResEmoteNet training..."

python3 train_enhanced_model.py \
    --train_dir ./extracted/emotion/train \
    --test_dir ./extracted/emotion/test \
    --output_dir ./models \
    --backbone resnet18 \
    --dropout_rate 0.4 \
    --use_fpn \
    --use_landmarks \
    --use_contrastive \
    --contrastive_weight 0.3 \
    --batch_size 24 \
    --epochs 150 \
    --lr 0.003 \
    --warmup_epochs 15 \
    --weight_decay 0.001 \
    --optimizer adamw \
    --use_focal_loss \
    --use_weighted_sampling \
    --use_cosine_scheduler \
    --patience 25 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --seed 42

echo "Training complete! Model saved to ./models/best_enhanced_model.pth" 