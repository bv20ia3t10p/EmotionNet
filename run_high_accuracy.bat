@echo off
echo Starting high-performance emotion recognition training with improved augmentations...

REM Install required packages if not already installed
pip install timm albumentations opencv-python

python high_accuracy_training.py ^
  --batch_size 16 ^
  --image_size 224 ^
  --epochs 50 ^
  --patience 15 ^
  --learning_rate 0.0001 ^
  --class_weights_boost 2.0 ^
  --backbones efficientnet_b0 resnet18

echo Training completed!
pause 