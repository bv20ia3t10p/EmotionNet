@echo off
echo Starting emotion recognition training with larger batch size and without bias correction...

python integrate_bias_correction.py ^
  --batch_size 64 ^
  --no_bias_correction ^
  --learning_rate 0.0005 ^
  --epochs 30 ^
  --patience 10

echo Training completed!
pause 