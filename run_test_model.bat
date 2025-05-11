@echo off
echo Testing high-performance emotion recognition model...

python test_high_accuracy_model.py ^
  --model_path ./models/high_accuracy_model.pth ^
  --test_dir ./extracted/emotion/test ^
  --output_dir ./results

echo Testing completed!
pause 