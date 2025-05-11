Write-Host "Starting emotion recognition training with increased batch size and without bias correction..." -ForegroundColor Green

python integrate_bias_correction.py `
  --batch_size 64 `
  --no_bias_correction `
  --learning_rate 0.0005 `
  --backbone "resnet18" `
  --epochs 30 `
  --patience 10

Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null 