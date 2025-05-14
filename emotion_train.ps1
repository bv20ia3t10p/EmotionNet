# PowerShell script for training emotion recognition model on Windows
# Usage: .\emotion_train.ps1 [fer2013|rafdb]

param (
    [string]$dataset = "fer2013"
)

# Set PYTHONPATH environment variable
$env:PYTHONPATH = "."

# Configuration
$DATA_DIR = "./test_dataset/$dataset"
$MODEL_DIR = "./models/${dataset}_sota_stable"
$BATCH_SIZE = 16
$EPOCHS = 50  # More epochs with early stopping
$LEARNING_RATE = 0.000008  # Significantly reduced learning rate
$IMAGE_SIZE = 256
$BACKBONE = "resnet34"
$ARCHITECTURE = "sota_resemote_medium"
$EMBEDDING_SIZE = 512
$WEIGHT_DECAY = 0.00005  # Increased weight decay for stability

# Create model directory
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null

# Print info
Write-Host "Starting training for $dataset dataset using SOTA ResEmote model with STABLE settings..."
Write-Host "Using architecture: $ARCHITECTURE with backbone: $BACKBONE"
Write-Host "Learning rate: $LEARNING_RATE (reduced for stability)"
Write-Host "Weight decay: $WEIGHT_DECAY"
Write-Host "Data directory: $DATA_DIR"
Write-Host "Model directory: $MODEL_DIR"

# Run training with stability-focused settings
python emotion_net/train.py `
    --dataset_name $dataset `
    --data_dir $DATA_DIR `
    --model_dir $MODEL_DIR `
    --num_epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --image_size $IMAGE_SIZE `
    --backbones $BACKBONE `
    --loss_type "cross_entropy" `  # Use simpler loss initially
    --label_smoothing 0.2 `  # Increased smoothing
    --scheduler_type "cosine_annealing" `
    --architecture $ARCHITECTURE `
    --embedding_size $EMBEDDING_SIZE `
    --optimizer "adamw" `  # Switch to AdamW
    --weight_decay $WEIGHT_DECAY `
    --class_weights `
    --gem_pooling `
    --gradient_clip 0.1 `  # Add aggressive gradient clipping
    --warmup_epochs 5 `  # Add warmup
    --drop_path_rate 0.05 `  # Add stochastic depth
    --pretrained

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully!"
    Write-Host "Model saved to $MODEL_DIR"
    
    # Run evaluation
    Write-Host "Running evaluation..."
    python emotion_net/evaluate.py `
        --dataset_name $dataset `
        --data_dir $DATA_DIR `
        --model_path "$MODEL_DIR/best_model.pth" `
        --architecture $ARCHITECTURE `
        --image_size $IMAGE_SIZE `
        --batch_size 32
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Evaluation completed successfully!"
    } else {
        Write-Host "Evaluation failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE"
} 