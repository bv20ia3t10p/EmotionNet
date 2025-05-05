#!/bin/bash

# Clean up previous log file
if [ -f nohup.out ]; then
    echo "Removing existing nohup.out file..."
    rm nohup.out
fi

# Run training with nohup to keep it running in background
echo "Starting training with reduced batch size (150)..."
nohup ./train.sh &

# Print instructions
echo "Training started in background. Monitor progress with:"
echo "  tail -f nohup.out" 