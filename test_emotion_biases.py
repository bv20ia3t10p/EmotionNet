#!/usr/bin/env python3
# Test script for different bias settings in emotion recognition

import os
import cv2
import numpy as np
import argparse
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

def test_bias_combinations(image_path, model_path='./models/hybrid_model.pth', output_dir='bias_test_results'):
    """Test different combinations of angry and neutral bias settings"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).rsplit('.', 1)[0]
    
    # Bias combinations to test
    angry_bias_values = [1.0, 2.0, 4.0, 6.0, 8.0]  # 1.0 = no bias
    neutral_bias_values = [1.0, 2.0, 3.0, 4.0]  # 1.0 = no bias
    
    results = []
    
    # Run the prediction with different bias settings
    print(f"Testing {len(angry_bias_values) * len(neutral_bias_values)} different bias combinations...")
    for angry_bias in tqdm(angry_bias_values, desc="Testing angry bias"):
        for neutral_bias in neutral_bias_values:
            # Create output filenames
            bias_str = f"a{angry_bias:.1f}_n{neutral_bias:.1f}"
            output_image = os.path.join(output_dir, f"{base_name}_{bias_str}.jpg")
            output_plot = os.path.join(output_dir, f"{base_name}_{bias_str}_plot.png")
            
            # Build and run the command
            cmd = [
                "python3", "predict_emotion.py",
                "--image", image_path,
                "--model", model_path,
                "--angry_bias", str(angry_bias),
                "--neutral_bias", str(neutral_bias),
                "--output", output_image,
                "--plot", output_plot
            ]
            
            # Run the command and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            # Parse the output to get prediction
            output = stdout.decode('utf-8')
            
            # Extract the prediction
            for line in output.splitlines():
                if line.startswith("Face") or line.startswith("Prediction"):
                    # Extract emotion and confidence
                    parts = line.split(": ")[1].split(" (")
                    emotion = parts[0]
                    confidence = float(parts[1].rstrip("%)"))
                    
                    results.append({
                        'angry_bias': angry_bias,
                        'neutral_bias': neutral_bias,
                        'emotion': emotion,
                        'confidence': confidence,
                        'output_image': output_image,
                        'output_plot': output_plot
                    })
                    break  # Take the first prediction
    
    # Create a summary visualization
    create_summary_plot(results, base_name, output_dir)
    
    return results

def create_summary_plot(results, base_name, output_dir):
    """Create a summary visualization of the bias results"""
    # Group results by predicted emotion
    emotions = {}
    for result in results:
        emotion = result['emotion']
        if emotion not in emotions:
            emotions[emotion] = []
        emotions[emotion].append(result)
    
    # Create a grid with angry_bias on x-axis and neutral_bias on y-axis
    angry_biases = sorted(set([r['angry_bias'] for r in results]))
    neutral_biases = sorted(set([r['neutral_bias'] for r in results]))
    
    # Create a figure with size proportional to the grid
    fig, ax = plt.subplots(figsize=(len(angry_biases)*2 + 2, len(neutral_biases)*1.5 + 2))
    
    # Dict to map emotions to colors
    emotion_colors = {
        'angry': 'red',
        'disgust': 'orange',
        'fear': 'yellow',
        'happy': 'green',
        'sad': 'blue',
        'surprise': 'purple',
        'neutral': 'gray'
    }
    
    # Create a grid showing predicted emotions for each bias combination
    grid = np.zeros((len(neutral_biases), len(angry_biases)), dtype=object)
    conf_grid = np.zeros((len(neutral_biases), len(angry_biases)))
    
    for result in results:
        x_idx = angry_biases.index(result['angry_bias'])
        y_idx = neutral_biases.index(result['neutral_bias'])
        grid[y_idx, x_idx] = result['emotion']
        conf_grid[y_idx, x_idx] = result['confidence']
    
    # Plot the grid
    for i in range(len(neutral_biases)):
        for j in range(len(angry_biases)):
            emotion = grid[i, j]
            confidence = conf_grid[i, j]
            
            # Get color for the emotion
            color = emotion_colors.get(emotion, 'black')
            
            # Plot a colored rectangle with emotion and confidence
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, alpha=min(confidence/100.0, 0.8))
            ax.add_patch(rect)
            
            # Add text with emotion and confidence
            ax.text(j + 0.5, i + 0.5, f"{emotion}\n{confidence:.1f}%", 
                    ha='center', va='center', 
                    color='white' if color in ['blue', 'purple', 'red'] else 'black',
                    fontweight='bold')
    
    # Set axis labels and limits
    ax.set_xticks([i + 0.5 for i in range(len(angry_biases))])
    ax.set_xticklabels(angry_biases)
    ax.set_yticks([i + 0.5 for i in range(len(neutral_biases))])
    ax.set_yticklabels(neutral_biases)
    
    ax.set_xlim(0, len(angry_biases))
    ax.set_ylim(0, len(neutral_biases))
    
    ax.set_xlabel('Angry Bias')
    ax.set_ylabel('Neutral Bias')
    
    plt.title(f'Predicted Emotions with Different Bias Settings\n{base_name}')
    plt.tight_layout()
    
    # Save the plot
    summary_path = os.path.join(output_dir, f"{base_name}_bias_summary.png")
    plt.savefig(summary_path, dpi=100)
    print(f"Summary visualization saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Test different bias settings for emotion recognition')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the input image')
    parser.add_argument('--model', type=str, default='./models/hybrid_model.pth',
                        help='Path to the model weights')
    parser.add_argument('--output_dir', type=str, default='bias_test_results',
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Run tests with different bias combinations
    results = test_bias_combinations(args.image, args.model, args.output_dir)
    
    print(f"Testing complete. {len(results)} bias combinations tested.")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 