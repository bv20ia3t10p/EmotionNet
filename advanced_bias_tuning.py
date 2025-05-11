#!/usr/bin/env python3
# Advanced bias parameter tuning for facial emotion recognition

import os
import cv2
import numpy as np
import argparse
import subprocess
import multiprocessing
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import itertools
import optuna
from scipy.optimize import differential_evolution

from enhanced_resemotenet import EnhancedResEmoteNet
from train_enhanced_model import extract_features, load_hybrid_model
from predict_emotion import EMOTIONS, preprocess_image, predict_emotion

class BiasOptimizer:
    """Class for optimizing bias parameters for facial emotion recognition"""
    
    def __init__(self, 
                 image_dir,
                 model_path='./models/hybrid_model.pth',
                 output_dir='bias_optimization_results',
                 target_distribution=None,
                 use_ground_truth=False,
                 ground_truth_file=None,
                 optimization_method='grid_search',
                 num_trials=100,
                 parallel_jobs=4,
                 custom_classes=None):
        
        self.image_dir = image_dir
        self.model_path = model_path
        self.output_dir = output_dir
        self.optimization_method = optimization_method
        self.num_trials = num_trials
        self.parallel_jobs = parallel_jobs
        self.use_ground_truth = use_ground_truth
        self.ground_truth_file = ground_truth_file
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Classes to focus on for optimization
        self.custom_classes = custom_classes or [0, 6]  # Default: focus on angry and neutral
        
        # Set target distribution (ideal class prediction ratios)
        self.target_distribution = target_distribution or {
            'angry': 0.15,     # Target ~15% angry predictions
            'disgust': 0.05,   # Target ~5% disgust predictions  
            'fear': 0.10,      # Target ~10% fear predictions
            'happy': 0.25,     # Target ~25% happy predictions
            'sad': 0.20,       # Target ~20% sad predictions
            'surprise': 0.10,  # Target ~10% surprise predictions
            'neutral': 0.15    # Target ~15% neutral predictions
        }
        
        # For tracking the best parameters
        self.best_params = None
        self.best_score = float('-inf')
        self.best_distribution = None
        self.results = []
        
        # Load ground truth annotations if available
        self.ground_truth = {}
        if use_ground_truth and ground_truth_file:
            self._load_ground_truth(ground_truth_file)
        
        # List all images for testing
        self.image_files = self._get_image_files()
        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")
        print(f"Found {len(self.image_files)} images for optimization")
    
    def _get_image_files(self):
        """Get all image files in the directory"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(self.image_dir).glob(ext)))
        return [str(p) for p in image_files]
    
    def _load_ground_truth(self, ground_truth_file):
        """Load ground truth annotations from file"""
        try:
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded ground truth data for {len(self.ground_truth)} images")
        except Exception as e:
            print(f"Warning: Failed to load ground truth file: {e}")
            self.use_ground_truth = False
    
    def _evaluate_bias_setting(self, angry_bias, neutral_bias):
        """Evaluate a specific bias setting by running the model on test images"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        if self.model_path.endswith('hybrid_model.pth'):
            model = load_hybrid_model(self.model_path, device)
        else:
            model = EnhancedResEmoteNet(num_classes=7, backbone='resnet18', use_fpn=True, use_landmarks=True)
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model = model.to(device)
        
        # Set bias parameters
        model.angry_bias = angry_bias
        model.neutral_bias = neutral_bias
        
        # Track predictions
        all_preds = []
        correct_preds = 0
        total_tested = 0
        
        # Process each image
        for img_path in self.image_files[:min(100, len(self.image_files))]:  # Limit to 100 images for speed
            tensor, _ = preprocess_image(img_path)
            emotion, confidence, all_confidences, _ = predict_emotion(model, tensor, device, use_hybrid=True)
            
            # Store prediction
            all_preds.append(emotion)
            
            # Check against ground truth if available
            if self.use_ground_truth:
                img_name = os.path.basename(img_path)
                if img_name in self.ground_truth:
                    true_emotion = self.ground_truth[img_name]
                    if emotion == true_emotion:
                        correct_preds += 1
                    total_tested += 1
        
        # Calculate distribution of predictions
        pred_distribution = {emotion: 0 for emotion in EMOTIONS.values()}
        for pred in all_preds:
            pred_distribution[pred] += 1
        
        # Convert to percentages
        total = len(all_preds)
        for emotion in pred_distribution:
            pred_distribution[emotion] = pred_distribution[emotion] / total
        
        # Calculate accuracy if ground truth is available
        accuracy = correct_preds / total_tested if total_tested > 0 else 0
        
        # Calculate score based on how close the distribution is to target
        distribution_score = 0
        for emotion, target_pct in self.target_distribution.items():
            actual_pct = pred_distribution[emotion]
            # Smaller penalty for being over target for underrepresented classes
            if emotion in ['angry', 'neutral', 'disgust'] and actual_pct > target_pct:
                penalty = 0.5 * abs(actual_pct - target_pct)
            else:
                penalty = abs(actual_pct - target_pct)
            distribution_score -= penalty
        
        # Calculate a special score for focus classes (angry, neutral)
        focus_score = 0
        for class_idx in self.custom_classes:
            emotion = EMOTIONS[class_idx]
            target = self.target_distribution[emotion]
            actual = pred_distribution[emotion]
            
            # We want these classes to be recognized, so heavily penalize if below target
            if actual < target:
                focus_score -= 3.0 * (target - actual)
            else:
                # Small bonus for being slightly above target
                focus_score += 0.5 * min(0.05, actual - target)
        
        # Combine scores - accuracy matters more if ground truth is available
        if self.use_ground_truth:
            final_score = (0.6 * accuracy) + (0.3 * distribution_score) + (0.1 * focus_score)
        else:
            final_score = (0.7 * distribution_score) + (0.3 * focus_score)
        
        result = {
            'angry_bias': angry_bias,
            'neutral_bias': neutral_bias,
            'distribution': pred_distribution,
            'accuracy': accuracy if self.use_ground_truth else None,
            'distribution_score': distribution_score,
            'focus_score': focus_score,
            'final_score': final_score
        }
        
        self.results.append(result)
        
        # Update best parameters if this setting is better
        if final_score > self.best_score:
            self.best_score = final_score
            self.best_params = (angry_bias, neutral_bias)
            self.best_distribution = pred_distribution
        
        return final_score
    
    def _optuna_objective(self, trial):
        """Objective function for Optuna optimization"""
        angry_bias = trial.suggest_float('angry_bias', 1.0, 10.0)
        neutral_bias = trial.suggest_float('neutral_bias', 1.0, 6.0)
        
        return self._evaluate_bias_setting(angry_bias, neutral_bias)
    
    def _differential_evolution_objective(self, params):
        """Objective function for differential evolution"""
        angry_bias, neutral_bias = params
        
        # Invert the score because differential_evolution minimizes
        return -self._evaluate_bias_setting(angry_bias, neutral_bias)
    
    def grid_search(self):
        """Perform grid search to find optimal bias parameters"""
        print("Performing grid search for optimal bias parameters...")
        
        # Define parameter grids
        angry_bias_values = np.linspace(1.0, 10.0, 10)  # 1.0 to 10.0
        neutral_bias_values = np.linspace(1.0, 6.0, 6)  # 1.0 to 6.0
        
        # Generate parameter combinations
        param_combinations = list(itertools.product(angry_bias_values, neutral_bias_values))
        
        # Run evaluations in parallel
        if self.parallel_jobs > 1:
            with multiprocessing.Pool(self.parallel_jobs) as pool:
                results = list(tqdm(
                    pool.starmap(
                        self._evaluate_bias_setting, 
                        param_combinations
                    ),
                    total=len(param_combinations)
                ))
        else:
            # Run sequentially
            for angry_bias, neutral_bias in tqdm(param_combinations):
                self._evaluate_bias_setting(angry_bias, neutral_bias)
        
        # Results are already stored in self.results during evaluation
        print(f"Grid search complete. Best parameters: angry_bias={self.best_params[0]:.2f}, neutral_bias={self.best_params[1]:.2f}")
        print(f"Best score: {self.best_score:.4f}")
        
        return self.best_params
    
    def optuna_search(self):
        """Use Optuna for Bayesian optimization of bias parameters"""
        print("Performing Bayesian optimization with Optuna...")
        
        # Create a study object and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(self._optuna_objective, n_trials=self.num_trials)
        
        # Get best parameters
        best_params = study.best_params
        self.best_params = (best_params['angry_bias'], best_params['neutral_bias'])
        
        print(f"Optuna search complete. Best parameters: angry_bias={self.best_params[0]:.2f}, neutral_bias={self.best_params[1]:.2f}")
        print(f"Best score: {self.best_score:.4f}")
        
        return self.best_params
    
    def evolutionary_search(self):
        """Use differential evolution to find optimal parameters"""
        print("Performing differential evolution optimization...")
        
        # Define parameter bounds
        bounds = [(1.0, 10.0), (1.0, 6.0)]  # (angry_bias, neutral_bias)
        
        # Run differential evolution
        result = differential_evolution(
            self._differential_evolution_objective,
            bounds,
            popsize=10,
            maxiter=self.num_trials // 10,
            disp=True,
            workers=self.parallel_jobs if self.parallel_jobs > 1 else 1
        )
        
        # Results are already stored in self.results during evaluation
        print(f"Evolutionary search complete. Best parameters: angry_bias={self.best_params[0]:.2f}, neutral_bias={self.best_params[1]:.2f}")
        print(f"Best score: {self.best_score:.4f}")
        
        return self.best_params
    
    def run_optimization(self):
        """Run the selected optimization method"""
        start_time = time.time()
        
        if self.optimization_method == 'grid_search':
            self.grid_search()
        elif self.optimization_method == 'optuna':
            self.optuna_search()
        elif self.optimization_method == 'evolutionary':
            self.evolutionary_search()
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
        
        # Save results
        self.save_results()
        
        # Display results
        self.visualize_results()
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        return self.best_params, self.best_score, self.best_distribution
    
    def save_results(self):
        """Save optimization results to files"""
        # Save all results to JSON
        results_file = os.path.join(self.output_dir, "optimization_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'best_params': {
                    'angry_bias': self.best_params[0],
                    'neutral_bias': self.best_params[1]
                },
                'best_score': self.best_score,
                'best_distribution': self.best_distribution,
                'all_results': self.results
            }, f, indent=4)
        
        print(f"Results saved to {results_file}")
        
        # Save best parameters to a separate file for easy access
        best_params_file = os.path.join(self.output_dir, "best_bias_params.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                'angry_bias': self.best_params[0],
                'neutral_bias': self.best_params[1]
            }, f, indent=4)
        
        print(f"Best parameters saved to {best_params_file}")
    
    def visualize_results(self):
        """Create visualizations of the optimization results"""
        # Filter results for plotting (use a subset if there are too many)
        plot_results = self.results
        if len(plot_results) > 100:
            plot_results = sorted(plot_results, key=lambda x: x['final_score'], reverse=True)[:100]
        
        # Extract data for plotting
        angry_biases = [r['angry_bias'] for r in plot_results]
        neutral_biases = [r['neutral_bias'] for r in plot_results]
        scores = [r['final_score'] for r in plot_results]
        
        # Create a scatter plot of parameter space
        plt.figure(figsize=(10, 8))
        plt.scatter(angry_biases, neutral_biases, c=scores, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Score')
        plt.xlabel('Angry Bias')
        plt.ylabel('Neutral Bias')
        plt.title('Parameter Space Exploration')
        
        # Mark the best parameters
        plt.scatter([self.best_params[0]], [self.best_params[1]], c='red', s=100, marker='*')
        plt.annotate(f'Best: ({self.best_params[0]:.2f}, {self.best_params[1]:.2f})',
                    (self.best_params[0], self.best_params[1]),
                    xytext=(10, 10), textcoords='offset points')
        
        plt.tight_layout()
        scatter_plot_path = os.path.join(self.output_dir, "parameter_space.png")
        plt.savefig(scatter_plot_path)
        print(f"Parameter space visualization saved to {scatter_plot_path}")
        plt.close()
        
        # Create a heatmap for grid search results (if enough data points)
        if len(self.results) > 20:
            # Prepare data for heatmap
            angry_unique = sorted(list(set([round(r['angry_bias'], 2) for r in self.results])))
            neutral_unique = sorted(list(set([round(r['neutral_bias'], 2) for r in self.results])))
            
            if len(angry_unique) > 1 and len(neutral_unique) > 1:
                # Create meshgrid
                X, Y = np.meshgrid(angry_unique, neutral_unique)
                Z = np.zeros_like(X, dtype=float)
                
                # Fill in scores
                for i, neutral in enumerate(neutral_unique):
                    for j, angry in enumerate(angry_unique):
                        # Find matching result
                        matches = [r for r in self.results 
                                if round(r['angry_bias'], 2) == angry 
                                and round(r['neutral_bias'], 2) == neutral]
                        if matches:
                            Z[i, j] = matches[0]['final_score']
                
                # Create heatmap
                plt.figure(figsize=(12, 10))
                plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
                plt.colorbar(label='Score')
                plt.xlabel('Angry Bias')
                plt.ylabel('Neutral Bias')
                plt.title('Parameter Score Heatmap')
                
                # Mark the best parameters
                plt.scatter([self.best_params[0]], [self.best_params[1]], c='red', s=100, marker='*')
                
                plt.tight_layout()
                heatmap_path = os.path.join(self.output_dir, "score_heatmap.png")
                plt.savefig(heatmap_path)
                print(f"Score heatmap saved to {heatmap_path}")
                plt.close()
        
        # Plot the best distribution compared to target
        self._plot_distribution_comparison()
    
    def _plot_distribution_comparison(self):
        """Plot the best distribution compared to the target distribution"""
        if not self.best_distribution:
            return
        
        plt.figure(figsize=(12, 6))
        
        emotions = list(self.target_distribution.keys())
        target_values = [self.target_distribution[e] * 100 for e in emotions]
        actual_values = [self.best_distribution[e] * 100 for e in emotions]
        
        x = np.arange(len(emotions))
        width = 0.35
        
        plt.bar(x - width/2, target_values, width, label='Target Distribution')
        plt.bar(x + width/2, actual_values, width, label='Achieved Distribution')
        
        plt.xlabel('Emotion')
        plt.ylabel('Percentage (%)')
        plt.title('Emotion Distribution Comparison')
        plt.xticks(x, emotions)
        plt.legend()
        
        # Annotate bars with values
        for i, v in enumerate(target_values):
            plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
        
        for i, v in enumerate(actual_values):
            plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        dist_plot_path = os.path.join(self.output_dir, "distribution_comparison.png")
        plt.savefig(dist_plot_path)
        print(f"Distribution comparison saved to {dist_plot_path}")
        plt.close()

def run_test_with_optimal_params(image_path, model_path, bias_params_file):
    """Run a test with the optimal parameters"""
    # Load bias parameters
    with open(bias_params_file, 'r') as f:
        bias_params = json.load(f)
    
    # Build command
    cmd = [
        "python3", "predict_emotion.py",
        "--image", image_path,
        "--model", model_path,
        "--angry_bias", str(bias_params['angry_bias']),
        "--neutral_bias", str(bias_params['neutral_bias'])
    ]
    
    # Run command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    print(stdout.decode('utf-8'))
    if stderr:
        print("Errors:")
        print(stderr.decode('utf-8'))

def create_ground_truth_file(image_dir, output_file):
    """
    Create a ground truth JSON file by inferring emotions from directory structure
    Assumes images are organized in directories named by emotion
    """
    ground_truth = {}
    
    # Scan through emotion directories
    for emotion in EMOTIONS.values():
        emotion_dir = os.path.join(image_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in Path(emotion_dir).glob(ext):
                img_name = img_path.name
                ground_truth[img_name] = emotion
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=4)
    
    print(f"Created ground truth file with {len(ground_truth)} annotations: {output_file}")
    return ground_truth

def main():
    parser = argparse.ArgumentParser(description='Advanced bias parameter tuning for facial emotion recognition')
    parser.add_argument('--image_dir', type=str, required=True, 
                      help='Path to directory with test images')
    parser.add_argument('--model', type=str, default='./models/hybrid_model.pth',
                      help='Path to the model weights')
    parser.add_argument('--output_dir', type=str, default='bias_optimization_results',
                      help='Directory to save optimization results')
    parser.add_argument('--method', type=str, default='grid_search',
                      choices=['grid_search', 'optuna', 'evolutionary'],
                      help='Optimization method to use')
    parser.add_argument('--trials', type=int, default=60,
                      help='Number of trials for optimization (for optuna and evolutionary)')
    parser.add_argument('--jobs', type=int, default=4,
                      help='Number of parallel jobs for optimization')
    parser.add_argument('--use_ground_truth', action='store_true',
                      help='Use ground truth annotations for optimization')
    parser.add_argument('--ground_truth_file', type=str,
                      help='Path to ground truth file (JSON format)')
    parser.add_argument('--create_ground_truth', action='store_true',
                      help='Create ground truth file from directory structure')
    parser.add_argument('--test', type=str,
                      help='Run a test with optimal parameters on this image')
    
    args = parser.parse_args()
    
    # Create ground truth file if requested
    if args.create_ground_truth:
        ground_truth_file = os.path.join(args.output_dir, "ground_truth.json")
        create_ground_truth_file(args.image_dir, ground_truth_file)
        args.ground_truth_file = ground_truth_file
        args.use_ground_truth = True
    
    # Run optimization
    optimizer = BiasOptimizer(
        image_dir=args.image_dir,
        model_path=args.model,
        output_dir=args.output_dir,
        optimization_method=args.method,
        num_trials=args.trials,
        parallel_jobs=args.jobs,
        use_ground_truth=args.use_ground_truth,
        ground_truth_file=args.ground_truth_file
    )
    
    best_params, best_score, best_distribution = optimizer.run_optimization()
    
    # Run a test with optimal parameters if requested
    if args.test:
        best_params_file = os.path.join(args.output_dir, "best_bias_params.json")
        run_test_with_optimal_params(args.test, args.model, best_params_file)

if __name__ == '__main__':
    main() 