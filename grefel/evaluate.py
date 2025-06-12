import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse
from datetime import datetime
from model import EnhancedGReFEL
from dataset import FERPlusDataset

def setup_output_dir(base_dir, model_name):
    # Create timestamped directory name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    output_dir = os.path.join(base_dir, f'eval_{model_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class ModelEvaluator:
    def __init__(self, model_path, val_dir, output_dir, batch_size=32, num_workers=4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        
        print(f"Loading model from: {model_path}")
        print(f"Validation data directory: {val_dir}")
        print(f"Results will be saved to: {output_dir}")
        
        # Load model with same parameters as train.py
        try:
            self.model = EnhancedGReFEL(
                num_classes=8,
                num_anchors=16  # Match train.py default
            )
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Create dataset and loader
        try:
            self.val_dataset = FERPlusDataset(val_dir, train=False)
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            print(f"Validation dataset loaded with {len(self.val_dataset)} samples")
        except Exception as e:
            print(f"Error loading validation dataset: {str(e)}")
            raise
        
        # Set output directory
        self.output_dir = output_dir
        
        # Save configuration
        self.save_config(model_path, val_dir, batch_size, num_workers)
    
    def save_config(self, model_path, val_dir, batch_size, num_workers):
        config = {
            'model_path': os.path.abspath(model_path),
            'val_dir': os.path.abspath(val_dir),
            'batch_size': batch_size,
            'num_workers': num_workers,
            'device': str(self.device),
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_config.txt'), 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

    def evaluate(self):
        all_preds = []
        all_labels = []
        all_probs = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(images)
                probs = torch.softmax(logits, dim=1)
                
                # Get predictions
                _, predicted = logits.max(1)
                
                # Update metrics
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs, correct / total

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
    def plot_class_distribution(self, y_true):
        plt.figure(figsize=(12, 6))
        pd.Series(y_true).map(dict(enumerate(self.emotion_labels))).value_counts().plot(kind='bar')
        plt.title('Class Distribution in Validation Set')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'))
        plt.close()
        
    def plot_confidence_distribution(self, probs):
        plt.figure(figsize=(12, 6))
        max_probs = probs.max(axis=1)
        sns.histplot(max_probs, bins=50)
        plt.title('Model Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confidence_distribution.png'))
        plt.close()
        
    def analyze_errors(self, y_true, y_pred, probs):
        errors = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    'true_label': self.emotion_labels[y_true[i]],
                    'pred_label': self.emotion_labels[y_pred[i]],
                    'confidence': probs[i][y_pred[i]],
                    'true_prob': probs[i][y_true[i]]
                })
        
        error_df = pd.DataFrame(errors)
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'most_confused_pairs': error_df.groupby(['true_label', 'pred_label']).size().sort_values(ascending=False).head(10),
            'avg_error_confidence': error_df['confidence'].mean(),
            'high_confidence_errors': len(error_df[error_df['confidence'] > 0.9])
        }
        
        return error_analysis
    
    def generate_report(self):
        print("Starting comprehensive model evaluation...")
        
        # Get predictions and metrics
        predictions, labels, probabilities, accuracy = self.evaluate()
        
        # Generate classification report
        report = classification_report(
            labels, 
            predictions, 
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        # Plot visualizations
        self.plot_confusion_matrix(labels, predictions)
        self.plot_class_distribution(labels)
        self.plot_confidence_distribution(probabilities)
        
        # Analyze errors
        error_analysis = self.analyze_errors(labels, predictions, probabilities)
        
        # Save detailed report
        with open(os.path.join(self.output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("=== Model Evaluation Report ===\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            metrics_df = pd.DataFrame(report).T
            f.write(metrics_df.to_string())
            f.write("\n\n")
            
            f.write("Error Analysis:\n")
            f.write(f"Total Errors: {error_analysis['total_errors']}\n")
            f.write(f"Error Rate: {error_analysis['error_rate']:.4f}\n")
            f.write(f"Average Error Confidence: {error_analysis['avg_error_confidence']:.4f}\n")
            f.write(f"High Confidence Errors (>0.9): {error_analysis['high_confidence_errors']}\n\n")
            
            f.write("Most Confused Pairs:\n")
            f.write(error_analysis['most_confused_pairs'].to_string())
            
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")
        
        # Return summary metrics
        return {
            'accuracy': accuracy,
            'per_class_metrics': report,
            'error_analysis': error_analysis
        }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of GReFEL model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--val_dir', type=str, default='../FERPlus/data/FER2013Valid', 
                       help='Path to validation data directory (should contain "Valid" or "PublicTest" in name)')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation', help='Base directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    
    try:
        # Convert relative paths to absolute paths
        args.model_path = os.path.abspath(args.model_path)
        args.val_dir = os.path.abspath(args.val_dir)
        args.output_dir = os.path.abspath(args.output_dir)
        
        # Verify validation directory structure
        if not os.path.exists(args.val_dir):
            raise ValueError(f"Validation directory not found: {args.val_dir}")
        
        dir_name = os.path.basename(args.val_dir)
        if not ('Valid' in dir_name or 'PublicTest' in dir_name):
            print("Warning: Validation directory should contain 'Valid' or 'PublicTest' in its name")
        
        # Check for fer2013new.csv
        ferplus_path = os.path.join(os.path.dirname(os.path.dirname(args.val_dir)), 'fer2013new.csv')
        if not os.path.exists(ferplus_path):
            raise ValueError(f"FER+ labels file not found: {ferplus_path}\nExpected structure:\n"
                           f"root/\n"
                           f"  ├── fer2013new.csv\n"
                           f"  └── FERPlus/\n"
                           f"      ├── Training/\n"
                           f"      └── PublicTest/")
        
        # Create output directory with timestamp
        eval_dir = setup_output_dir(args.output_dir, args.model_path)
        print(f"\nEvaluation results will be saved to: {eval_dir}")
        
        # Create evaluator and generate report
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            val_dir=args.val_dir,
            output_dir=eval_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        metrics = evaluator.generate_report()
        
        # Print summary
        print("\nEvaluation Summary:")
        print("=" * 50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class F1 Scores:")
        print("-" * 30)
        for emotion, scores in metrics['per_class_metrics'].items():
            if emotion in evaluator.emotion_labels:
                print(f"{emotion:10s}: {scores['f1-score']:.4f}")
        
        print("\nError Analysis:")
        print("-" * 30)
        print(f"Total Errors: {metrics['error_analysis']['total_errors']}")
        print(f"Error Rate: {metrics['error_analysis']['error_rate']:.4f}")
        print(f"Avg Error Confidence: {metrics['error_analysis']['avg_error_confidence']:.4f}")
        
        print(f"\nDetailed results saved to: {eval_dir}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == '__main__':
    main() 