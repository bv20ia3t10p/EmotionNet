import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# import face_recognition
# Note: Using dummy landmarks for demo purposes
from tqdm import tqdm

from grefel_implementation import GReFELModel
from train_grefel import FacialExpressionDataset, DataAugmentation

class GReFELEvaluator:
    """Comprehensive evaluation for GReFEL model"""
    
    def __init__(self, model_path, num_classes=8, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        
        # Load model
        self.model = GReFELModel(num_classes=num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        
        # Image transforms
        self.transform = DataAugmentation.get_val_transforms()
        
    def extract_landmarks(self, image):
        """Extract facial landmarks from image using dummy generation"""
        try:
            if isinstance(image, Image.Image):
                width, height = image.size
            else:
                height, width = image.shape[:2]
            
            # Generate realistic dummy landmarks for a centered face
            center_x, center_y = width // 2, height // 2
            face_width, face_height = width * 0.6, height * 0.8
            
            landmarks = []
            
            # Face contour (17 points)
            for i in range(17):
                x = center_x + (i - 8) * face_width / 16
                y = center_y + abs(i - 8) * face_height / 32
                landmarks.append([x, y])
            
            # Right eyebrow (5 points)
            for i in range(5):
                x = center_x - face_width/4 + i * face_width/8
                y = center_y - face_height/4
                landmarks.append([x, y])
            
            # Left eyebrow (5 points)
            for i in range(5):
                x = center_x + face_width/8 + i * face_width/8
                y = center_y - face_height/4
                landmarks.append([x, y])
            
            # Nose bridge and tip (9 points)
            for i in range(9):
                x = center_x + (i - 4) * face_width/20
                y = center_y - face_height/8 + i * face_height/16
                landmarks.append([x, y])
            
            # Right eye (6 points)
            for i in range(6):
                x = center_x - face_width/4 + i * face_width/12
                y = center_y - face_height/8
                landmarks.append([x, y])
            
            # Left eye (6 points)
            for i in range(6):
                x = center_x + face_width/6 + i * face_width/12
                y = center_y - face_height/8
                landmarks.append([x, y])
            
            # Outer lip (12 points)
            for i in range(12):
                angle = i * 2 * np.pi / 12
                x = center_x + np.cos(angle) * face_width/6
                y = center_y + face_height/4 + np.sin(angle) * face_height/12
                landmarks.append([x, y])
            
            # Inner lip (8 points)
            for i in range(8):
                angle = i * 2 * np.pi / 8
                x = center_x + np.cos(angle) * face_width/8
                y = center_y + face_height/4 + np.sin(angle) * face_height/16
                landmarks.append([x, y])
            
            landmark_array = np.array(landmarks[:68])  # Ensure exactly 68 points
            
            # Normalize to [0, 1] range
            landmark_array[:, 0] /= width
            landmark_array[:, 1] /= height
                
        except Exception as e:
            print(f"Error generating landmarks: {e}")
            # Return center-based dummy landmarks if generation fails
            landmark_array = np.random.rand(68, 2) * 0.1 + 0.45
            
        return landmark_array.astype(np.float32)
    
    def predict_single_image(self, image_path):
        """Predict emotion for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        landmarks = self.extract_landmarks(image)
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor, landmarks_tensor)
            
        # Get probabilities and prediction
        probabilities = outputs['final_probs'].cpu().numpy()[0]
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        return {
            'prediction': prediction,
            'emotion': self.emotion_labels[prediction],
            'confidence': confidence,
            'probabilities': probabilities,
            'primary_probs': outputs['primary_probs'].cpu().numpy()[0],
            'geometric_correction': outputs['geometric_correction'].cpu().numpy()[0],
            'attentive_correction': outputs['attentive_correction'].cpu().numpy()[0]
        }
    
    def predict_batch(self, data_loader):
        """Predict emotions for a batch of images"""
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                outputs = self.model(images, landmarks)
                probabilities = outputs['final_probs'].cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                confidences = np.max(probabilities, axis=1)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_confidences.extend(confidences)
                all_probabilities.extend(probabilities)
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_confidences), np.array(all_probabilities)
    
    def compute_metrics(self, predictions, labels):
        """Compute comprehensive evaluation metrics"""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
        
        # Overall metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        return {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }
    
    def plot_confusion_matrix(self, predictions, labels, save_path='confusion_matrix_eval.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels[:self.num_classes],
                   yticklabels=self.emotion_labels[:self.num_classes])
        plt.title('Confusion Matrix - GReFEL Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_class_wise_metrics(self, metrics, save_path='class_metrics.png'):
        """Plot class-wise performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        emotions = self.emotion_labels[:self.num_classes]
        
        # Precision
        axes[0, 0].bar(emotions, metrics['precision_per_class'])
        axes[0, 0].set_title('Precision per Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[0, 1].bar(emotions, metrics['recall_per_class'])
        axes[0, 1].set_title('Recall per Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1, 0].bar(emotions, metrics['f1_per_class'])
        axes[1, 0].set_title('F1-Score per Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Support
        axes[1, 1].bar(emotions, metrics['support_per_class'])
        axes[1, 1].set_title('Support per Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_confidence_distribution(self, confidences, predictions, labels, save_path='confidence_analysis.png'):
        """Analyze confidence distribution"""
        correct_mask = (predictions == labels)
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Overall confidence distribution
        plt.subplot(2, 2, 1)
        plt.hist(confidences, bins=50, alpha=0.7, label='All predictions')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Overall Confidence Distribution')
        plt.legend()
        
        # Subplot 2: Correct vs Incorrect predictions
        plt.subplot(2, 2, 2)
        plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence by Correctness')
        plt.legend()
        
        # Subplot 3: Confidence vs Accuracy
        plt.subplot(2, 2, 3)
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(predictions[mask] == labels[mask])
                bin_accuracies.append(bin_accuracy)
                bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
        
        plt.plot(bin_centers, bin_accuracies, 'o-', color='blue')
        plt.plot([0, 1], [0, 1], '--', color='red', label='Perfect calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Confidence Calibration')
        plt.legend()
        
        # Subplot 4: Per-class confidence
        plt.subplot(2, 2, 4)
        class_confidences = []
        for class_idx in range(self.num_classes):
            class_mask = labels == class_idx
            if np.sum(class_mask) > 0:
                class_confidences.append(confidences[class_mask])
        
        plt.boxplot(class_confidences, labels=self.emotion_labels[:self.num_classes])
        plt.xlabel('Emotion Class')
        plt.ylabel('Confidence')
        plt.title('Confidence by Class')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, image_paths, predictions_data, save_path='prediction_examples.png'):
        """Visualize prediction examples"""
        n_examples = min(8, len(image_paths))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(n_examples):
            image = Image.open(image_paths[i]).convert('RGB')
            pred_data = predictions_data[i]
            
            axes[i].imshow(image)
            axes[i].set_title(f"Pred: {pred_data['emotion']}\nConf: {pred_data['confidence']:.3f}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_examples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_reliability_balancing(self, probabilities_data):
        """Analyze the effect of reliability balancing"""
        primary_probs = np.array([p['primary_probs'] for p in probabilities_data])
        geometric_corrections = np.array([p['geometric_correction'] for p in probabilities_data])
        attentive_corrections = np.array([p['attentive_correction'] for p in probabilities_data])
        final_probs = np.array([p['probabilities'] for p in probabilities_data])
        
        # Compare predictions before and after reliability balancing
        primary_predictions = np.argmax(primary_probs, axis=1)
        final_predictions = np.argmax(final_probs, axis=1)
        
        # Calculate difference
        changed_predictions = np.sum(primary_predictions != final_predictions)
        total_predictions = len(primary_predictions)
        change_rate = changed_predictions / total_predictions
        
        print(f"Reliability Balancing Analysis:")
        print(f"Total predictions: {total_predictions}")
        print(f"Changed predictions: {changed_predictions}")
        print(f"Change rate: {change_rate:.3f}")
        
        return {
            'primary_predictions': primary_predictions,
            'final_predictions': final_predictions,
            'change_rate': change_rate,
            'changed_predictions': changed_predictions
        }
    
    def compare_with_baseline(self, test_loader, baseline_predictions=None):
        """Compare GReFEL performance with baseline methods"""
        predictions, labels, confidences, probabilities = self.predict_batch(test_loader)
        grefel_metrics = self.compute_metrics(predictions, labels)
        
        print("=== GReFEL Performance ===")
        print(f"Accuracy: {grefel_metrics['accuracy']:.4f}")
        print(f"Macro F1: {grefel_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {grefel_metrics['weighted_f1']:.4f}")
        
        if baseline_predictions is not None:
            baseline_metrics = self.compute_metrics(baseline_predictions, labels)
            print("\n=== Baseline Comparison ===")
            print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
            print(f"Improvement: {grefel_metrics['accuracy'] - baseline_metrics['accuracy']:.4f}")
        
        return grefel_metrics
    
    def evaluate_dataset(self, test_csv, img_dir=None):
        """Complete evaluation on a test dataset"""
        print("Starting comprehensive evaluation...")
        
        # Create test dataset
        test_dataset = FacialExpressionDataset(
            csv_file=test_csv,
            img_dir=img_dir,
            transform=self.transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        # Get predictions
        predictions, labels, confidences, probabilities = self.predict_batch(test_loader)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, labels)
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        
        print("\n=== Per-Class Results ===")
        for i, emotion in enumerate(self.emotion_labels[:self.num_classes]):
            print(f"{emotion}: P={metrics['precision_per_class'][i]:.3f}, "
                  f"R={metrics['recall_per_class'][i]:.3f}, "
                  f"F1={metrics['f1_per_class'][i]:.3f}, "
                  f"Support={metrics['support_per_class'][i]}")
        
        # Generate visualizations
        self.plot_confusion_matrix(predictions, labels)
        self.plot_class_wise_metrics(metrics)
        self.analyze_confidence_distribution(confidences, predictions, labels)
        
        return metrics, predictions, labels, confidences

def main():
    """Main evaluation function"""
    # Configuration
    model_path = 'grefel_best_model.pth'
    test_csv = 'val_split.csv'  # Use validation split for testing
    img_dir = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Create evaluator
    evaluator = GReFELEvaluator(model_path, device=device)
    
    # Single image prediction example
    print("=== Single Image Prediction Example ===")
    # You can test with any image file
    # result = evaluator.predict_single_image('test_image.jpg')
    # print(f"Prediction: {result['emotion']} (confidence: {result['confidence']:.3f})")
    
    # Full dataset evaluation
    if os.path.exists(test_csv):
        print(f"\n=== Dataset Evaluation ===")
        metrics, predictions, labels, confidences = evaluator.evaluate_dataset(test_csv, img_dir)
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'true_label': labels,
            'predicted_label': predictions,
            'confidence': confidences,
            'correct': predictions == labels
        })
        results_df.to_csv('evaluation_results.csv', index=False)
        print("Results saved to evaluation_results.csv")
    else:
        print(f"Test dataset not found at {test_csv}")

if __name__ == "__main__":
    main() 