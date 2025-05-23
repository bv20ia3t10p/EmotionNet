import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

class DynamicClassWeighting:
    """
    Dynamically adjusts class weights based on performance metrics.
    """
    def __init__(
        self, 
        num_classes: int = 7, 
        initial_weights: Optional[List[float]] = None,
        alpha: float = 0.7,  # Smoothing factor for weight updates
        min_weight: float = 0.5,
        max_weight: float = 5.0
    ):
        self.num_classes = num_classes
        
        # Initialize weights
        if initial_weights is None:
            self.weights = torch.ones(num_classes)
        else:
            if len(initial_weights) != num_classes:
                raise ValueError(f"Expected {num_classes} weights, got {len(initial_weights)}")
            self.weights = torch.tensor(initial_weights)
        
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # History of metrics
        self.f1_history = []
        self.weight_history = []
        self.confusion_matrices = []
        
        # Save initial weights
        self.weight_history.append(self.weights.clone())
    
    def update_weights(self, f1_scores: List[float], confusion_mat: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Update weights based on F1 scores and confusion matrix.
        Lower F1 scores get higher weights.
        
        Args:
            f1_scores: F1 score for each class
            confusion_mat: Confusion matrix (optional)
            
        Returns:
            Updated weights tensor
        """
        # Convert to tensor if not already
        f1 = torch.tensor(f1_scores)
        
        # Save F1 history
        self.f1_history.append(f1.clone())
        
        # Calculate inverse F1 scores (lower F1 -> higher weight)
        inverse_f1 = 1.0 - f1
        
        # Apply smoothing with previous weights
        new_weights = self.alpha * self.weights + (1 - self.alpha) * (1.0 + inverse_f1)
        
        # If confusion matrix is provided, incorporate class confusion
        if confusion_mat is not None:
            # Save confusion matrix
            self.confusion_matrices.append(confusion_mat.copy())
            
            # Normalize confusion matrix by row (true labels)
            cm_norm = confusion_mat.astype(float) / confusion_mat.sum(axis=1, keepdims=True)
            
            # Replace NaN with 0
            cm_norm = np.nan_to_num(cm_norm, 0)
            
            # For each class, calculate average confusion with other classes
            for i in range(self.num_classes):
                # Skip diagonal (self-prediction)
                confusion_score = np.sum(cm_norm[i, :]) - cm_norm[i, i]
                
                # Add weighted confusion score to class weight
                # Higher confusion means higher weight
                new_weights[i] += 0.3 * confusion_score
        
        # Clip weights to valid range
        new_weights = torch.clamp(new_weights, min=self.min_weight, max=self.max_weight)
        
        # Normalize weights to have mean = 1.0
        new_weights = new_weights * (self.num_classes / new_weights.sum())
        
        # Update weights
        self.weights = new_weights
        
        # Save weight history
        self.weight_history.append(self.weights.clone())
        
        # Log the updated weights
        print("Updated class weights:")
        for i, w in enumerate(self.weights):
            print(f"  Class {i}: {w:.4f}")
        
        return self.weights
    
    def get_focal_gamma(self, f1_scores: List[float]) -> List[float]:
        """
        Calculate class-specific focal loss gamma values based on F1 scores.
        Lower F1 scores get higher gamma values for more focus.
        
        Args:
            f1_scores: F1 score for each class
            
        Returns:
            List of gamma values for each class
        """
        f1 = torch.tensor(f1_scores)
        
        # Scale inverse F1 to gamma range [1.0, 3.0]
        inverse_f1 = 1.0 - f1
        gammas = 1.0 + 2.0 * inverse_f1
        
        return gammas.tolist()
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot the history of weights and F1 scores.
        """
        # Convert histories to numpy
        weights = torch.stack(self.weight_history).numpy()
        f1s = torch.stack(self.f1_history).numpy() if self.f1_history else np.array([])
        
        epochs = range(len(weights))
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot weights
        for i in range(self.num_classes):
            ax1.plot(epochs, weights[:, i], label=f'Class {i}')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weight')
        ax1.set_title('Class Weights Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot F1 scores if available
        if len(f1s) > 0:
            for i in range(self.num_classes):
                ax2.plot(range(len(f1s)), f1s[:, i], label=f'Class {i}')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Class F1 Scores Over Time')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, epoch: int, class_names: List[str], save_path: Optional[str] = None):
        """
        Plot confusion matrix for a specific epoch.
        """
        if epoch >= len(self.confusion_matrices):
            print(f"No confusion matrix available for epoch {epoch}")
            return
        
        cm = self.confusion_matrices[epoch]
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class FocalLossWithDynamicGamma(nn.Module):
    """
    Focal Loss with class-specific gamma values that can be updated dynamically.
    """
    def __init__(
        self, 
        num_classes: int = 7, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: Optional[torch.Tensor] = None
    ):
        super(FocalLossWithDynamicGamma, self).__init__()
        self.num_classes = num_classes
        
        # Initialize alpha (class weights)
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = alpha
        
        # Initialize gamma (focusing parameter)
        if gamma is None:
            self.gamma = torch.ones(num_classes) * 2.0
        else:
            self.gamma = gamma
    
    def update_parameters(self, alpha: Optional[torch.Tensor] = None, gamma: Optional[torch.Tensor] = None):
        """
        Update alpha and gamma parameters.
        """
        if alpha is not None:
            self.alpha = alpha
        
        if gamma is not None:
            self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with class-specific gamma values.
        
        Args:
            inputs: Model predictions, shape [N, C]
            targets: Ground truth labels, shape [N]
            
        Returns:
            Focal loss value
        """
        # Get device
        device = inputs.device
        
        # Move parameters to device
        alpha = self.alpha.to(device)
        gamma = self.gamma.to(device)
        
        # Compute softmax and log_softmax
        log_prob = F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        
        # Gather the targets
        batch_size = inputs.size(0)
        target_onehot = F.one_hot(targets, num_classes=self.num_classes)
        
        # Extract the probabilities of the target classes
        prob_gt = (prob * target_onehot.float()).sum(1)
        
        # Calculate focal weights
        # Use class-specific gamma values
        gamma_tensor = gamma[targets]
        focal_weight = (1 - prob_gt).pow(gamma_tensor)
        
        # Calculate weighted loss
        alpha_tensor = alpha[targets]
        focal_loss = -alpha_tensor * focal_weight * torch.log(prob_gt + 1e-8)
        
        return focal_loss.mean()

class ClassAnalyzer:
    """
    Analyzes class performance and provides recommendations
    for class-specific strategies.
    """
    def __init__(self, num_classes: int = 7, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Performance tracking
        self.f1_history = []
        self.precision_history = []
        self.recall_history = []
        self.confusion_matrices = []
        
        # Problem class thresholds
        self.low_f1_threshold = 0.4
        self.high_confusion_threshold = 0.3
    
    def update_metrics(
        self, 
        f1_scores: List[float], 
        precision_scores: Optional[List[float]] = None,
        recall_scores: Optional[List[float]] = None,
        confusion_mat: Optional[np.ndarray] = None
    ):
        """
        Update performance metrics.
        """
        self.f1_history.append(f1_scores)
        
        if precision_scores:
            self.precision_history.append(precision_scores)
        
        if recall_scores:
            self.recall_history.append(recall_scores)
        
        if confusion_mat is not None:
            self.confusion_matrices.append(confusion_mat)
    
    def identify_problem_classes(self) -> Dict[str, List[int]]:
        """
        Identify classes with potential issues.
        
        Returns:
            Dictionary of problem types and corresponding class indices
        """
        if not self.f1_history:
            return {}
        
        # Get most recent F1 scores
        recent_f1 = self.f1_history[-1]
        
        problems = {
            "low_f1": [],
            "high_confusion": [],
            "low_recall": [],
            "low_precision": []
        }
        
        # Find classes with low F1
        for i, f1 in enumerate(recent_f1):
            if f1 < self.low_f1_threshold:
                problems["low_f1"].append(i)
        
        # Find classes with high confusion (if confusion matrix is available)
        if self.confusion_matrices:
            cm = self.confusion_matrices[-1]
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            
            for i in range(self.num_classes):
                # Skip diagonal (self-prediction)
                confusion = np.sum(cm_norm[i, :]) - cm_norm[i, i]
                if confusion > self.high_confusion_threshold:
                    problems["high_confusion"].append(i)
        
        # Find classes with precision/recall issues
        if self.precision_history and self.recall_history:
            recent_precision = self.precision_history[-1]
            recent_recall = self.recall_history[-1]
            
            for i in range(self.num_classes):
                if recent_precision[i] < 0.4 and recent_precision[i] < recent_recall[i] * 0.7:
                    problems["low_precision"].append(i)
                
                if recent_recall[i] < 0.4 and recent_recall[i] < recent_precision[i] * 0.7:
                    problems["low_recall"].append(i)
        
        return problems
    
    def get_class_recommendations(self) -> Dict[int, List[str]]:
        """
        Get recommendations for each problem class.
        
        Returns:
            Dictionary mapping class indices to list of recommendations
        """
        problems = self.identify_problem_classes()
        recommendations = {}
        
        # Process each class and its problems
        for problem_type, class_indices in problems.items():
            for class_idx in class_indices:
                if class_idx not in recommendations:
                    recommendations[class_idx] = []
                
                if problem_type == "low_f1":
                    recommendations[class_idx].append("Increase class weight")
                    recommendations[class_idx].append("Apply stronger augmentation")
                    recommendations[class_idx].append("Increase focal loss gamma")
                
                elif problem_type == "high_confusion":
                    recommendations[class_idx].append("Increase margin in loss function")
                    recommendations[class_idx].append("Train a specialized classifier")
                    
                    # Find most confused classes
                    if self.confusion_matrices:
                        cm = self.confusion_matrices[-1]
                        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
                        
                        # Get top confused classes (excluding self)
                        confused_with = []
                        for j in range(self.num_classes):
                            if j != class_idx:
                                confused_with.append((j, cm_norm[class_idx, j]))
                        
                        confused_with.sort(key=lambda x: x[1], reverse=True)
                        top_confused = confused_with[:2]
                        
                        for j, conf_val in top_confused:
                            if conf_val > 0.1:  # Only significant confusion
                                recommendations[class_idx].append(
                                    f"Address confusion with {self.class_names[j]}"
                                )
                
                elif problem_type == "low_precision":
                    recommendations[class_idx].append("Focus on reducing false positives")
                    recommendations[class_idx].append("Increase precision-oriented metrics")
                
                elif problem_type == "low_recall":
                    recommendations[class_idx].append("Focus on reducing false negatives")
                    recommendations[class_idx].append("Increase recall-oriented metrics")
        
        return recommendations
    
    def print_analysis(self):
        """
        Print a comprehensive analysis of class performance.
        """
        if not self.f1_history:
            print("No data available for analysis yet.")
            return
        
        problems = self.identify_problem_classes()
        recommendations = self.get_class_recommendations()
        
        print("\n=== Class Performance Analysis ===")
        print(f"{'Class':<15} {'F1 Score':<10} {'Issues':<30} {'Recommendations':<40}")
        print("-" * 95)
        
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            recent_f1 = self.f1_history[-1][i]
            
            # Compile issues
            issues = []
            for problem_type, class_indices in problems.items():
                if i in class_indices:
                    issues.append(problem_type.replace("_", " ").title())
            
            # Get recommendations
            class_recommendations = recommendations.get(i, [])
            
            # Print row
            issues_str = ", ".join(issues) if issues else "None"
            recommendations_str = "; ".join(class_recommendations) if class_recommendations else "None"
            
            print(f"{class_name:<15} {recent_f1:.4f}      {issues_str:<30} {recommendations_str:<40}")
        
        print("\n") 