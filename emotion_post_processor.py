import torch
import torch.nn.functional as F
from model import CBAM


class EmotionPostProcessor:
    """
    Post-processor for emotion recognition that refines predictions
    using attention mapping and confidence weighting.

    Features:
    - Attention visualization for interpretability
    - Confidence-based decision refinement
    - Multi-class confusion handling
    - Emotion-specific decision thresholds
    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Emotion-specific confidence thresholds based on observed data patterns
        self.emotion_thresholds = {
            0: 0.35,  # 'angry' - moderately strict
            1: 0.45,  # 'disgust' - strict (often confused with anger)
            2: 0.30,  # 'fear' - lenient (hard to detect)
            3: 0.25,  # 'happy' - lenient (easy to detect)
            4: 0.35,  # 'neutral' - moderately strict
            5: 0.40,  # 'sad' - moderately strict
            6: 0.30,  # 'surprise' - lenient
        }

        # Emotion name mapping
        self.emotion_names = [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]

        # Confusion matrix-based refinement
        # Higher value means these classes are often confused
        self.confusion_pairs = [
            (0, 1, 0.4),  # angry-disgust
            (0, 4, 0.3),  # angry-neutral
            (0, 5, 0.3),  # angry-sad
            (2, 6, 0.4),  # fear-surprise
            (4, 5, 0.3),  # neutral-sad
        ]

        # Secondary classifier for refinement
        self.attention_refiner = CBAM(1024, reduction_ratio=4).to(self.device)

    def refine_prediction(self, logits, image_features=None):
        """
        Refine model prediction using confidence thresholds and confusion pairs

        Args:
            logits: Model output logits (batch_size, num_classes)
            image_features: Optional intermediate features for attention mapping

        Returns:
            refined_probs: Refined probability distribution
            refined_pred: Refined class prediction
            refinement_info: Dictionary with refinement information
        """
        # Get initial probabilities and predictions
        probs = F.softmax(logits, dim=1)
        initial_pred = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        # Initialize refined probabilities to original
        refined_probs = probs.clone()

        # Apply attention-based refinement if features are provided
        if image_features is not None:
            attention_weights = self.attention_refiner(image_features)
            # Use attention maps to refine probabilities
            # This is a simplified approach - in practice would need hooks to extract features
            batch_size = probs.size(0)
            for i in range(batch_size):
                # Example of how we could use attention to refine
                # In practice, this would be more sophisticated
                attention_score = attention_weights[i].mean().item()
                # Adjust confidence based on attention
                refined_probs[i] = refined_probs[i] * \
                    (1.0 + 0.2 * attention_score)
                refined_probs[i] = refined_probs[i] / refined_probs[i].sum()

        # Apply confidence thresholds
        batch_size = probs.size(0)
        refined_pred = initial_pred.clone()
        refinement_info = []

        for i in range(batch_size):
            pred_class = initial_pred[i].item()
            conf = confidence[i].item()
            threshold = self.emotion_thresholds[pred_class]

            info = {
                "initial_pred": pred_class,
                "initial_emotion": self.emotion_names[pred_class],
                "confidence": conf,
                "threshold": threshold,
                "was_refined": False,
                "refinement_reason": None
            }

            # If confidence is below threshold, check for confusion pairs
            if conf < threshold:
                # Get top-2 predictions
                top2_values, top2_indices = torch.topk(probs[i], 2)

                # Check if this is a known confusion pair
                for class1, class2, conf_threshold in self.confusion_pairs:
                    if ((pred_class == class1 and top2_indices[1].item() == class2) or
                            (pred_class == class2 and top2_indices[1].item() == class1)):

                        # If confidence difference is small, prefer the class
                        # that is generally more accurately predicted
                        if (top2_values[0] - top2_values[1]) < conf_threshold:
                            # For this simple example, we arbitrarily choose class2
                            # In a real implementation, you would use validation statistics
                            preferred_class = class2 if pred_class == class1 else class1

                            # Update prediction and record reason
                            refined_pred[i] = preferred_class
                            info["was_refined"] = True
                            info["refinement_reason"] = f"Confusion pair {self.emotion_names[class1]}-{self.emotion_names[class2]}"
                            info["new_emotion"] = self.emotion_names[preferred_class]

                            # Also update probabilities to reflect our decision
                            refined_probs[i, preferred_class] = max(
                                refined_probs[i, preferred_class], 0.6)
                            # Normalize
                            refined_probs[i] = refined_probs[i] / \
                                refined_probs[i].sum()
                            break

            refinement_info.append(info)

        return refined_probs, refined_pred, refinement_info

    @staticmethod
    def ensemble_predictions(model_outputs, weights=None):
        """
        Ensemble predictions from multiple models or augmentations

        Args:
            model_outputs: List of logits from different models/runs
            weights: Optional weights for each model

        Returns:
            ensemble_probs: Ensembled probability distribution
            ensemble_pred: Ensembled class prediction
        """
        if weights is None:
            weights = [1.0] * len(model_outputs)

        # Normalize weights
        weights = torch.tensor(weights) / sum(weights)

        # Convert logits to probabilities
        all_probs = [F.softmax(output, dim=1) for output in model_outputs]

        # Weighted average of probabilities
        ensemble_probs = torch.zeros_like(all_probs[0])
        for i, probs in enumerate(all_probs):
            ensemble_probs += probs * weights[i]

        # Get prediction
        ensemble_pred = torch.argmax(ensemble_probs, dim=1)

        return ensemble_probs, ensemble_pred

    @staticmethod
    def get_calibrated_predictions(logits, temperature=1.0):
        """
        Apply temperature scaling for better calibrated probabilities

        Args:
            logits: Model output logits
            temperature: Scaling temperature (>1 for smoother, <1 for sharper)

        Returns:
            calibrated_probs: Temperature-scaled probabilities
        """
        return F.softmax(logits / temperature, dim=1)


def apply_post_processing(model_outputs, ensemble=True, refine=True, temperature=1.2):
    """
    Apply the full post-processing pipeline to model outputs

    Args:
        model_outputs: Single tensor or list of tensors (logits)
        ensemble: Whether to ensemble (if multiple outputs)
        refine: Whether to apply refinement
        temperature: Temperature scaling factor

    Returns:
        final_probs: Final probability distribution
        final_pred: Final class prediction
        info: Processing information
    """
    device = model_outputs[0].device if isinstance(
        model_outputs, list) else model_outputs.device
    processor = EmotionPostProcessor(device)

    # Apply ensembling if needed
    if isinstance(model_outputs, list) and len(model_outputs) > 1 and ensemble:
        probs, pred = processor.ensemble_predictions(model_outputs)
        info = {"ensembled": True, "num_models": len(model_outputs)}
    else:
        # If single output or no ensembling
        logits = model_outputs[0] if isinstance(
            model_outputs, list) else model_outputs
        probs = processor.get_calibrated_predictions(logits, temperature)
        pred = torch.argmax(probs, dim=1)
        info = {"ensembled": False, "temperature": temperature}

    # Apply refinement if requested
    if refine:
        final_probs, final_pred, refinement_info = processor.refine_prediction(
            torch.log(probs)  # Convert back to logits for refinement
        )
        info["refinement"] = refinement_info
    else:
        final_probs, final_pred = probs, pred
        info["refinement"] = None

    return final_probs, final_pred, info


# Function to add to test_model.py
def evaluate_with_post_processing(model, val_loader, device, ensemble_models=None):
    """
    Evaluate model with post-processing for improved accuracy

    Args:
        model: Primary model to evaluate
        val_loader: Validation data loader
        device: Computation device
        ensemble_models: Optional list of additional models for ensembling

    Returns:
        accuracy: Overall accuracy with post-processing
        class_accuracies: Per-class accuracies
    """
    model.eval()
    if ensemble_models:
        for m in ensemble_models:
            m.eval()

    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(7)}
    class_total = {i: 0 for i in range(7)}

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get outputs from primary model
            outputs = model(inputs)

            # If ensemble models are provided, get their outputs too
            if ensemble_models:
                all_outputs = [outputs]
                for m in ensemble_models:
                    all_outputs.append(m(inputs))

                # Apply post-processing with ensembling
                _, predictions, _ = apply_post_processing(
                    all_outputs, ensemble=True, refine=True
                )
            else:
                # Apply post-processing without ensembling
                _, predictions, _ = apply_post_processing(
                    [outputs], ensemble=False, refine=True
                )

            # Calculate accuracy
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

            # Calculate per-class accuracy
            for i in range(7):
                mask = targets == i
                class_total[i] += mask.sum().item()
                class_correct[i] += ((predictions == i) & mask).sum().item()

    accuracy = 100 * correct / total
    class_accuracies = {i: 100 * class_correct[i] / max(1, class_total[i])
                        for i in range(7)}

    return accuracy, class_accuracies
