import os
import numpy as np
import tensorflow as tf

from src.config import DataConfig, TrainingConfig, AugmentationConfig, ModelConfig
from src.data import FER2013Processor
from src.models import ModelBuilder
from src.training import ModelTrainer, TestTimeAugmentation, ModelEvaluator

# Emotion class names
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def setup_gpu_memory():
    """Set up GPU memory growth to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth set for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")


def main():
    """Main function to orchestrate the entire process."""
    # Set up GPU
    setup_gpu_memory()
    
    # Initialize configurations
    data_config = DataConfig(
        data_path="data/fer2013/train.csv",
        cached_data_path="data/fer2013/preprocessed_data.npz"
    )
    
    training_config = TrainingConfig(
        batch_size=64,
        epochs=100,
        learning_rate=0.0001
    )
    
    augmentation_config = AugmentationConfig(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    model_config = ModelConfig(
        input_shape=(48, 48, 1),
        num_classes=7,
        dropout_rate=0.5
    )
    
    print("Configurations initialized")
    
    # Initialize data processor
    processor = FER2013Processor(data_config, training_config, augmentation_config)
    
    # Load and preprocess data
    print("Loading data...")
    data_dict = processor.load_data()
    
    print("Preprocessing data...")
    processed_data = processor.preprocess(data_dict)
    
    # Create data generators
    print("Creating data generators...")
    generators = processor.create_generators(processed_data)
    
    # Compute class weights to handle imbalance
    print("Computing class weights...")
    class_weights = processor.compute_class_weights(processed_data['train'][1])
    
    # Build model
    print("Building model...")
    model_builder = ModelBuilder(model_config)
    model = model_builder.build()
    
    # Print model summary
    model.summary()
    
    # Initialize trainer and compile model
    print("Initializing trainer...")
    trainer = ModelTrainer(model, training_config)
    trainer.compile_model()
    
    # Create directory for model checkpoints
    os.makedirs("models", exist_ok=True)
    
    # Train model
    print("Training model...")
    history = trainer.train(
        generators['train_generator'],
        generators['val_generator'],
        generators['train_steps'],
        generators['val_steps'],
        class_weights,
        validation_data=generators['val_data'],
        train_data=generators['train_data'],
        class_names=EMOTION_LABELS,
        checkpoint_path="models/best_model.h5"
    )
    
    # Initialize evaluator
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, EMOTION_LABELS)
    
    # Evaluate on validation set
    loss, accuracy = evaluator.evaluate(processed_data['val'][0], processed_data['val'][1])
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred = evaluator.predict(processed_data['val'][0])
    
    # Print classification report
    print("\nClassification Report:")
    print(evaluator.compute_classification_report(processed_data['val'][1], y_pred))
    
    # Plot confusion matrix
    print("Plotting final confusion matrix...")
    evaluator.plot_confusion_matrix(processed_data['val'][1], y_pred, normalize=True)
    
    # Plot training history
    print("Plotting training history...")
    evaluator.plot_training_history(history.history)
    
    # Initialize test-time augmentation
    print("Running test-time augmentation...")
    tta = TestTimeAugmentation(model, augmentation_config)
    
    # Evaluate with TTA
    tta_predictions = tta.predict_batch_with_tta(processed_data['val'][0])
    
    # Print classification report with TTA
    print("\nClassification Report (with Test-Time Augmentation):")
    print(evaluator.compute_classification_report(processed_data['val'][1], tta_predictions))
    
    # Plot confusion matrix with TTA
    print("Plotting confusion matrix with TTA...")
    evaluator.plot_confusion_matrix(processed_data['val'][1], tta_predictions, normalize=True)
    
    print("Done!")


if __name__ == "__main__":
    main() 