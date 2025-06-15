import matplotlib.pyplot as plt
import numpy as np
import os

# Create eval_results directory if it doesn't exist
os.makedirs('eval_results', exist_ok=True)

# Training data from logs
epochs = list(range(1, 58))  # Fixed length to match data points

# Overall accuracy data
val_accs = [0.7623, 0.7682, 0.8128, 0.8250, 0.8267, 0.8459, 0.8351, 0.8476, 0.8509, 0.8479, 0.8495, 0.8465, 0.8501, 0.8373, 0.8479, 0.8537, 0.8557, 0.8651, 0.8587, 0.8590, 0.8537, 0.8529, 0.8582, 0.8598, 0.8590, 0.8546, 0.8607, 0.8629, 0.8551, 0.8668, 0.8674, 0.8638, 0.8668, 0.8635, 0.8598, 0.8610, 0.8671, 0.8638, 0.8643, 0.8638, 0.8632, 0.8582, 0.8596, 0.8665, 0.8604, 0.8596, 0.8615, 0.8621, 0.8624, 0.8576, 0.8612, 0.8638, 0.8649, 0.8559, 0.8540, 0.8654, 0.8582][:57]
ema_val_accs = [0.5043, 0.3631, 0.5004, 0.5729, 0.6211, 0.6576, 0.6916, 0.7180, 0.7417, 0.7593, 0.7712, 0.7863, 0.7969, 0.8030, 0.8119, 0.8181, 0.8222, 0.8247, 0.8292, 0.8348, 0.8384, 0.8409, 0.8440, 0.8476, 0.8479, 0.8487, 0.8515, 0.8534, 0.8546, 0.8565, 0.8598, 0.8607, 0.8640, 0.8646, 0.8663, 0.8660, 0.8668, 0.8679, 0.8690, 0.8699, 0.8693, 0.8685, 0.8699, 0.8738, 0.8727, 0.8741, 0.8732, 0.8743, 0.8755, 0.8752, 0.8743, 0.8724, 0.8721, 0.8716, 0.8716, 0.8716, 0.8716][:57]

# Class-wise accuracy data (sampled at key epochs)
class_names = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
class_accs = {
    'Neutral':   [0.7642, 0.7848, 0.8519, 0.8312, 0.8558, 0.8712, 0.8858],
    'Happiness': [0.9644, 0.8999, 0.9466, 0.9544, 0.9355, 0.9355, 0.9355],
    'Surprise':  [0.8235, 0.9455, 0.9259, 0.7996, 0.8237, 0.9237, 0.9237],
    'Sadness':   [0.5861, 0.5981, 0.6029, 0.7440, 0.7584, 0.7584, 0.7584],
    'Anger':     [0.6531, 0.6219, 0.6719, 0.7875, 0.8000, 0.8000, 0.8000],
    'Disgust':   [0.0000, 0.0000, 0.0000, 0.0000, 0.4722, 0.4722, 0.4722],
    'Fear':      [0.0000, 0.0000, 0.2400, 0.6000, 0.6533, 0.6533, 0.6533],
    'Contempt':  [0.0000, 0.0000, 0.0000, 0.0000, 0.2800, 0.2800, 0.2800]
}

# Create a figure with two subplots
plt.figure(figsize=(15, 10))

# Plot 1: Overall accuracy
plt.subplot(2, 1, 1)
plt.plot(epochs, val_accs, label='Validation Accuracy', marker='o', markersize=3)
plt.plot(epochs, ema_val_accs, label='EMA Validation Accuracy', marker='o', markersize=3)

plt.title('Overall Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add vertical lines at best performance points
best_val_epoch = 30  # Epoch with best validation accuracy
best_ema_epoch = 50  # Epoch with best EMA validation accuracy
plt.axvline(x=best_val_epoch, color='r', linestyle='--', alpha=0.3, label=f'Best Val Acc (86.74%)')
plt.axvline(x=best_ema_epoch, color='g', linestyle='--', alpha=0.3, label=f'Best EMA Val Acc (87.55%)')

# Plot 2: Class-wise accuracy
plt.subplot(2, 1, 2)
sample_epochs = [1, 5, 10, 20, 30, 40, 50]  # Epochs where we have class accuracy data

for class_name in class_names:
    plt.plot(sample_epochs, class_accs[class_name], label=class_name, marker='o')

plt.title('Class-wise Accuracy Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('eval_results/training_progress_with_classes.png', dpi=300, bbox_inches='tight')
plt.close() 