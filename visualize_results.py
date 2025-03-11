import matplotlib.pyplot as plt
import yaml

# Load file kết quả training
with open('runs/train/exp/results.yaml', 'r') as f:
    results = yaml.safe_load(f)

# Vẽ Loss
plt.figure(figsize=(10,5))
plt.plot(results['train_loss'], label='Train Loss')
plt.plot(results['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.show()

# Vẽ mAP, Precision, Recall
plt.figure(figsize=(10,5))
plt.plot(results['metrics/mAP_50'], label='mAP@50')
plt.plot(results['metrics/precision'], label='Precision')
plt.plot(results['metrics/recall'], label='Recall')
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.title("Evaluation Metrics Curve")
plt.show()
