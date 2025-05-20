import matplotlib.pyplot as plt
import numpy as np
import json

with open("assets/metrics.json", "r") as f:
    metrics = json.load(f)

losses = metrics["losses"]
accuracies = metrics["accuracies"]
epochs = np.arange(1, len(losses) + 1)

def moving_average(x, window=3):
    return np.convolve(x, np.ones(window)/window, mode='valid')

ma_loss = moving_average(losses)

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Loss', marker='o')
plt.plot(epochs[1:-1], ma_loss, label='Moving Avg (w=3)', linestyle='--')
plt.plot(epochs, np.array(accuracies) * max(losses), label='Accuracy (scaled)', linestyle=':', color='green')

plt.title('Transformer Training Overview')
plt.xlabel('Epoch')
plt.ylabel('Loss / Scaled Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("assets/transformer_training_combined_plot.png")
plt.show()
