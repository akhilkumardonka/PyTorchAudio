import numpy as np
import matplotlib.pyplot as plt

def lossPlot(train_losses, valid_losses):
    tl = np.array(train_losses).mean(axis=1)
    vl = np.array(valid_losses).mean(axis=1)
    epochs = np.arange(0, len(tl), dtype=int)

    plt.figure(figsize=(12, 8))
    plt.plot(tl, color='r', label='Train Loss')
    plt.plot(vl, color='b', label='Valid Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig('loss_result.jpg')