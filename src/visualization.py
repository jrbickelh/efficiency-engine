import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss(history, title, filename="loss_plot.png"):
    """
    Plots the training and validation loss curves and saves the image.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.ylim([0, 10])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

    print(f"Saving plot to {filename}...")
    plt.savefig(filename)
    plt.close() # Close figure to free memory
