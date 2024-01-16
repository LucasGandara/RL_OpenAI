import io

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def make_simple_figure(y, xlabel="Epoch", ylabel="loss", show_grid=False):
    # Make the graphs of the policy and value function
    figure = plt.figure(figsize=(4, 4))
    figure.suptitle("Policy loss throug epochs")
    plt.plot(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(show_grid)

    return figure


if __name__ == "__main__":
    pass
