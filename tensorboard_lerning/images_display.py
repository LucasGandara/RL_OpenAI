from datetime import datetime

import numpy as np
import tensorflow as tf
from packaging import version
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert (
    version.parse(tf.__version__).release[0] >= 2
), "This notebook requires TensorFlow 2.0 or above."

if __name__ == "__main__":
    # Download the data. The data is already divided into train and test.
    # The labels are integers representing classes.
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    print("Shape: ", train_images[0].shape)
    print("Label: ", train_labels[0], "->", class_names[train_labels[0]])

    # Notice that the shape of each image in the data set is a rank-2 tensor of shape (28, 28), representing the height and the width.
    # However, tf.summary.image() expects a rank-4 tensor containing (batch_size, height, width, channels). Therefore, the tensors need to be reshaped.

    # Reshape the image for the Summary API.
    img = np.reshape(train_images[0], (-1, 28, 28, 1))
    logdir = "logs/train_data_display_imgs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(train_images[0:25], (-1, 28, 28, 1))
        tf.summary.image("25 training data examples", images, max_outputs=25, step=0)
