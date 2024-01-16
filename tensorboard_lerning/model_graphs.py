from datetime import datetime

import tensorboard
import tensorflow as tf
from packaging import version
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
print("TensorFlow version: ", tf.__version__)
assert (
    version.parse(tf.__version__).release[0] >= 2
), "This notebook requires TensorFlow 2.0 or above."


# The function to be traced.
@tf.function
def my_tf_func(x, y):
    # A simple hand-rolled layer.
    return tf.nn.relu(tf.matmul(x, y))


def model_graphs():
    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0

    logdir = "logs/model_graphs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Train the model.
    model.fit(
        train_images,
        train_labels,
        batch_size=64,
        epochs=5,
        callbacks=[tensorboard_callback],
    )


def tf_functions_graphs():
    # Set up logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/tf_functions/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Sample data for your function.
    x = tf.random.uniform((3, 3))
    y = tf.random.uniform((3, 3))

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = my_tf_func(x, y)
    with writer.as_default():
        tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=logdir)


if __name__ == "__main__":
    model_graphs()
    # tf_functions_graphs()
