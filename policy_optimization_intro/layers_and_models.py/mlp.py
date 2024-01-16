import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from my_keras import MLP


@tf.function
def train_step(model, images, labels, loss_object, optmizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optmizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

def main(epochs):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize data

    # Add channels dimension
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')

    train_ds = tf.data.Dataset.from_tensor_slices( (x_train, y_train) ).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices( (x_test, y_test) ).batch(32)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optmizer = tf.keras.optimizers.legacy.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model = MLP()

    flatten = tf.keras.layers.Flatten()

    for epoch in range(epochs):
        test_loss.reset_states()
        test_accuracy.reset_states()
        train_accuracy.reset_states()
        train_loss.reset_states()

        for images, labels in train_ds:
            train_step(model, images, labels, loss_object, optmizer, train_loss, train_accuracy)

        for test_images, test_labels in test_ds:
            test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

    # Show one test
    iterator = iter(test_ds)
    random_img = iterator.get_next()[0].numpy()[random.randint(0, 32)]
    prediction = model(tf.expand_dims(random_img, axis=0))
    print(tf.argmax(tf.nn.softmax(prediction), 1).numpy()[0])

    plt.imshow(random_img)

    plt.show()


if __name__ == '__main__':
    EPOCHS = 5
    main(EPOCHS)