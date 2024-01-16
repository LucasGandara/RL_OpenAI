import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(DenseLayer, self).__init__()
        self.units = units
        self.flatten = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(DenseLayer, self).build(input_shape)

    def call(self, inputs):
        flattened_inputs = self.flatten(inputs)
        z = tf.matmul(flattened_inputs, self.w) + self.b
        g = tf.keras.activations.relu(z)
        return g

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class MLP(tf.keras.Model, object):
    def __init__(self):
        super(MLP, self).__init__(name='Multi_layer_perceptron')
        self.conv2D = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.layer_1 = DenseLayer(units=128)
        self.layer_2 = DenseLayer(units=10)
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, x):
        y = self.conv2D(x)
        y = self.flatten(y)
        y = self.d1(y)
        y = self.d2(y)
        return y

if __name__ == '__main__':
    dense_layer = DenseLayer(units=2)
    print(dense_layer(tf.constant([[1, 2, 3]], dtype=tf.float32)))