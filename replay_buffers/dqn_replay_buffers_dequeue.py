# Author: Lucas Gandara
from collections import deque

import gymnasium as gym
import keras


class Model(keras.Model):
    def __init__(self, input_size, output_size, model_name):
        super(Model, self).__init__(name=model_name)
        self.input_layer = keras.layers.InputLayer(input_shape=(input_size,))
        self.dense1 = keras.layers.Dense(128, activation="relu")
        self.dense2 = keras.layers.Dense(128, activation="relu")
        self.output_layer = keras.layers.Dense(output_size, activation="linear")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    replay_buffer = deque(maxlen=1000)

    env = gym.make("CartPole-v1", render_mode="human")

    target_model = Model(
        env.observation_space.shape[0], env.action_space.n, "target_model"
    )

    target_model.summary()
