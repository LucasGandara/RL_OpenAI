import ale_py
import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from keras import layers

num_actions = 4


class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        # Define layers in the constructor
        self.lambda_layer = layers.Lambda(
            lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
            output_shape=(84, 84, 4),
        )

        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu", name="conv1")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu", name="conv2")
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation="relu", name="conv3")

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.output_layer = layers.Dense(num_actions, activation="linear")

    def call(self, inputs):
        # Define the forward pass
        x = self.lambda_layer(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)


gym.register_envs(ale_py)
env = gym.make("BreakoutNoFrameskip-v4")  # , render_mode="human")
# Environment preprocessing
env = AtariPreprocessing(env)
# Stack four frames
env = FrameStackObservation(env, 4)

observation, _ = env.reset()
state = np.array(observation)


model = Model(num_actions)

state_tensor = keras.ops.convert_to_tensor(state, dtype=tf.float32)
state_tensor = keras.ops.expand_dims(state_tensor, 0)
action_probs = model(state_tensor, training=False)


model.compile(optimizer="adam", loss="mse")
model.summary()
