# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN)
Author: Lucas Gandara
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import random

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf

print("Keras is using {} as backend".format(keras.backend.backend()))


class Hyperparameters(object):
    num_epochs = 500
    epsilon = 0.1
    epsilon_decay = 0.95
    min_epsilon = 0.1
    gamma = 0.99  # discount factor
    learning_rate = 0.001
    batch_size = 32
    max_episode_duration = 2000
    update_target_every = 15


def create_model(output_size, name):
    return keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(output_size, activation="linear"),
        ],
        name=name,
    )


def train_one_epoch(
    done_history,
    state_history,
    rewards_history,
    action_history,
    state_next_history,
    actor_model,
    env,
    loss_function=keras.losses.mean_squared_error,
):
    indices = np.random.choice(
        range(len(done_history)), size=Hyperparameters.batch_size
    )

    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = [rewards_history[i] for i in indices]
    action_sample = [action_history[i] for i in indices]
    done_sample = keras.ops.convert_to_tensor([float(done_history[i]) for i in indices])

    future_rewards = actor_model(state_next_sample)

    # Q value = reward + discount factor * expected future reward
    updated_q_values = rewards_sample + Hyperparameters.gamma * keras.ops.amax(
        future_rewards, axis=1
    )

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values
    masks = keras.ops.one_hot(action_sample, env.action_space.n)

    with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = actor_model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value
        loss = loss_function(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, actor_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode=None)

    actor_model = create_model(env.action_space.n, "actor_model")
    target_model = create_model(env.action_space.n, "target_model")

    actor_model.compile(optimizer="adam", loss="mse")
    target_model.compile(optimizer="adam", loss="mse")

    target_model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    # Experience replay buffers: Arrays
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []

    max_memory_length = 100000

    epsilon = Hyperparameters.epsilon

    for epoch in range(Hyperparameters.num_epochs):
        observation, _ = env.reset()
        state = np.array(observation)

        episode_reward = 0

        for episode in range(Hyperparameters.max_episode_duration):
            random_sample = random.random()
            if random_sample < Hyperparameters.epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = keras.ops.convert_to_tensor(state)
                state_tensor = keras.ops.expand_dims(state_tensor, 0)
                action_probs = actor_model(state_tensor, training=False)
                # Take action with max Q value
                action = keras.ops.argmax(action_probs[0]).numpy()

                epsilon *= Hyperparameters.epsilon_decay
                epsilon = max(epsilon, Hyperparameters.min_epsilon)

                state_next, reward, done, truncated, _ = env.step(action)
                state_next = np.array(state_next)

                if done:
                    reward = -100
                elif truncated:
                    reward = 100

                episode_reward += reward

                # Save state, actions, rewards and next states
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                state = state_next

                train_one_epoch(
                    done_history,
                    state_history,
                    rewards_history,
                    action_history,
                    state_next_history,
                    actor_model,
                    env,
                    loss_function,
                )

                if done:
                    break

        print("Epoch {}; Episode return: {}".format(epoch, episode_reward))
