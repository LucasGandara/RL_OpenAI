import os
import sys
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../utils")
from tensorboard_utils import make_simple_figure, plot_to_image


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(
    sizes,
    activation=tf.keras.activations.tanh,
    output_activation=tf.identity,
    name="Sequential",
):
    """Build a feedforward neural network."""
    layers = []
    layers.append(
        tf.keras.layers.Input(
            (sizes[0]),
        )
    )
    for size in sizes[1:-1]:
        layers.append(tf.keras.layers.Dense(units=size, activation=activation))
    layers.append(tf.keras.layers.Dense(units=sizes[-1], activation=output_activation))
    return tf.keras.Sequential(layers, name=name)


def get_policy(obs, logits_net):
    # Make function to compute action distribution
    logits = logits_net(obs)
    return tfp.distributions.Categorical(logits=logits)


def get_action(obs, logits_net):
    return get_policy(tf.expand_dims(obs, axis=0), logits_net).sample().numpy().item()


def compute_policy_loss(batch_obs, batch_acts, advantages, logits_net):
    # make loss function whose gradient, for the right data, is policy gradient
    logp = get_policy(batch_obs, logits_net).log_prob(batch_acts)
    return -tf.reduce_mean(logp * advantages)


def compute_value_loss(batch_obs, batch_returns, value_function_net):
    vpred = value_function_net(batch_obs)
    return tf.reduce_mean((vpred - batch_returns) ** 2)


def reward_to_go(rews, discount=1):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def collect_experience(
    env: gym.Env,
    batch_size: list,
    logits_net: tf.keras.Sequential,
    value_function_net: tf.keras.Sequential,
    gamma=0.95,
    lamda=0.99,
):
    # make some empty lists for logging.
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_discounted_rewards = []  # list for rewards accrued throughout ep
    batch_advantages = []  # for measuring advantage

    # reset episode-specific variables
    obs, _ = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_discounted_rewards = []  # list for rewards accrued throughout ep
    ep_value_function = []  # for measuring baseline

    step = 0
    # collect experience by acting in the environment with current policy
    while True:
        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        action = get_action(tf.constant(obs, dtype=tf.float32), logits_net)
        step += 1
        obs, reward, done, _, _ = env.step(action)

        ep_discounted_rewards.append(reward * gamma**step)

        baseline_estimate = value_function_net(tf.expand_dims(obs, axis=0))

        # save action, reward and value function
        batch_acts.append(action)

        ep_value_function.append(baseline_estimate)

        if done or step >= 200:
            # If episode is over, record info about episode
            ep_ret, ep_len = sum(ep_discounted_rewards), len(ep_discounted_rewards)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # The rewards to go are the truth of the value functions
            # batch_rewards_to_go += list(reward_to_go(ep_rews))
            batch_ep_returns = [ep_ret] * ep_len

            # Calculate the GAE-Lambda
            deltas = ep_discounted_rewards[:-1] + tf.squeeze(
                tf.constant(lamda) * ep_value_function[1:]
            )

            batch_advantages += list(discount_cumsum(deltas, gamma))

            batch_discounted_rewards += list(reward_to_go(ep_discounted_rewards, gamma))

            # reset episode-specific variables
            step = 0
            obs, done, ep_discounted_rewards, ep_value_function = (
                env.reset(),
                False,
                [],
                [],
            )
            obs = obs[0]

            # The last state doesn't have advantage since there is no further value of the action
            batch_obs.pop()
            batch_acts.pop()
            batch_discounted_rewards.pop()

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    return (
        batch_obs,
        batch_acts,
        batch_advantages,
        batch_discounted_rewards,
        batch_rets,
        batch_lens,
    )


def train_one_epoch(
    env: gym.Env,
    batch_size: int,
    logits_optimizer: tf.keras.optimizers.legacy.Adam,
    value_function_optimizer: tf.keras.optimizers.legacy.Adam,
    logits_net: tf.keras.models.Sequential,
    value_function_net: tf.keras.models.Sequential,
    discount_factor=0.95,
    gamma=0.99,
):
    (
        batch_obs,
        batch_acts,
        batch_advantages,
        batch_discounted_rewards,
        batch_rets,
        batch_lens,
    ) = collect_experience(
        env, batch_size, logits_net, value_function_net, discount_factor, gamma
    )

    # Reset the optmizers
    for var in logits_optimizer.variables():
        var.assign(tf.zeros_like(var))
    for var in value_function_optimizer.variables():
        var.assign(tf.zeros_like(var))

    # Gradients for automatic differentiation

    # For the policy
    with tf.GradientTape() as tape:
        batch_policy_loss = compute_policy_loss(
            batch_obs=tf.constant(batch_obs, dtype=tf.dtypes.float32),
            batch_acts=tf.constant(batch_acts, dtype=tf.dtypes.float32),
            advantages=tf.constant(batch_advantages, dtype=tf.dtypes.float32),
            logits_net=logits_net,
        )

    # Perform backpropagation
    gradients = tape.gradient(batch_policy_loss, logits_net.trainable_variables)
    logits_optimizer.apply_gradients(zip(gradients, logits_net.trainable_variables))

    # For the value function
    with tf.GradientTape() as tape:
        batch_value_function_loss = compute_value_loss(
            batch_obs=tf.constant(batch_obs, dtype=tf.dtypes.float32),
            batch_returns=tf.constant(batch_advantages, dtype=tf.dtypes.float32),
            value_function_net=value_function_net,
        )
    gradients = tape.gradient(
        batch_value_function_loss, value_function_net.trainable_variables
    )
    value_function_optimizer.apply_gradients(
        zip(gradients, value_function_net.trainable_variables)
    )

    return batch_value_function_loss, batch_policy_loss, batch_rets, batch_lens


def train(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    discount_factor=0.95,
    log_writter=None,
):
    env = gym.make(env_name)

    assert isinstance(
        env.observation_space, gym.spaces.Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    print(f"Observation dimention: {obs_dim}")
    n_acts = env.action_space.n
    print(f"Number of possible actions: {n_acts}\n")

    # make core of policy network and value function
    sizes = [obs_dim] + hidden_sizes + [n_acts]
    logits_net = mlp(sizes=sizes, name="logits_net")
    logits_net.summary()

    print("\n\n")

    sizes = [obs_dim] + hidden_sizes + [1]
    value_function_net = mlp(sizes=sizes, name="value_function_net")
    value_function_net.summary()

    # make optimizer
    logits_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    value_function_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    print("\nStart training...\n")

    # Initialize some variables for tensorboard logging
    batch_policy_loss = np.zeros(epochs)
    batch_value_function_loss = np.zeros(epochs)

    # Train loop
    for i in range(epochs):
        (
            value_function_loss,
            policy_loss,
            batch_rets,
            batch_lens,
        ) = train_one_epoch(
            env,
            batch_size,
            logits_optimizer,
            value_function_optimizer,
            logits_net,
            value_function_net,
            discount_factor,
        )
        batch_policy_loss[i] = policy_loss.numpy()
        batch_value_function_loss[i] = value_function_loss.numpy()
        print(
            "epoch: %3d \t policy loss: %.3f \t Value function loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (
                i,
                policy_loss,
                value_function_loss,
                np.mean(batch_rets),
                np.mean(batch_lens),
            )
        )

    policy_loss_figure = make_simple_figure(
        batch_policy_loss, xlabel="Epoch", ylabel="Policy loss", show_grid=True
    )
    value_function_loss_figure = make_simple_figure(
        batch_value_function_loss,
        xlabel="Epoch",
        ylabel="Value Function loss",
        show_grid=True,
    )

    if log_writter:
        with file_writer.as_default():
            tf.summary.image("Policy loss", plot_to_image(policy_loss_figure), step=0)
            tf.summary.image(
                "Value functions loss",
                plot_to_image(value_function_loss_figure),
                step=0,
            )

    env = gym.make(env_name, render_mode="human")
    (
        value_function_loss,
        policy_loss,
        batch_rets,
        batch_lens,
    ) = train_one_epoch(
        env,
        batch_size,
        logits_optimizer,
        value_function_optimizer,
        logits_net,
        value_function_net,
        discount_factor,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    logdir = (
        "logs/gae_lambda/"
        + f"{args.env_name}_{args.lr}_{args.epochs}_"
        + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    file_writer = tf.summary.create_file_writer(logdir)

    print(
        f"\nUsing vanilla policy gradient. Environment: {args.env_name}\n, epochs: {args.epochs}, lr={args.lr}"
    )
    train(
        env_name=args.env_name,
        lr=args.lr,
        discount_factor=0.99,
        epochs=args.epochs,
        log_writter=file_writer,
    )
