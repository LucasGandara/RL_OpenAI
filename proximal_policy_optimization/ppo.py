import datetime
import os

import gymnasium as gym
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

print(f"\nTensorflow version: {tf.__version__}")


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


def get_policy_entropy(obs: tf.Tensor, logits_net: tf.keras.Sequential):
    logits = logits_net(obs)
    return tfp.distributions.Categorical(logits=logits).entropy()


def get_action(obs, net):
    return get_policy(tf.expand_dims(obs, axis=0), net).sample().numpy().item()


def compute_policy_loss(
    old_logp: list,
    batch_obs: list,
    batch_acts: list,
    advantages: list,
    clip_ratio: float,
    policy_network: tf.keras.Sequential,
):
    # Policy loss
    logp = get_policy(batch_obs, policy_network).log_prob(batch_acts)
    ratio = tf.exp(logp - old_logp)
    clip_adv = (
        tf.clip_by_value(
            ratio, clip_value_min=1 - clip_ratio, clip_value_max=1 + clip_ratio
        )
        * advantages
    )
    loss_pi = tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))

    # Extra information
    approx_kl = tf.reduce_mean(logp - old_logp)
    # Compute wheter or not the ratio were clipped
    clipped = tf.greater(ratio, 1 + clip_ratio) | tf.less(ratio, 1 - clip_ratio)
    entropy = get_policy_entropy(batch_obs, policy_network)
    clip_frac = tf.reduce_mean(tf.cast(clipped, tf.float32))
    pi_info = dict(kl=approx_kl, ent=entropy, cf=clip_frac)

    return loss_pi, pi_info


def compute_value_loss(
    batch_obs: list, batch_returns: list, value_function_net: tf.keras.Sequential
):
    vpred: value_function_net = value_function_net(batch_obs)
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
    discount_factor: float,
    gamma=0.95,
    lamda=0.99,
):
    # Make some empty lists for logging
    batch_rets = []  # For measuring episode returns
    batch_lens = []  # For measuring episode lengths
    batch_obs = []  # For observations
    batch_acts = []  # For actions
    batch_discounted_rewards = []  # list for rewards accrued throughout ep
    batch_advantages = []  # for measuring advantage
    batch_logp_pi = []  # for measuring logp(pi)

    obs, _ = env.reset()  # first obs comes from starting distribution
    done = False
    ep_discounted_rewards = []  # list for rewards accrued throughout ep
    ep_value_function = []  # for measuring baseline

    step = 0

    # collect experience by acting in the environment with current policy
    # collect experience by acting in the environment with current policy
    while True:
        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        action = get_action(tf.constant(obs, dtype=tf.float32), logits_net)
        # save action, reward and value function
        batch_acts.append(action)

        baseline_estimate = value_function_net(tf.expand_dims(obs, axis=0))
        ep_value_function.append(baseline_estimate)

        step += 1
        obs, reward, done, _, _ = env.step(action)

        ep_discounted_rewards.append(reward * gamma**step)

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

    batch_logp_pi = get_policy(tf.expand_dims(obs, axis=0), logits_net).log_prob(
        batch_acts
    )
    return (
        batch_obs,
        batch_acts,
        batch_advantages,
        batch_discounted_rewards,
        batch_rets,
        batch_lens,
        batch_logp_pi,
    )


def update_ppo(
    pi_l_old: list,
    batch_obs: list,
    batch_acts: list,
    batch_advantages: list,
    batch_log_pi: list,
    policy_network: tf.keras.Sequential,
    policy_optimizer: tf.keras.optimizers.legacy.Adam,
    value_function_net: tf.keras.Sequential,
    value_function_optimizer: tf.keras.optimizers.legacy.Adam,
    clip_ratio: float,
):
    pi_l_old, pi_info_old = compute_policy_loss(
        old_logp=tf.constant(batch_log_pi, dtype=tf.float32),  # dummy value, not used
        batch_obs=tf.constant(batch_obs, dtype=tf.float32),
        batch_acts=tf.constant(batch_acts, dtype=tf.float32),
        advantages=tf.constant(batch_advantages, dtype=tf.float32),
        clip_ratio=clip_ratio,
        policy_network=policy_network,
    )

    # Reset the optmizers
    for var in policy_optimizer.variables():
        var.assign(tf.zeros_like(var))
    for var in value_function_optimizer.variables():
        var.assign(tf.zeros_like(var))

    # 80: train pi iters parameter
    for i in range(80):
        with tf.GradientTape() as tape:
            batch_policy_loss, pi_info = compute_policy_loss(
                old_logp=tf.constant(batch_log_pi, dtype=tf.float32),
                batch_obs=tf.constant(batch_obs, dtype=tf.float32),
                batch_acts=tf.constant(batch_acts, dtype=tf.float32),
                advantages=tf.constant(batch_advantages, dtype=tf.float32),
                clip_ratio=clip_ratio,
                policy_network=policy_network,
            )
        kl = pi_info["kl"]
        gradients = tape.gradient(batch_policy_loss, policy_network.trainable_variables)
        policy_optimizer.apply_gradients(
            zip(gradients, policy_network.trainable_variables)
        )
        # Target KL parameter
        if kl > 1.5 * 0.01:
            print(f"Early stopping at step {kl.numpy()} due to reaching max kl.")
            break

    # Train the criticlue function
    for i in range(80):
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

    return pi_l_old, batch_value_function_loss, kl.numpy()


def ppo(
    env_name="CartPole-v1",
    lr=0.01,
    discount_factor=0.96,
    hidden_sizes=[32],
    epochs=10,
    local_steps_per_epoch=200,
    gamma=0.95,
    lamda=0.99,
    clip_ratio=0.2,
):
    """
    Proximal Policy Optimization (by clipping)
    """
    env = gym.make(env_name)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"\nObservation dimention: {observation_dim}\naction dim: {action_dim}")

    # Create actor network
    sizes = [observation_dim] + hidden_sizes + [action_dim]
    policy_network = mlp(sizes=sizes, name="policy_network")
    policy_network.summary()

    # Creates critic network
    sizes = [observation_dim] + hidden_sizes + [1]
    value_function_network = mlp(sizes=sizes, name="value_function_network")
    value_function_network.summary()

    # Define optimizers
    policy_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    value_function_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    # Training loop
    for i in range(epochs):
        (
            batch_obs,
            batch_acts,
            batch_advantages,
            batch_discounted_rewards,
            batch_rets,
            batch_lens,
            batch_log_pi,
        ) = collect_experience(
            env,
            1,
            policy_network,
            value_function_network,
            discount_factor,
            gamma,
            lamda,
        )

        policy_loss, value_function_loss, kl = update_ppo(
            tf.constant(0.0, dtype=tf.float32),
            batch_obs,
            batch_acts,
            batch_advantages,
            batch_log_pi,
            policy_network,
            policy_optimizer,
            value_function_network,
            value_function_optimizer,
            clip_ratio,
        )

        tf.summary.scalar("policy_loss", data=policy_loss, step=i)

        print(
            "epoch: %3d \t policy loss: %.3f \t Value function loss: %.3f \tkl %.4f \t return: %.3f \t ep_len: %.3f"
            % (
                i,
                policy_loss,
                value_function_loss,
                kl,
                np.mean(batch_rets),
                np.mean(batch_lens),
            )
        )

    # Render:
    done = False
    env = gym.make(env_name, render_mode="human")
    while not done:
        obs, _ = env.reset()
        for t in range(200):
            env.render()
            action = get_action(tf.constant(obs, dtype=tf.float32), policy_network)
            obs, reward, done, _, _ = env.step(action)
            if done:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--local_steps_per_epoch", type=int, default=200)
    parser.add_argument("--clip", type=float, default=0.2)

    args = parser.parse_args()

    print(
        f"\nUsing PPO policy gradient. Environment: {args.env_name}, epochs: {args.epochs} lr: {args.lr}"
    )

    path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(
        path,
        f"logs/ppo_{args.epochs}epochs_{datetime.datetime.now().strftime('%Y%m%d-%H%H')}",
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    train_summary_writer = tf.summary.create_file_writer(log_dir)

    ppo(
        env_name=args.env_name,
        lr=args.lr,
        discount_factor=args.discount_factor,
        epochs=args.epochs,
        hidden_sizes=args.hidden_sizes,
        local_steps_per_epoch=args.local_steps_per_epoch,
        logger=None,
    )
