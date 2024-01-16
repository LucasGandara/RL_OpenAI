import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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
        layers.append(tf.keras.layers.Dropout(0.2))
    layers.append(tf.keras.layers.Dense(units=sizes[-1], activation=output_activation))
    return tf.keras.Sequential(layers, name=name)


def get_policy(obs, logits_net):
    # Make function to compute action distribution
    logits = logits_net(obs)
    return tfp.distributions.Categorical(logits=logits)


def get_action(obs, logits_net):
    return get_policy(tf.expand_dims(obs, axis=0), logits_net).sample().numpy().item()


def compute_policy_loss(batch_obs, batch_acts, batch_advantages, logits_net):
    # make loss function whose gradient, for the right data, is policy gradient
    logp = get_policy(batch_obs, logits_net).log_prob(batch_acts)
    return -tf.reduce_mean(logp * batch_advantages)


def compute_value_loss(batch_obs, batch_returns, value_function_net):
    vpred = value_function_net(batch_obs)
    return tf.reduce_mean((vpred - batch_returns) ** 2)


def reward_to_go(rews):
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
    discount_factor=0.95,
):
    # make some empty lists for logging.
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_expected_value_function = []  # for measuring baseline
    batch_advantages = []  # for measuring advantage
    batch_rewards_to_go = []  # for measuring reward to go

    # reset episode-specific variables
    obs, _ = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    step = 0

    # collect experience by acting in the environment with current policy
    while True:
        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        action = get_action(tf.constant(obs, dtype=tf.float32), logits_net)
        step += 1
        obs, reward, done, _, _ = env.step(action)

        discounted_reward = reward * discount_factor**step
        baseline_estimate = value_function_net(tf.expand_dims(obs, axis=0))

        # save action, reward and value function
        batch_acts.append(action)
        ep_rews.append(discounted_reward)
        batch_expected_value_function.append(baseline_estimate)

        # The weight for eacht logprob(a|s) is At = (Discounted R(Tau) - baseline estimate)
        Advantage = discounted_reward - baseline_estimate
        batch_advantages.append(Advantage)

        if done:
            # If episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # The rewards to go are the truth of the value functions
            batch_rewards_to_go += list(reward_to_go(ep_rews))

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []
            obs = obs[0]
            step = 0

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    return (
        batch_obs,
        batch_acts,
        batch_advantages,
        batch_rewards_to_go,
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
):
    (
        batch_obs,
        batch_acts,
        batch_advantages,
        batch_rewards_to_go,
        batch_rets,
        batch_lens,
    ) = collect_experience(
        env, batch_size, logits_net, value_function_net, discount_factor
    )

    # Reset the optmizers
    for var in logits_optimizer.variables():
        var.assign(tf.zeros_like(var))
    for var in value_function_optimizer.variables():
        var.assign(tf.zeros_like(var))

    # Gradients for automatic differentiation

    # For the policy
    with tf.GradientTape() as tape:
        policy_loss_by_epoch = compute_policy_loss(
            batch_obs=tf.constant(batch_obs, dtype=tf.dtypes.float32),
            batch_acts=tf.constant(batch_acts, dtype=tf.dtypes.float32),
            batch_advantages=batch_advantages,
            logits_net=logits_net,
        )

    # Perform backpropagation
    gradients = tape.gradient(policy_loss_by_epoch, logits_net.trainable_variables)
    logits_optimizer.apply_gradients(zip(gradients, logits_net.trainable_variables))

    # For the value function
    with tf.GradientTape() as tape:
        batch_value_function_loss = compute_value_loss(
            batch_obs=tf.constant(batch_obs, dtype=tf.dtypes.float32),
            batch_returns=tf.constant(batch_rewards_to_go, dtype=tf.dtypes.float32),
            value_function_net=value_function_net,
        )
    gradients = tape.gradient(
        batch_value_function_loss, value_function_net.trainable_variables
    )
    value_function_optimizer.apply_gradients(
        zip(gradients, value_function_net.trainable_variables)
    )

    return batch_value_function_loss, policy_loss_by_epoch, batch_rets, batch_lens


def train(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    discount_factor=0.95,
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

    # Train loop
    for i in range(epochs):
        (
            batch_value_function_loss,
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
        print(
            "epoch: %3d \t policy loss: %.3f \t Value function loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (
                i,
                policy_loss,
                batch_value_function_loss,
                np.mean(batch_rets),
                np.mean(batch_lens),
            )
        )

    env = gym.make(env_name, render_mode="human")
    (
        batch_value_function_loss,
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

    print(f"\nUsing vanilla policy gradient. Environment: {args.env_name}\n")
    train(env_name=args.env_name, lr=args.lr, discount_factor=0.99, epochs=50)
