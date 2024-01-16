import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def mlp(sizes, activation=tf.keras.activations.tanh, output_activation=tf.identity):
    # Build a feedforward neural network.
    layers = []
    layers.append(tf.keras.layers.Input((sizes[0]), ))
    for size in sizes[1:-1]:
        layers.append(tf.keras.layers.Dense(units=size, activation=activation))
    layers.append(tf.keras.layers.Dense(
        units=sizes[-1], activation=output_activation))
    return tf.keras.Sequential(layers)

# make function to compute action distribution


@tf.function
def get_policy(obs, logits_net):
    logits = logits_net(obs)
    return tfp.distributions.Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)


def get_action(obs, logits_net):
    return get_policy(tf.expand_dims(obs, axis=0), logits_net).sample().numpy().item()

# make loss function whose gradient, for the right data, is policy gradient


def compute_loss(batch_obs, batch_acts, batch_weights, logits_net):
    logp = get_policy(batch_obs, logits_net).log_prob(batch_acts)
    return -tf.reduce_mean(logp * batch_weights)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def collect_experience(env, batch_size, logits_net):
    # make some empty lists for logging.
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths
    batch_obs = []          # for observations
    batch_acts = []         # for actions

    # reset episode-specific variables
    obs, _ = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # collect experience by acting in the environment with current policy
    while True:

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(tf.constant(obs, dtype=tf.float32), logits_net)
        obs, rew, done, _, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a_t|s_t) is reward-to-go from t
            batch_weights += list(reward_to_go(ep_rews))

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []
            obs = obs[0]

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    return batch_obs, batch_acts, batch_weights, batch_rets, batch_lens


def train_one_epoch(env, batch_size, optimizer: tf.keras.optimizers.legacy.Adam, logits_net):
    batch_obs, batch_acts, batch_weights, batch_rets, batch_lens = collect_experience(
        env=env, batch_size=batch_size, logits_net=logits_net)

    # Reset the optimizer
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))

    # Gradients for automatic differentiation
    with tf.GradientTape() as tape:
        batch_loss = compute_loss(
            batch_obs=tf.constant(batch_obs, dtype=tf.dtypes.float32),
            batch_acts=tf.constant(batch_acts, dtype=tf.dtypes.float32),
            batch_weights=tf.constant(batch_weights, dtype=tf.dtypes.float32),
            logits_net=logits_net
        )

    # Perform backpropagation
    gradients = tape.gradient(batch_loss, logits_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_net.trainable_variables))
    return batch_loss, batch_rets, batch_lens


def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)

    assert isinstance(env.observation_space, gym.spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    print(f'Observation dimention: {obs_dim}')
    n_acts = env.action_space.n
    print(f'Number of possible actions: {n_acts}')

    # make core of policy network
    sizes = [obs_dim]+hidden_sizes+[n_acts]
    logits_net = mlp(sizes=sizes)

    # make optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    # Train loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(
            env, batch_size, optimizer, logits_net)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

    env = gym.make(env_name, render_mode='human')
    train_one_epoch(env, batch_size, optimizer, logits_net)

    return {
        'obs_dim': obs_dim,
        'n_acts': n_acts,
        'sizes': sizes
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
