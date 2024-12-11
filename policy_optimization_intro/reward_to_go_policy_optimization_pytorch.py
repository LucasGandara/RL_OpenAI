import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.optim import Adam


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# make function to compute action distribution
def get_policy(obs, logits_net):
    logits = logits_net(obs)
    return Categorical(logits=logits)


# Don't let the pass distract you
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


# make action selection function (outputs int actions, sampled from policy)
def get_action(obs, logits_net):
    return get_policy(obs, logits_net).sample().item()


# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights, logits_net):
    logp = get_policy(obs, logits_net).log_prob(act)
    return -(logp * weights).mean()


def collect_experience(env, batch_size, logits_net):
    # make some empty lists for logging.
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths
    batch_obs = []  # for observations
    batch_acts = []  # for actions

    # reset episode-specific variables
    obs, _ = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    # collect experience by acting in the environment with current policy
    while True:

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32), logits_net)
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
            # won't render again this epoch
            finished_rendering_this_epoch = True
            env.close()

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    return batch_obs, batch_acts, batch_weights, batch_rets, batch_lens


# for training policy
def train_one_epoch(env, batch_size, logits_net, optimizer):
    batch_obs, batch_acts, batch_weights, batch_rets, batch_lens = collect_experience(
        env=env, batch_size=batch_size, logits_net=logits_net
    )

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(
        obs=torch.as_tensor(batch_obs, dtype=torch.float32),
        act=torch.as_tensor(batch_acts, dtype=torch.int32),
        weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        logits_net=logits_net,
    )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens


def train(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    render=False,
):

    print(f"Training in env: {env_name}")
    # make environment, check spaces, get obs / act dims
    env = gymnasium.make(env_name)
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    sizes = [obs_dim] + hidden_sizes + [n_acts]
    logits_net = mlp(sizes=sizes)

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(
            env=env, batch_size=batch_size, logits_net=logits_net, optimizer=optimizer
        )
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )

    env = gymnasium.make(env_name, render_mode="human")
    batch_loss, batch_rets, batch_lens = train_one_epoch(
        env=env, batch_size=1, logits_net=logits_net, optimizer=optimizer
    )

    return {"obs_dim": obs_dim, "n_acts": n_acts, "sizes": sizes}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    print("\nUsing simplest formulation of policy gradient.\n")
    train(env_name=args.env_name, render=args.render, lr=args.lr)
