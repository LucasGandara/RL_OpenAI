# Author: Lucas Gandara

import enum
import math
import random
import tempfile

import gymnasium as gym
import tensordict
import torch
import torch.nn.modules
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObservationType(enum.Enum):
    cart_position = 0
    cart_velocity = 1
    pole_angle = 2
    pole_angular_velocity = 3


class Hyperparameters(object):
    num_epochs = 10
    epsilon = 1
    epsilon_decay = 0.99
    gamma = 0.99
    batch_size = 32
    max_episode_duration = 2000


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, 24)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(24, 24)
        self.activation2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(24, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.output(x)

        return x


def train_one_epoch(
    replay_buffer: TensorDictReplayBuffer,
    actor_model: Model,
    target_model: Model,
    optimizer: torch.optim.Optimizer,
) -> None:
    if len(replay_buffer) < Hyperparameters.batch_size:
        return
    # Train one epoch
    transitions = replay_buffer.sample(Hyperparameters.batch_size)

    rewards = transitions["reward"].to(device)
    actions = transitions["action"].to(device)
    states = transitions["state"].to(device)

    state_action_values = actor_model(states)


def main():
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(1000, scratch_dir=tempfile.TemporaryDirectory()),
        sampler=PrioritizedSampler(
            max_capacity=100, alpha=0.0, beta=1.1
        ),  # TODO: change this to 0.8; 0 is the uniform case
        batch_size=1,
        collate_fn=lambda x: x,
    )

    env = gym.make("CartPole-v1", render_mode="human")
    target_model = Model(
        input_size=env.observation_space.shape[0], output_size=env.action_space.n
    ).to(device)
    actor_model = Model(
        input_size=env.observation_space.shape[0], output_size=env.action_space.n
    ).to(device)
    target_model.load_state_dict(actor_model.state_dict())

    optimizer = torch.optim.Adam(actor_model.parameters(), lr=0.001)

    for epochs in range(Hyperparameters.num_epochs):
        epsilon = Hyperparameters.epsilon
        (
            state,
            _,
        ) = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        for episode in range(Hyperparameters.max_episode_duration):
            epsilon *= Hyperparameters.epsilon_decay

            random_sample = random.random()
            action = None
            if random_sample < epsilon:
                action = torch.tensor(
                    [[env.action_space.sample()]], device=device, dtype=torch.int
                ).item()
            else:
                with torch.no_grad():
                    action = actor_model(state).max(0).indices.view(1, 1).item()

            observation, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                next_state = None
                reward -= 100
            if truncated:
                next_state = None
                reward += 100
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).view(1, -1)

            reward = torch.tensor([reward], device=device)

            replay_buffer.add(
                tensordict.TensorDict(
                    {
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                        "reward": reward,
                    }
                ).to(device)
            )
            state = next_state

            train_one_epoch(replay_buffer, actor_model, target_model, optimizer)


if __name__ == "__main__":
    main()
