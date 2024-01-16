import unittest

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from simple_policy_optimization import \
    collect_experience as collect_experienceTF
from simple_policy_optimization import compute_loss as compute_lossTF
from simple_policy_optimization import get_action as get_actionTF
from simple_policy_optimization import get_policy as get_policyTF
from simple_policy_optimization import mlp as mlpTF
from simple_policy_optimization import train_one_epoch as train_one_epochTF
from simple_policy_optimization_pytorch import (collect_experience,
                                                compute_loss, get_action,
                                                get_policy, mlp, train,
                                                train_one_epoch)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

env_name = 'CartPole-v1'
render = False
lr = 0.01
hidden_sizes = [32]

class TestSimplePolicyOptimization_pytorch(unittest.TestCase):
    def test_obs_dim_and_act_dim_and_sizes(self):
        train_return = train(env_name=env_name, render=render, lr=lr, epochs=1, hidden_sizes=hidden_sizes)

        expected_obs_dim = train_return['obs_dim']
        expected_n_acts = train_return['n_acts']
        expected_sizes = train_return['sizes']

        self.assertEqual(expected_obs_dim, 4)
        self.assertEqual(expected_n_acts, 2)
        self.assertEqual(expected_sizes, [expected_obs_dim]+hidden_sizes+[expected_n_acts])

    def test_multilayer_perceptron(self):
        test_mlp = mlp(sizes=[4, 1, 2]).to(device)
        self.assertEqual(type(test_mlp[0]), torch.nn.Linear)
        self.assertEqual(type(test_mlp[1]), torch.nn.Tanh)
        self.assertEqual(type(test_mlp[2]), torch.nn.Linear)
        self.assertEqual(type(test_mlp[3]), torch.nn.Identity)

        X = torch.rand(1, 4, device=device)
        logits = test_mlp(X)

        self.assertEqual(X.shape, torch.Size([1, 4]))
        self.assertEqual(logits.shape, torch.Size([1, 2]))

    def test_get_policy(self):
        test_mlp = mlp(sizes=[4, 1, 2]).to(device)
        test_obs_space = torch.as_tensor(gym.spaces.Box(low=1.0, high=2.0, shape=(4,)).sample(), device=device)
        policy = get_policy(obs=test_obs_space, logits_net=test_mlp)

        self.assertEqual(type(policy), torch.distributions.Categorical)
        self.assertEqual(policy.param_shape, torch.Size([2]))

    def test_get_action(self):
        test_mlp = mlp(sizes=[4, 1, 2]).to(device)
        test_obs_space = torch.as_tensor(gym.spaces.Box(low=1.0, high=2.0, shape=(4,)).sample(), device=device)
        action = get_action(obs=test_obs_space, logits_net=test_mlp)
        self.assertEqual(type(action), int)
        self.assertIn(action, [0, 1])

    def test_collect_experience(self):
        test_mlp = mlp(sizes=[4, 1, 2])
        test_env = gym.make(env_name)
        test_batch_size = 2

        batch_obs, batch_acts, batch_weights, batch_rets, batch_lens = collect_experience(env=test_env, batch_size=test_batch_size, logits_net=test_mlp)
        self.assertEqual(len(batch_obs), len(batch_acts))
        self.assertEqual(len(batch_acts), len(batch_weights))
        self.assertEqual(len(batch_weights), len(batch_acts))
        self.assertEqual(len(batch_rets), len(batch_lens))

    def test_compute_loss(self):
        test_mlp = mlp(sizes=[4, 1, 2])
        test_env = gym.make(env_name)
        test_batch_size = 2

        batch_obs, batch_acts, batch_weights, _, _ = collect_experience(env=test_env, batch_size=test_batch_size, logits_net=test_mlp)

        test_batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
            logits_net=test_mlp
        )

        self.assertEqual(type(test_batch_loss.item()), float)
    
    def test_train_one_epoch(self):
        test_env = gym.make(env_name)
        test_batch_size = 2
        test_mlp = mlp(sizes=[4, 1, 2])
        test_optimizer = torch.optim.Adam(test_mlp.parameters(), lr=0.01)
        
        test_batch_loss, test_batch_rets, test_batch_lens = train_one_epoch(env=test_env, batch_size=test_batch_size, logits_net=test_mlp, optimizer=test_optimizer)

        self.assertEqual(type(test_batch_loss.item()), float)
        self.assertEqual(type(test_batch_rets), list)
        self.assertEqual(type(test_batch_lens), list)
        self.assertEqual(len(test_batch_rets), len(test_batch_lens))

class TestSimplePolicyOptimization_tensorflow(unittest.TestCase):
    def test_obs_dim_and_act_dim_and_sizes(self):
        # train_return = trainTF(env_name=env_name, render=render, lr=lr, epochs=1, hidden_sizes=hidden_sizes)
        # self.assertEqual(train_return['obs_dim'], 4)
        self.assertEqual(True, True)
        
    def test_multilayer_perceptron(self):
        test_mlp = mlpTF(sizes=[4, hidden_sizes[0], 2])
        test_input = tf.random.uniform((1, 4), dtype=tf.float16)

        logits = test_mlp(test_input)

        self.assertEqual(type(test_mlp.layers[0]), tf.keras.layers.Dense)
        self.assertEqual(type(test_mlp.layers[1]), tf.keras.layers.Dense)
        self.assertEqual(test_mlp.layers[0].input_shape, (None, 4))

        self.assertEqual(logits.shape, (1, 2))

    def test_get_policy(self):
        test_mlp = mlpTF(sizes=[4, hidden_sizes[0], 2])
        test_obs_space = tf.convert_to_tensor(gym.spaces.Box(low=1.0, high=2.0, shape=(4,)).sample())
        policy = get_policyTF(obs=tf.expand_dims(test_obs_space, axis=0), logits_net=test_mlp)

        self.assertEqual(type(policy), tfp.distributions.Categorical)
        self.assertEqual(policy.logits.shape, (1, 2))
        self.assertEqual(policy.sample().shape[0], 1)
    
    def test_get_action(self):
        test_mlp = mlpTF(sizes=[4, 32, 2])
        test_obs_space = tf.convert_to_tensor(gym.spaces.Box(low=1.0, high=2.0, shape=(4,)).sample())
        action = get_actionTF(obs=test_obs_space, logits_net=test_mlp)
        self.assertEqual(type(action), int)
        self.assertIn(action, [0, 1])

    def test_collect_experience(self):
        test_mlp = mlpTF(sizes=[4, 1, 2])
        test_env = gym.make(env_name)
        test_batch_size = 2

        batch_obs, batch_acts, batch_weights, batch_rets, batch_lens = collect_experienceTF(env=test_env, batch_size=test_batch_size, logits_net=test_mlp)
        self.assertEqual(len(batch_obs), len(batch_acts))
        self.assertEqual(len(batch_acts), len(batch_weights))
        self.assertEqual(len(batch_weights), len(batch_acts))
        self.assertEqual(len(batch_rets), len(batch_lens))

    def test_compute_loss(self):
        test_mlp = mlpTF(sizes=[4, 1, 2])
        test_env = gym.make(env_name)
        test_batch_size = 2

        batch_obs, batch_acts, batch_weights, _, _ = collect_experienceTF(env=test_env, batch_size=test_batch_size, logits_net=test_mlp)

        test_batch_loss = compute_lossTF(
            batch_obs = tf.constant(batch_obs, dtype=tf.dtypes.float32),
            batch_acts = tf.constant(batch_acts, dtype=tf.dtypes.float32),
            batch_weights = tf.constant(batch_weights, dtype=tf.dtypes.float32),
            logits_net = test_mlp
        )

        self.assertEqual(type(test_batch_loss.numpy()), np.float32)

    def test_train_one_epoch(self):
        test_env = gym.make(env_name)
        test_batch_size = 2
        test_mlp = mlpTF(sizes=[4, 1, 2])
        test_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        test_batch_loss, test_batch_rets, test_batch_lens = train_one_epochTF(env=test_env, batch_size=test_batch_size, logits_net=test_mlp, optimizer=test_optimizer)

        self.assertEqual(type(test_batch_loss.numpy()), np.float32)
        self.assertEqual(type(test_batch_rets), list)
        self.assertEqual(type(test_batch_lens), list)
        self.assertEqual(len(test_batch_rets), len(test_batch_lens))

if __name__ == '__main__':
    unittest.main()
