import copy

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from common import CreateInputLayer, make_dist
from tensorlayer.layers import Dense
from tensorlayer.models import Model


class StochasticPolicyNetwork(Model):
    def __init__(
        self,
        state_space,
        action_space,
        hidden_dim_list,
        w_init=tf.keras.initializers.glorot_normal(),
        activation=tf.nn.relu,
        output_activation=tf.nn.tanh,
        log_std_min=-20,
        log_std_max=2,
        trainable=True,
        name=None,
        state_conditioned=False,
    ):
        """ 
        Stochastic continuous/discrete policy network with multiple fully-connected layers 
        
        :param state_space: (gym.spaces) space of the state from gym environments
        :param action_space: (gym.spaces) space of the action from gym environments
        :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
        :param w_init: (callable) weights initialization
        :param activation: (callable) activation function
        :param output_activation: (callable or None) output activation function
        :param log_std_min: (float) lower bound of standard deviation of action
        :param log_std_max: (float) upper bound of standard deviation of action
        :param trainable: (bool) set training and evaluation mode

        Tips: We recommend to use tf.nn.tanh for output_activation, especially for continuous action space, \
            to ensure the final action range is exactly the same as declared in action space after action normalization.
        """
        self._state_space, self._action_space = state_space, action_space

        if isinstance(self._action_space, gym.spaces.Discrete):
            self._action_shape = (self._action_space.n,)
            self.policy_dist = make_dist(
                self._action_space
            )  # create action distribution
        elif isinstance(self._action_space, gym.spaces.Box):  # normalize action
            assert len(self._action_space.shape) == 1
            self._action_shape = self._action_space.shape

            assert all(self._action_space.low < self._action_space.high)
            action_bounds = [self._action_space.low, self._action_space.high]
            self._action_mean = np.mean(action_bounds, 0)
            self._action_scale = action_bounds[1] - self._action_mean

            self.policy_dist = make_dist(
                self._action_space
            )  # create action distribution
            self.policy_dist.action_mean = self._action_mean
            self.policy_dist.action_scale = self._action_scale
        else:
            raise NotImplementedError

        self._state_conditioned = state_conditioned

        obs_inputs, current_layer, self._state_shape = CreateInputLayer(state_space)

        # build structure
        if isinstance(state_space, gym.spaces.Dict):
            assert isinstance(obs_inputs, dict)
            assert isinstance(current_layer, dict)
            self.input_dict = obs_inputs
            obs_inputs = list(obs_inputs.values())
            current_layer = tl.layers.Concat(-1)(list(current_layer.values()))

        with tf.name_scope("MLP"):
            for i, dim in enumerate(hidden_dim_list):
                current_layer = Dense(
                    n_units=dim,
                    act=activation,
                    W_init=w_init,
                    name="hidden_layer%d" % (i + 1),
                )(current_layer)

        with tf.name_scope("Output"):
            if isinstance(action_space, gym.spaces.Discrete):
                outputs = Dense(
                    n_units=self.policy_dist.ndim, act=output_activation, W_init=w_init
                )(current_layer)
            elif isinstance(action_space, gym.spaces.Box):
                mu = Dense(
                    n_units=self.policy_dist.ndim, act=output_activation, W_init=w_init
                )(current_layer)

                if self._state_conditioned:
                    log_sigma = Dense(
                        n_units=self.policy_dist.ndim, act=None, W_init=w_init
                    )(current_layer)
                    log_sigma = tl.layers.Lambda(
                        lambda x: tf.clip_by_value(x, log_std_min, log_std_max)
                    )(log_sigma)
                    outputs = [mu, log_sigma]
                else:
                    outputs = mu
                    self._log_sigma = tf.Variable(
                        np.zeros(self.policy_dist.ndim, dtype=np.float32)
                    )
            else:
                raise NotImplementedError

        # make model
        super().__init__(inputs=obs_inputs, outputs=outputs, name=name)
        if (
            isinstance(self._action_space, gym.spaces.Box)
            and not self._state_conditioned
        ):
            self.trainable_weights.append(self._log_sigma)

        if trainable:
            self.train()
        else:
            self.eval()

    def __call__(self, states, *args, greedy=False, **kwargs):
        if isinstance(self._state_space, gym.spaces.Dict):
            states = np.array(states).transpose([1, 0]).tolist()
        else:
            if np.shape(states)[1:] != self.state_shape:
                raise ValueError(
                    "Input state shape error. Shape should be {} but your shape is {}".format(
                        (None,) + self.state_shape, np.shape(states)
                    )
                )
            states = np.array(states, dtype=np.float32)
        params = super().__call__(states, *args, **kwargs)
        if (
            isinstance(self._action_space, gym.spaces.Box)
            and not self._state_conditioned
        ):
            params = params, self._log_sigma
        self.policy_dist.set_param(params)
        if greedy:
            result = self.policy_dist.greedy_sample()
        else:
            result = self.policy_dist.sample()

        if isinstance(self._action_space, gym.spaces.Box):  # normalize action
            if greedy:
                result = result * self._action_scale + self._action_mean
            else:
                result, explore = result
                result = result * self._action_scale + self._action_mean + explore

            result = tf.clip_by_value(
                result, self._action_space.low, self._action_space.high
            )
        return result

    def random_sample(self):
        """generate random actions for exploration"""

        if isinstance(self._action_space, gym.spaces.Discrete):
            return np.random.choice(self._action_space.n, 1)[0]
        else:
            return np.random.uniform(
                self._action_space.low, self._action_space.high, self._action_shape
            )

    @property
    def state_space(self):
        return copy.deepcopy(self._state_space)

    @property
    def action_space(self):
        return copy.deepcopy(self._action_space)

    @property
    def state_shape(self):
        return copy.deepcopy(self._state_shape)

    @property
    def action_shape(self):
        return copy.deepcopy(self._action_shape)
