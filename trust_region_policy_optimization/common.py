import collections
import copy
import os
from functools import wraps

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Input


def CNN(input_shape, conv_kwargs=None):
    """Multiple convolutional layers for approximation
    Default setting is equal to architecture used in DQN

    :param input_shape: (tuple[int]) (H, W, C)
    :param conv_kwargs: (list[param]) list of conv parameters for tl.layers.Conv2d

    Return:
        input tensor, output tensor
    """
    if not conv_kwargs:
        in_channels = input_shape[-1]
        conv_kwargs = [
            {
                "in_channels": in_channels,
                "n_filter": 32,
                "act": tf.nn.relu,
                "filter_size": (8, 8),
                "strides": (4, 4),
                "padding": "VALID",
                "W_init": tf.initializers.GlorotUniform(),
            },
            {
                "in_channels": 32,
                "n_filter": 64,
                "act": tf.nn.relu,
                "filter_size": (4, 4),
                "strides": (2, 2),
                "padding": "VALID",
                "W_init": tf.initializers.GlorotUniform(),
            },
            {
                "in_channels": 64,
                "n_filter": 64,
                "act": tf.nn.relu,
                "filter_size": (3, 3),
                "strides": (1, 1),
                "padding": "VALID",
                "W_init": tf.initializers.GlorotUniform(),
            },
        ]
    l = inputs = tl.layers.Input((1,) + input_shape)

    for i, kwargs in enumerate(conv_kwargs):
        # kwargs['name'] = kwargs.get('name', 'cnn_layer{}'.format(i + 1))
        l = tl.layers.Conv2d(**kwargs)(l)
    outputs = tl.layers.Flatten()(l)

    return inputs, outputs


def MLP(
    input_dim,
    hidden_dim_list,
    w_init=tf.initializers.Orthogonal(0.2),
    activation=tf.nn.relu,
    *args,
    **kwargs
):
    """Multiple fully-connected layers for approximation

    :param input_dim: (int) size of input tensor
    :param hidden_dim_list: (list[int]) a list of dimensions of hidden layers
    :param w_init: (callable) initialization method for weights
    :param activation: (callable) activation function of hidden layers

    Return:
        input tensor, output tensor
    """

    l = inputs = Input([None, input_dim])
    for i in range(len(hidden_dim_list)):
        l = Dense(n_units=hidden_dim_list[i], act=activation, W_init=w_init)(l)
    outputs = l

    return inputs, outputs


def CreateInputLayer(state_space, conv_kwargs=None):
    def CreateSingleInput(single_state_space):
        single_state_shape = single_state_space.shape
        # build structure
        if len(single_state_shape) == 1:
            l = inputs = Input((None,) + single_state_shape, name="input_layer")
        else:
            with tf.name_scope("CNN"):
                inputs, l = CNN(single_state_shape, conv_kwargs=conv_kwargs)
        return inputs, l, single_state_shape

    if isinstance(state_space, gym.spaces.Dict):
        input_dict, layer_dict, shape_dict = (
            collections.OrderedDict(),
            collections.OrderedDict(),
            collections.OrderedDict(),
        )
        for k, v in state_space.spaces.items():
            input_dict[k], layer_dict[k], shape_dict[k] = CreateSingleInput(v)
        return input_dict, layer_dict, shape_dict
    if isinstance(state_space, gym.spaces.Space):
        return CreateSingleInput(state_space)
    else:
        raise ValueError("state space error")


def expand_dims(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result = tf.expand_dims(result, axis=-1)
        return result

    return wrapper


class Distribution(object):
    """A particular probability distribution"""

    def set_param(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sampling from distribution. Allow explore parameters."""
        raise NotImplementedError

    def logp(self, x):
        """Calculate log probability of a sample."""
        return -self.neglogp(x)

    def neglogp(self, x):
        """Calculate negative log probability of a sample."""
        raise NotImplementedError

    def kl(self, *parameters):
        """Calculate Kullback–Leibler divergence"""
        raise NotImplementedError

    def entropy(self):
        """Calculate the entropy of distribution."""
        raise NotImplementedError


class Categorical(Distribution):
    """Creates a categorical distribution"""

    def __init__(self, ndim, logits=None):
        """
        Args:
            ndim (int): total number of actions
            logits (tensor): logits variables
        """
        self._ndim = ndim
        self._logits = logits
        self.param = self._logits

    @property
    def ndim(self):
        return copy.copy(self._ndim)

    def set_param(self, logits):
        """
        Args:
            logits (tensor): logits variables to set
        """
        self._logits = logits
        self.param = self._logits

    def get_param(self):
        return copy.deepcopy(self._logits)

    def sample(self):
        """Sample actions from distribution, using the Gumbel-Softmax trick"""
        u = np.array(
            np.random.uniform(0, 1, size=np.shape(self._logits)), dtype=np.float32
        )
        res = tf.argmax(self._logits - tf.math.log(-tf.math.log(u)), axis=-1)
        return res

    def greedy_sample(self):
        """Get actions greedily"""
        _probs = tf.nn.softmax(self._logits)
        return tf.argmax(_probs, axis=-1)

    def logp(self, x):
        return -self.neglogp(x)

    @expand_dims
    def neglogp(self, x):
        x = np.array(x)
        if np.any(x % 1):
            raise ValueError("Input float actions in discrete action space")
        x = tf.convert_to_tensor(x, tf.int32)
        x = tf.one_hot(x, self._ndim, axis=-1)
        return tf.nn.softmax_cross_entropy_with_logits(x, self._logits)

    @expand_dims
    def kl(self, logits):
        """
        Args:
            logits (tensor): logits variables of another distribution
        """
        a0 = self._logits - tf.reduce_max(self._logits, axis=-1, keepdims=True)
        a1 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1
        )

    @expand_dims
    def entropy(self):
        a0 = self._logits - tf.reduce_max(self._logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)


class DiagGaussian(Distribution):
    """Creates a diagonal Gaussian distribution"""

    def __init__(self, ndim, mean_logstd=None):
        """
        Args:
            ndim (int): the dimenstion of actions
            mean_logstd (tensor): mean and logstd stacked on the last axis
        """
        self._ndim = ndim
        self.mean = None
        self.logstd = None
        self.std = None
        self.action_mean = None
        self.action_scale = None
        self.param = self.mean, self.logstd
        if mean_logstd is not None:
            self.set_param(mean_logstd)

    @property
    def ndim(self):
        return copy.copy(self._ndim)

    def set_param(self, mean_logstd):
        """
        Args:
            mean_logstd (tensor): mean and log std
        """
        self.mean, self.logstd = mean_logstd
        self.std = tf.math.exp(self.logstd)
        self.param = self.mean, self.logstd

    def get_param(self):
        """Get parameters"""
        return copy.deepcopy(self.mean), copy.deepcopy(self.logstd)

    def sample(self):
        """Get actions in deterministic or stochastic manner"""
        return self.mean, self.std * np.random.normal(0, 1, np.shape(self.mean))

    def greedy_sample(self):
        """Get actions greedily/deterministically"""
        return self.mean

    def logp(self, x):
        return -self.neglogp(x)

    @expand_dims
    def neglogp(self, x):
        # here we reverse the action normalization to make the computation of negative log probability correct
        x = (x - self.action_mean) / self.action_scale

        return (
            0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1)
            + 0.5 * np.log(2.0 * np.pi) * float(self._ndim)
            + tf.reduce_sum(self.logstd, axis=-1)
        )

    @expand_dims
    def kl(self, mean_logstd):
        """
        Args:
            mean_logstd (tensor): mean and logstd of another distribution
        """
        mean, logstd = mean_logstd
        return tf.reduce_sum(
            logstd
            - self.logstd
            + (tf.square(self.std) + tf.square(self.mean - mean))
            / (2.0 * tf.square(tf.math.exp(logstd)))
            - 0.5,
            axis=-1,
        )

    @expand_dims
    def entropy(self):
        return tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def make_dist(ac_space):
    """Get distribution based on action space

    :param ac_space: gym.spaces.Space
    """
    if isinstance(ac_space, gym.spaces.Discrete):
        return Categorical(ac_space.n)
    elif isinstance(ac_space, gym.spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussian(ac_space.shape[0])
    else:
        raise NotImplementedError


def plot(episode_rewards, algorithm_name, env_name):
    """
    plot the learning curve, saved as ./img/algorithm_name-env_name.png

    :param episode_rewards: array of floats
    :param algorithm_name: string
    :param env_name: string
    """
    path = os.path.join(".", "img")
    name = algorithm_name + "-" + env_name
    plt.figure(figsize=(10, 5))
    plt.title(name)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, name + ".png"))
    plt.close()


def plot_save_log(episode_rewards, algorithm_name, env_name):
    """
    plot the learning curve, saved as ./img/algorithm_name-env_name.png,
    and save the rewards log as ./log/algorithm_name-env_name.npy

    :param episode_rewards: array of floats
    :param algorithm_name: string
    :param env_name: string
    """
    path = os.path.join(".", "log")
    name = algorithm_name + "-" + env_name
    plot(episode_rewards, algorithm_name, env_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, name), episode_rewards)


def save_model(model, model_name, algorithm_name, env_name):
    """
    save trained neural network model

    :param model: tensorlayer.models.Model
    :param model_name: string, e.g. 'model_sac_q1'
    :param algorithm_name: string, e.g. 'SAC'
    """
    name = algorithm_name + "-" + env_name
    path = os.path.join(".", "model", name)
    if not os.path.exists(path):
        os.makedirs(path)
    # tl.files.save_npz(model.trainable_weights, os.path.join(path, model_name))


def load_model(model, model_name, algorithm_name, env_name):
    """
    load saved neural network model

    :param model: tensorlayer.models.Model
    :param model_name: string, e.g. 'model_sac_q1'
    :param algorithm_name: string, e.g. 'SAC'
    """
    name = algorithm_name + "-" + env_name
    path = os.path.join(".", "model", name)
    try:
        param = tl.files.load_npz(path, model_name + ".npz")
        for p0, p1 in zip(model.trainable_weights, param):
            p0.assign(p1)
    except Exception as e:
        print("Load Model Fails!")
        raise e
