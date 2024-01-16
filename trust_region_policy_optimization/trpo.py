# example taken from: https://github.com/tensorlayer/RLzoo/blob/master/rlzoo/algorithms/trpo/trpo.py#L329
import copy
import time

import gymnasium as gym
import numpy as np
import tensorflow as tf
from ActorNetwork import StochasticPolicyNetwork
from common import load_model, make_dist, plot_save_log, save_model
from ValueNetwork import ValueNetwork

EPS = 1e-8


class TRPO:
    """
    trpo class
    """

    def __init__(
        self, net_list, optimizers_list, damping_coeff=0.1, cg_iters=10, delta=0.01
    ):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param damping_coeff: Artifact for numerical stability
        :param cg_iters: Number of iterations of conjugate gradient to perform
        :param delta: KL-divergence limit for TRPO update.
        """
        assert len(net_list) == 2
        assert len(optimizers_list) == 1

        self.name = "trpo"

        self.critic, self.actor = net_list

        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

        self.damping_coeff, self.cg_iters = damping_coeff, cg_iters
        self.delta = delta

        self.critic_optimizer = optimizers_list[0]

        self.old_dist = make_dist(self.actor.action_space)

    @staticmethod
    def flat_concat(xs):
        """
        flat concat input
        :param xs: a list of tensor
        :return: flat tensor
        """
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    @staticmethod
    def assign_params_from_flat(x, params):
        """
        assign params from flat input

        :param x:
        :param params:

        :return: group
        """
        flat_size = lambda p: int(
            np.prod(p.shape.as_list())
        )  # the 'int' is important for scalars
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])

    def get_action(self, state):
        """
        Choose action

        :param s: state

        :return: clipped act
        """
        return self.actor([state])[0].numpy()

    def get_v(self, state):
        """
        Compute value

        :param state: state

        :return: value
        """
        if state.ndim < 2:
            state = state[np.newaxis, :]
        res = self.critic(state)[0, 0]
        return res

    def train_critic(self, tfdc_r, s):
        """
        Update critic network

        :param tfdc_r: cumulative reward
        :param s: state

        :return: None
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v
            closs = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

    def save_ckpt(self, env_name):
        """
        save trained weights

        :return: None
        """
        save_model(self.actor, "actor", self.name, env_name)
        save_model(self.critic, "critic", self.name, env_name)

    def load_ckpt(self, env_name):
        """
        load trained weights

        :return: None
        """
        load_model(self.actor, "actor", self.name, env_name)
        load_model(self.critic, "critic", self.name, env_name)

    def calculate_adventages(self, tfs, tfdc_r):
        """
        Calculate advantage

        :param tfs: state
        :param tfdc_r: cumulative reward

        :return: advantage
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def set_pi_params(self, v_ph):
        """
        set actor trainable parameters

        :param v_ph: inputs

        :return: None
        """
        pi_params = self.actor.trainable_weights
        self.assign_params_from_flat(v_ph, pi_params)

    def conjugate_gradient(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = copy.deepcopy(
            b
        )  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)

        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def eval(self, batch_states, batch_actions, batch_advantages, oldpi_prob):
        """
        Function to evaluate kl divergence and surrogate loss
        with tf.autograd
        """
        _ = self.actor(batch_states)
        pi_prob = tf.exp(self.actor.policy_dist.logp(batch_actions))
        ratio = pi_prob / (oldpi_prob + EPS)

        surrogate = ratio * batch_advantages
        aloss = -tf.reduce_mean(surrogate)
        kl = self.old_dist.kl(self.actor.policy_dist.param)
        # kl = tfp.distributions.kl_divergence(oldpi, pi)
        kl = tf.reduce_mean(kl)
        return aloss, kl

    def train_actor(
        self,
        batch_states,
        batch_actions,
        batch_advantages,
        oldpi_prob,
        backtrack_iters,
        backtrack_coeff,
    ):
        batch_states = np.array(batch_states, dtype=np.float32)
        batch_actions = np.array(batch_actions, dtype=np.float32)
        batch_advantages = np.array(batch_advantages, dtype=np.float32)
        with tf.GradientTape() as tape:
            aloss, kl = self.eval(
                batch_states, batch_actions, batch_advantages, oldpi_prob
            )

        actor_grads = tape.gradient(aloss, self.actor.trainable_weights)
        actor_grads = self.flat_concat(actor_grads)

        pi_l_old = aloss

        Hx = lambda x: self.hessian_vector_products(
            batch_states, batch_actions, batch_advantages, oldpi_prob, x
        )

        x = self.conjugate_gradient(Hx, actor_grads)

        alpha = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + EPS))

        old_params = self.flat_concat(self.actor.trainable_weights)

        for j in range(backtrack_iters):
            self.set_pi_params(old_params - alpha * x * backtrack_coeff**j)
            kl, pi_l_new = self.eval(
                batch_states, batch_actions, batch_advantages, oldpi_prob
            )
            if kl <= self.delta and pi_l_new <= pi_l_old:
                # Accepting new params at step j of line search.
                break

            if j == backtrack_iters - 1:
                # Line search failed! Keeping old params.
                self.set_pi_params(old_params)

    def hessian_vector_products(
        self, batch_states, batch_actions, batch_advantages, oldpi_prob, v_ph
    ):
        """
        Hessian vector products
        """
        params = self.actor.trainable_weights

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                aloss, kl = self.eval(
                    batch_states, batch_actions, batch_advantages, oldpi_prob
                )
            gradient = tape0.gradient(aloss, params)
            gradient = self.flat_concat(gradient)

            assert v_ph.shape == gradient.shape

            v = tf.reduce_sum(gradient * v_ph)

        grad = tape1.gradient(v, params)
        hvp = self.flat_concat(grad)

        if self.damping_coeff > 0:
            hvp += self.damping_coeff * v_ph

        return hvp

    def update(
        self,
        batch_states,
        batch_actions,
        batch_rewards,
        train_critic_iters,
        backtrack_iters,
        backtrack_coeff,
    ):
        """
        update trpo

        :return: None
        """
        adv = self.calculate_adventages(batch_states, batch_rewards)
        _ = self.actor(batch_states)
        oldpi_prob = tf.exp(self.actor.policy_dist.logp(batch_actions))
        oldpi_prob = tf.stop_gradient(oldpi_prob)

        oldpi_param = self.actor.policy_dist.get_param()
        self.old_dist.set_param(oldpi_param)

        self.train_actor(
            batch_states,
            batch_actions,
            adv,
            oldpi_prob,
            backtrack_iters,
            backtrack_coeff,
        )

        for _ in range(train_critic_iters):
            self.train_critic(batch_rewards, batch_states)

    def learn(
        self,
        env,
        train_episodes=200,
        test_episodes=100,
        max_steps=200,
        save_interval=10,
        gamma=0.9,
        mode="train",
        render=False,
        batch_size=32,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        train_critic_iters=80,
        plot_func=None,
    ):
        """
        learn function

        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param gamma: reward discount factor
        :param mode: train or test
        :param render: render each step
        :param batch_size: update batch size
        :param backtrack_iters: Maximum number of steps allowed in the backtracking line search
        :param backtrack_coeff: How far back to step during backtracking line search
        :param train_critic_iters: critic update iteration steps

        :return: None
        """

        t0 = time.time()

        if mode == "train":
            print(
                "Training...  | Algorithm: {}  | Environment: {}".format(
                    self.name, env.spec.id
                )
            )

            reward_buffer = []
            for ep in range(1, train_episodes + 1):
                state, _ = env.reset()
                buffer_states, buffer_actions, buffer_rewards = [], [], []
                ep_rewards_sum = 0

                for t in range(max_steps):  # in one episode
                    action = self.get_action(state)

                    next_state, reward, done, truncated, _ = env.step(action)

                    buffer_states.append(state)
                    buffer_actions.append(action)
                    buffer_rewards.append(reward)

                    state = next_state

                    ep_rewards_sum += reward

                    # Update
                    if (t + 1) % batch_size == 0 or t == max_steps - 1 or done:
                        if done:
                            value = 0
                        else:
                            try:
                                value = self.get_value(next_state)
                            except:
                                value = self.get_v(
                                    next_state[np.newaxis, :]
                                )  # for raw-pixel input

                        discounted_reward = []

                        # Discount the rewards
                        for r in buffer_rewards[::-1]:
                            value = r + gamma * value
                            discounted_reward.append(value)
                        discounted_reward.reverse()
                        bs = buffer_states
                        ba, br = (
                            buffer_actions,
                            np.array(discounted_reward)[:, np.newaxis],
                        )
                        buffer_states, buffer_actions, buffer_rewards = [], [], []
                        self.update(
                            bs,
                            ba,
                            br,
                            train_critic_iters,
                            backtrack_iters,
                            backtrack_coeff,
                        )

                    if done:
                        break

                print(
                    "Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}".format(
                        ep, train_episodes, ep_rewards_sum, time.time() - t0
                    )
                )

                reward_buffer.append(ep_rewards_sum)
                if plot_func is not None:
                    plot_func(reward_buffer)
                if ep and not ep % save_interval:
                    self.save_ckpt(env_name=env.spec.id)
                    plot_save_log(reward_buffer, self.name, env.spec.id)

            self.save_ckpt(env_name=env.spec.id)
            plot_save_log(reward_buffer, self.name, env.spec.id)


def main(args):
    env = gym.make(args.env_name).unwrapped
    hidden_layers = [64] * args.hidden_layers

    critic = ValueNetwork(env.observation_space, hidden_layers)

    actor = StochasticPolicyNetwork(
        env.observation_space, env.action_space, hidden_layers
    )

    net_list = critic, actor

    critic_lr = args.critic_lr
    optimizers_list = [tf._optimizers.Adam(critic_lr)]

    # Create the model
    model = TRPO(net_list, optimizers_list, damping_coeff=0.1, cg_iters=10, delta=0.01)

    model.learn(
        env,
        mode="train",
        render=False,
        train_episodes=args.epochs,
        max_steps=200,
        save_interval=100,
        gamma=0.9,
        batch_size=args.batch_size,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        train_critic_iters=80,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1500)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"\nUsing TRPO. Environment: {args.env_name}, learning rate: {args.lr}")
    main(args)
