import tensorflow as tf
import numpy as np
from copy import deepcopy
from agents.network import Network
from abc import ABCMeta, abstractmethod
from collections import deque, namedtuple

class Agent(metaclass=ABCMeta):
    """Abstract Agent Class"""

    def __init__(self, id=None, name=None, training=None, policy=None):
        self.id = id
        self.name = name
        self.training = training
        self.policy = policy
        self.reward_history = []

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def get_reward(self, reward):
        pass

    @abstractmethod
    def observe(self, next_state):
        pass

class DQNAgent(Agent):
    """
        DQNエージェント
    """
    def __init__(self, gamma=0.99, alpha_decay_rate=0.999, actions=None, memory=None, memory_interval=1,train_interval=1, 
                 batch_size=32, update_interval=10, nb_steps_warmup=100, observation=None,
                 input_shape=None, 
                 **kwargs):

        super().__init__(**kwargs)
        self.actions = actions
        self.gamma = gamma
        self.state = observation
        self.alpha_decay_rate = alpha_decay_rate
        self.recent_observation = observation
        self.update_interval = update_interval
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.recent_action_id = 0
        self.nb_steps_warmup = nb_steps_warmup
        self.sess = tf.InteractiveSession()
        self.net = Network(self.sess)
        self.model_inputs, self.model_outputs, self.model_max_outputs, self.model = self.build_model(input_shape, len(self.actions))
        self.target_model_inputs, self.target_model_outputs, self.target_model_max_outputs, self.target_model= self.build_model(input_shape, len(self.actions))
        target_model_weights = self.target_model.trainable_weights
        model_weights = self.model.trainable_weights
        self.update_target_model = [target_model_weights[i].assign(model_weights[i]) for i in range(len(target_model_weights))]
        self.train_interval = train_interval
        self.step = 0

    def build_model(self, input_shape, nb_output):
        model = tf.keras.models.Sequential()
        inputs = tf.placeholder(dtype=tf.float32, shape = [None,]+input_shape, name="input")
        model.add(tf.keras.layers.Dense(16, activation="relu", input_shape =[None,]+input_shape))
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(nb_output))
        outputs = model(inputs)
        max_outputs = tf.reduce_max(outputs, reduction_indices=1)
        return inputs, outputs, max_outputs, model

    def compile(self, optimizer=None):
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None, len(self.actions)], name="target_q")
        self.inputs= tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        mask = tf.one_hot(indices=self.inputs, depth=len(self.actions), on_value=1.0, off_value=0.0, name="action_one_hot")
        self.pred_q = tf.multiply(self.model_outputs, mask)
        self.delta = tf.pow(self.targets - self.pred_q, 2)

        # huber loss
        self.clipped_error = tf.where(self.delta < 1.0,
                                      0.5 * tf.square(self.delta),
                                      self.delta - 0.5, name="clipped_error")
        self.loss = tf.reduce_mean(self.clipped_error, name="loss")

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        else:
            optimizer = optimizer
        self.train = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def update_target_model_hard(self):
        """ copy q-network to target network """
        self.sess.run(self.update_target_model)

    def train_on_batch(self, state_batch, action_batch, targets):
        self.sess.run(self.train, feed_dict={self.model_inputs:state_batch, self.inputs:action_batch, self.targets:targets})

    def predict_on_batch(self, state1_batch):
        q_values = self.sess.run(self.target_model_max_outputs, feed_dict={self.target_model_inputs:state1_batch})
        return q_values

    def compute_q_values(self, state):
        q_values = self.sess.run(self.model_outputs, feed_dict={self.model_inputs:[state]})
        return q_values[0]

    def get_reward(self, reward, terminal):
        self.reward_history.append(reward)
        if self.training:
            self._update_q_value(reward, terminal)

        self.policy.decay_eps_rate()
        self.step += 1

    def _update_q_value(self, reward, terminal):
        self.backward(reward, terminal)

    def backward(self, reward, terminal):
        if self.step % self.memory_interval == 0:
            """ store experience """
            self.memory.append(self.recent_observation, self.recent_action_id, reward, terminal=terminal, training=self.training)

        if (self.step > self.nb_steps_warmup) and (self.step % self.train_interval == 0):
            experiences = self.memory.sample(self.batch_size)

            state0_batch = []
            reward_batch = []
            action_batch = []
            state1_batch = []
            terminal_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            reward_batch = np.array(reward_batch)
            target_q_values = np.array(self.predict_on_batch(state1_batch))   # compute maxQ'(s')

            targets = np.zeros((self.batch_size, len(self.actions)))

            discounted_reward_batch = (self.gamma * target_q_values)
            discounted_reward_batch *= terminal_batch
            Rs = reward_batch + discounted_reward_batch    # target = r + γ maxQ'(s')

            for idx, (target, R, action) in enumerate(zip(targets, Rs, action_batch)):
                target[action] = R  

            self.train_on_batch(state0_batch, action_batch, targets)

        if self.step % self.update_interval == 0:
            """ update target network """
            self.update_target_model_hard()

    def act(self):
        action_id = self.forward()
        action = self.actions[action_id]
        return action

    def forward(self):
        state = self.recent_observation
        q_values = self.compute_q_values(state)
        if self.training:
            action_id = self.policy.select_action(q_values=q_values)
        else:
            action_id = self.policy.select_greedy_action(q_values=q_values)

        self.recent_action_id = action_id
        return action_id

    def observe(self, next_state):
        self.recent_observation = next_state

    def reset(self):
        self.recent_observation = None
        self.recent_action_id = None
