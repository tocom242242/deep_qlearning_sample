import numpy as np
from copy import deepcopy
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal')

def sample_batch_indexes(low, high, size):
    if high - low >= size:
        try:
            r = range(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class Memory():
    """
        Experience replay用のメモリ 
    """
    def __init__(self, limit, maxlen):
        self.limit = limit
        # self.actions = RingBuffer(limit)
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)
        self.maxlen = maxlen
        self.recent_observations = deque(maxlen=maxlen)

    def sample(self, batch_size, batch_idxs=None):
        batch_idxs = sample_batch_indexes(0, len(self.observations) - 1, size=batch_size)
        # シンプルに[(s, a)->(s1, r)]の経験をサンプルリング
        for (i, idx) in enumerate(batch_idxs):
            terminal = self.terminals[idx-1]
            while terminal:
                idx = sample_batch_indexes(0, len(self.observations)-1, size=1)[0]
                batch_idxs[i] = idx
                terminal = self.terminals[idx-1]

        experiences = []
        for idx in batch_idxs:
            state0 = self.observations[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            state1 = self.observations[idx+1]
            experiences.append(Experience(state0=state0, action=action, reward=reward,state1=state1, terminal=terminal))

        return experiences

    def append(self, observation, action, reward, terminal=False, training=True):
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            self.recent_observations.append(observation)
