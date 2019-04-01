import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from collections import deque, namedtuple
from agents.dqn import DQNAgent
from agents.policy import EpsGreedyQPolicy
from agents.memory import Memory
import random

if __name__ == '__main__':
    env = gym.make('CartPole-v0')  # ゲームを指定して読み込む
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    actions = np.arange(nb_actions)
    policy = EpsGreedyQPolicy(eps=1.0, eps_decay_rate=0.99, min_eps=0.01)
    memory = Memory(limit=50000, maxlen=1)
    # 初期観測情報
    obs = env.reset()
    # エージェントの初期化
    agent = DQNAgent(actions=actions, memory=memory, update_interval=200, train_interval=1, batch_size=32,
                     memory_interval=1, observation=obs, input_shape=[len(obs)], id=1, name=None, training=True, policy=policy)
    agent.compile()

    result = []
    nb_epsiodes = 500   # エピソード数
    for episode in range(nb_epsiodes):
        agent.reset()
        observation =  env.reset() # 環境の初期化
        observation = deepcopy(observation)
        agent.observe(observation)
        done = False
        while not done:
            # env.render() # 表示
            action = deepcopy(agent.act())
            observation, reward, done, info = env.step(action) #　アクションを実行した結果の状態、報酬、ゲームをクリアしたかどうか、その他の情報を返す
            observation = deepcopy(observation)
            agent.get_reward(reward, done)
            agent.observe(observation)
            if done:
                break

        # 評価
        agent.training = False
        observation = env.reset() # 環境の初期化
        agent.observe(observation)
        done = False
        step = 0
        while not done:
            # env.render() # 表示
            step+=1
            action = agent.act()
            observation, reward, done, info = env.step(action)
            agent.observe(observation)
            if done:
                print("Episode {}: {} steps".format(episode, step))
                result.append(step)
                break

        agent.act()
        agent.get_reward(0, False)
        agent.training = True

    x = np.arange(len(result))
    plt.ylabel("time")
    plt.xlabel("episode")
    plt.plot(x, result)
    plt.savefig("result.png")
