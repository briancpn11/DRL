# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:01:40 2020

@author: pingn
"""

# For actor_critic_keras.py

import numpy as np
from actor_critic_keras import Agent
from utils import plotLearning
import gym

if __name__ == "__main__":
  env = gym.make('LunarLander-v2')
  n_episodes = 100
  agent = Agent(alpha = 0.00001, beta = 0.00005)

  score_history = []

  for i in range(n_episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
      env.render()
      action = agent.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      score += reward
      observation = observation_
      agent.learn(observation, action, reward, observation_, done)

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

  filename = 'lunar.png'
  plotLearning(score_history, filename=filename, window = 10)