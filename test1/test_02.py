import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Breakout-v4')
obs = np.array(env.reset())
plt.title('Breakout-v4')
plt.imshow(obs)