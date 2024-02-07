# 主要的入口文件，用于初始化和运行整个训练流程。
from agent import DQNAgent
from environment import Environment
from train import train
import gymnasium as gym

if __name__ == "__main__":
    # 初始化环境和代理
    env = gym.make('BreakoutNoFrameskip-v4')
    env = Environment(env, w=84, h=84, num_stack=4)
    agent = DQNAgent(input_size=env.observation_space.shape, output_size=env.action_space.n)

# 训练代理
    train(env, agent, num_episodes=1000)